"""
Classroom Orchestrator — v1 skeleton
=====================================
Phase state + salience routing for the classroom API.

The orchestrator sits between event diffing and SSE broadcast in
`classroom_api.py`'s `push_state` endpoint. It classifies each outgoing
event as AMBIENT / BROADCAST / DIRECTED based on the current session
phase and a declarative routing policy.

This module is deliberately standalone — no FastAPI, no Supabase, no
classroom_api.py imports. That means it can be unit-tested and
demo-run anywhere with plain Python.

Design notes: see tasks/orchestrator-v1.md for full context.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ── Primitives ───────────────────────────────────────────────────────────────


class Phase(str, Enum):
    """Session phase. UNKNOWN is pass-through (no orchestration)."""

    UNKNOWN = "unknown"
    ARRIVAL = "arrival"
    LECTURE = "lecture"
    ACTIVITY = "activity"
    CONCLUDE = "conclude"
    DEPARTURE = "departure"


class Salience(str, Enum):
    """How an event travels after routing."""

    AMBIENT = "ambient"        # written to log, NOT SSE-broadcast
    BROADCAST = "broadcast"    # SSE fan-out to all subscribers
    DIRECTED = "directed"      # SSE to specific project IDs only


@dataclass(frozen=True)
class Rule:
    """One routing rule within a phase policy."""

    event_types: frozenset[str]
    salience: Salience
    targets: tuple[str, ...] = ()   # only used for DIRECTED

    @classmethod
    def make(
        cls,
        event_types: Iterable[str],
        salience: Salience,
        targets: Iterable[str] = (),
    ) -> "Rule":
        return cls(
            event_types=frozenset(event_types),
            salience=salience,
            targets=tuple(targets),
        )


# ── Routing policy ───────────────────────────────────────────────────────────
#
# The whole declarative policy fits below. Read top-to-bottom to understand
# what the room does during each phase of a class session. Event types mirror
# the ones emitted by classroom_api.py's `detect_changes()` function.

POLICIES: dict[Phase, list[Rule]] = {
    Phase.ARRIVAL: [
        # People coming in is the main signal; everything else is background.
        Rule.make({"person_change"}, Salience.BROADCAST),
        Rule.make({"fatigue_change"}, Salience.AMBIENT),       # nobody's tired yet
        Rule.make({"whiteboard_change"}, Salience.AMBIENT),    # probably leftovers
        Rule.make({"probe_classification"}, Salience.AMBIENT),
        Rule.make({"anomaly_change"}, Salience.AMBIENT),
    ],
    Phase.LECTURE: [
        # Mid-lecture person traffic is noise. Whiteboard and probe are the
        # signal. Fatigue goes DM-only so we don't shame students publicly.
        Rule.make({"person_change"}, Salience.AMBIENT),
        Rule.make({"fatigue_change"}, Salience.DIRECTED, targets=("prof-dm",)),
        Rule.make({"whiteboard_change"}, Salience.BROADCAST),
        Rule.make({"probe_classification"}, Salience.BROADCAST),
        Rule.make({"anomaly_change"}, Salience.DIRECTED, targets=("prof-dm",)),
    ],
    Phase.ACTIVITY: [
        # Movement, collaboration, energy: everything matters, room is live.
        Rule.make({"person_change"}, Salience.BROADCAST),
        Rule.make({"fatigue_change"}, Salience.BROADCAST),
        Rule.make({"whiteboard_change"}, Salience.AMBIENT),    # activity, not notes
        Rule.make({"probe_classification"}, Salience.BROADCAST),
        Rule.make({"anomaly_change"}, Salience.BROADCAST),
    ],
    Phase.CONCLUDE: [
        # Wrap-up notes matter most. Person traffic is people packing up.
        Rule.make({"whiteboard_change"}, Salience.BROADCAST),
        Rule.make({"person_change"}, Salience.AMBIENT),
        Rule.make({"fatigue_change"}, Salience.AMBIENT),
        Rule.make({"probe_classification"}, Salience.AMBIENT),
        Rule.make({"anomaly_change"}, Salience.AMBIENT),
    ],
    Phase.DEPARTURE: [
        # Count people out, log the rest.
        Rule.make({"person_change"}, Salience.BROADCAST),
        Rule.make({"fatigue_change"}, Salience.AMBIENT),
        Rule.make({"whiteboard_change"}, Salience.AMBIENT),
        Rule.make({"probe_classification"}, Salience.AMBIENT),
        Rule.make({"anomaly_change"}, Salience.AMBIENT),
    ],
    Phase.UNKNOWN: [],  # empty -> pass-through (everything defaults to BROADCAST)
}


# Event types that are always BROADCAST regardless of phase. These are
# orchestrator-internal announcements that every subscriber needs to hear.
ALWAYS_BROADCAST: frozenset[str] = frozenset({"phase_change"})


def _normalize_targets(value) -> tuple[str, ...]:
    """Normalize a target or target list from an event envelope."""
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, IterableABC) and not isinstance(value, (bytes, dict)):
        return tuple(str(target) for target in value if target)
    return ()


def _explicit_targets(event: dict) -> tuple[str, ...]:
    """Return event-level routing targets if the publisher supplied them.

    This is the small "conversation" escape hatch: project/client events can
    ask for a named subscriber directly, while phase policy still handles the
    ambient/broadcast defaults for ordinary detector events.
    """
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    return (
        _normalize_targets(event.get("targets"))
        or _normalize_targets(event.get("target"))
        or _normalize_targets(payload.get("targets"))
        or _normalize_targets(payload.get("target"))
    )


# ── Routed events container ──────────────────────────────────────────────────


@dataclass
class RoutedEvents:
    """Result of routing a batch of events under the current phase policy."""

    ambient: list[dict] = field(default_factory=list)
    broadcast: list[dict] = field(default_factory=list)
    directed: dict[str, list[dict]] = field(default_factory=dict)

    def for_subscriber(self, subscriber_id: str) -> list[dict]:
        """Events this subscriber should receive over SSE.

        Ambient events are NEVER delivered over SSE — they live in the log only.
        """
        return list(self.broadcast) + list(self.directed.get(subscriber_id, []))

    def all_for_log(self) -> list[dict]:
        """Every event that should be written to the durable event log,
        regardless of salience. Log captures everything."""
        collected = list(self.ambient) + list(self.broadcast)
        for directed_list in self.directed.values():
            collected.extend(directed_list)
        return collected

    def counts(self) -> dict[str, int]:
        return {
            "ambient": len(self.ambient),
            "broadcast": len(self.broadcast),
            "directed": sum(len(v) for v in self.directed.values()),
        }


# ── Routing function (pure) ──────────────────────────────────────────────────


def _rule_for_event(event_type: str, phase: Phase) -> Optional[Rule]:
    """First rule that matches the event type for the given phase, else None."""
    for rule in POLICIES.get(phase, []):
        if event_type in rule.event_types:
            return rule
    return None


def route(events: list[dict], phase: Phase) -> RoutedEvents:
    """Classify each event's salience based on the active phase policy.

    Rules:
    - Events in ALWAYS_BROADCAST are always BROADCAST regardless of phase.
    - If the phase has no matching rule, the event defaults to BROADCAST
      (pass-through). That makes UNKNOWN phase a full no-op.
    - AMBIENT events are kept in the log only.
    - DIRECTED events go to their rule's target list.
    """
    routed = RoutedEvents()

    for event in events:
        event_type = event.get("event_type", "")

        if event_type in ALWAYS_BROADCAST:
            routed.broadcast.append(event)
            continue

        targets = _explicit_targets(event)
        if targets:
            for target in targets:
                routed.directed.setdefault(target, []).append(event)
            continue

        rule = _rule_for_event(event_type, phase)

        if rule is None:
            # No matching rule -> pass-through. Safe default.
            routed.broadcast.append(event)
            continue

        if rule.salience is Salience.AMBIENT:
            routed.ambient.append(event)
        elif rule.salience is Salience.BROADCAST:
            routed.broadcast.append(event)
        elif rule.salience is Salience.DIRECTED:
            for target in rule.targets:
                routed.directed.setdefault(target, []).append(event)

    return routed


# ── Phase state (mutable, persisted) ─────────────────────────────────────────

_STATE_LOCK = threading.Lock()
_STATE: dict = {
    "phase": Phase.UNKNOWN.value,
    "started_at": time.time(),
}

_DEFAULT_STATE_FILE = Path.home() / "oak-projects" / "classroom_session.json"


def _default_state_path() -> Path:
    return _DEFAULT_STATE_FILE


def load_phase_state(path: Optional[Path] = None) -> None:
    """Load phase state from disk. Silently does nothing if the file doesn't exist."""
    global _STATE
    path = path or _default_state_path()
    try:
        raw = json.loads(Path(path).read_text())
        with _STATE_LOCK:
            _STATE = {
                "phase": raw.get("phase", Phase.UNKNOWN.value),
                "started_at": float(raw.get("started_at", time.time())),
            }
        logger.info("Loaded phase state from %s: %s", path, _STATE["phase"])
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("Failed to load phase state from %s: %s", path, exc)


def save_phase_state(path: Optional[Path] = None) -> None:
    """Persist current phase state to disk."""
    path = path or _default_state_path()
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with _STATE_LOCK:
            snapshot = dict(_STATE)
        Path(path).write_text(json.dumps(snapshot, indent=2))
    except Exception as exc:
        logger.warning("Failed to save phase state to %s: %s", path, exc)


def current_phase() -> Phase:
    with _STATE_LOCK:
        return Phase(_STATE["phase"])


def phase_status() -> dict:
    """Serializable phase status for GET /phase."""
    with _STATE_LOCK:
        started = _STATE["started_at"]
        phase = _STATE["phase"]
    return {
        "phase": phase,
        "started_at": started,
        "duration_sec": max(0.0, time.time() - started),
    }


def set_phase(new_phase: Phase, *, persist_to: Optional[Path] = None) -> dict:
    """Transition to a new phase. Returns a `phase_change` event dict
    that callers should include in the next broadcast.

    The event is marked ALWAYS_BROADCAST, so `route()` will always send
    it out regardless of the target phase's policy.
    """
    if not isinstance(new_phase, Phase):
        new_phase = Phase(new_phase)

    with _STATE_LOCK:
        old_phase = _STATE["phase"]
        _STATE["phase"] = new_phase.value
        _STATE["started_at"] = time.time()

    save_phase_state(persist_to)

    return {
        "event_type": "phase_change",
        "payload": {
            "old_phase": old_phase,
            "new_phase": new_phase.value,
            "started_at": _STATE["started_at"],
        },
    }


def policy_snapshot() -> dict:
    """Serializable view of all policies, for `GET /phase/policy`
    and white-paper diagrams."""
    out: dict[str, list[dict]] = {}
    for phase, rules in POLICIES.items():
        out[phase.value] = [
            {
                "event_types": sorted(r.event_types),
                "salience": r.salience.value,
                "targets": list(r.targets),
            }
            for r in rules
        ]
    return out
