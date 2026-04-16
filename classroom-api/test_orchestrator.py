"""
Unit tests for the orchestrator skeleton.

Run from classroom-api/:
    python -m pytest test_orchestrator.py -v

No pytest? Fall back to running the file directly — it has a tiny
self-contained runner at the bottom.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import orchestrator as orch
from orchestrator import Phase, Rule, Salience


# ── Helpers ──────────────────────────────────────────────────────────────────


def ev(event_type: str, **payload) -> dict:
    return {"camera_id": "orbit", "event_type": event_type, "payload": payload}


# ── Pass-through behavior ────────────────────────────────────────────────────


def test_unknown_phase_is_pure_passthrough():
    """UNKNOWN phase must broadcast every event — it's the safe default."""
    events = [
        ev("person_change", old_count=0, new_count=1),
        ev("fatigue_change", fatigue_detected=True),
        ev("whiteboard_change", text_detected=True),
        ev("made_up_event_type"),
    ]
    routed = orch.route(events, Phase.UNKNOWN)
    assert len(routed.broadcast) == 4
    assert routed.ambient == []
    assert routed.directed == {}


def test_unmatched_event_type_defaults_to_broadcast():
    """An event type not named in any rule for the active phase
    should still pass through as BROADCAST (safe default)."""
    routed = orch.route([ev("something_new")], Phase.LECTURE)
    assert len(routed.broadcast) == 1
    assert routed.ambient == []


# ── Phase-specific routing ───────────────────────────────────────────────────


def test_arrival_broadcasts_person_change():
    routed = orch.route([ev("person_change", old_count=0, new_count=2)], Phase.ARRIVAL)
    assert len(routed.broadcast) == 1
    assert routed.ambient == []


def test_arrival_logs_fatigue_as_ambient():
    routed = orch.route([ev("fatigue_change", fatigue_detected=True)], Phase.ARRIVAL)
    assert routed.broadcast == []
    assert len(routed.ambient) == 1


def test_lecture_suppresses_person_traffic_to_ambient():
    """During a lecture, people walking around is noise, not signal."""
    routed = orch.route([ev("person_change", old_count=10, new_count=11)], Phase.LECTURE)
    assert routed.broadcast == []
    assert len(routed.ambient) == 1


def test_lecture_directs_fatigue_to_prof_only():
    """Fatigue during lecture goes DM-only — no public shaming."""
    routed = orch.route([ev("fatigue_change", fatigue_detected=True)], Phase.LECTURE)
    assert routed.broadcast == []
    assert routed.ambient == []
    assert "prof-dm" in routed.directed
    assert len(routed.directed["prof-dm"]) == 1


def test_lecture_broadcasts_whiteboard_and_probe():
    routed = orch.route(
        [
            ev("whiteboard_change", text_detected=True, text=["hi"]),
            ev("probe_classification", new_class="presentation"),
        ],
        Phase.LECTURE,
    )
    assert len(routed.broadcast) == 2


def test_activity_broadcasts_everything_that_matters():
    routed = orch.route(
        [
            ev("person_change", old_count=15, new_count=12),
            ev("fatigue_change", fatigue_detected=False),
            ev("probe_classification", new_class="discussion"),
        ],
        Phase.ACTIVITY,
    )
    assert len(routed.broadcast) == 3


def test_activity_makes_whiteboard_ambient():
    """In activity mode, writing on the board is background, not the signal."""
    routed = orch.route(
        [ev("whiteboard_change", text_detected=True, text=["group notes"])],
        Phase.ACTIVITY,
    )
    assert len(routed.ambient) == 1
    assert routed.broadcast == []


def test_conclude_prioritizes_whiteboard():
    routed = orch.route(
        [
            ev("whiteboard_change", text_detected=True, text=["Read: Ch 3"]),
            ev("person_change", old_count=12, new_count=10),
        ],
        Phase.CONCLUDE,
    )
    assert len(routed.broadcast) == 1
    assert routed.broadcast[0]["event_type"] == "whiteboard_change"
    assert len(routed.ambient) == 1


def test_departure_broadcasts_person_changes():
    routed = orch.route(
        [ev("person_change", old_count=10, new_count=0)],
        Phase.DEPARTURE,
    )
    assert len(routed.broadcast) == 1


# ── ALWAYS_BROADCAST events ──────────────────────────────────────────────────


def test_phase_change_events_always_broadcast():
    """phase_change must fan out to every subscriber, regardless of phase."""
    for phase in Phase:
        routed = orch.route([ev("phase_change", old_phase="arrival", new_phase="lecture")], phase)
        assert len(routed.broadcast) == 1, f"phase_change suppressed in {phase}"
        assert routed.ambient == []


# ── RoutedEvents helpers ─────────────────────────────────────────────────────


def test_for_subscriber_delivers_broadcast_plus_directed():
    """A named subscriber sees broadcasts + anything directed at it."""
    routed = orch.route(
        [
            ev("whiteboard_change", text_detected=True),   # BROADCAST in LECTURE
            ev("fatigue_change", fatigue_detected=True),    # DIRECTED to prof-dm
            ev("person_change", old_count=10, new_count=11),  # AMBIENT, not delivered
        ],
        Phase.LECTURE,
    )

    prof_view = routed.for_subscriber("prof-dm")
    assert len(prof_view) == 2  # whiteboard (broadcast) + fatigue (directed)
    types = {e["event_type"] for e in prof_view}
    assert types == {"whiteboard_change", "fatigue_change"}

    student_view = routed.for_subscriber("some-student-project")
    assert len(student_view) == 1  # whiteboard only
    assert student_view[0]["event_type"] == "whiteboard_change"


def test_event_target_routes_directed_to_named_subscriber():
    """Any event can carry a target for client-to-client conversation."""
    event = {
        "camera_id": "__project__:smart-stage",
        "event_type": "timer.offer",
        "target": "gesture-timer",
        "payload": {"minutes": 5},
    }
    routed = orch.route([event], Phase.LECTURE)

    assert routed.broadcast == []
    assert routed.ambient == []
    assert "gesture-timer" in routed.directed
    assert routed.for_subscriber("gesture-timer") == [event]
    assert routed.for_subscriber("some-other-project") == []


def test_payload_targets_route_to_multiple_subscribers():
    event = ev(
        "fiducial.request",
        targets=["horizon", "gravity"],
        marker_family="tag36h11",
    )
    routed = orch.route([event], Phase.ACTIVITY)

    assert routed.broadcast == []
    assert set(routed.directed.keys()) == {"horizon", "gravity"}
    assert routed.for_subscriber("horizon") == [event]
    assert routed.for_subscriber("gravity") == [event]


def test_all_for_log_captures_every_event():
    """The durable log gets everything, even ambient."""
    routed = orch.route(
        [
            ev("whiteboard_change"),
            ev("fatigue_change"),
            ev("person_change"),
        ],
        Phase.LECTURE,
    )
    assert len(routed.all_for_log()) == 3


def test_counts_sum_to_input_size():
    events = [
        ev("whiteboard_change"),
        ev("fatigue_change"),
        ev("person_change"),
        ev("probe_classification"),
    ]
    routed = orch.route(events, Phase.LECTURE)
    counts = routed.counts()
    assert counts["ambient"] + counts["broadcast"] + counts["directed"] == len(events)


# ── Phase state + persistence ────────────────────────────────────────────────


def test_set_phase_emits_phase_change_event():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "session.json"
        event = orch.set_phase(Phase.LECTURE, persist_to=path)
        assert event["event_type"] == "phase_change"
        assert event["payload"]["new_phase"] == "lecture"
        assert orch.current_phase() is Phase.LECTURE


def test_phase_state_persists_and_reloads():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "session.json"

        orch.set_phase(Phase.ACTIVITY, persist_to=path)

        raw = json.loads(path.read_text())
        assert raw["phase"] == "activity"
        assert "started_at" in raw

        # Simulate a restart by flipping state to something else
        orch.set_phase(Phase.UNKNOWN, persist_to=path)
        assert orch.current_phase() is Phase.UNKNOWN

        # Reload from the file we saved earlier — but we overwrote it,
        # so instead write a fresh file and reload
        path.write_text(json.dumps({"phase": "conclude", "started_at": 1000.0}))
        orch.load_phase_state(path)
        assert orch.current_phase() is Phase.CONCLUDE


def test_phase_status_shape():
    orch.set_phase(Phase.ARRIVAL, persist_to=Path(tempfile.mktemp()))
    status = orch.phase_status()
    assert set(status.keys()) == {"phase", "started_at", "duration_sec"}
    assert status["phase"] == "arrival"
    assert status["duration_sec"] >= 0


# ── Policy snapshot ──────────────────────────────────────────────────────────


def test_policy_snapshot_is_serializable():
    snap = orch.policy_snapshot()
    # Must round-trip through JSON — this is what `GET /phase/policy` returns.
    serialized = json.dumps(snap)
    reloaded = json.loads(serialized)
    assert set(reloaded.keys()) == {p.value for p in Phase}
    # Lecture policy should mention prof-dm as a DIRECTED target
    lecture_rules = reloaded["lecture"]
    directed = [r for r in lecture_rules if r["salience"] == "directed"]
    assert any("prof-dm" in r["targets"] for r in directed)


# ── Fallback runner (no pytest required) ─────────────────────────────────────


def _run_all():
    import inspect
    tests = [
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn) and inspect.isfunction(fn)
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except AssertionError as exc:
            print(f"  ✗ {name}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ✗ {name}: {type(exc).__name__}: {exc}")
            failed += 1
    total = passed + failed
    print(f"\n{passed}/{total} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(_run_all())
