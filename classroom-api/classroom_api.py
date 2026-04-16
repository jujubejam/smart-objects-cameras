"""
Smart Classroom API
===================
Thin FastAPI service wrapping Supabase for the Smart Classroom.
Students hit these endpoints. Detectors push updates here.

Runs alongside the V-JEPA server (port 8765) on the GPU PC.

Endpoints:
    GET  /health                        — service status
    GET  /state                         — all cameras + global room mode
    GET  /state/{camera_id}             — single camera state
    GET  /mode                          — just the room mode + person count
    GET  /events                        — query classroom events
    GET  /projects                      — list student projects
    GET  /projects/{project_id}/events  — a project's events
    POST /push/state                    — detectors push state updates
    POST /projects/{project_id}/events  — students publish events
    GET  /subscribe/state               — SSE stream of state changes
    GET  /subscribe/events              — SSE stream of classroom events

Usage:
    pip install fastapi uvicorn supabase sse-starlette python-dotenv
    python classroom_api.py

Environment:
    SUPABASE_URL          — your Supabase project URL
    SUPABASE_SERVICE_KEY  — service_role key (not anon)
    CLASSROOM_API_KEY     — shared secret for detector auth
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel
import uvicorn

try:
    from dotenv import load_dotenv
    load_dotenv()
    # Also try loading from the standard oak-projects location
    env_file = Path.home() / "oak-projects" / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

try:
    from sse_starlette.sse import EventSourceResponse
    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False

import orchestrator
from orchestrator import Phase, Salience, RoutedEvents

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("classroom_api")

# ── Config ───────────────────────────────────────────────────────────────────

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
CLASSROOM_API_KEY = os.getenv("CLASSROOM_API_KEY", "changeme")
PORT = int(os.getenv("CLASSROOM_API_PORT", "8766"))
API_DIR = Path(__file__).resolve().parent
REPO_ROOT = API_DIR.parent
CONSOLE_FILE = API_DIR / "console.html"
HEARTBEAT_FILE = API_DIR / "student_heartbeat.html"
CONTRACT_DOC = REPO_ROOT / "docs" / "CLASSROOM_CONTRACTS.md"
LOCAL_DATA_DIR = REPO_ROOT / ".local" / "classroom-api"
LOCAL_SNAPSHOT_FILE = LOCAL_DATA_DIR / "snapshot.json"
WEEK2_REPO = Path(
    os.getenv(
        "SMARTOBJECTS_WEEK2_REPO",
        str(REPO_ROOT.parent / "smartobjects-labs-week2"),
    )
)
WEEK2_OBJECTS_FILE = WEEK2_REPO / "server" / "objects.json"
MEDIA_FILES = {
    "gravity_photo.jpg": REPO_ROOT / "gravity_photo.jpg",
    "horizon_photo.jpg": REPO_ROOT / "horizon_photo.jpg",
    "gravity_depth_combined.jpg": REPO_ROOT / "gravity_depth_combined.jpg",
    "horizon_depth_overlay.jpg": REPO_ROOT / "horizon_depth_overlay.jpg",
}

# ── Supabase client ──────────────────────────────────────────────────────────

supabase = None

if SUPABASE_AVAILABLE and SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        log.info(f"Supabase connected: {SUPABASE_URL}")
    except Exception as e:
        log.warning(f"Supabase connection failed: {e}")
else:
    if not SUPABASE_URL:
        log.warning("SUPABASE_URL not set — running in local-only mode")
    if not SUPABASE_AVAILABLE:
        log.warning("supabase-py not installed — pip install supabase")

# ── In-memory state (works even without Supabase) ───────────────────────────

_camera_states: dict[str, dict] = {}  # camera_id -> latest state
_previous_states: dict[str, dict] = {}  # for change detection
_classroom_events: list[dict] = []  # local event log when Supabase is absent
_project_events: list[dict] = []  # local project event log when Supabase is absent
_classroom_event_seq = 0
_project_event_seq = 0
_project_status: dict[str, dict] = {}
_state_subscribers: list[asyncio.Queue] = []  # SSE subscribers
# event subscribers carry a subscriber_id so the orchestrator can route
# DIRECTED events to specific recipients (e.g. "prof-dm")
_event_subscribers: list[tuple[str, asyncio.Queue]] = []

PROJECT_CONTRACTS: dict[str, dict] = {
    "gesture-timer": {
        "display_name": "Gesture Classroom Timer",
        "student_name": "Phil",
        "description": "Timer that can be offered, started, stopped, and report done.",
        "consumes": ["phase_change", "room_mode_change", "timer.offer", "timer.command"],
        "emits": ["timer.started", "timer.done", "timer.cancelled", "timer.declined"],
        "commands": {
            "timer.offer": {"minutes": "number", "reason": "string", "mode": "string"},
            "timer.command": {"action": "start | stop | pause | resume", "minutes": "number?"},
        },
        "mock": {"can_run_without_hardware": True, "default_response": "timer.started"},
    },
    "smart-stage": {
        "display_name": "Smart Stage",
        "student_name": "Gordon",
        "description": "Stage setup, recording, captions, and presentation affordances.",
        "consumes": ["person_change", "probe_classification", "room_mode_change", "fiducial.detected"],
        "emits": ["timer.offer", "fiducial.request", "stage.ready", "stage.needs_setup"],
        "mock": {"can_run_without_hardware": True, "default_response": "stage.ready"},
    },
    "horizon": {
        "display_name": "Horizon Fiducial Camera",
        "student_name": "System",
        "description": "Camera agent that can run fiducial detection for tagged classroom objects.",
        "consumes": ["fiducial.request", "capability.request"],
        "emits": ["fiducial.detected", "fiducial.zone_entered", "fiducial.lost", "capability.available"],
        "mock": {"can_run_without_hardware": True, "default_response": "fiducial.detected"},
    },
    "gravity": {
        "display_name": "Gravity Fiducial Camera",
        "student_name": "System",
        "description": "Second camera agent for depth, object, and fiducial positioning.",
        "consumes": ["fiducial.request", "capability.request"],
        "emits": ["fiducial.detected", "fiducial.zone_entered", "depth.observed", "capability.available"],
        "mock": {"can_run_without_hardware": True, "default_response": "fiducial.detected"},
    },
    "assignment-tracker": {
        "display_name": "Assignment Progress Tracking",
        "student_name": "Shuyang Tian",
        "description": "Assignment scanning and reminder agent.",
        "consumes": ["whiteboard_change", "phase_change", "assignment.request"],
        "emits": ["assignment.suggested", "assignment.created", "assignment.reminder"],
        "mock": {"can_run_without_hardware": True, "default_response": "assignment.suggested"},
    },
    "prof-dm": {
        "display_name": "Professor Direct Channel",
        "student_name": "Instructor",
        "description": "Private directed route for sensitive alerts and confirmations.",
        "consumes": ["fatigue_change", "anomaly_change", "timer.done", "agent.message"],
        "emits": ["phase.command", "agent.confirmed"],
        "mock": {"can_run_without_hardware": True, "default_response": "agent.confirmed"},
    },
    "calmball": {
        "display_name": "CalmBall",
        "student_name": "Ramon Naula",
        "description": "Stress regulation squeeze ball and calming sound object.",
        "consumes": ["room_mode_change", "person_change"],
        "emits": ["calm.activated", "calm.deactivated"],
        "mock": {"can_run_without_hardware": True, "default_response": "calm.activated"},
    },
    "gus-mode": {
        "display_name": "Gus Mode",
        "student_name": "Juju",
        "description": "Virtual dog presence via VIAM rover livestream projected life-size in the classroom.",
        "consumes": ["person_change", "room_mode_change"],
        "emits": ["gus.present", "gus.excited", "gus.sleeping"],
        "mock": {"can_run_without_hardware": True, "default_response": "gus.present"},
    },
    "echodesk": {
        "display_name": "EchoDesk",
        "student_name": "Kathy Choi",
        "description": "Shared question board and conversational desk object.",
        "consumes": ["room_mode_change", "agent.message"],
        "emits": ["question.submitted", "question.displayed"],
        "mock": {"can_run_without_hardware": True, "default_response": "question.displayed"},
    },
    "focus-beam": {
        "display_name": "Focus Beam",
        "student_name": "Feifey",
        "description": "Assistive classroom spotlight following instructor gestures.",
        "consumes": ["probe_classification", "room_mode_change"],
        "emits": ["beam.pointed", "beam.off"],
        "mock": {"can_run_without_hardware": True, "default_response": "beam.pointed"},
    },
    "forest-classroom": {
        "display_name": "Forest in the Classroom",
        "student_name": "Sophie Lee",
        "description": "Living forest projection reacting to presence and emotion.",
        "consumes": ["person_change", "room_mode_change"],
        "emits": ["forest.mood_set", "forest.pulse"],
        "mock": {"can_run_without_hardware": True, "default_response": "forest.mood_set"},
    },
    "imprint": {
        "display_name": "Imprint",
        "student_name": "Darren Chia",
        "description": "Reads handwriting from surfaces and saves it to notes.",
        "consumes": ["whiteboard_change", "fiducial.detected"],
        "emits": ["handwriting.captured", "note.saved"],
        "mock": {"can_run_without_hardware": True, "default_response": "handwriting.captured"},
    },
    "seren-room": {
        "display_name": "A Room (Context-Aware Classroom)",
        "student_name": "Yuxuan (Seren)",
        "description": "Reads the room and changes ambience by occupancy and activity mode.",
        "consumes": ["person_change", "probe_classification", "room_mode_change"],
        "emits": ["ambience.changed", "room.adapted"],
        "mock": {"can_run_without_hardware": True, "default_response": "ambience.changed"},
    },
}

EVENT_SCHEMAS: dict[str, dict[str, str]] = {
    "timer.offer": {
        "minutes": "number",
        "reason": "string",
        "mode": "string?",
    },
    "timer.command": {
        "action": "string",
        "minutes": "number?",
    },
    "timer.started": {
        "minutes": "number",
        "started_by": "string?",
        "ends_at": "string?",
    },
    "timer.done": {
        "minutes": "number",
        "completed": "boolean?",
    },
    "timer.cancelled": {
        "reason": "string?",
    },
    "timer.declined": {
        "reason": "string?",
    },
    "fiducial.request": {
        "marker_family": "string",
        "needed_for": "string?",
        "zones": "list?",
    },
    "fiducial.detected": {
        "tag_id": "number",
        "name": "string",
        "x": "number",
        "y": "number",
        "zone": "string?",
        "confidence": "number?",
    },
    "fiducial.zone_entered": {
        "tag_id": "number",
        "name": "string?",
        "zone": "string",
    },
    "capability.request": {
        "capability": "string",
        "needed_by": "string?",
        "reason": "string?",
    },
    "capability.available": {
        "capability": "string",
        "provider": "string?",
    },
    "assignment.suggested": {
        "title": "string",
        "due": "string?",
    },
    "assignment.created": {
        "title": "string",
        "due": "string?",
    },
    "assignment.reminder": {
        "title": "string",
        "due": "string?",
    },
    "agent.message": {
        "text": "string",
        "tone": "string?",
    },
    "stage.ready": {
        "message": "string?",
    },
    "stage.needs_setup": {
        "reason": "string",
    },
    "calm.activated": {
        "sound": "string?",
        "intensity": "number?",
    },
    "calm.deactivated": {
        "reason": "string?",
    },
    "question.submitted": {
        "text": "string",
        "student": "string?",
    },
    "question.displayed": {
        "text": "string",
        "display_id": "string?",
    },
    "beam.pointed": {
        "x": "number?",
        "y": "number?",
        "target": "string?",
    },
    "beam.off": {
        "reason": "string?",
    },
    "forest.mood_set": {
        "mood": "string",
        "intensity": "number?",
    },
    "forest.pulse": {
        "count": "number?",
    },
    "handwriting.captured": {
        "text": "string",
        "surface": "string?",
    },
    "note.saved": {
        "text": "string",
        "title": "string?",
    },
    "ambience.changed": {
        "mode": "string",
        "intensity": "number?",
    },
    "room.adapted": {
        "adaptation": "string",
        "reason": "string?",
    },
    "gus.present": {
        "activity": "string?",
        "energy": "number?",
    },
    "gus.excited": {
        "trigger": "string?",
    },
    "gus.sleeping": {
        "reason": "string?",
    },
}

EVENT_EXAMPLES: dict[str, dict] = {
    "timer.offer": {"minutes": 5, "reason": "activity started", "mode": "group"},
    "timer.command": {"action": "start", "minutes": 5},
    "timer.started": {"minutes": 5, "started_by": "gesture"},
    "timer.done": {"minutes": 5, "completed": True},
    "fiducial.request": {
        "marker_family": "tag36h11",
        "needed_for": "smart-stage",
        "zones": ["podium", "whiteboard"],
    },
    "fiducial.detected": {
        "tag_id": 3,
        "name": "Whiteboard Marker",
        "x": 0.52,
        "y": 0.08,
        "zone": "whiteboard",
        "confidence": 0.92,
    },
    "assignment.suggested": {
        "title": "Connect your project to the classroom bus",
        "due": "Monday",
    },
    "agent.message": {"text": "Can you use the timer?", "tone": "helpful"},
}

LOCAL_PROJECTS: dict[str, dict] = {
    "seren-room": {
        "display_name": "A Room (Context-Aware Classroom)",
        "student_name": "Yuxuan (Seren)",
        "description": "Reads the room and changes ambience by occupancy and activity mode.",
        "subscribed_events": ["person_change", "probe_classification", "room_mode_change"],
    },
    "echodesk": {
        "display_name": "EchoDesk",
        "student_name": "Kathy Choi",
        "description": "Shared question board and conversational desk object.",
        "subscribed_events": ["room_mode_change", "agent.message"],
    },
    "calmball": {
        "display_name": "CalmBall",
        "student_name": "Ramon Naula",
        "description": "Stress regulation squeeze ball and calming sound object.",
        "subscribed_events": ["room_mode_change", "person_change"],
    },
    "gus-mode": {
        "display_name": "Gus Mode",
        "student_name": "Juju",
        "description": "Virtual presence of Gus the dog — VIAM rover streams live video, segmented and projected life-size onto the classroom wall.",
        "subscribed_events": ["person_change", "room_mode_change"],
    },
    "focus-beam": {
        "display_name": "Focus Beam",
        "student_name": "Feifey",
        "description": "Assistive classroom spotlight following instructor gestures.",
        "subscribed_events": ["probe_classification", "room_mode_change"],
    },
    "forest-classroom": {
        "display_name": "Forest in the Classroom",
        "student_name": "Sophie Lee",
        "description": "Living forest projection reacting to presence and emotion.",
        "subscribed_events": ["person_change", "room_mode_change"],
    },
    "imprint": {
        "display_name": "Imprint",
        "student_name": "Darren Chia",
        "description": "Reads handwriting from surfaces and saves it to notes.",
        "subscribed_events": ["whiteboard_change", "fiducial.detected"],
    },
}

for _project_id, _contract in PROJECT_CONTRACTS.items():
    LOCAL_PROJECTS.setdefault(
        _project_id,
        {
            "display_name": _contract["display_name"],
            "student_name": _contract["student_name"],
            "description": _contract["description"],
            "subscribed_events": _contract.get("consumes", []),
        },
    )
    _project_status[_project_id] = {
        "project_id": _project_id,
        "status": "mock",
        "capabilities": sorted({
            *_contract.get("consumes", []),
            *_contract.get("emits", []),
            *([
                "timer"
            ] if "timer" in _project_id else []),
            *([
                "fiducials"
            ] if _project_id in {"horizon", "gravity"} else []),
        }),
        "consumes": _contract.get("consumes", []),
        "emits": _contract.get("emits", []),
        "message": "registered mock contract",
        "last_seen": None,
        "meta": {"mock": True},
    }

# Load persisted phase state at module load so requests see the right phase
# even before __main__ runs (e.g., when served by `uvicorn classroom_api:app`)
orchestrator.load_phase_state()

# ── Pydantic models ──────────────────────────────────────────────────────────


class PushStateRequest(BaseModel):
    camera_id: str
    person_detected: Optional[bool] = None
    person_count: Optional[int] = None
    fatigue_detected: Optional[bool] = None
    anomaly_score: Optional[float] = None
    anomaly_level: Optional[str] = None
    predicted_class: Optional[str] = None
    prediction_confidence: Optional[float] = None
    class_probs: Optional[dict] = None
    whiteboard_text: Optional[list[str]] = None
    whiteboard_text_detected: Optional[bool] = None
    detector_host: Optional[str] = None
    detector_user: Optional[str] = None


class ProjectEventRequest(BaseModel):
    event_type: str
    payload: dict = {}
    target: Optional[Union[str, list[str]]] = None


class ProjectHeartbeatRequest(BaseModel):
    status: str = "online"
    capabilities: list[str] = []
    consumes: list[str] = []
    emits: list[str] = []
    message: Optional[str] = None
    meta: dict = {}


class ValidateEventRequest(BaseModel):
    project_id: str
    event_type: str
    payload: dict = {}
    target: Optional[Union[str, list[str]]] = None


class MockScenarioRequest(BaseModel):
    scenario: str = "full"


class CapabilityRouteRequest(BaseModel):
    capability: str
    needed_by: Optional[str] = None
    reason: Optional[str] = None
    prefer: Optional[list[str]] = None


class LabsImportRequest(BaseModel):
    provider: str = "horizon"
    target: Optional[Union[str, list[str]]] = None


class SetPhaseRequest(BaseModel):
    phase: str  # must match a Phase enum value


# ── Room mode computation ────────────────────────────────────────────────────

def compute_room_mode(states: dict[str, dict]) -> dict:
    """
    Aggregate across all cameras to determine the global classroom mode.

    Priority: presentation > focus > group > duo > solo > empty
    """
    if not states:
        return {"room_mode": "unknown", "total_persons": 0,
                "whiteboard_active": False, "probe_classes": []}

    active_states = [s for s in states.values() if s.get("running", True)]

    total_persons = sum(s.get("person_count", 0) for s in active_states)
    any_whiteboard = any(s.get("whiteboard_text_detected", False) for s in active_states)

    # Gather probe predictions with reasonable confidence
    probe_classes = [
        s.get("predicted_class")
        for s in active_states
        if s.get("predicted_class") and s.get("prediction_confidence", 0) > 0.5
    ]

    # Determine mode
    if any(c == "presentation" for c in probe_classes):
        mode = "presentation"
    elif any_whiteboard:
        mode = "focus"
    elif total_persons == 0:
        mode = "empty"
    elif total_persons == 1:
        mode = "solo"
    elif total_persons == 2:
        mode = "duo"
    else:
        mode = "group"

    return {
        "room_mode": mode,
        "total_persons": total_persons,
        "whiteboard_active": any_whiteboard,
        "probe_classes": probe_classes,
    }


# ── Change detection ─────────────────────────────────────────────────────────

TRACKED_FIELDS = [
    "person_count", "person_detected", "fatigue_detected",
    "predicted_class", "anomaly_level", "whiteboard_text_detected",
]


def detect_changes(camera_id: str, new_state: dict) -> list[dict]:
    """Compare new state to previous; return list of events to emit."""
    old = _previous_states.get(camera_id, {})
    events = []

    if old.get("person_count") != new_state.get("person_count"):
        events.append({
            "camera_id": camera_id,
            "event_type": "person_change",
            "payload": {
                "old_count": old.get("person_count", 0),
                "new_count": new_state.get("person_count", 0),
                "detected": new_state.get("person_detected", False),
            },
        })

    if old.get("predicted_class") != new_state.get("predicted_class"):
        events.append({
            "camera_id": camera_id,
            "event_type": "probe_classification",
            "payload": {
                "old_class": old.get("predicted_class"),
                "new_class": new_state.get("predicted_class"),
                "confidence": new_state.get("prediction_confidence", 0),
                "class_probs": new_state.get("class_probs", {}),
            },
        })

    if old.get("fatigue_detected") != new_state.get("fatigue_detected"):
        events.append({
            "camera_id": camera_id,
            "event_type": "fatigue_change",
            "payload": {
                "fatigue_detected": new_state.get("fatigue_detected", False),
            },
        })

    if old.get("anomaly_level") != new_state.get("anomaly_level"):
        events.append({
            "camera_id": camera_id,
            "event_type": "anomaly_change",
            "payload": {
                "old_level": old.get("anomaly_level"),
                "new_level": new_state.get("anomaly_level"),
                "score": new_state.get("anomaly_score", 0),
            },
        })

    if old.get("whiteboard_text_detected") != new_state.get("whiteboard_text_detected"):
        events.append({
            "camera_id": camera_id,
            "event_type": "whiteboard_change",
            "payload": {
                "text_detected": new_state.get("whiteboard_text_detected", False),
                "text": new_state.get("whiteboard_text", []),
            },
        })

    # Check if room mode changed
    old_mode_info = compute_room_mode(_camera_states)
    # Temporarily update state for new computation
    old_camera = _camera_states.get(camera_id)
    _camera_states[camera_id] = new_state
    new_mode_info = compute_room_mode(_camera_states)
    # Restore if needed (the caller will set it after)
    if old_camera is not None:
        _camera_states[camera_id] = old_camera
    else:
        del _camera_states[camera_id]

    if old_mode_info.get("room_mode") != new_mode_info.get("room_mode"):
        events.append({
            "camera_id": camera_id,
            "event_type": "room_mode_change",
            "payload": {
                "old_mode": old_mode_info.get("room_mode"),
                "new_mode": new_mode_info.get("room_mode"),
                "total_persons": new_mode_info.get("total_persons"),
                "trigger": "state_push",
            },
        })

    return events


# ── SSE helpers ──────────────────────────────────────────────────────────────

async def broadcast_state(state_snapshot: dict):
    """Push state update to all SSE subscribers."""
    dead = []
    for q in _state_subscribers:
        try:
            q.put_nowait(state_snapshot)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _state_subscribers.remove(q)


async def broadcast_events_routed(routed: RoutedEvents):
    """Deliver events to SSE subscribers according to their salience.

    - BROADCAST events go to every subscriber.
    - DIRECTED events go only to subscribers whose subscriber_id is in the
      rule's targets list.
    - AMBIENT events are NOT delivered over SSE — they live in the log only.
    """
    dead: list[tuple[str, asyncio.Queue]] = []
    for sub_id, q in _event_subscribers:
        events_for_sub = routed.for_subscriber(sub_id)
        for event in events_for_sub:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append((sub_id, q))
                break
    for item in dead:
        if item in _event_subscribers:
            _event_subscribers.remove(item)


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Classroom API",
    description="Shared data layer for the ixD Smart Classroom",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Student projects can be anywhere
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth helpers ─────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def console_root():
    if not CONSOLE_FILE.exists():
        raise HTTPException(status_code=404, detail="console.html not found")
    return FileResponse(CONSOLE_FILE)


@app.get("/console", include_in_schema=False)
def console_page():
    if not CONSOLE_FILE.exists():
        raise HTTPException(status_code=404, detail="console.html not found")
    return FileResponse(CONSOLE_FILE)


@app.get("/heartbeat", include_in_schema=False)
def heartbeat_page():
    if not HEARTBEAT_FILE.exists():
        raise HTTPException(status_code=404, detail="student_heartbeat.html not found")
    return FileResponse(HEARTBEAT_FILE)


@app.get("/media/{filename}", include_in_schema=False)
def console_media(filename: str):
    path = MEDIA_FILES.get(filename)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="media not found")
    return FileResponse(path)


@app.get("/contract-doc", include_in_schema=False)
def contract_doc():
    if not CONTRACT_DOC.exists():
        raise HTTPException(status_code=404, detail="contract doc not found")
    return FileResponse(CONTRACT_DOC, media_type="text/markdown")


def verify_detector_key(x_api_key: str = Header(None)):
    """Verify the shared detector API key."""
    if x_api_key != CLASSROOM_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def verify_project_key(project_id: str, x_api_key: str = Header(None)):
    """Verify a student project's API key."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")

    # Instructor console and mock drivers can use the shared classroom key.
    if x_api_key == CLASSROOM_API_KEY:
        return

    if supabase:
        result = supabase.table("student_projects").select("api_key").eq(
            "project_id", project_id
        ).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
        if result.data[0]["api_key"] != x_api_key:
            raise HTTPException(status_code=401, detail="Invalid project API key")
    else:
        # Local-only mode: accept any key
        pass


# ── READ ENDPOINTS ───────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_local_snapshot() -> None:
    """Persist local-only state so demos survive API restarts."""
    if supabase:
        return
    try:
        LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "camera_states": _camera_states,
            "previous_states": _previous_states,
            "classroom_events": _classroom_events[:500],
            "project_events": _project_events[:500],
            "classroom_event_seq": _classroom_event_seq,
            "project_event_seq": _project_event_seq,
            "project_status": _project_status,
            "saved_at": _now_iso(),
        }
        LOCAL_SNAPSHOT_FILE.write_text(json.dumps(payload, indent=2))
    except Exception as exc:
        log.warning("Failed to save local snapshot: %s", exc)


def load_local_snapshot() -> None:
    """Load persisted local-only state if Supabase is absent."""
    global _classroom_event_seq, _project_event_seq
    if supabase or not LOCAL_SNAPSHOT_FILE.exists():
        return
    try:
        raw = json.loads(LOCAL_SNAPSHOT_FILE.read_text())
        _camera_states.update(raw.get("camera_states", {}))
        _previous_states.update(raw.get("previous_states", {}))
        _classroom_events[:] = raw.get("classroom_events", [])[:500]
        _project_events[:] = raw.get("project_events", [])[:500]
        _classroom_event_seq = int(raw.get("classroom_event_seq", len(_classroom_events)))
        _project_event_seq = int(raw.get("project_event_seq", len(_project_events)))
        for project_id, status in raw.get("project_status", {}).items():
            _project_status[project_id] = status
        log.info("Loaded local classroom snapshot from %s", LOCAL_SNAPSHOT_FILE)
    except Exception as exc:
        log.warning("Failed to load local snapshot: %s", exc)


def _remember_classroom_event(event: dict, source: str) -> dict:
    """Store a classroom/detector event in the local in-memory event log."""
    global _classroom_event_seq
    _classroom_event_seq += 1
    enriched = dict(event)
    enriched.setdefault("id", _classroom_event_seq)
    enriched.setdefault("created_at", _now_iso())
    enriched.setdefault("source", source)
    enriched.setdefault("camera_id", "__system__")
    _classroom_events.insert(0, enriched)
    del _classroom_events[500:]
    save_local_snapshot()
    return enriched


def _remember_project_event(project_id: str, event: dict) -> dict:
    """Store a project/client event in the local in-memory event log."""
    global _project_event_seq
    _project_event_seq += 1
    enriched = dict(event)
    enriched.setdefault("id", _project_event_seq)
    enriched.setdefault("created_at", _now_iso())
    enriched.setdefault("source", "project")
    enriched.setdefault("project_id", project_id)
    _project_events.insert(0, enriched)
    del _project_events[500:]
    save_local_snapshot()
    return enriched


def _project_with_contract(project_id: str, project: dict) -> dict:
    out = dict(project)
    out.setdefault("project_id", project_id)
    out.setdefault("is_active", True)
    out.setdefault("config", {})
    contract = PROJECT_CONTRACTS.get(project_id)
    if contract:
        out["contract"] = contract
        out["config"] = {**out.get("config", {}), "contract": contract}
        out.setdefault("subscribed_events", contract.get("consumes", []))
    return out


def _local_projects() -> list[dict]:
    return [
        _project_with_contract(project_id, project)
        for project_id, project in sorted(LOCAL_PROJECTS.items())
    ]


def _filter_event_list(
    events: list[dict],
    *,
    limit: int,
    event_type: Optional[str] = None,
    camera_id: Optional[str] = None,
    project_id: Optional[str] = None,
    since: Optional[str] = None,
) -> list[dict]:
    filtered = []
    for event in events:
        if event_type and event.get("event_type") != event_type:
            continue
        if camera_id and event.get("camera_id") != camera_id:
            continue
        if project_id and event.get("project_id") != project_id:
            continue
        if since and event.get("created_at", "") < since:
            continue
        filtered.append(event)
    return filtered[:limit]


def _heartbeat_age(status: dict) -> Optional[float]:
    last_seen = status.get("last_seen")
    if not last_seen:
        return None
    try:
        return max(0.0, time.time() - datetime.fromisoformat(last_seen).timestamp())
    except Exception:
        return None


def _status_with_age(project_id: str, status: dict) -> dict:
    out = dict(status)
    out.setdefault("project_id", project_id)
    age = _heartbeat_age(out)
    out["age_sec"] = age
    out["is_live"] = (
        out.get("status") in {"online", "ready", "busy"}
        and age is not None
        and age < 120
    )
    return out


def project_status_snapshot() -> list[dict]:
    return [
        _status_with_age(project_id, status)
        for project_id, status in sorted(_project_status.items())
    ]


def capability_index() -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    for project_id, raw in _project_status.items():
        status = _status_with_age(project_id, raw)
        for capability in status.get("capabilities", []):
            index.setdefault(capability, []).append(status)
    for providers in index.values():
        providers.sort(key=lambda item: (not item.get("is_live"), item.get("project_id", "")))
    return index


def project_event_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in _project_events:
        project_id = event.get("project_id")
        if project_id:
            counts[project_id] = counts.get(project_id, 0) + 1
    return counts


def project_readiness_snapshot() -> dict:
    projects_by_id = {project["project_id"]: project for project in _local_projects()}
    for project_id in _project_status.keys():
        projects_by_id.setdefault(
            project_id,
            {
                "project_id": project_id,
                "display_name": project_id,
                "student_name": "external",
                "description": "Registered by heartbeat.",
                "subscribed_events": [],
            },
        )

    counts = project_event_counts()
    rows = []
    for project_id, project in sorted(projects_by_id.items()):
        status = _status_with_age(project_id, _project_status.get(project_id, {}))
        contract = PROJECT_CONTRACTS.get(project_id)
        has_heartbeat = bool(status.get("last_seen"))
        emitted_count = counts.get(project_id, 0)
        has_contract = bool(contract)
        emits = status.get("emits") or (contract.get("emits", []) if contract else [])
        has_declared_io = bool(
            status.get("capabilities")
            or status.get("consumes")
            or emits
            or project.get("subscribed_events")
        )

        if status.get("is_live"):
            level = "live"
        elif has_heartbeat:
            level = "stale"
        elif status.get("meta", {}).get("mock") or has_contract:
            level = "mock-ready"
        else:
            level = "needs-heartbeat"

        score = sum([
            1 if has_contract else 0,
            1 if has_declared_io else 0,
            1 if has_heartbeat else 0,
            1 if emitted_count > 0 else 0,
            1 if status.get("is_live") else 0,
        ])

        rows.append({
            "project_id": project_id,
            "display_name": project.get("display_name", project_id),
            "student_name": project.get("student_name"),
            "description": project.get("description"),
            "level": level,
            "score": score,
            "score_max": 5,
            "has_contract": has_contract,
            "has_declared_io": has_declared_io,
            "has_heartbeat": has_heartbeat,
            "is_live": bool(status.get("is_live")),
            "last_seen": status.get("last_seen"),
            "age_sec": status.get("age_sec"),
            "status": status.get("status") or "missing",
            "capabilities": status.get("capabilities") or [],
            "consumes": status.get("consumes") or project.get("subscribed_events", []),
            "emits": emits,
            "event_count": emitted_count,
            "message": status.get("message"),
        })

    summary = {
        "total": len(rows),
        "live": sum(1 for row in rows if row["level"] == "live"),
        "stale": sum(1 for row in rows if row["level"] == "stale"),
        "mock_ready": sum(1 for row in rows if row["level"] == "mock-ready"),
        "needs_heartbeat": sum(1 for row in rows if row["level"] == "needs-heartbeat"),
    }
    return {
        "summary": summary,
        "projects": rows,
        "timestamp": _now_iso(),
    }


def build_project_nudges() -> dict:
    readiness = project_readiness_snapshot()
    nudges = []
    for project in readiness["projects"]:
        project_id = project["project_id"]
        if project["level"] == "needs-heartbeat":
            message = (
                f"{project_id}: start with the minimum heartbeat. Open "
                f"http://localhost:{PORT}/heartbeat, enter PROJECT_ID={project_id}, "
                "list what your project consumes/emits, and send one heartbeat."
            )
            action = "send-heartbeat"
        elif project["level"] == "stale":
            message = (
                f"{project_id}: your project checked in before, but it is stale. "
                "Restart the heartbeat or refresh the browser heartbeat page before critique."
            )
            action = "refresh-heartbeat"
        elif project["level"] == "mock-ready":
            message = (
                f"{project_id}: the room has a mock/contract path. Next, emit one real or mock event "
                "so your interaction leaves evidence in the bus."
            )
            action = "emit-evidence"
        else:
            message = (
                f"{project_id}: you are live. Capture one consume/emit loop and use the report as "
                "portfolio process evidence."
            )
            action = "capture-evidence"

        nudges.append({
            "project_id": project_id,
            "level": project["level"],
            "action": action,
            "message": message,
        })

    return {
        "summary": readiness["summary"],
        "nudges": nudges,
        "timestamp": _now_iso(),
    }


def find_project_readiness(project_id: str) -> Optional[dict]:
    for project in project_readiness_snapshot()["projects"]:
        if project["project_id"] == project_id:
            return project
    return None


def build_student_packet(project_id: str) -> dict:
    project = find_project_readiness(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Unknown project '{project_id}'")

    contract = PROJECT_CONTRACTS.get(project_id)
    examples = {}
    for event_type in project.get("emits", []):
        if event_type in EVENT_EXAMPLES:
            examples[event_type] = EVENT_EXAMPLES[event_type]

    heartbeat_command = (
        "set CLASSROOM_API=http://localhost:8766\n"
        f"set PROJECT_ID={project_id}\n"
        "set PROJECT_API_KEY=testkey\n"
        f"set CAPABILITIES={','.join(project.get('capabilities') or [project_id])}\n"
        f"set CONSUMES={','.join(project.get('consumes') or [])}\n"
        f"set EMITS={','.join(project.get('emits') or ['project.heartbeat'])}\n"
        "python student_heartbeat.py"
    )

    return {
        "project": project,
        "contract": contract,
        "examples": examples,
        "heartbeat_url": "http://localhost:8766/heartbeat",
        "console_url": "http://localhost:8766/console",
        "heartbeat_command": heartbeat_command,
        "next_actions": [
            item for item in build_project_nudges()["nudges"]
            if item["project_id"] == project_id
        ],
    }


def build_student_packet_markdown(project_id: str) -> str:
    packet = build_student_packet(project_id)
    project = packet["project"]
    lines = [
        f"# Smart Classroom Packet: {project_id}",
        "",
        f"Display name: {project.get('display_name')}",
        f"Student: {project.get('student_name') or 'TBD'}",
        f"Status: {project.get('level')} ({project.get('score')}/{project.get('score_max')})",
        "",
        "## Minimum Check-In",
        "",
        "Open:",
        "",
        packet["heartbeat_url"],
        "",
        "Or run:",
        "",
        "```bat",
        packet["heartbeat_command"],
        "```",
        "",
        "## What Your Project Listens For",
        "",
    ]
    consumes = project.get("consumes") or []
    lines.extend([f"- `{item}`" for item in consumes] or ["- Declare this with a heartbeat."])

    lines.extend(["", "## What Your Project Emits", ""])
    emits = project.get("emits") or []
    lines.extend([f"- `{item}`" for item in emits] or ["- Declare this with a heartbeat."])

    lines.extend(["", "## Next Action", ""])
    nudges = packet.get("next_actions") or []
    lines.extend([f"- {item['message']}" for item in nudges] or ["- Send a heartbeat."])

    if packet["examples"]:
        lines.extend(["", "## Example Payloads", ""])
        for event_type, payload in packet["examples"].items():
            lines.extend([
                f"### `{event_type}`",
                "",
                "```json",
                json.dumps(payload, indent=2),
                "```",
                "",
            ])

    lines.extend([
        "## Portfolio Evidence",
        "",
        "Your project does not need to be perfect today. It needs evidence:",
        "",
        "- heartbeat",
        "- one event you consume or intend to consume",
        "- one event you emit or intend to emit",
        "- a screenshot or screen recording of the room bus seeing your project",
        "",
    ])
    return "\n".join(lines)


def build_roster_csv() -> str:
    rows = project_readiness_snapshot()["projects"]
    headers = [
        "project_id",
        "display_name",
        "student_name",
        "level",
        "score",
        "score_max",
        "has_contract",
        "has_heartbeat",
        "is_live",
        "event_count",
        "last_seen",
        "capabilities",
    ]
    lines = [",".join(headers)]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, list):
                value = ";".join(map(str, value))
            text = str(value if value is not None else "")
            text = '"' + text.replace('"', '""') + '"'
            values.append(text)
        lines.append(",".join(values))
    return "\n".join(lines)


def build_demo_script_markdown() -> str:
    context = build_room_context()
    readiness = project_readiness_snapshot()
    return "\n".join([
        "# Smart Classroom Demo Script",
        "",
        "## 1. Open The Console",
        "",
        "http://localhost:8766/console",
        "",
        "Point out phase, room mode, feeling, live bus, contracts, and readiness.",
        "",
        "## 2. Reset And Run The Room",
        "",
        "Press `Run full script` or call:",
        "",
        "```bash",
        "curl -X POST http://localhost:8766/mock/scenario -H \"Content-Type: application/json\" -H \"X-API-Key: testkey\" -d \"{\\\"scenario\\\":\\\"full\\\"}\"",
        "```",
        "",
        "## 3. Show Student Check-In",
        "",
        "Open:",
        "",
        "http://localhost:8766/heartbeat",
        "",
        "Explain: heartbeat is the minimum proof that a project can participate.",
        "",
        "## 4. Show Labs / Prototype Bridge",
        "",
        "Press `Import labs objects` or call `/labs/import`.",
        "",
        "This turns object positions and zones into `fiducial.detected`, `fiducial.zone_entered`, and `labs.rule_triggered` bus events.",
        "",
        "## 5. Show Evidence",
        "",
        "Open:",
        "",
        "http://localhost:8766/showcase/report",
        "",
        f"Current phase: {context['phase']}",
        f"Current room feeling: {context['feeling']}",
        f"Readiness: {readiness['summary']}",
        "",
        "## 6. Teaching Line",
        "",
        "A polished portfolio artifact starts with a working interaction contract. If the hardware fails, the room still records the intended interaction and gives us something to critique.",
        "",
    ])


def choose_provider(capability: str, prefer: Optional[list[str]] = None) -> Optional[dict]:
    providers = capability_index().get(capability, [])
    if not providers:
        return None
    preferred = prefer or []
    for project_id in preferred:
        match = next((p for p in providers if p.get("project_id") == project_id), None)
        if match:
            return match
    live = next((p for p in providers if p.get("is_live")), None)
    return live or providers[0]


def update_project_status(project_id: str, req: ProjectHeartbeatRequest) -> dict:
    contract = PROJECT_CONTRACTS.get(project_id, {})
    capabilities = sorted(set(
        req.capabilities
        or contract.get("emits", [])
        or LOCAL_PROJECTS.get(project_id, {}).get("subscribed_events", [])
    ))
    status = {
        "project_id": project_id,
        "status": req.status,
        "capabilities": capabilities,
        "consumes": req.consumes or contract.get("consumes", []),
        "emits": req.emits or contract.get("emits", []),
        "message": req.message,
        "last_seen": _now_iso(),
        "meta": req.meta,
    }
    _project_status[project_id] = status
    LOCAL_PROJECTS.setdefault(
        project_id,
        {
            "display_name": project_id,
            "student_name": "external",
            "description": "Registered itself by heartbeat.",
            "subscribed_events": status["consumes"],
        },
    )
    save_local_snapshot()
    return _status_with_age(project_id, status)


def _matches_type(value, expected: str) -> bool:
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "list":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def validate_project_event_payload(
    project_id: str,
    event_type: str,
    payload: dict,
    target=None,
) -> dict:
    """Validate known event contracts without blocking unknown experiments."""
    contract = PROJECT_CONTRACTS.get(project_id)
    schema = EVENT_SCHEMAS.get(event_type)
    errors: list[dict] = []
    warnings: list[str] = []

    if not isinstance(payload, dict):
        errors.append({
            "field": "payload",
            "message": "payload must be a JSON object",
            "expected": "object",
        })
        payload = {}

    if contract and event_type not in contract.get("emits", []):
        warnings.append(
            f"{project_id} does not list {event_type} in its emits contract yet"
        )

    if schema:
        for field, expected in schema.items():
            optional = expected.endswith("?")
            base_type = expected[:-1] if optional else expected
            if field not in payload:
                if not optional:
                    errors.append({
                        "field": field,
                        "message": f"missing required field '{field}'",
                        "expected": base_type,
                    })
                continue
            if not _matches_type(payload[field], base_type):
                errors.append({
                    "field": field,
                    "message": f"field '{field}' should be {base_type}",
                    "expected": base_type,
                    "received": type(payload[field]).__name__,
                })
    else:
        warnings.append(
            f"{event_type} has no schema yet; event will be accepted as a custom event"
        )

    target_list = []
    if target:
        target_list = target if isinstance(target, list) else [target]
        unknown_targets = [
            item for item in target_list
            if item not in LOCAL_PROJECTS and item not in PROJECT_CONTRACTS
        ]
        if unknown_targets:
            warnings.append(
                "unknown target subscriber(s): " + ", ".join(map(str, unknown_targets))
            )

    return {
        "ok": not errors,
        "event_type": event_type,
        "project_id": project_id,
        "target": target,
        "errors": errors,
        "warnings": warnings,
        "example": EVENT_EXAMPLES.get(event_type),
        "known_schema": schema,
    }


def _raise_contract_error_if_needed(project_id: str, req: ProjectEventRequest) -> dict:
    validation = validate_project_event_payload(
        project_id=project_id,
        event_type=req.event_type,
        payload=req.payload,
        target=req.target,
    )
    if not validation["ok"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    f"{req.event_type} does not match the classroom contract. "
                    "Fix the payload fields below and send it again."
                ),
                **validation,
            },
        )
    return validation


def load_labs_config() -> dict:
    """Load the companion labs/prototype repo object-zone config if present."""
    if not WEEK2_OBJECTS_FILE.exists():
        return {
            "available": False,
            "path": str(WEEK2_OBJECTS_FILE),
            "objects": [],
            "zones": [],
            "rules": [],
        }
    try:
        raw = json.loads(WEEK2_OBJECTS_FILE.read_text())
        raw["available"] = True
        raw["path"] = str(WEEK2_OBJECTS_FILE)
        return raw
    except Exception as exc:
        return {
            "available": False,
            "path": str(WEEK2_OBJECTS_FILE),
            "error": str(exc),
            "objects": [],
            "zones": [],
            "rules": [],
        }


def object_zone(obj: dict, zones: list[dict]) -> Optional[str]:
    x = obj.get("x")
    y = obj.get("y")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    for zone in zones:
        if (
            zone.get("x", 0) <= x <= zone.get("x", 0) + zone.get("w", 0)
            and zone.get("y", 0) <= y <= zone.get("y", 0) + zone.get("h", 0)
        ):
            return zone.get("name")
    return None


def evaluate_labs_rules(config: dict) -> list[dict]:
    objects = config.get("objects", [])
    zones = config.get("zones", [])
    rules = config.get("rules", [])
    by_name = {obj.get("name"): obj for obj in objects}
    triggered = []

    for rule in rules:
        when = rule.get("when", {})
        if "object" in when and "in_zone" in when:
            obj = by_name.get(when["object"])
            if obj and object_zone(obj, zones) == when["in_zone"]:
                triggered.append({
                    "rule": rule.get("name"),
                    "actions": rule.get("then", {}),
                    "reason": f"{when['object']} in {when['in_zone']}",
                })

        if "min_objects_in_zone" in when and "zone" in when:
            count = sum(1 for obj in objects if object_zone(obj, zones) == when["zone"])
            if count >= when["min_objects_in_zone"]:
                triggered.append({
                    "rule": rule.get("name"),
                    "actions": rule.get("then", {}),
                    "reason": f"{count} objects in {when['zone']}",
                })

    return triggered


def build_room_context() -> dict:
    """Shared room context object that downstream agents can subscribe to."""
    mode_info = compute_room_mode(_camera_states)
    phase = orchestrator.current_phase().value
    total = mode_info["total_persons"]
    fatigue = any(s.get("fatigue_detected", False) for s in _camera_states.values())
    whiteboard = mode_info["whiteboard_active"]
    probe_classes = mode_info.get("probe_classes", [])

    if total == 0:
        feeling = "empty"
    elif fatigue:
        feeling = "strained"
    elif phase == "activity" or mode_info["room_mode"] == "group":
        feeling = "collaborative"
    elif whiteboard or mode_info["room_mode"] == "focus":
        feeling = "focused"
    elif mode_info["room_mode"] == "presentation":
        feeling = "attentive"
    else:
        feeling = "settling"

    capabilities = capability_index()
    active_capabilities = []
    if total > 0:
        active_capabilities.append("presence")
    if whiteboard:
        active_capabilities.append("whiteboard")
    if probe_classes:
        active_capabilities.append("jepa_probe")
    active_capabilities.extend(["timer", "fiducials", "directed_events"])
    active_capabilities.extend(capabilities.keys())

    merged_recent = list(_classroom_events) + list(_project_events)
    merged_recent.sort(key=lambda event: event.get("created_at", ""), reverse=True)
    recent = _filter_event_list(merged_recent, limit=8)

    return {
        "phase": phase,
        "room_mode": mode_info["room_mode"],
        "feeling": feeling,
        "temporal": {
            "phase": phase,
            "duration_sec": orchestrator.phase_status()["duration_sec"],
        },
        "social": {
            "total_persons": total,
            "group_present": total >= 3,
        },
        "task": {
            "whiteboard_active": whiteboard,
            "probe_classes": probe_classes,
        },
        "historical": {
            "recent_events": recent,
        },
        "active_capabilities": sorted(set(active_capabilities)),
        "providers": capabilities,
        "suggested_prompts": [
            "Do you want a timer for this activity?",
            "Should Horizon or Gravity look for fiducials?",
            "Should this alert go to the professor only?",
        ],
        "timestamp": _now_iso(),
    }


def build_showcase_report() -> dict:
    readiness = project_readiness_snapshot()
    nudges = build_project_nudges()
    context = build_room_context()
    events = get_bus_events(limit=25)["events"]
    labs = load_labs_config()
    labs_rules = evaluate_labs_rules(labs)
    return {
        "title": "Smart Classroom Runtime Report",
        "generated_at": _now_iso(),
        "context": context,
        "readiness": readiness,
        "nudges": nudges,
        "recent_events": events,
        "labs": {
            "available": labs.get("available", False),
            "path": labs.get("path"),
            "objects": labs.get("objects", []),
            "zones": labs.get("zones", []),
            "triggered_rules": labs_rules,
        },
        "contracts": PROJECT_CONTRACTS,
    }


def build_showcase_report_markdown(report: dict) -> str:
    context = report["context"]
    readiness = report["readiness"]
    lines = [
        "# Smart Classroom Runtime Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Room Context",
        "",
        f"- Phase: {context['phase']}",
        f"- Mode: {context['room_mode']}",
        f"- Feeling: {context['feeling']}",
        f"- People: {context['social']['total_persons']}",
        f"- Whiteboard active: {context['task']['whiteboard_active']}",
        f"- Active capabilities: {', '.join(context['active_capabilities'])}",
        "",
        "## Project Readiness",
        "",
        (
            f"- Total: {readiness['summary']['total']} | "
            f"Live: {readiness['summary']['live']} | "
            f"Stale: {readiness['summary']['stale']} | "
            f"Mock-ready: {readiness['summary']['mock_ready']} | "
            f"Needs heartbeat: {readiness['summary']['needs_heartbeat']}"
        ),
        "",
        "| Project | Level | Score | Last Seen | Capabilities |",
        "| --- | --- | ---: | --- | --- |",
    ]

    for project in readiness["projects"]:
        caps = ", ".join(project.get("capabilities", [])[:6])
        lines.append(
            "| "
            + " | ".join([
                str(project["project_id"]),
                str(project["level"]),
                f"{project['score']}/{project['score_max']}",
                str(project.get("last_seen") or "-"),
                caps or "-",
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Next Action Nudges",
        "",
    ])
    for nudge in report["nudges"]["nudges"]:
        lines.append(f"- {nudge['message']}")

    lines.extend([
        "",
        "## Recent Runtime Events",
        "",
    ])
    for event in report["recent_events"][:15]:
        owner = event.get("project_id") or event.get("camera_id") or event.get("source")
        lines.append(
            f"- `{event.get('event_type')}` from `{owner}` at {event.get('created_at')}"
        )

    lines.extend([
        "",
        "## Labs / Prototype Bridge",
        "",
        f"- Config available: {report['labs']['available']}",
        f"- Config path: {report['labs'].get('path')}",
        f"- Objects: {len(report['labs']['objects'])}",
        f"- Zones: {len(report['labs']['zones'])}",
        "",
    ])
    if report["labs"]["triggered_rules"]:
        lines.append("Triggered rules:")
        for rule in report["labs"]["triggered_rules"]:
            lines.append(f"- {rule.get('rule')}: {rule.get('reason')}")
    else:
        lines.append("Triggered rules: none")

    lines.extend([
        "",
        "## Teaching Frame",
        "",
        (
            "The portfolio artifact starts with a working interaction contract. "
            "If the final hardware or visual layer fails, the heartbeat, contract, "
            "mock path, and bus events still preserve the interaction design for critique."
        ),
        "",
    ])
    return "\n".join(lines)


load_local_snapshot()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "supabase": supabase is not None,
        "sse": SSE_AVAILABLE,
        "cameras": list(_camera_states.keys()),
        "phase": orchestrator.current_phase().value,
        "state_subscribers": len(_state_subscribers),
        "event_subscribers": len(_event_subscribers),
        "local_classroom_events": len(_classroom_events),
        "local_project_events": len(_project_events),
        "projects": len(LOCAL_PROJECTS),
        "status_entries": len(_project_status),
        "snapshot": str(LOCAL_SNAPSHOT_FILE),
        "labs_repo": str(WEEK2_REPO),
        "labs_config": WEEK2_OBJECTS_FILE.exists(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/state")
def get_state():
    """All cameras + aggregated global room mode."""
    mode_info = compute_room_mode(_camera_states)
    context = build_room_context()
    return {
        "cameras": _camera_states,
        "room_mode": mode_info["room_mode"],
        "phase": orchestrator.current_phase().value,
        "feeling": context["feeling"],
        "context": context,
        "total_persons": mode_info["total_persons"],
        "whiteboard_active": mode_info["whiteboard_active"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/state/{camera_id}")
def get_camera_state(camera_id: str):
    """Single camera state."""
    if camera_id not in _camera_states:
        # Try Supabase
        if supabase:
            result = supabase.table("classroom_state").select("*").eq(
                "camera_id", camera_id
            ).execute()
            if result.data:
                return result.data[0]
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")
    return _camera_states[camera_id]


@app.get("/mode")
def get_mode():
    """Just the room mode, person count, and per-camera detail."""
    mode_info = compute_room_mode(_camera_states)
    detail = {}
    for cam_id, state in _camera_states.items():
        detail[cam_id] = {
            "persons": state.get("person_count", 0),
            "probe": state.get("predicted_class", "unknown"),
            "confidence": state.get("prediction_confidence", 0),
        }
    return {
        "room_mode": mode_info["room_mode"],
        "total_persons": mode_info["total_persons"],
        "whiteboard_active": mode_info["whiteboard_active"],
        "probe_consensus": max(
            mode_info["probe_classes"], key=mode_info["probe_classes"].count
        ) if mode_info["probe_classes"] else None,
        "detail": detail,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/events")
def get_events(
    limit: int = Query(50, ge=1, le=500),
    event_type: Optional[str] = None,
    camera_id: Optional[str] = None,
    since: Optional[str] = None,
):
    """Query classroom events with optional filters."""
    if not supabase:
        return {
            "events": _filter_event_list(
                _classroom_events,
                limit=limit,
                event_type=event_type,
                camera_id=camera_id,
                since=since,
            ),
            "note": "local in-memory event log",
        }

    query = supabase.table("classroom_events").select("*").order(
        "created_at", desc=True
    ).limit(limit)

    if event_type:
        query = query.eq("event_type", event_type)
    if camera_id:
        query = query.eq("camera_id", camera_id)
    if since:
        query = query.gte("created_at", since)

    result = query.execute()
    return {"events": result.data}


@app.get("/projects")
def get_projects():
    """List all student projects (without API keys)."""
    if supabase:
        result = supabase.table("student_projects").select(
            "project_id, display_name, student_name, description, "
            "subscribed_events, is_active, config"
        ).execute()
        projects = [
            _project_with_contract(row["project_id"], row)
            for row in result.data
        ]
        seen = {p["project_id"] for p in projects}
        projects.extend(p for p in _local_projects() if p["project_id"] not in seen)
        return {"projects": projects}
    return {"projects": _local_projects(), "note": "local project registry"}


@app.get("/projects/{project_id}/events")
def get_project_events(
    project_id: str,
    limit: int = Query(20, ge=1, le=200),
):
    """A project's published events."""
    if not supabase:
        return {
            "events": _filter_event_list(_project_events, limit=limit, project_id=project_id),
            "note": "local in-memory project event log",
        }

    result = supabase.table("project_events").select("*").eq(
        "project_id", project_id
    ).order("created_at", desc=True).limit(limit).execute()

    return {"events": result.data}


# ── WRITE ENDPOINTS (detectors) ──────────────────────────────────────────────

@app.get("/contracts")
def get_contracts():
    """Project contracts for the mockable classroom bus."""
    return {
        "contract_version": "classroom-contracts/v0.1",
        "contracts": PROJECT_CONTRACTS,
        "event_schemas": EVENT_SCHEMAS,
        "examples": EVENT_EXAMPLES,
        "projects": _local_projects(),
    }


@app.get("/projects/status")
def get_projects_status():
    """Live/mock status and capabilities for project agents."""
    return {"projects": project_status_snapshot()}


@app.get("/projects/readiness")
def get_projects_readiness():
    """Critique-ready status: who has a contract, heartbeat, and evidence."""
    return project_readiness_snapshot()


@app.get("/projects/nudges")
def get_project_nudges():
    """Plain-language next actions for each student project."""
    return build_project_nudges()


@app.get("/projects/nudges.md", response_class=PlainTextResponse)
def get_project_nudges_markdown():
    """Markdown next actions that can be pasted into Discord or critique notes."""
    nudges = build_project_nudges()
    lines = [
        "# Smart Classroom Next Actions",
        "",
        (
            f"Live: {nudges['summary']['live']} | "
            f"Stale: {nudges['summary']['stale']} | "
            f"Mock-ready: {nudges['summary']['mock_ready']} | "
            f"Needs heartbeat: {nudges['summary']['needs_heartbeat']}"
        ),
        "",
    ]
    for item in nudges["nudges"]:
        lines.append(f"- {item['message']}")
    return "\n".join(lines)


@app.get("/projects/roster.csv")
def get_projects_roster_csv():
    """CSV roster/check-in export for grading or attendance."""
    return Response(
        build_roster_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=smart-classroom-roster.csv"},
    )


@app.get("/projects/{project_id}/packet")
def get_student_packet_json(project_id: str):
    """JSON packet for one project."""
    return build_student_packet(project_id)


@app.get("/projects/{project_id}/packet.md", response_class=PlainTextResponse)
def get_student_packet_markdown(project_id: str):
    """Markdown packet for one project."""
    return build_student_packet_markdown(project_id)


@app.post("/projects/{project_id}/heartbeat")
async def project_heartbeat(
    project_id: str,
    req: ProjectHeartbeatRequest,
    x_api_key: str = Header(None),
):
    """Let a student project announce that it is alive and what it can handle."""
    await verify_project_key(project_id, x_api_key)
    status = update_project_status(project_id, req)
    event = _remember_project_event(
        project_id,
        {
            "project_id": project_id,
            "event_type": "project.heartbeat",
            "payload": {
                "status": status["status"],
                "capabilities": status["capabilities"],
                "message": status.get("message"),
            },
            "created_at": status["last_seen"],
            "source": "project",
        },
    )
    await broadcast_events_routed(orchestrator.route([event], orchestrator.current_phase()))
    return {"ok": True, "status": status}


@app.get("/capabilities")
def get_capabilities():
    """Capability index: which project can currently handle what."""
    return {"capabilities": capability_index()}


@app.post("/capabilities/route")
def route_capability(req: CapabilityRouteRequest):
    """Resolve a capability request into the best current project target."""
    provider = choose_provider(req.capability, req.prefer)
    if not provider:
        raise HTTPException(
            status_code=404,
            detail=f"No registered provider for capability '{req.capability}'",
        )
    return {
        "ok": True,
        "capability": req.capability,
        "target": provider["project_id"],
        "provider": provider,
        "request": req.model_dump(),
    }


@app.post("/contracts/validate")
def validate_event_contract(req: ValidateEventRequest):
    """Validate an event payload and return fixable errors without publishing it."""
    return validate_project_event_payload(
        project_id=req.project_id,
        event_type=req.event_type,
        payload=req.payload,
        target=req.target,
    )


@app.get("/room/context")
def get_room_context():
    """The shared context object every agent can use instead of guessing."""
    return build_room_context()


@app.get("/bus/events")
def get_bus_events(
    limit: int = Query(100, ge=1, le=500),
    event_type: Optional[str] = None,
    source: Optional[str] = None,
    project_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    since: Optional[str] = None,
):
    """Merged local bus history for the console."""
    merged = list(_classroom_events) + list(_project_events)
    merged.sort(key=lambda event: event.get("created_at", ""), reverse=True)
    events = _filter_event_list(
        merged,
        limit=limit,
        event_type=event_type,
        project_id=project_id,
        camera_id=camera_id,
        since=since,
    )
    if source:
        events = [event for event in events if event.get("source") == source][:limit]
    return {"events": events}


@app.get("/showcase/report.json")
def get_showcase_report_json():
    """JSON evidence packet for critique, demo, or portfolio documentation."""
    return build_showcase_report()


@app.get("/showcase/report", response_class=PlainTextResponse)
def get_showcase_report():
    """Markdown evidence packet for critique, demo, or portfolio documentation."""
    return build_showcase_report_markdown(build_showcase_report())


@app.get("/showcase/demo-script", response_class=PlainTextResponse)
def get_demo_script():
    """One-page script for running the classroom demo/critique."""
    return build_demo_script_markdown()


@app.get("/labs/config")
def get_labs_config():
    """Read object/zone/rule config from the companion labs/prototype repo."""
    config = load_labs_config()
    config["triggered_rules"] = evaluate_labs_rules(config)
    return config


@app.post("/labs/import")
async def import_labs_objects(
    req: LabsImportRequest,
    x_api_key: str = Header(None),
):
    """Turn labs repo object positions into classroom bus fiducial events."""
    verify_detector_key(x_api_key)
    config = load_labs_config()
    if not config.get("available"):
        raise HTTPException(
            status_code=404,
            detail=f"Labs objects config not found at {WEEK2_OBJECTS_FILE}",
        )

    emitted = []
    zones = config.get("zones", [])
    provider = req.provider
    update_project_status(
        provider,
        ProjectHeartbeatRequest(
            status="online",
            capabilities=["fiducials", "fiducial.detected", "fiducial.zone_entered"],
            consumes=["fiducial.request", "capability.request"],
            emits=["fiducial.detected", "fiducial.zone_entered"],
            message="imported from labs object config",
            meta={"source": str(WEEK2_OBJECTS_FILE)},
        ),
    )

    for obj in config.get("objects", []):
        zone = object_zone(obj, zones)
        detected = {
            "tag_id": obj.get("id", 0),
            "name": obj.get("name", f"tag-{obj.get('id', '?')}"),
            "x": obj.get("x", 0.0),
            "y": obj.get("y", 0.0),
            "zone": zone,
            "confidence": 1.0,
            "source": "labs-config",
        }
        await _mock_publish(provider, "fiducial.detected", detected, target=req.target)
        emitted.append({"event_type": "fiducial.detected", "payload": detected})
        if zone:
            zone_event = {
                "tag_id": detected["tag_id"],
                "name": detected["name"],
                "zone": zone,
            }
            await _mock_publish(provider, "fiducial.zone_entered", zone_event, target=req.target)
            emitted.append({"event_type": "fiducial.zone_entered", "payload": zone_event})

    for rule in evaluate_labs_rules(config):
        await _mock_publish("smart-stage", "labs.rule_triggered", rule, target=req.target)
        emitted.append({"event_type": "labs.rule_triggered", "payload": rule})

    return {
        "ok": True,
        "provider": provider,
        "path": str(WEEK2_OBJECTS_FILE),
        "emitted": emitted,
    }


@app.post("/push/state")
async def push_state(req: PushStateRequest, x_api_key: str = Header(None)):
    """
    Detectors push state updates here.
    Upserts classroom_state, detects changes, emits events.
    """
    verify_detector_key(x_api_key)

    now = datetime.now(timezone.utc).isoformat()
    camera_id = req.camera_id

    # Build the state dict from non-None fields
    state = {"camera_id": camera_id, "updated_at": now, "running": True}
    for field, value in req.model_dump(exclude_none=True).items():
        if field != "camera_id":
            state[field] = value

    # Merge with existing state (so partial updates work)
    existing = _camera_states.get(camera_id, {})
    merged = {**existing, **state}

    # Compute room mode
    temp_states = {**_camera_states, camera_id: merged}
    mode_info = compute_room_mode(temp_states)
    merged["room_mode"] = mode_info["room_mode"]

    # Detect changes and emit events
    events = detect_changes(camera_id, merged)
    events = [_remember_classroom_event(event, "detector") for event in events]

    # Update in-memory state
    _previous_states[camera_id] = _camera_states.get(camera_id, {})
    _camera_states[camera_id] = merged
    save_local_snapshot()

    # Push to Supabase
    if supabase:
        try:
            supabase.table("classroom_state").upsert(
                merged, on_conflict="camera_id"
            ).execute()
        except Exception as e:
            log.error(f"Supabase upsert failed: {e}")

        for event in events:
            try:
                supabase.table("classroom_events").insert({
                    "camera_id": event["camera_id"],
                    "event_type": event["event_type"],
                    "payload": event["payload"],
                    "source": "detector",
                }).execute()
            except Exception as e:
                log.error(f"Supabase event insert failed: {e}")

    # Broadcast to SSE subscribers
    state_snapshot = get_state()
    await broadcast_state(state_snapshot)

    # Orchestrator routes events by salience based on the active phase.
    # AMBIENT events are persisted (above) but suppressed from SSE fan-out;
    # DIRECTED events go only to matching subscribers.
    active_phase = orchestrator.current_phase()
    routed = orchestrator.route(events, active_phase)
    await broadcast_events_routed(routed)

    counts = routed.counts()
    log.info(
        f"[{camera_id}] pushed — persons={merged.get('person_count', '?')}, "
        f"probe={merged.get('predicted_class', '?')}, "
        f"mode={merged.get('room_mode', '?')}, "
        f"phase={active_phase.value}, "
        f"events={len(events)} "
        f"(b={counts['broadcast']} a={counts['ambient']} d={counts['directed']})"
    )

    return {
        "ok": True,
        "camera_id": camera_id,
        "room_mode": merged["room_mode"],
        "phase": active_phase.value,
        "events_emitted": len(events),
        "routing": counts,
    }


# ── WRITE ENDPOINTS (students) ───────────────────────────────────────────────

@app.post("/mock/reset")
async def reset_mock_state(x_api_key: str = Header(None)):
    """Clear local room state and event memory for repeatable demos."""
    verify_detector_key(x_api_key)

    _camera_states.clear()
    _previous_states.clear()
    _classroom_events.clear()
    _project_events.clear()
    global _classroom_event_seq, _project_event_seq
    _classroom_event_seq = 0
    _project_event_seq = 0
    phase_event = orchestrator.set_phase(Phase.UNKNOWN)
    remembered = _remember_classroom_event(
        {
            **phase_event,
            "camera_id": "__orchestrator__",
        },
        "orchestrator",
    )
    await broadcast_state(get_state())
    await broadcast_events_routed(orchestrator.route([remembered], Phase.UNKNOWN))
    return {
        "ok": True,
        "phase": "unknown",
        "cameras": 0,
        "events": 1,
    }


@app.post("/projects/{project_id}/events")
async def publish_project_event(
    project_id: str,
    req: ProjectEventRequest,
    x_api_key: str = Header(None),
):
    """Students publish events from their projects.

    Project events are logged to `project_events` and also put on the live
    `/subscribe/events` stream so projects can talk to each other. If `target`
    is supplied, only that subscriber_id receives the event.
    """
    await verify_project_key(project_id, x_api_key)
    validation = _raise_contract_error_if_needed(project_id, req)

    now = datetime.now(timezone.utc).isoformat()
    event = {
        "project_id": project_id,
        "event_type": req.event_type,
        "payload": req.payload,
        "created_at": now,
        "source": "project",
    }
    if req.target:
        event["target"] = req.target
    event = _remember_project_event(project_id, event)

    result_id = None
    if supabase:
        try:
            result = supabase.table("project_events").insert({
                "project_id": project_id,
                "event_type": req.event_type,
                "payload": req.payload,
            }).execute()
            if result.data:
                result_id = result.data[0].get("id")
        except Exception as e:
            log.error(f"Supabase project event insert failed: {e}")

    routed = orchestrator.route([event], orchestrator.current_phase())
    await broadcast_events_routed(routed)
    counts = routed.counts()

    target_note = f" target={req.target}" if req.target else ""
    log.info(f"[{project_id}] event: {req.event_type}{target_note} route={counts}")
    return {
        "ok": True,
        "id": result_id,
        "created_at": now,
        "routing": counts,
        "validation": validation,
    }


# ── SSE ENDPOINTS ────────────────────────────────────────────────────────────

@app.get("/subscribe/state")
async def subscribe_state(request: Request):
    """Server-Sent Events stream of classroom state changes."""
    if not SSE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="SSE not available. Install: pip install sse-starlette",
        )

    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    _state_subscribers.append(queue)

    async def event_generator():
        try:
            # Send current state immediately
            yield {"event": "state", "data": json.dumps(get_state())}
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30)
                    yield {"event": "state", "data": json.dumps(data)}
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": ""}
        finally:
            if queue in _state_subscribers:
                _state_subscribers.remove(queue)

    return EventSourceResponse(event_generator())


@app.get("/subscribe/events")
async def subscribe_events(
    request: Request,
    subscriber_id: str = Query("anonymous", description="identifier for DIRECTED routing"),
    event_type: Optional[str] = None,
):
    """Server-Sent Events stream of classroom events.

    Pass `subscriber_id` as a query param to receive DIRECTED events targeted
    at that ID (e.g. `?subscriber_id=prof-dm`). Defaults to "anonymous", which
    only receives BROADCAST events.
    """
    if not SSE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="SSE not available. Install: pip install sse-starlette",
        )

    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    subscriber = (subscriber_id, queue)
    _event_subscribers.append(subscriber)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30)
                    # Filter by event_type if requested
                    if event_type and data.get("event_type") != event_type:
                        continue
                    yield {"event": "classroom_event", "data": json.dumps(data)}
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
        finally:
            if subscriber in _event_subscribers:
                _event_subscribers.remove(subscriber)

    return EventSourceResponse(event_generator())


# ── PHASE / ORCHESTRATOR ENDPOINTS ───────────────────────────────────────────

@app.get("/phase")
def get_phase():
    """Current session phase, start time, and duration."""
    return orchestrator.phase_status()


@app.post("/phase")
async def set_phase(req: SetPhaseRequest, x_api_key: str = Header(None)):
    """Transition to a new phase. Emits a `phase_change` event that
    is always broadcast to every SSE subscriber."""
    verify_detector_key(x_api_key)

    try:
        new_phase = Phase(req.phase.lower())
    except ValueError:
        valid = ", ".join(p.value for p in Phase)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid phase '{req.phase}'. Valid: {valid}",
        )

    phase_change_event = orchestrator.set_phase(new_phase)
    phase_change_event = _remember_classroom_event(
        {
            **phase_change_event,
            "camera_id": "__orchestrator__",
        },
        "orchestrator",
    )

    # Persist to Supabase event log if available
    if supabase:
        try:
            supabase.table("classroom_events").insert({
                "camera_id": "__orchestrator__",
                "event_type": "phase_change",
                "payload": phase_change_event["payload"],
                "source": "orchestrator",
            }).execute()
        except Exception as e:
            log.error(f"Supabase phase_change insert failed: {e}")

    # Broadcast the phase_change event. ALWAYS_BROADCAST ensures it hits
    # every subscriber regardless of the new phase's policy.
    routed = orchestrator.route([phase_change_event], new_phase)
    await broadcast_events_routed(routed)

    log.info(
        f"[orchestrator] phase → {new_phase.value} "
        f"(was {phase_change_event['payload']['old_phase']})"
    )

    return {
        "ok": True,
        "phase": new_phase.value,
        "started_at": phase_change_event["payload"]["started_at"],
    }


async def _mock_set_phase(phase: str) -> None:
    await set_phase(SetPhaseRequest(phase=phase), x_api_key=CLASSROOM_API_KEY)


async def _mock_push(camera_id: str, **fields) -> None:
    await push_state(PushStateRequest(camera_id=camera_id, **fields), x_api_key=CLASSROOM_API_KEY)


async def _mock_publish(project_id: str, event_type: str, payload: dict, target=None) -> None:
    await publish_project_event(
        project_id,
        ProjectEventRequest(event_type=event_type, payload=payload, target=target),
        x_api_key=CLASSROOM_API_KEY,
    )


@app.post("/mock/scenario")
async def run_mock_scenario(req: MockScenarioRequest, x_api_key: str = Header(None)):
    """Run a built-in scenario through the real API paths."""
    verify_detector_key(x_api_key)

    scenario = req.scenario.lower()
    valid = {"reset", "arrival", "lecture", "activity", "conclude", "full"}
    if scenario not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario '{req.scenario}'. Valid: {', '.join(sorted(valid))}",
        )

    if scenario in {"reset", "full"}:
        await reset_mock_state(x_api_key=CLASSROOM_API_KEY)

    if scenario in {"arrival", "full"}:
        await _mock_set_phase("arrival")
        await _mock_push(
            "orbit",
            person_detected=False,
            person_count=0,
            predicted_class="empty",
            prediction_confidence=0.7,
            fatigue_detected=False,
            whiteboard_text_detected=False,
            whiteboard_text=[],
        )
        await _mock_push(
            "orbit",
            person_detected=True,
            person_count=12,
            predicted_class="arrival",
            prediction_confidence=0.76,
            fatigue_detected=False,
            whiteboard_text_detected=False,
            whiteboard_text=[],
        )

    if scenario in {"lecture", "full"}:
        await _mock_set_phase("lecture")
        await _mock_push(
            "orbit",
            person_detected=True,
            person_count=13,
            predicted_class="presentation",
            prediction_confidence=0.88,
            whiteboard_text_detected=True,
            whiteboard_text=["contracts", "room state", "SSE"],
        )
        await _mock_push("orbit", fatigue_detected=True)

    if scenario in {"activity", "full"}:
        await _mock_set_phase("activity")
        await _mock_push(
            "orbit",
            person_detected=True,
            person_count=14,
            predicted_class="discussion",
            prediction_confidence=0.91,
            whiteboard_text_detected=False,
            whiteboard_text=[],
        )
        await _mock_publish(
            "smart-stage",
            "timer.offer",
            {"minutes": 5, "reason": "activity mode started", "mode": "group"},
            target="gesture-timer",
        )
        await _mock_publish(
            "smart-stage",
            "fiducial.request",
            {
                "marker_family": "tag36h11",
                "needed_for": "smart-stage",
                "zones": ["podium", "whiteboard", "collaboration"],
            },
            target="horizon",
        )
        await _mock_publish(
            "gesture-timer",
            "timer.started",
            {"minutes": 5, "started_by": "mock", "ends_at": _now_iso()},
        )
        await _mock_publish(
            "horizon",
            "fiducial.detected",
            {
                "tag_id": 3,
                "name": "Whiteboard Marker",
                "x": 0.52,
                "y": 0.08,
                "zone": "whiteboard",
                "confidence": 0.92,
            },
        )
        await _mock_publish(
            "gesture-timer",
            "timer.done",
            {"minutes": 5, "completed": True},
            target="prof-dm",
        )

    if scenario in {"conclude", "full"}:
        await _mock_set_phase("conclude")
        await _mock_push(
            "orbit",
            person_detected=True,
            person_count=12,
            predicted_class="wrap_up",
            prediction_confidence=0.8,
            whiteboard_text_detected=True,
            whiteboard_text=["next week: bring working contract"],
        )
        await _mock_publish(
            "assignment-tracker",
            "assignment.suggested",
            {"title": "Connect your project to the classroom bus", "due": "Monday"},
        )

    return {
        "ok": True,
        "scenario": scenario,
        "context": build_room_context(),
        "state": get_state(),
    }


@app.get("/phase/policy")
def get_phase_policy():
    """Full routing policy by phase. Useful for debugging, the TUI
    phase strip, and the white paper's routing diagram."""
    return {
        "current_phase": orchestrator.current_phase().value,
        "policies": orchestrator.policy_snapshot(),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info(f"Starting Smart Classroom API on port {PORT}")
    log.info(f"Supabase: {'connected' if supabase else 'not connected (local-only mode)'}")
    log.info(f"SSE: {'available' if SSE_AVAILABLE else 'not available'}")

    # Load existing state from Supabase on startup
    if supabase:
        try:
            result = supabase.table("classroom_state").select("*").execute()
            for row in result.data:
                _camera_states[row["camera_id"]] = row
                _previous_states[row["camera_id"]] = row.copy()
            log.info(f"Loaded {len(result.data)} camera states from Supabase")
        except Exception as e:
            log.warning(f"Could not load initial state: {e}")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
