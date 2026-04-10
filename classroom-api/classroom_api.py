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
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
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

# ── Supabase client ──────────────────────────────────────────────────────────

supabase: Optional[Client] = None

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
_state_subscribers: list[asyncio.Queue] = []  # SSE subscribers
_event_subscribers: list[asyncio.Queue] = []  # SSE event subscribers

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


async def broadcast_event(event: dict):
    """Push a classroom event to all SSE event subscribers."""
    dead = []
    for q in _event_subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _event_subscribers.remove(q)


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

def verify_detector_key(x_api_key: str = Header(None)):
    """Verify the shared detector API key."""
    if x_api_key != CLASSROOM_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def verify_project_key(project_id: str, x_api_key: str = Header(None)):
    """Verify a student project's API key."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")

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

@app.get("/health")
def health():
    return {
        "status": "ok",
        "supabase": supabase is not None,
        "sse": SSE_AVAILABLE,
        "cameras": list(_camera_states.keys()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/state")
def get_state():
    """All cameras + aggregated global room mode."""
    mode_info = compute_room_mode(_camera_states)
    return {
        "cameras": _camera_states,
        "room_mode": mode_info["room_mode"],
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
        return {"events": [], "note": "Supabase not connected"}

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
        return {"projects": result.data}
    return {"projects": [], "note": "Supabase not connected"}


@app.get("/projects/{project_id}/events")
def get_project_events(
    project_id: str,
    limit: int = Query(20, ge=1, le=200),
):
    """A project's published events."""
    if not supabase:
        return {"events": [], "note": "Supabase not connected"}

    result = supabase.table("project_events").select("*").eq(
        "project_id", project_id
    ).order("created_at", desc=True).limit(limit).execute()

    return {"events": result.data}


# ── WRITE ENDPOINTS (detectors) ──────────────────────────────────────────────

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

    # Update in-memory state
    _previous_states[camera_id] = _camera_states.get(camera_id, {})
    _camera_states[camera_id] = merged

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
    for event in events:
        await broadcast_event(event)

    log.info(
        f"[{camera_id}] pushed — persons={merged.get('person_count', '?')}, "
        f"probe={merged.get('predicted_class', '?')}, "
        f"mode={merged.get('room_mode', '?')}, "
        f"events={len(events)}"
    )

    return {
        "ok": True,
        "camera_id": camera_id,
        "room_mode": merged["room_mode"],
        "events_emitted": len(events),
    }


# ── WRITE ENDPOINTS (students) ───────────────────────────────────────────────

@app.post("/projects/{project_id}/events")
async def publish_project_event(
    project_id: str,
    req: ProjectEventRequest,
    x_api_key: str = Header(None),
):
    """Students publish events from their projects."""
    await verify_project_key(project_id, x_api_key)

    now = datetime.now(timezone.utc).isoformat()
    event = {
        "project_id": project_id,
        "event_type": req.event_type,
        "payload": req.payload,
        "created_at": now,
    }

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

    log.info(f"[{project_id}] event: {req.event_type}")
    return {"ok": True, "id": result_id, "created_at": now}


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
    event_type: Optional[str] = None,
):
    """Server-Sent Events stream of classroom events."""
    if not SSE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="SSE not available. Install: pip install sse-starlette",
        )

    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    _event_subscribers.append(queue)

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
            if queue in _event_subscribers:
                _event_subscribers.remove(queue)

    return EventSourceResponse(event_generator())


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
