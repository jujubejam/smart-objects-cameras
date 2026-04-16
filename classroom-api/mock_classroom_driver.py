#!/usr/bin/env python3
"""
Mock classroom driver for Monday demos.

Runs a short class-session script against classroom_api.py:
arrival -> lecture -> activity -> conclude. It pushes fake detector state and
publishes project events using the same contracts real student projects use.

Start the API first:
    CLASSROOM_API_KEY=testkey python classroom_api.py

Then run:
    CLASSROOM_API_URL=http://localhost:8766 CLASSROOM_API_KEY=testkey python mock_classroom_driver.py

Listen from another terminal:
    curl -N "http://localhost:8766/subscribe/events?subscriber_id=gesture-timer"
    curl -N "http://localhost:8766/subscribe/events?subscriber_id=horizon"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    sys.stderr.write("requests not installed. Run: pip install requests\n")
    sys.exit(2)


API_URL = os.getenv("CLASSROOM_API_URL", "http://localhost:8766")
API_KEY = os.getenv("CLASSROOM_API_KEY", "testkey")
PROJECT_API_KEY = os.getenv("PROJECT_API_KEY", API_KEY)

DETECTOR_HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
}
PROJECT_HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": PROJECT_API_KEY,
}


def post(path: str, payload: dict, *, headers: dict = DETECTOR_HEADERS) -> dict:
    response = requests.post(f"{API_URL}{path}", json=payload, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json()


def set_phase(phase: str) -> None:
    data = post("/phase", {"phase": phase})
    print(f"phase -> {data['phase']}")


def reset_room() -> None:
    data = post("/mock/reset", {})
    print(f"reset -> {data['phase']}")


def push_state(camera_id: str, **fields) -> None:
    data = post("/push/state", {"camera_id": camera_id, **fields})
    print(
        f"state {camera_id}: mode={data.get('room_mode')} "
        f"phase={data.get('phase')} routing={data.get('routing')}"
    )


def publish(project_id: str, event_type: str, payload: dict, target=None) -> None:
    body = {"event_type": event_type, "payload": payload}
    if target:
        body["target"] = target
    data = post(f"/projects/{project_id}/events", body, headers=PROJECT_HEADERS)
    target_note = f" -> {target}" if target else ""
    print(f"event {project_id}:{event_type}{target_note} routing={data.get('routing')}")


def pause(seconds: float, speed: float) -> None:
    time.sleep(max(0.0, seconds / speed))


def run_script(speed: float) -> None:
    print(f"Mock classroom driver -> {API_URL}")
    print()

    reset_room()

    set_phase("arrival")
    push_state(
        "orbit",
        person_detected=False,
        person_count=0,
        predicted_class="empty",
        prediction_confidence=0.7,
        fatigue_detected=False,
        whiteboard_text_detected=False,
        whiteboard_text=[],
    )
    pause(1, speed)
    push_state(
        "orbit",
        person_detected=True,
        person_count=12,
        predicted_class="arrival",
        prediction_confidence=0.76,
        fatigue_detected=False,
        whiteboard_text_detected=False,
        whiteboard_text=[],
    )
    pause(1, speed)

    set_phase("lecture")
    push_state(
        "orbit",
        person_detected=True,
        person_count=13,
        predicted_class="presentation",
        prediction_confidence=0.88,
        whiteboard_text_detected=True,
        whiteboard_text=["contracts", "room state", "SSE"],
    )
    # Presentation mode: focus-beam points at the board; seren-room shifts to focus
    publish("focus-beam", "beam.pointed",    {"x": 0.5, "y": 0.2, "focus": "whiteboard"})
    publish("seren-room", "ambience.changed", {"mode": "focus", "intensity": 0.6})
    publish("gus-mode",   "gus.sleeping",    {"reason": "lecture in progress"})
    # Imprint captures the whiteboard text
    publish("imprint", "handwriting.captured", {"text": "contracts, room state, SSE", "surface": "whiteboard"})
    publish("imprint", "note.saved",           {"text": "contracts, room state, SSE", "title": "Lecture notes"})
    pause(1, speed)
    push_state("orbit", fatigue_detected=True)
    pause(1, speed)

    set_phase("activity")
    push_state(
        "orbit",
        person_detected=True,
        person_count=14,
        predicted_class="discussion",
        prediction_confidence=0.91,
        whiteboard_text_detected=False,
    )
    # Activity mode: forest reacts to group energy; calmball activates; seren-room adapts
    publish("gus-mode",         "gus.excited",     {"trigger": "group activity started"})
    publish("forest-classroom", "forest.mood_set", {"mood": "active", "intensity": 0.8})
    publish("forest-classroom", "forest.pulse",    {"count": 14})
    publish("calmball",         "calm.activated",  {"sound": "forest-ambience", "intensity": 0.5})
    publish("seren-room",       "ambience.changed", {"mode": "group", "intensity": 0.7})
    publish("seren-room",       "room.adapted",    {"adaptation": "warm collaborative lighting", "reason": "group activity"})
    # Focus beam turns off during open discussion
    publish("focus-beam", "beam.off", {"reason": "activity mode"})
    publish(
        "smart-stage",
        "timer.offer",
        {
            "minutes": 5,
            "reason": "activity mode started",
            "mode": "group",
        },
        target="gesture-timer",
    )
    publish(
        "smart-stage",
        "fiducial.request",
        {
            "marker_family": "tag36h11",
            "needed_for": "smart-stage",
            "zones": ["podium", "whiteboard", "collaboration"],
        },
        target="horizon",
    )
    pause(1, speed)

    publish(
        "gesture-timer",
        "timer.started",
        {
            "minutes": 5,
            "started_by": "mock",
            "ends_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    publish(
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
    # Imprint captures the fiducial-detected zone
    publish("imprint", "handwriting.captured", {"text": "zone: whiteboard [tag 3]", "surface": "fiducial"})
    # EchoDesk surfaces a student question during group work
    publish("echodesk", "question.displayed", {"text": "What does your project consume?", "display_id": "q-001"})
    pause(2, speed)
    publish("gesture-timer", "timer.done", {"minutes": 5, "completed": True})
    # CalmBall deactivates when timer is done (lower energy)
    publish("calmball", "calm.deactivated", {"reason": "timer done"})
    pause(1, speed)

    set_phase("conclude")
    push_state(
        "orbit",
        person_detected=True,
        person_count=12,
        predicted_class="wrap_up",
        prediction_confidence=0.8,
        whiteboard_text_detected=True,
        whiteboard_text=["next week: bring working contract"],
    )
    # Seren-room shifts back to quiet mode; forest winds down
    publish("seren-room",       "ambience.changed",   {"mode": "wrap_up", "intensity": 0.4})
    publish("forest-classroom", "forest.mood_set",    {"mood": "quiet", "intensity": 0.3})
    # Imprint captures the closing whiteboard content
    publish("imprint", "handwriting.captured", {"text": "next week: bring working contract", "surface": "whiteboard"})
    publish(
        "assignment-tracker",
        "assignment.suggested",
        {
            "title": "Connect your project to the classroom bus",
            "due": "Monday",
        },
    )

    print()
    print("done")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=1.0, help="higher means shorter pauses")
    args = parser.parse_args()

    try:
        requests.get(f"{API_URL}/health", timeout=5).raise_for_status()
    except Exception as exc:
        sys.stderr.write(f"API is not reachable at {API_URL}: {exc}\n")
        return 1

    run_script(speed=args.speed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
