"""
Student Project Template — Smart Classroom
============================================
Replace PROJECT_ID with your project, then implement on_state_change().

This template connects your project to the shared classroom system:
  - Read current classroom state (who's here, what's happening)
  - Subscribe to real-time changes (SSE stream)
  - Publish events from your project back to the classroom

Setup:
    pip install requests sseclient-py

    export CLASSROOM_API=http://<pc-ip>:8766
    export PROJECT_API_KEY=<your-key-from-the-student_projects-table>

    python my_project.py

Your API key:
    Ask the instructor, or check Supabase: student_projects table → api_key
"""

import os
import json
import time
import threading
from datetime import datetime

import requests

# ── Config (CHANGE THESE) ────────────────────────────────────────────────────

PROJECT_ID = "your-project-id"  # e.g. "seren-room", "echodesk", "calmball"
API_BASE = os.getenv("CLASSROOM_API", "http://localhost:8766")
API_KEY = os.getenv("PROJECT_API_KEY", "")

HEADERS = {"X-API-Key": API_KEY}


# ── Read classroom state ─────────────────────────────────────────────────────

def get_state() -> dict:
    """
    Get current classroom state (all cameras + global room mode).

    Returns:
        {
            "cameras": {
                "orbit": {"person_count": 2, "predicted_class": "discussion", ...},
                "gravity": {...},
            },
            "room_mode": "group",
            "total_persons": 5,
            "whiteboard_active": false,
            "timestamp": "..."
        }
    """
    r = requests.get(f"{API_BASE}/state", timeout=5)
    r.raise_for_status()
    return r.json()


def get_room_mode() -> dict:
    """
    Get just the current room mode.

    Returns:
        {
            "room_mode": "group",        # solo | duo | group | focus | presentation | empty
            "total_persons": 5,
            "whiteboard_active": false,
            "probe_consensus": "discussion",
            "detail": {
                "orbit": {"persons": 2, "probe": "discussion", "confidence": 0.85},
                ...
            }
        }
    """
    r = requests.get(f"{API_BASE}/mode", timeout=5)
    r.raise_for_status()
    return r.json()


def get_events(event_type: str = None, limit: int = 20) -> list:
    """
    Get recent classroom events.

    event_type options:
        "person_change", "probe_classification", "room_mode_change",
        "fatigue_change", "anomaly_change", "whiteboard_change"
    """
    params = {"limit": limit}
    if event_type:
        params["event_type"] = event_type
    r = requests.get(f"{API_BASE}/events", params=params, timeout=5)
    r.raise_for_status()
    return r.json().get("events", [])


# ── Publish your own events ──────────────────────────────────────────────────

def publish_event(event_type: str, payload: dict = None) -> dict:
    """
    Publish an event from your project.

    Examples:
        publish_event("mode_change", {"mode": "party"})
        publish_event("calmball_squeeze", {"intensity": 0.8})
        publish_event("timer_started", {"minutes": 5})
        publish_event("echodesk_message", {"text": "Can you repeat that?"})
    """
    r = requests.post(
        f"{API_BASE}/projects/{PROJECT_ID}/events",
        json={"event_type": event_type, "payload": payload or {}},
        headers=HEADERS,
        timeout=5,
    )
    r.raise_for_status()
    return r.json()


# ── Subscribe to real-time changes (SSE) ─────────────────────────────────────

def subscribe_to_state(callback):
    """
    Subscribe to classroom state changes via Server-Sent Events.
    callback(data: dict) is called for each state change.
    Runs in a background thread.

    Usage:
        def my_handler(state):
            print(f"Room mode: {state['room_mode']}")

        subscribe_to_state(my_handler)
    """
    try:
        import sseclient
    except ImportError:
        print("Install sseclient-py for real-time updates: pip install sseclient-py")
        print("Falling back to polling mode (every 5 seconds)")
        _poll_fallback(callback)
        return

    def _listen():
        while True:
            try:
                response = requests.get(
                    f"{API_BASE}/subscribe/state",
                    stream=True,
                    headers={"Accept": "text/event-stream"},
                    timeout=None,
                )
                client = sseclient.SSEClient(response)
                for event in client.events():
                    if event.event == "ping":
                        continue
                    try:
                        data = json.loads(event.data)
                        callback(data)
                    except Exception as e:
                        print(f"SSE parse error: {e}")
            except Exception as e:
                print(f"SSE connection lost: {e}, reconnecting in 5s...")
                time.sleep(5)

    thread = threading.Thread(target=_listen, daemon=True)
    thread.start()
    return thread


def _poll_fallback(callback):
    """Fallback: poll the API if SSE is not available."""
    def _poll():
        last_mode = None
        while True:
            try:
                state = get_state()
                if state.get("room_mode") != last_mode:
                    callback(state)
                    last_mode = state.get("room_mode")
            except Exception:
                pass
            time.sleep(5)

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    return thread


# ══════════════════════════════════════════════════════════════════════════════
# YOUR PROJECT LOGIC GOES BELOW
# ══════════════════════════════════════════════════════════════════════════════

def on_state_change(state: dict):
    """
    Called every time the classroom state changes.
    Implement your project's response here.

    `state` contains:
        state["room_mode"]        → "solo" | "duo" | "group" | "focus" | "presentation" | "empty"
        state["total_persons"]    → int
        state["whiteboard_active"] → bool
        state["cameras"]          → dict of per-camera states
    """
    mode = state.get("room_mode", "unknown")
    persons = state.get("total_persons", 0)
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] Room mode: {mode}  |  People: {persons}")

    # ── Example: Seren's Room ─────────────────────────────────────────────
    #
    # if mode == "solo":
    #     set_lights("warm", brightness=80)
    #     play_music("upbeat")
    #     ai_speak("Hey -- you're solo. Party mode?")
    #
    # elif mode == "duo":
    #     set_lights("dim-warm", brightness=40)
    #     play_music("lofi")
    #
    # elif mode == "group":
    #     set_lights("bright", brightness=100)
    #     play_music("ambient")
    #     show_icebreaker()
    #
    # elif mode == "focus":
    #     stop_music()
    #     set_lights("neutral", brightness=70)
    #
    # elif mode == "presentation":
    #     stop_music()
    #     set_lights("presentation", brightness=50)
    #
    # publish_event("mode_change", {"mode": mode})


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"=== {PROJECT_ID} ===")
    print(f"API: {API_BASE}")
    print()

    # 1. Get initial state
    try:
        state = get_state()
        print(f"Current room mode: {state.get('room_mode', '?')}")
        print(f"People in room: {state.get('total_persons', '?')}")
        print()
    except Exception as e:
        print(f"Could not get initial state: {e}")
        print("Is classroom_api.py running?")
        print()

    # 2. Subscribe to real-time changes
    subscribe_to_state(on_state_change)
    print("Listening for classroom changes...\n")

    # 3. Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
