#!/usr/bin/env python3
"""
Minimum Smart Classroom heartbeat client.

This is the smallest useful student integration:
  1. Pick a PROJECT_ID.
  2. Say what your project can handle.
  3. Keep sending a heartbeat so the orchestrator knows you exist.

Run:
    set CLASSROOM_API=http://localhost:8766
    set PROJECT_ID=gesture-timer
    set PROJECT_API_KEY=testkey
    python student_heartbeat.py

Optional:
    set CAPABILITIES=timer,timer.offer,timer.command
    set CONSUMES=timer.offer,timer.command
    set EMITS=timer.started,timer.done
    set MESSAGE=timer prototype running on my laptop
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime

try:
    import requests
except ImportError:
    sys.stderr.write("Install requests first: pip install requests\n")
    sys.exit(2)


API_BASE = os.getenv("CLASSROOM_API", "http://localhost:8766").rstrip("/")
PROJECT_ID = os.getenv("PROJECT_ID", "student-project")
API_KEY = os.getenv("PROJECT_API_KEY", os.getenv("CLASSROOM_API_KEY", "testkey"))
INTERVAL_SEC = float(os.getenv("HEARTBEAT_INTERVAL_SEC", "30"))


def csv_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


CAPABILITIES = csv_env("CAPABILITIES", "student-project")
CONSUMES = csv_env("CONSUMES")
EMITS = csv_env("EMITS", "project.heartbeat")
MESSAGE = os.getenv("MESSAGE", f"{PROJECT_ID} heartbeat running")


def heartbeat() -> dict:
    response = requests.post(
        f"{API_BASE}/projects/{PROJECT_ID}/heartbeat",
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        json={
            "status": "online",
            "capabilities": CAPABILITIES,
            "consumes": CONSUMES,
            "emits": EMITS,
            "message": MESSAGE,
            "meta": {
                "client": "student_heartbeat.py",
                "hostname": os.getenv("COMPUTERNAME") or os.getenv("HOSTNAME"),
            },
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def main() -> int:
    print("Smart Classroom heartbeat")
    print(f"  API: {API_BASE}")
    print(f"  project: {PROJECT_ID}")
    print(f"  capabilities: {', '.join(CAPABILITIES) or '(none)'}")
    print()

    while True:
        try:
            result = heartbeat()
            status = result.get("status", {})
            ts = datetime.now().strftime("%H:%M:%S")
            age = status.get("age_sec")
            age_text = f"{age:.1f}s" if isinstance(age, (int, float)) else "unknown"
            print(
                f"[{ts}] heartbeat ok: "
                f"{status.get('project_id', PROJECT_ID)} "
                f"status={status.get('status')} "
                f"live={status.get('is_live')} "
                f"age={age_text}"
            )
        except KeyboardInterrupt:
            print("\nStopped")
            return 0
        except Exception as exc:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] heartbeat failed: {exc}")
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    sys.exit(main())
