#!/usr/bin/env python3
"""
Webcam Person Detector (YOLOv8)
===============================
Detects people using the laptop webcam and YOLOv8-nano.
Writes the same status files as the OAK-D version so the
Discord bot works unchanged.

Usage:
    python3 webcam/webcam_person_detector.py              # Basic detection
    python3 webcam/webcam_person_detector.py --display     # Show live video
    python3 webcam/webcam_person_detector.py --discord     # Enable Discord notifications
    python3 webcam/webcam_person_detector.py --log         # Log events to file
"""

import sys
import argparse
import time
import os
import json
import cv2
import socket
import getpass
from pathlib import Path
from datetime import datetime

# Add parent directory for discord_notifier import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load environment variables from ~/oak-projects/.env (per-user)
try:
    from dotenv import load_dotenv
    load_dotenv(Path.home() / "oak-projects" / ".env")
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Import Discord notifier
try:
    from discord_notifier import send_notification
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

# Parse arguments
parser = argparse.ArgumentParser(
    description='Webcam Person Detector (YOLOv8)')
parser.add_argument('--log', action='store_true', help='Log events to file')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Detection confidence threshold (0-1)')
parser.add_argument('--discord', action='store_true',
                    help='Enable Discord notifications')
parser.add_argument('--discord-quiet', action='store_true',
                    help='Only send Discord notifications for person detected (not when clear)')
parser.add_argument('--display', action='store_true',
                    help='Show live video window with bounding boxes')
parser.add_argument('--camera', type=int, default=0,
                    help='Camera index (default: 0 for built-in webcam)')
parser.add_argument('--fps-limit', type=int, default=15,
                    help='FPS limit to reduce CPU usage (default: 15)')
args = parser.parse_args()

# Global state tracking
last_status = None
last_count = 0
log_file = None

# Temporal smoothing to prevent flickering
pending_state = None
pending_state_time = None
DEBOUNCE_SECONDS = 1.5  # State must persist for 1.5 seconds before triggering

# Status file for Discord bot integration
STATUS_FILE = Path.home() / "oak-projects" / "camera_status.json"
STATUS_UPDATE_INTERVAL = 10  # Update status file every 10 seconds even if no change
last_status_update_time = 0

# Screenshot for Discord bot
SCREENSHOT_FILE = Path.home() / "oak-projects" / "latest_frame.jpg"
SCREENSHOT_UPDATE_INTERVAL = 5  # Save screenshot every 5 seconds
last_screenshot_time = 0

# Classroom API integration
CLASSROOM_API_URL = os.getenv("CLASSROOM_API_URL", "")
CLASSROOM_API_KEY = os.getenv("CLASSROOM_API_KEY", "")


def push_to_classroom(camera_id, detected, count, username=None, hostname=None):
    """Push person detection to the classroom API (fire-and-forget)."""
    if not CLASSROOM_API_URL:
        return
    try:
        import requests
        requests.post(
            f"{CLASSROOM_API_URL.rstrip('/')}/push/state",
            json={
                "camera_id": camera_id,
                "person_detected": detected,
                "person_count": count,
                "detector_host": hostname,
                "detector_user": username,
            },
            headers={"X-API-Key": CLASSROOM_API_KEY},
            timeout=2,
        )
    except Exception:
        pass  # Never block the detection loop


def log_event(message):
    """Print and optionally log an event."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)

    if log_file:
        log_file.write(line + "\n")
        log_file.flush()


def send_discord_notification(message, force=False):
    """Send Discord notification if enabled."""
    if not args.discord and not force:
        return

    if not DISCORD_AVAILABLE:
        return

    if not os.getenv('DISCORD_WEBHOOK_URL'):
        if force:
            log_event("WARNING: Discord notifications requested but DISCORD_WEBHOOK_URL not set")
        return

    send_notification(message, add_timestamp=False)


def update_status_file(detected, count, running=True, username=None, hostname=None):
    """Update status file for Discord bot integration."""
    try:
        status_data = {
            "detected": detected,
            "count": count,
            "timestamp": datetime.now().isoformat(),
            "running": running
        }
        if username:
            status_data["username"] = username
        if hostname:
            status_data["hostname"] = hostname

        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(json.dumps(status_data, indent=2))
    except Exception as e:
        log_event(f"WARNING: Could not update status file: {e}")


def run_detection():
    """Main detection loop using webcam + YOLOv8."""
    global log_file, last_status, last_count
    global pending_state, pending_state_time
    global last_status_update_time, last_screenshot_time

    # Import YOLO here so startup errors are clear
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.")
        print("   Install with: pip install ultralytics")
        sys.exit(1)

    # Get user and hostname
    try:
        username = getpass.getuser()
    except Exception:
        username = os.getenv('USER', 'unknown')

    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = 'unknown'

    # Open log file if requested
    if args.log:
        log_filename = f"person_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file = open(log_filename, 'w')
        log_event(f"Logging to {log_filename}")

    log_event("Person detector started (Webcam + YOLOv8-nano)")
    log_event(f"Confidence threshold: {args.threshold}")
    if args.discord:
        log_event("Discord notifications: ENABLED")
    log_event("Press Ctrl+C to exit\n")

    # Initialize status file
    update_status_file(detected=False, count=0, running=True,
                       username=username, hostname=hostname)
    last_status_update_time = time.time()

    # Send startup notification
    if args.discord:
        discord_startup = f"🎥 **{username}** is now running webcam_person_detector.py on **{hostname}**"
        send_discord_notification(discord_startup)

    # Load YOLO model (auto-downloads yolov8n.pt on first run, ~6MB)
    log_event("Loading YOLOv8-nano model...")
    model = YOLO("yolov8n.pt")
    log_event("Model loaded.")

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        log_event(f"ERROR: Could not open camera {args.camera}")
        log_event("   Check System Preferences > Privacy > Camera")
        sys.exit(1)

    log_event(f"Camera {args.camera} opened.")
    log_event("Detection started. Monitoring for people...\n")

    frame_interval = 1.0 / args.fps_limit

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                log_event("WARNING: Could not read frame from camera")
                time.sleep(0.1)
                continue

            # Run YOLO detection (class 0 = person in COCO)
            results = model(frame, classes=[0], conf=args.threshold, verbose=False)

            person_count = len(results[0].boxes)
            person_detected = person_count > 0
            current_time = time.time()

            # ── Debouncing logic ────────────────────────────────────────
            if person_detected != last_status:
                if pending_state == person_detected:
                    if current_time - pending_state_time >= DEBOUNCE_SECONDS:
                        if person_detected:
                            discord_msg = "Students detected in classroom"
                            log_msg = f"PERSON DETECTED (count: {person_count})"
                            log_event(log_msg)
                            send_discord_notification(discord_msg)
                        else:
                            discord_msg = "Classroom is empty"
                            log_msg = "No person detected - area clear"
                            log_event(log_msg)
                            if not args.discord_quiet:
                                send_discord_notification(discord_msg)

                        last_status = person_detected
                        last_count = person_count
                        pending_state = None
                        pending_state_time = None

                        update_status_file(
                            person_detected, person_count, running=True,
                            username=username, hostname=hostname)
                        push_to_classroom(hostname, person_detected,
                                          person_count, username, hostname)
                else:
                    pending_state = person_detected
                    pending_state_time = current_time
            else:
                pending_state = None
                pending_state_time = None

            # Update count for display purposes
            if person_count != last_count and person_detected and last_status:
                log_event(f"   Count changed: {person_count} people")
                last_count = person_count

            # ── Console status ──────────────────────────────────────────
            status_str = "DETECTED" if person_detected else "clear"
            print(
                f"\r  Persons: {person_count} | Status: {status_str}  ",
                end="", flush=True
            )

            # ── Periodic status file update ─────────────────────────────
            if current_time - last_status_update_time >= STATUS_UPDATE_INTERVAL:
                detected = last_status if last_status is not None else False
                count = last_count if last_status else 0
                update_status_file(detected, count, running=True,
                                   username=username, hostname=hostname)
                push_to_classroom(hostname, detected, count, username, hostname)
                last_status_update_time = current_time

            # ── Periodic screenshot ─────────────────────────────────────
            if current_time - last_screenshot_time >= SCREENSHOT_UPDATE_INTERVAL:
                try:
                    cv2.imwrite(str(SCREENSHOT_FILE), frame)
                    last_screenshot_time = current_time
                except Exception as e:
                    log_event(f"WARNING: Could not save screenshot: {e}")

            # ── Display window ──────────────────────────────────────────
            if args.display:
                # Draw YOLO bounding boxes
                annotated = results[0].plot()
                cv2.imshow("Person Detector (Webcam)", annotated)
                if cv2.waitKey(1) == ord('q'):
                    break

            # FPS limiting
            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        shutdown_msg = "Person detector stopped"
        log_event(f"\n{shutdown_msg}")
        if args.discord:
            discord_shutdown = f"📴 **{username}** stopped webcam_person_detector.py on **{hostname}** - camera is free"
            send_discord_notification(discord_shutdown)

    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        update_status_file(False, 0, running=False,
                           username=username, hostname=hostname)
        if log_file:
            log_file.close()


if __name__ == "__main__":
    if args.discord and not DISCORD_AVAILABLE:
        print("ERROR: Discord notifications requested but discord_notifier.py not found")
        print("   Make sure discord_notifier.py is in the repository root")
        sys.exit(1)

    if args.discord and not DOTENV_AVAILABLE:
        print("WARNING: python-dotenv not installed - ensure DISCORD_WEBHOOK_URL is in environment")
        print("   Install with: pip install python-dotenv")

    run_detection()
