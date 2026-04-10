#!/usr/bin/env python3
"""
Webcam Fatigue Detector (MediaPipe FaceLandmarker)
==================================================
Monitors eye closure (EAR) and head tilt to detect student fatigue.
Uses MediaPipe Tasks API FaceLandmarker for face + iris landmarks.
Writes the same fatigue_status.json so discord_dm_notifier.py works unchanged.

Usage:
    python3 webcam/webcam_fatigue_detector.py                # Basic detection
    python3 webcam/webcam_fatigue_detector.py --display       # Show live video
    python3 webcam/webcam_fatigue_detector.py --log           # Log to file
"""

import sys
import argparse
import time
import os
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

# Add parent directory for discord_notifier import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load environment variables from ~/oak-projects/.env (per-user)
try:
    from dotenv import load_dotenv
    load_dotenv(Path.home() / "oak-projects" / ".env")
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

from webcam_face_utils import (
    mediapipe_to_pixel_coords, determine_fatigue, get_face_landmarker_model_path,
)

# Parse arguments
parser = argparse.ArgumentParser(
    description='Webcam Fatigue Detector (MediaPipe FaceLandmarker)')
parser.add_argument('--log', action='store_true', help='Log events to file')
parser.add_argument('--fps-limit', type=int, default=15,
                    help='FPS limit to reduce CPU usage (default: 15)')
parser.add_argument('--pitch-threshold', type=int, default=20,
                    help='Head tilt angle threshold in degrees (default: 20)')
parser.add_argument('--ear-threshold', type=float, default=0.15,
                    help='Eye aspect ratio threshold (default: 0.15)')
parser.add_argument('--display', action='store_true',
                    help='Show live video window (requires display)')
parser.add_argument('--camera', type=int, default=0,
                    help='Camera index (default: 0 for built-in webcam)')
args = parser.parse_args()

# Global state tracking
log_file = None

# Fatigue state tracking
last_fatigue_status = None  # None = unknown, True = fatigued, False = alert
last_eyes_closed = None
last_head_tilted = None

# Temporal smoothing (rolling windows like OAK-D version)
closed_eye_history = deque(maxlen=30)
head_tilted_history = deque(maxlen=30)
FATIGUE_THRESHOLD = 0.75  # 75% of frames must show fatigue

# Debouncing for state transitions
pending_state = None
pending_state_time = None
DEBOUNCE_SECONDS = 1.5

# Status file for Discord bot integration (DM bot watches this file)
STATUS_FILE = Path.home() / "oak-projects" / "fatigue_status.json"
STATUS_UPDATE_INTERVAL = 10
last_status_update_time = 0

# Screenshot for Discord bot
SCREENSHOT_FILE = Path.home() / "oak-projects" / "latest_fatigue_frame.jpg"
SCREENSHOT_UPDATE_INTERVAL = 5
last_screenshot_time = 0


def log_event(message):
    """Print and optionally log an event."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)

    if log_file:
        log_file.write(line + "\n")
        log_file.flush()


def update_status_file(faces_detected, fatigue_detected, eyes_closed,
                       head_tilted, fatigue_percent, running=True):
    """Update status file for Discord bot integration."""
    try:
        status_data = {
            "faces_detected": faces_detected,
            "fatigue_detected": fatigue_detected,
            "eyes_closed": eyes_closed,
            "head_tilted": head_tilted,
            "fatigue_percent": round(fatigue_percent, 2),
            "timestamp": datetime.now().isoformat(),
            "running": running
        }
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(json.dumps(status_data, indent=2))
    except Exception as e:
        log_event(f"WARNING: Could not update status file: {e}")


def draw_landmarks_on_frame(frame, face_landmarks, w, h):
    """Draw face landmark dots on the frame for visualization."""
    # Draw key landmarks: eyes, nose, mouth outline
    key_indices = [
        # Left eye contour
        33, 160, 158, 133, 144, 153, 7, 163, 246,
        # Right eye contour
        263, 387, 385, 362, 373, 380, 249, 390, 466,
        # Nose
        1, 4, 5, 6, 195, 197,
        # Mouth
        61, 291, 0, 17, 78, 308,
        # Iris
        468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
    ]
    for idx in key_indices:
        if idx < len(face_landmarks):
            lm = face_landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            color = (0, 255, 255) if idx >= 468 else (0, 200, 0)
            radius = 2 if idx < 468 else 3
            cv2.circle(frame, (x, y), radius, color, -1)


def run_detection():
    """Main fatigue detection loop using webcam + MediaPipe FaceLandmarker."""
    global log_file, last_fatigue_status, last_eyes_closed, last_head_tilted
    global pending_state, pending_state_time
    global last_status_update_time, last_screenshot_time

    # Import MediaPipe here so startup errors are clear
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker, FaceLandmarkerOptions, RunningMode,
        )
    except ImportError:
        print("ERROR: mediapipe not installed.")
        print("   Install with: pip install mediapipe")
        sys.exit(1)

    # Open log file if requested
    if args.log:
        log_filename = f"fatigue_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file = open(log_filename, 'w')
        log_event(f"Logging to {log_filename}")

    log_event("Fatigue detector started (Webcam + MediaPipe FaceLandmarker)")
    log_event(f"EAR threshold: {args.ear_threshold}, Pitch threshold: {args.pitch_threshold}")
    log_event("Press Ctrl+C to exit (or 'q' in display window)\n")

    # Initialize status file
    update_status_file(
        faces_detected=0, fatigue_detected=False,
        eyes_closed=False, head_tilted=False,
        fatigue_percent=0.0, running=True
    )
    last_status_update_time = time.time()

    # Download model if needed, then create FaceLandmarker
    model_path = get_face_landmarker_model_path()
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        log_event(f"ERROR: Could not open camera {args.camera}")
        log_event("   Check System Preferences > Privacy > Camera")
        sys.exit(1)

    log_event(f"Camera {args.camera} opened.")
    log_event("Detection started. Monitoring for fatigue...\n")

    frame_interval = 1.0 / args.fps_limit
    frame_count = 0

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            frame_count += 1

            # Convert BGR to RGB and create MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Run face landmark detection
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            faces_detected = 0
            current_eyes_closed = False
            current_head_tilted = False

            if result.face_landmarks:
                faces_detected = len(result.face_landmarks)
                face_lms = result.face_landmarks[0]  # First face

                # Convert to pixel coordinates
                landmarks = mediapipe_to_pixel_coords(face_lms, w, h)

                # Determine fatigue
                head_tilted, eyes_closed = determine_fatigue(
                    (h, w), landmarks,
                    pitch_angle=args.pitch_threshold,
                    ear_threshold=args.ear_threshold,
                )

                closed_eye_history.append(eyes_closed)
                head_tilted_history.append(head_tilted)

                current_eyes_closed = eyes_closed
                current_head_tilted = head_tilted

            # Calculate fatigue percentages from rolling window
            percent_eyes_closed = (
                sum(closed_eye_history) / len(closed_eye_history)
                if closed_eye_history else 0.0
            )
            percent_head_tilted = (
                sum(head_tilted_history) / len(head_tilted_history)
                if head_tilted_history else 0.0
            )
            fatigue_percent = max(percent_eyes_closed, percent_head_tilted)
            fatigue_detected = fatigue_percent >= FATIGUE_THRESHOLD

            current_time = time.time()

            # Console status line (overwrite in place)
            eyes_str = "CLOSED" if current_eyes_closed else "open"
            head_str = "TILTED" if current_head_tilted else "up"
            print(
                f"\r  Faces: {faces_detected} | "
                f"Eyes: {eyes_str} ({percent_eyes_closed:.0%}) | "
                f"Head: {head_str} ({percent_head_tilted:.0%}) | "
                f"Fatigue: {fatigue_percent:.0%}  ",
                end="", flush=True
            )

            # ── Update status file on state transitions ─────────────────
            state_changed = False
            if current_eyes_closed != last_eyes_closed and last_eyes_closed is not None:
                state_changed = True
            if current_head_tilted != last_head_tilted and last_head_tilted is not None:
                state_changed = True

            if state_changed:
                update_status_file(
                    faces_detected, fatigue_detected,
                    current_eyes_closed, current_head_tilted,
                    fatigue_percent
                )

            last_eyes_closed = current_eyes_closed
            last_head_tilted = current_head_tilted

            # ── Fatigue state change with debouncing ────────────────────
            if fatigue_detected != last_fatigue_status:
                if pending_state == fatigue_detected:
                    if current_time - pending_state_time >= DEBOUNCE_SECONDS:
                        if fatigue_detected:
                            reasons = []
                            if percent_eyes_closed >= FATIGUE_THRESHOLD:
                                reasons.append("eyes closed")
                            if percent_head_tilted >= FATIGUE_THRESHOLD:
                                reasons.append("head tilted")
                            reason_str = " / ".join(reasons)
                            log_event(f"\nFATIGUE DETECTED ({reason_str})")
                        else:
                            log_event("\nAttention restored - student alert")

                        last_fatigue_status = fatigue_detected
                        pending_state = None
                        pending_state_time = None

                        update_status_file(
                            faces_detected, fatigue_detected,
                            current_eyes_closed, current_head_tilted,
                            fatigue_percent
                        )
                else:
                    pending_state = fatigue_detected
                    pending_state_time = current_time
            else:
                pending_state = None
                pending_state_time = None

            # ── Periodic status file update ─────────────────────────────
            if current_time - last_status_update_time >= STATUS_UPDATE_INTERVAL:
                fatigue = last_fatigue_status if last_fatigue_status is not None else False
                eyes = last_eyes_closed if last_eyes_closed is not None else False
                head = last_head_tilted if last_head_tilted is not None else False
                pct = max(
                    sum(closed_eye_history) / len(closed_eye_history) if closed_eye_history else 0.0,
                    sum(head_tilted_history) / len(head_tilted_history) if head_tilted_history else 0.0,
                )
                update_status_file(0, fatigue, eyes, head, pct)
                last_status_update_time = current_time

            # ── Screenshot save + live display ──────────────────────────
            if current_time - last_screenshot_time >= SCREENSHOT_UPDATE_INTERVAL:
                try:
                    cv2.imwrite(str(SCREENSHOT_FILE), frame)
                    last_screenshot_time = current_time
                except Exception as e:
                    log_event(f"WARNING: Could not save screenshot: {e}")

            if args.display:
                fatigue = last_fatigue_status if last_fatigue_status is not None else False
                color = (0, 0, 255) if fatigue else (0, 255, 0)
                status = "FATIGUED" if fatigue else "ALERT"
                cv2.putText(frame, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                if current_eyes_closed:
                    cv2.putText(frame, "Eyes Closed", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if current_head_tilted:
                    cv2.putText(frame, "Head Tilted", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Draw face landmarks if available
                if result.face_landmarks:
                    draw_landmarks_on_frame(frame, result.face_landmarks[0], w, h)

                cv2.imshow("Fatigue Detector (Webcam)", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            # FPS limiting
            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        log_event("\nFatigue detector stopped")

    finally:
        landmarker.close()
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        update_status_file(0, False, False, False, 0.0, running=False)
        if log_file:
            log_file.close()


if __name__ == "__main__":
    run_detection()
