#!/usr/bin/env python3
"""
Webcam Gaze Direction Detector (MediaPipe FaceLandmarker)
=========================================================
Estimates where a person is looking using iris landmark positions.
Uses MediaPipe Tasks API FaceLandmarker for face + iris landmarks.
Writes the same gaze_status.json for Discord bot integration.

Usage:
    python3 webcam/webcam_gaze_detector.py                # Basic detection
    python3 webcam/webcam_gaze_detector.py --display       # Show live video with gaze arrows
    python3 webcam/webcam_gaze_detector.py --log           # Log to file
"""

import sys
import argparse
import time
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webcam_face_utils import (
    mediapipe_to_pixel_coords, estimate_gaze_from_iris,
    get_face_landmarker_model_path,
    LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER,
)

# Parse arguments
parser = argparse.ArgumentParser(
    description='Webcam Gaze Direction Detector (MediaPipe FaceLandmarker)')
parser.add_argument('--log', action='store_true', help='Log events to file')
parser.add_argument('--fps-limit', type=int, default=15,
                    help='FPS limit to reduce CPU usage (default: 15)')
parser.add_argument('--display', action='store_true',
                    help='Show live video window with gaze vectors (requires display)')
parser.add_argument('--camera', type=int, default=0,
                    help='Camera index (default: 0 for built-in webcam)')
args = parser.parse_args()

# Global state
log_file = None

# Status file for integration
STATUS_FILE = Path.home() / "oak-projects" / "gaze_status.json"
STATUS_UPDATE_INTERVAL = 2
last_status_update_time = 0

# Screenshot
SCREENSHOT_FILE = Path.home() / "oak-projects" / "latest_gaze_frame.jpg"
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


def classify_gaze(gaze_x, gaze_y):
    """Classify gaze vector into a human-readable direction.

    Args:
        gaze_x: Horizontal gaze component (positive = right)
        gaze_y: Vertical gaze component (positive = up)

    Returns:
        Direction string like "center", "left", "right", "up", "down",
        or combinations like "up-left".
    """
    threshold = 0.15

    h_dir = ""
    v_dir = ""

    if gaze_x < -threshold:
        h_dir = "left"
    elif gaze_x > threshold:
        h_dir = "right"

    if gaze_y > threshold:
        v_dir = "up"
    elif gaze_y < -threshold:
        v_dir = "down"

    if h_dir and v_dir:
        return f"{v_dir}-{h_dir}"
    elif h_dir:
        return h_dir
    elif v_dir:
        return v_dir
    else:
        return "center"


def update_status_file(faces_detected, gaze_direction, gaze_x, gaze_y, gaze_z,
                       head_yaw, head_pitch, head_roll, running=True):
    """Update status file for external integration."""
    try:
        status_data = {
            "faces_detected": faces_detected,
            "gaze_direction": gaze_direction,
            "gaze_x": round(float(gaze_x), 4),
            "gaze_y": round(float(gaze_y), 4),
            "gaze_z": round(float(gaze_z), 4),
            "head_yaw": round(float(head_yaw), 1),
            "head_pitch": round(float(head_pitch), 1),
            "head_roll": round(float(head_roll), 1),
            "timestamp": datetime.now().isoformat(),
            "running": running,
        }
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(json.dumps(status_data, indent=2))
    except Exception as e:
        log_event(f"WARNING: Could not update status file: {e}")


def draw_gaze_arrow(frame, iris_pos, gaze_x, gaze_y, color=(0, 255, 0)):
    """Draw a gaze direction arrow from an iris position."""
    scale = 200  # Arrow length in pixels
    start = (int(iris_pos[0]), int(iris_pos[1]))
    end = (
        int(start[0] + gaze_x * scale),
        int(start[1] - gaze_y * scale),  # Y is inverted in image space
    )
    cv2.arrowedLine(frame, start, end, color, 2, tipLength=0.3)


def run_detection():
    """Main gaze detection loop using webcam + MediaPipe FaceLandmarker."""
    global log_file, last_status_update_time, last_screenshot_time

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

    if args.log:
        log_filename = f"gaze_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file = open(log_filename, 'w')
        log_event(f"Logging to {log_filename}")

    log_event("Gaze detector started (Webcam + MediaPipe FaceLandmarker + Iris)")
    log_event("Press Ctrl+C to exit (or 'q' in display window)\n")

    # Initialize status file
    update_status_file(0, "unknown", 0, 0, 0, 0, 0, 0, running=True)
    last_status_update_time = time.time()

    # Track last gaze for console output
    last_gaze_direction = "unknown"
    last_gaze_x = 0.0
    last_gaze_y = 0.0
    last_gaze_z = 0.0
    last_head_yaw = 0.0
    last_head_pitch = 0.0
    last_head_roll = 0.0

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
    log_event("Detection started. Monitoring gaze direction...\n")

    frame_interval = 1.0 / args.fps_limit

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]

            # Convert BGR to RGB and create MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Run face landmark detection
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            faces_detected = 0

            if result.face_landmarks:
                faces_detected = len(result.face_landmarks)
                face_lms = result.face_landmarks[0]  # First face

                # Convert to pixel coordinates
                landmarks = mediapipe_to_pixel_coords(face_lms, w, h)

                # Estimate gaze from iris position
                gaze_x, gaze_y, gaze_z, head_yaw, head_pitch, head_roll = (
                    estimate_gaze_from_iris(landmarks, (h, w))
                )

                gaze_direction = classify_gaze(gaze_x, gaze_y)

                last_gaze_direction = gaze_direction
                last_gaze_x = gaze_x
                last_gaze_y = gaze_y
                last_gaze_z = gaze_z
                last_head_yaw = head_yaw
                last_head_pitch = head_pitch
                last_head_roll = head_roll

                # Console status line
                print(
                    f"\r  Faces: {faces_detected} | "
                    f"Gaze: {gaze_direction:>10} "
                    f"(x:{gaze_x:+.2f} y:{gaze_y:+.2f} z:{gaze_z:+.2f})  ",
                    end="", flush=True
                )

                # Display: draw gaze arrows
                if args.display:
                    left_iris = landmarks[LEFT_IRIS_CENTER]
                    right_iris = landmarks[RIGHT_IRIS_CENTER]

                    draw_gaze_arrow(frame, left_iris, gaze_x, gaze_y)
                    draw_gaze_arrow(frame, right_iris, gaze_x, gaze_y)

                    # Draw iris circles
                    cv2.circle(frame, tuple(left_iris), 3, (0, 255, 255), -1)
                    cv2.circle(frame, tuple(right_iris), 3, (0, 255, 255), -1)
            else:
                print(
                    f"\r  Faces: 0 | Gaze: --           "
                    f"                              ",
                    end="", flush=True
                )

            current_time = time.time()

            # ── Update status file periodically ─────────────────────────
            if current_time - last_status_update_time >= STATUS_UPDATE_INTERVAL:
                update_status_file(
                    faces_detected, last_gaze_direction,
                    last_gaze_x, last_gaze_y, last_gaze_z,
                    last_head_yaw, last_head_pitch, last_head_roll,
                )
                last_status_update_time = current_time

            # ── Screenshot ──────────────────────────────────────────────
            if current_time - last_screenshot_time >= SCREENSHOT_UPDATE_INTERVAL:
                try:
                    cv2.imwrite(str(SCREENSHOT_FILE), frame)
                    last_screenshot_time = current_time
                except Exception as e:
                    log_event(f"WARNING: Could not save screenshot: {e}")

            # ── Display window ──────────────────────────────────────────
            if args.display:
                direction = last_gaze_direction.upper()
                color = (0, 255, 0) if direction == "CENTER" else (0, 165, 255)
                cv2.putText(
                    frame, f"Gaze: {direction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
                )

                cv2.imshow("Gaze Detector (Webcam)", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            # FPS limiting
            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        log_event("\nGaze detector stopped")

    finally:
        landmarker.close()
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        update_status_file(0, "unknown", 0, 0, 0, 0, 0, 0, running=False)
        if log_file:
            log_file.close()


if __name__ == "__main__":
    run_detection()
