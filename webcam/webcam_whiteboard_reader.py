#!/usr/bin/env python3
"""
Webcam Whiteboard OCR Reader (Tesseract)
========================================
Reads text from whiteboards using the laptop webcam and Tesseract OCR.
Writes the same status files as the OAK-D version so the Discord bot
works unchanged.

Requires: brew install tesseract

Usage:
    python3 webcam/webcam_whiteboard_reader.py                # Basic OCR
    python3 webcam/webcam_whiteboard_reader.py --display       # Show live window
    python3 webcam/webcam_whiteboard_reader.py --discord       # Enable Discord notifications
    python3 webcam/webcam_whiteboard_reader.py --log           # Log detected text
"""

import sys
import argparse
import time
import os
import json
import cv2
import numpy as np
import socket
import getpass
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

# Import Discord notifier
try:
    from discord_notifier import send_notification
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

# Parse arguments
parser = argparse.ArgumentParser(
    description='Webcam Whiteboard OCR Reader (Tesseract)')
parser.add_argument('--log', action='store_true', help='Log detected text to file')
parser.add_argument('--discord', action='store_true',
                    help='Enable Discord notifications for text changes')
parser.add_argument('--discord-quiet', action='store_true',
                    help='Only send Discord notifications when new text appears (not when cleared)')
parser.add_argument('--display', action='store_true',
                    help='Show live detection window with text boxes')
parser.add_argument('--confidence', type=int, default=60,
                    help='Minimum OCR confidence threshold 0-100 (default: 60)')
parser.add_argument('--camera', type=int, default=0,
                    help='Camera index (default: 0 for built-in webcam)')
parser.add_argument('--fps-limit', type=int, default=5,
                    help='FPS limit — OCR is slow, keep this low (default: 5)')
args = parser.parse_args()

# Global state tracking
log_file = None
last_text_content = []
last_text_detected = False

# Temporal smoothing for text detection
text_detection_history = deque(maxlen=5)

# Debouncing for Discord notifications
pending_state = None
pending_state_time = None
DEBOUNCE_SECONDS = 2.0

# Status file for Discord bot integration
STATUS_FILE = Path.home() / "oak-projects" / "whiteboard_status.json"
STATUS_UPDATE_INTERVAL = 10
last_status_update_time = 0

# Screenshot for Discord bot
SCREENSHOT_FILE = Path.home() / "oak-projects" / "latest_whiteboard_frame.jpg"
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


def update_status_file(text_detected, text_content, num_regions,
                       running=True, username=None, hostname=None):
    """Update status file for Discord bot integration."""
    try:
        status_data = {
            "text_detected": text_detected,
            "text_content": text_content,
            "num_text_regions": num_regions,
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


def preprocess_for_ocr(frame):
    """Preprocess a frame for better OCR accuracy on whiteboard text.

    Converts to grayscale and applies adaptive thresholding to handle
    varying lighting conditions typical of whiteboard photos.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding handles uneven lighting on whiteboards
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    return thresh


def extract_text(frame, confidence_threshold):
    """Run Tesseract OCR on a frame and extract text with bounding boxes.

    Args:
        frame: BGR OpenCV frame
        confidence_threshold: Minimum confidence (0-100) to accept a word

    Returns:
        (text_lines, boxes) where:
          text_lines: list of strings (one per detected line)
          boxes: list of (x, y, w, h, text, conf) tuples for drawing
    """
    import pytesseract

    processed = preprocess_for_ocr(frame)

    # Get word-level data with positions and confidence
    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)

    # Group words into lines
    lines = {}  # (block_num, line_num) -> list of words
    boxes = []

    n_words = len(data['text'])
    for i in range(n_words):
        conf = int(data['conf'][i])
        text = data['text'][i].strip()

        if conf < confidence_threshold or not text:
            continue

        block = data['block_num'][i]
        line = data['line_num'][i]
        key = (block, line)

        if key not in lines:
            lines[key] = []
        lines[key].append(text)

        # Store box for visualization
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        boxes.append((x, y, w, h, text, conf))

    # Build text lines
    text_lines = [" ".join(words) for words in lines.values()]
    # Filter out very short lines (likely noise)
    text_lines = [line for line in text_lines if len(line) >= 3]

    return text_lines, boxes


def draw_info_banner(frame, info_items):
    """Draw a semi-transparent info banner at the bottom of the frame."""
    h, w = frame.shape[:2]
    banner_h = 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    text = " | ".join(info_items)
    cv2.putText(frame, text, (10, h - banner_h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def run_detection():
    """Main OCR detection loop using webcam + Tesseract."""
    global log_file, last_text_content, last_text_detected
    global pending_state, pending_state_time
    global last_status_update_time, last_screenshot_time

    # Import pytesseract here so startup errors are clear
    try:
        import pytesseract
        # Test that tesseract binary is available
        pytesseract.get_tesseract_version()
    except ImportError:
        print("ERROR: pytesseract not installed.")
        print("   Install with: pip install pytesseract")
        sys.exit(1)
    except EnvironmentError:
        print("ERROR: Tesseract OCR engine not found.")
        print("   Install with: brew install tesseract")
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
        log_filename = f"whiteboard_ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file = open(log_filename, 'w')
        log_event(f"Logging to {log_filename}")

    log_event("Whiteboard OCR reader started (Webcam + Tesseract)")
    log_event(f"OCR confidence threshold: {args.confidence}")
    if args.discord:
        log_event("Discord notifications: ENABLED")
    if args.display:
        log_event("Live display: ENABLED (press 'q' to quit)")
    log_event("Press Ctrl+C to exit\n")

    # Initialize status file
    update_status_file(text_detected=False, text_content=[], num_regions=0,
                       running=True, username=username, hostname=hostname)
    last_status_update_time = time.time()

    # Send startup notification
    if args.discord:
        discord_startup = f"📋 **{username}** is now running webcam_whiteboard_reader.py on **{hostname}**"
        send_discord_notification(discord_startup)

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        log_event(f"ERROR: Could not open camera {args.camera}")
        log_event("   Check System Preferences > Privacy > Camera")
        sys.exit(1)

    log_event(f"Camera {args.camera} opened.")
    log_event("OCR detection started. Monitoring whiteboard...\n")

    frame_interval = 1.0 / args.fps_limit
    num_regions = 0

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Run OCR
            text_lines, boxes = extract_text(frame, args.confidence)
            num_regions = len(boxes)
            text_detected = len(text_lines) > 0

            text_detection_history.append(text_detected)

            # Smoothed detection (majority vote from last 5 frames)
            smoothed_detection = sum(text_detection_history) >= len(text_detection_history) / 2

            current_time = time.time()

            # Console status line
            preview = text_lines[0][:40] if text_lines else ""
            print(
                f"\r  Text regions: {num_regions} | "
                f"Detected: {'YES' if smoothed_detection else 'NO'} "
                f"| {preview}  ",
                end="", flush=True
            )

            # ── Debouncing logic for state changes ──────────────────────
            if smoothed_detection != last_text_detected:
                if pending_state == smoothed_detection:
                    if current_time - pending_state_time >= DEBOUNCE_SECONDS:
                        if smoothed_detection:
                            log_event(f"\nTEXT DETECTED ({num_regions} words, {len(text_lines)} lines)")
                            for line in text_lines[:5]:
                                log_event(f"   > {line}")
                            if args.discord:
                                send_discord_notification(
                                    f"📝 Text detected on whiteboard ({len(text_lines)} lines)")
                        else:
                            log_event("\nWhiteboard cleared - no text detected")
                            if args.discord and not args.discord_quiet:
                                send_discord_notification("🗑️ Whiteboard cleared")

                        last_text_detected = smoothed_detection
                        last_text_content = text_lines
                        pending_state = None
                        pending_state_time = None

                        update_status_file(
                            smoothed_detection, text_lines, num_regions,
                            running=True, username=username, hostname=hostname)
                else:
                    pending_state = smoothed_detection
                    pending_state_time = current_time
            else:
                pending_state = None
                pending_state_time = None

            # ── Display window ──────────────────────────────────────────
            if args.display:
                display_frame = frame.copy()

                # Draw bounding boxes around detected words
                for (x, y, w, h, text, conf) in boxes:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{text} ({conf}%)", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # Status text
                cv2.putText(display_frame,
                            f"Text Regions: {num_regions}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame,
                            f"User: {username}@{hostname}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("Whiteboard OCR (Webcam)", display_frame)
                if cv2.waitKey(1) == ord('q'):
                    log_event("Display window closed by user")
                    break

            # ── Periodic status file update ─────────────────────────────
            if current_time - last_status_update_time >= STATUS_UPDATE_INTERVAL:
                detected = last_text_detected
                update_status_file(detected, last_text_content, num_regions,
                                   running=True, username=username, hostname=hostname)
                last_status_update_time = current_time

            # ── Periodic screenshot ─────────────────────────────────────
            if current_time - last_screenshot_time >= SCREENSHOT_UPDATE_INTERVAL:
                try:
                    screenshot = frame.copy()
                    # Draw boxes on screenshot
                    for (x, y, w, h, text, conf) in boxes:
                        cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Add info banner
                    info_items = [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        f"{num_regions} regions",
                        f"{username}@{hostname}",
                    ]
                    screenshot = draw_info_banner(screenshot, info_items)

                    cv2.imwrite(str(SCREENSHOT_FILE), screenshot)
                    last_screenshot_time = current_time
                except Exception as e:
                    log_event(f"WARNING: Could not save screenshot: {e}")

            # FPS limiting (OCR is slow, so this mainly prevents spinning
            # when OCR finishes faster than expected)
            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        log_event("\nWhiteboard OCR reader stopped")
        if args.discord:
            discord_shutdown = f"📴 **{username}** stopped webcam_whiteboard_reader.py on **{hostname}** - camera is free"
            send_discord_notification(discord_shutdown)

    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        if log_file:
            log_file.close()
        update_status_file(False, [], 0, running=False,
                           username=username, hostname=hostname)


if __name__ == "__main__":
    if args.discord and not DISCORD_AVAILABLE:
        print("ERROR: Discord notifications requested but discord_notifier.py not found")
        print("   Make sure discord_notifier.py is in the repository root")
        sys.exit(1)

    if args.discord and not DOTENV_AVAILABLE:
        print("WARNING: python-dotenv not installed - ensure DISCORD_WEBHOOK_URL is in environment")
        print("   Install with: pip install python-dotenv")

    run_detection()
