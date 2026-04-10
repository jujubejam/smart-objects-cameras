#!/usr/bin/env python3
"""
Simple Person Detector for OAK-D (DepthAI 3.x)
===============================================
Outputs timestamped person detection events using YOLO.
Optionally sends notifications to Discord.

Usage:
    python3 person_detector.py              # Basic detection
    python3 person_detector.py --log        # Log to file
    python3 person_detector.py --discord    # Enable Discord notifications
"""

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
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
    description='OAK-D Person Detector (DepthAI 3.x)')
parser.add_argument('--log', action='store_true', help='Log events to file')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Detection confidence threshold (0-1)')
parser.add_argument('--discord', action='store_true',
                    help='Enable Discord notifications')
parser.add_argument('--discord-quiet', action='store_true',
                    help='Only send Discord notifications for person detected (not when clear)')
parser.add_argument('--model', type=str, default='luxonis/yolov6-nano:r2-coco-512x288',
                    help='Model reference from Luxonis Hub')
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

# COCO class names - person is class 0
COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle',
                'airplane', 'bus', 'train', 'truck', 'boat']

# ── Classroom API integration ────────────────────────────────────────────────
CLASSROOM_API_URL = os.getenv("CLASSROOM_API_URL", "")
CLASSROOM_API_KEY = os.getenv("CLASSROOM_API_KEY", "")


def push_to_classroom(camera_id: str, detected: bool, count: int,
                      username: str = None, hostname: str = None):
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


def log_event(message: str):
    """Print and optionally log an event."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)

    if log_file:
        log_file.write(line + "\n")
        log_file.flush()


def send_discord_notification(message: str, force: bool = False):
    """Send Discord notification if enabled."""
    if not args.discord and not force:
        return

    if not DISCORD_AVAILABLE:
        return

    if not os.getenv('DISCORD_WEBHOOK_URL'):
        if force:  # Only warn on startup
            log_event(
                "WARNING: Discord notifications requested but DISCORD_WEBHOOK_URL not set")
        return

    # Send notification (don't add timestamp as it's already in the message)
    send_notification(message, add_timestamp=False)


def update_status_file(detected: bool, count: int, running: bool = True, username: str = None, hostname: str = None):
    """Update status file for Discord bot integration."""
    try:
        status_data = {
            "detected": detected,
            "count": count,
            "timestamp": datetime.now().isoformat(),
            "running": running
        }

        # Add user and hostname if provided
        if username:
            status_data["username"] = username
        if hostname:
            status_data["hostname"] = hostname

        STATUS_FILE.write_text(json.dumps(status_data, indent=2))
    except Exception as e:
        log_event(f"WARNING: Could not update status file: {e}")


def run_detection():
    """Main detection loop using DepthAI 3.x."""
    global log_file, last_status, last_count, pending_state, pending_state_time, last_status_update_time, last_screenshot_time

    # Get user and hostname for smart object announcements
    try:
        username = getpass.getuser()
    except:
        username = os.getenv('USER', 'unknown')

    try:
        hostname = socket.gethostname()
    except:
        hostname = 'unknown'

    # Open log file if requested
    if args.log:
        log_filename = f"person_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file = open(log_filename, 'w')
        log_event(f"Logging to {log_filename}")

    startup_msg = f"Person detector started (DepthAI 3.x with {args.model})"
    log_event(startup_msg)
    log_event(f"Confidence threshold: {args.threshold}")
    if args.discord:
        log_event("Discord notifications: ENABLED")
    log_event("Press Ctrl+C to exit\n")

    # Initialize status file
    update_status_file(detected=False, count=0, running=True, username=username, hostname=hostname)
    last_status_update_time = time.time()  # Initialize timestamp

    # Send startup notification to Discord with user and host info
    if args.discord:
        discord_startup = f"🎥 **{username}** is now running person_detector.py on **{hostname}**"
        send_discord_notification(discord_startup)

    try:
        # Connect to device
        device = dai.Device()
        platform = device.getPlatformAsString()
        log_event(f"Connected to device: {device.getDeviceId()}")
        log_event(f"Platform: {platform}")

        # Create pipeline
        with dai.Pipeline(device) as pipeline:
            log_event("Creating pipeline...")

            # Load model from Luxonis Hub
            model_description = dai.NNModelDescription(
                args.model, platform=platform)
            nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))

            # Create camera input (using ColorCamera - Camera node API is different)
            cam_rgb = pipeline.create(dai.node.ColorCamera)
            cam_rgb.setPreviewSize(512, 288)  # Match model input size
            cam_rgb.setInterleaved(False)
            cam_rgb.setFps(15)

            # Create neural network with parser
            nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
                cam_rgb.preview, nn_archive
            )

            # Get output queues (MUST be created before pipeline.start())
            q_det = nn_with_parser.out.createOutputQueue(
                maxSize=4, blocking=False)
            # Get preview frames directly from camera for screenshots
            q_preview = cam_rgb.preview.createOutputQueue(
                maxSize=4, blocking=False)

            log_event("Pipeline created.")

            # Start pipeline
            pipeline.start()

            log_event("Detection started. Monitoring for people...\n")

            while pipeline.isRunning():
                # Get detection results
                detections_msg = q_det.tryGet()

                # Get preview frame for screenshot
                preview_frame = q_preview.tryGet()

                if detections_msg is not None:
                    # Filter for person detections only (class 0 in COCO)
                    if hasattr(detections_msg, 'detections'):
                        all_detections = detections_msg.detections
                        person_detections = [d for d in all_detections
                                             if d.label == 0 and d.confidence >= args.threshold]
                        person_count = len(person_detections)
                    else:
                        person_count = 0

                    person_detected = person_count > 0
                    current_time = time.time()

                    # Debouncing logic: only change state if it persists
                    if person_detected != last_status:
                        # State is different from confirmed state
                        if pending_state == person_detected:
                            # Same pending state - check if enough time has passed
                            if current_time - pending_state_time >= DEBOUNCE_SECONDS:
                                # Confirm the state change
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

                                # Update status file
                                update_status_file(
                                    person_detected, person_count, running=True, username=username, hostname=hostname)
                                push_to_classroom(hostname, person_detected, person_count, username, hostname)
                        else:
                            # New pending state - start the timer
                            pending_state = person_detected
                            pending_state_time = current_time
                    else:
                        # State matches confirmed state - reset pending
                        pending_state = None
                        pending_state_time = None

                    # Update count for display purposes (but don't trigger notifications)
                    if person_count != last_count and person_detected and last_status:
                        log_event(f"   Count changed: {person_count} people")
                        last_count = person_count

                # Periodic status file update (even when nothing changes)
                current_time = time.time()
                if current_time - last_status_update_time >= STATUS_UPDATE_INTERVAL:
                    # Update status file with current state
                    detected = last_status if last_status is not None else False
                    count = last_count if last_status else 0
                    update_status_file(detected, count, running=True, username=username, hostname=hostname)
                    push_to_classroom(hostname, detected, count, username, hostname)
                    last_status_update_time = current_time

                # Periodic screenshot save (for Discord bot)
                if preview_frame is not None and current_time - last_screenshot_time >= SCREENSHOT_UPDATE_INTERVAL:
                    try:
                        # Get frame data as numpy array
                        frame = preview_frame.getCvFrame()
                        # Save as JPEG
                        cv2.imwrite(str(SCREENSHOT_FILE), frame)
                        last_screenshot_time = current_time
                    except Exception as e:
                        log_event(f"WARNING: Could not save screenshot: {e}")

                # Small sleep to prevent CPU spinning
                time.sleep(0.01)

    except KeyboardInterrupt:
        shutdown_msg = "Person detector stopped"
        log_event(f"\n{shutdown_msg}")
        # Send shutdown notification to Discord with user and host info
        if args.discord:
            discord_shutdown = f"📴 **{username}** stopped person_detector.py on **{hostname}** - camera is free"
            send_discord_notification(discord_shutdown)

    finally:
        if log_file:
            log_file.close()


if __name__ == "__main__":
    # Check if Discord is requested but not available
    if args.discord and not DISCORD_AVAILABLE:
        print("ERROR: Discord notifications requested but discord_notifier.py not found")
        print("   Make sure discord_notifier.py is in the same directory")
        import sys
        sys.exit(1)

    if args.discord and not DOTENV_AVAILABLE:
        print("WARNING: python-dotenv not installed - ensure DISCORD_WEBHOOK_URL is in environment")
        print("   Install with: pip install python-dotenv")

    run_detection()
