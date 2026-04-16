#!/usr/bin/env python3
"""
Whiteboard OCR Reader for OAK-D (DepthAI 3.x)
==============================================
Two-stage OCR pipeline: PaddlePaddle text detection + recognition.
Reads text from whiteboards and supports Discord bot integration.

Adapted from Luxonis oak-examples general-ocr.

Usage:
    python3 whiteboard_reader.py                # Basic OCR
    python3 whiteboard_reader.py --log          # Log detected text
    python3 whiteboard_reader.py --discord      # Enable Discord notifications
    python3 whiteboard_reader.py --display      # Show live window with text boxes
"""

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, GatherData
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

# Load environment variables for Discord webhook
try:
    from dotenv import load_dotenv
    load_dotenv()
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
    description='OAK-D Whiteboard OCR Reader (DepthAI 3.x)')
parser.add_argument('--log', action='store_true', help='Log detected text to file')
parser.add_argument('--discord', action='store_true',
                    help='Enable Discord notifications for text changes')
parser.add_argument('--discord-quiet', action='store_true',
                    help='Only send Discord notifications when new text appears (not when cleared)')
parser.add_argument('--display', action='store_true',
                    help='Show live detection window with text boxes')
parser.add_argument('--fps-limit', type=int, default=None,
                    help='FPS limit (default: 5 for RVC2, 30 for RVC4)')
parser.add_argument('--device', type=str, default=None,
                    help='Optional DeviceID or IP of the camera')
parser.add_argument('--focus', type=str, default='continuous',
                    help='Focus mode: "continuous" (default), "auto" (one-shot), '
                         'or 0-255 for manual lens position (0=far, 255=close)')
args = parser.parse_args()

# Camera resolution (larger than model input to keep detail)
REQ_WIDTH, REQ_HEIGHT = 1152, 640

# Global state tracking
log_file = None
last_text_content = []  # List of detected text lines
last_text_detected = False

# Temporal smoothing for text detection
text_detection_history = deque(maxlen=5)  # Track last 5 frames

# Debouncing for Discord notifications
pending_state = None
pending_state_time = None
DEBOUNCE_SECONDS = 2.0  # Wait 2 seconds before notifying

# Status file for Discord bot integration
STATUS_FILE = Path.home() / "oak-projects" / "whiteboard_status.json"
STATUS_UPDATE_INTERVAL = 10  # Update status file every 10 seconds
last_status_update_time = 0

# Screenshot for Discord bot
SCREENSHOT_FILE = Path.home() / "oak-projects" / "latest_whiteboard_frame.jpg"
SCREENSHOT_UPDATE_INTERVAL = 5  # Save screenshot every 5 seconds
last_screenshot_time = 0


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

    send_notification(message, add_timestamp=False)


def update_status_file(text_detected: bool, text_content: list, num_regions: int,
                       running: bool = True, username: str = None, hostname: str = None):
    """Update status file for Discord bot integration."""
    try:
        status_data = {
            "text_detected": text_detected,
            "text_content": text_content,
            "num_text_regions": num_regions,
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


def decode_text(text_bytes):
    """Decode recognized text from network output."""
    # PaddleOCR recognition output is a list of character probabilities
    # We need to decode the most likely sequence
    if not text_bytes or len(text_bytes) == 0:
        return ""

    # This is a simplified decoder - the actual paddle model uses CTC decoding
    # For now, return a placeholder that indicates text was found
    return "[Text detected - see visualization]"


def process_detections(det_msg, rec_results):
    """Process detection and recognition results to extract text."""
    detected_texts = []

    if not hasattr(det_msg, 'detections'):
        return detected_texts

    for i, detection in enumerate(det_msg.detections):
        if i < len(rec_results):
            # Get recognition result
            rec_msg = rec_results[i]

            # Decode text (simplified - actual paddle model has its own decoder)
            text = decode_text(rec_msg)

            if text and text.strip():
                detected_texts.append({
                    'text': text.strip(),
                    'bbox': {
                        'x1': detection.xmin,
                        'y1': detection.ymin,
                        'x2': detection.xmax,
                        'y2': detection.ymax
                    },
                    'confidence': getattr(detection, 'confidence', 0.0)
                })

    return detected_texts


def draw_text_boxes(frame, detections):
    """Draw bounding boxes around detected text regions."""
    if not detections:
        return frame

    h, w = frame.shape[:2]

    for det in detections:
        bbox = det['bbox']
        # Convert normalized coordinates to pixels
        x1 = int(bbox['x1'] * w)
        y1 = int(bbox['y1'] * h)
        x2 = int(bbox['x2'] * w)
        y2 = int(bbox['y2'] * h)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw text label
        text = det['text'][:30]  # Truncate long text
        label_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Background for text
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), (0, 255, 0), -1)

        # Text
        cv2.putText(frame, text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame


def draw_info_banner(frame, info_items):
    """Draw a semi-transparent info banner at the bottom of the frame.

    Args:
        frame: OpenCV BGR image (numpy array)
        info_items: list of strings to display, pipe-separated
    Returns:
        frame with banner drawn
    """
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
    """Main OCR detection loop using DepthAI 3.x."""
    global log_file, last_text_content, last_text_detected
    global pending_state, pending_state_time
    global last_status_update_time, last_screenshot_time

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
        log_filename = f"whiteboard_ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file = open(log_filename, 'w')
        log_event(f"Logging to {log_filename}")

    startup_msg = "Whiteboard OCR reader started (DepthAI 3.x with PaddlePaddle OCR)"
    log_event(startup_msg)
    if args.discord:
        log_event("Discord notifications: ENABLED")
    if args.display:
        log_event("Live display: ENABLED (press 'q' to quit)")
    log_event("Press Ctrl+C to exit\n")

    # Initialize status file
    update_status_file(text_detected=False, text_content=[], num_regions=0,
                      running=True, username=username, hostname=hostname)
    last_status_update_time = time.time()

    # Send startup notification to Discord
    if args.discord:
        discord_startup = f"📋 **{username}** is now running whiteboard_reader.py on **{hostname}**"
        send_discord_notification(discord_startup)

    try:
        # Connect to device
        if args.device:
            device = dai.Device(dai.DeviceInfo(args.device))
        else:
            device = dai.Device()

        platform = device.getPlatform().name
        log_event(f"Connected to device: {device.getDeviceId()}")
        log_event(f"Platform: {platform}")

        # Set FPS limit based on platform
        fps_limit = args.fps_limit
        if fps_limit is None:
            fps_limit = 5 if platform == "RVC2" else 30
            log_event(f"FPS limit set to {fps_limit} for {platform}")

        frame_type = (
            dai.ImgFrame.Type.BGR888i if platform == "RVC4"
            else dai.ImgFrame.Type.BGR888p
        )

        # Create pipeline
        with dai.Pipeline(device) as pipeline:
            log_event("Creating OCR pipeline...")

            # Load text detection model (Stage 1)
            det_model_path = Path(__file__).parent / "depthai_models" / f"paddle_text_detection.{platform}.yaml"
            if not det_model_path.exists():
                log_event(f"ERROR: Detection model not found at {det_model_path}")
                log_event("Please copy model files from oak-examples/neural-networks/ocr/general-ocr/depthai_models/")
                return

            det_model_description = dai.NNModelDescription.fromYamlFile(str(det_model_path))
            det_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(det_model_description))
            det_model_w, det_model_h = det_model_nn_archive.getInputSize()

            # Load text recognition model (Stage 2)
            rec_model_path = Path(__file__).parent / "depthai_models" / f"paddle_text_recognition.{platform}.yaml"
            if not rec_model_path.exists():
                log_event(f"ERROR: Recognition model not found at {rec_model_path}")
                log_event("Please copy model files from oak-examples/neural-networks/ocr/general-ocr/depthai_models/")
                return

            rec_model_description = dai.NNModelDescription.fromYamlFile(str(rec_model_path))
            rec_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(rec_model_description))
            rec_model_w, rec_model_h = rec_model_nn_archive.getInputSize()

            # Camera input
            cam = pipeline.create(dai.node.Camera).build()

            # Configure focus
            try:
                focus_val = int(args.focus)
                # Manual focus: 0 = infinity/far, 255 = close-up
                cam.initialControl.setManualFocus(max(0, min(255, focus_val)))
                log_event(f"Focus: manual (lens position {focus_val})")
            except ValueError:
                if args.focus == 'continuous':
                    cam.initialControl.setAutoFocusMode(
                        dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO
                    )
                    log_event("Focus: continuous autofocus")
                else:
                    # 'auto' or anything else — one-shot autofocus (default)
                    cam.initialControl.setAutoFocusMode(
                        dai.CameraControl.AutoFocusMode.AUTO
                    )
                    log_event("Focus: one-shot autofocus")

            cam_out = cam.requestOutput(
                size=(REQ_WIDTH, REQ_HEIGHT), type=frame_type, fps=fps_limit
            )

            # Resize to detection model input size
            resize_node = pipeline.create(dai.node.ImageManip)
            resize_node.initialConfig.setOutputSize(det_model_w, det_model_h)
            resize_node.initialConfig.setReusePreviousImage(False)
            resize_node.inputImage.setBlocking(True)
            cam_out.link(resize_node.inputImage)

            # Text detection neural network
            det_nn = pipeline.create(ParsingNeuralNetwork).build(
                resize_node.out, det_model_nn_archive
            )

            # Get output queues (MUST be created before pipeline.start())
            q_det = det_nn.out.createOutputQueue(maxSize=4, blocking=False)
            q_preview = cam_out.createOutputQueue(maxSize=4, blocking=False)

            log_event("Pipeline created.")

            # Start pipeline
            pipeline.start()
            log_event("OCR detection started. Monitoring whiteboard...\n")

            # Create window if display is enabled
            if args.display:
                cv2.namedWindow("Whiteboard OCR", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Whiteboard OCR", 1152, 640)

            # Track latest detections
            latest_detections = []

            while pipeline.isRunning():
                # Get detection results
                det_msg = q_det.tryGet()

                # Get preview frame
                preview_frame = q_preview.tryGet()

                if det_msg is not None:
                    # Count text regions detected
                    num_regions = 0
                    if hasattr(det_msg, 'detections'):
                        num_regions = len(det_msg.detections)

                    text_detected = num_regions > 0
                    text_detection_history.append(text_detected)

                    # Smoothed detection (majority vote from last 5 frames)
                    smoothed_detection = sum(text_detection_history) >= len(text_detection_history) / 2

                    current_time = time.time()

                    # Update console status
                    print(f"\r  Text regions: {num_regions} | Detected: {'YES' if smoothed_detection else 'NO'}  ",
                          end="", flush=True)

                    # Debouncing logic for state changes
                    if smoothed_detection != last_text_detected:
                        if pending_state == smoothed_detection:
                            if current_time - pending_state_time >= DEBOUNCE_SECONDS:
                                # Confirm state change
                                if smoothed_detection:
                                    log_event(f"\nTEXT DETECTED ({num_regions} regions)")
                                    if args.discord:
                                        send_discord_notification(
                                            f"📝 Text detected on whiteboard ({num_regions} regions)")
                                else:
                                    log_event("\nWhiteboard cleared - no text detected")
                                    if args.discord and not args.discord_quiet:
                                        send_discord_notification("🗑️ Whiteboard cleared")

                                last_text_detected = smoothed_detection
                                pending_state = None
                                pending_state_time = None

                                # Update status file
                                update_status_file(
                                    smoothed_detection, [], num_regions,
                                    running=True, username=username, hostname=hostname)
                        else:
                            # New pending state
                            pending_state = smoothed_detection
                            pending_state_time = current_time
                    else:
                        # State matches - reset pending
                        pending_state = None
                        pending_state_time = None

                # Display frame with text boxes if enabled
                if args.display and preview_frame is not None:
                    frame = preview_frame.getCvFrame()

                    # Draw text boxes if we have detection message
                    if det_msg is not None and hasattr(det_msg, 'detections'):
                        # Draw simple boxes (full recognition would need stage 2)
                        for detection in det_msg.detections:
                            x1 = int(detection.xmin * frame.shape[1])
                            y1 = int(detection.ymin * frame.shape[0])
                            x2 = int(detection.xmax * frame.shape[1])
                            y2 = int(detection.ymax * frame.shape[0])

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Add status text
                    status_text = f"Text Regions: {num_regions if det_msg else 0}"
                    cv2.putText(frame, status_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"User: {username}@{hostname}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    # Show frame
                    cv2.imshow("Whiteboard OCR", frame)

                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        log_event("Display window closed by user")
                        break

                # Periodic status file update
                current_time = time.time()
                if current_time - last_status_update_time >= STATUS_UPDATE_INTERVAL:
                    detected = last_text_detected
                    regions = num_regions if det_msg else 0
                    update_status_file(detected, [], regions,
                                     running=True, username=username, hostname=hostname)
                    last_status_update_time = current_time

                # Periodic screenshot save
                if preview_frame is not None and current_time - last_screenshot_time >= SCREENSHOT_UPDATE_INTERVAL:
                    try:
                        frame = preview_frame.getCvFrame()

                        # Draw text boxes on screenshot
                        if det_msg is not None and hasattr(det_msg, 'detections'):
                            for detection in det_msg.detections:
                                x1 = int(detection.xmin * frame.shape[1])
                                y1 = int(detection.ymin * frame.shape[0])
                                x2 = int(detection.xmax * frame.shape[1])
                                y2 = int(detection.ymax * frame.shape[0])

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Add info banner
                        info_items = [
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            f"{len(det_msg.detections) if det_msg is not None and hasattr(det_msg, 'detections') else 0} regions",
                            f"{username}@{hostname}",
                        ]
                        frame = draw_info_banner(frame, info_items)

                        cv2.imwrite(str(SCREENSHOT_FILE), frame)
                        last_screenshot_time = current_time
                    except Exception as e:
                        log_event(f"WARNING: Could not save screenshot: {e}")

                # Small sleep to prevent CPU spinning
                time.sleep(0.01)

    except KeyboardInterrupt:
        shutdown_msg = "Whiteboard OCR reader stopped"
        log_event(f"\n{shutdown_msg}")
        if args.discord:
            discord_shutdown = f"📴 **{username}** stopped whiteboard_reader.py on **{hostname}** - camera is free"
            send_discord_notification(discord_shutdown)

    finally:
        if args.display:
            cv2.destroyAllWindows()
        if log_file:
            log_file.close()

        # Mark as not running in status file
        update_status_file(False, [], 0, running=False, username=username, hostname=hostname)


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
