#!/usr/bin/env python3
"""
Whiteboard OCR Reader with FULL Text Recognition (DepthAI 3.x)
================================================================
Complete two-stage OCR pipeline: PaddlePaddle text detection + recognition.
Reads AND extracts text content from whiteboards.

Adapted from Luxonis oak-examples general-ocr.

Usage:
    python3 whiteboard_reader_full.py                # Full OCR with text content
    python3 whiteboard_reader_full.py --log          # Log detected text
    python3 whiteboard_reader_full.py --discord      # Enable Discord notifications
    python3 whiteboard_reader_full.py --display      # Show live window with text
"""

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, GatherData
from utils.ocr_crop_creator import CropConfigsCreator
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
from difflib import SequenceMatcher
import random

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
    description='OAK-D Whiteboard OCR Reader - Full Text Recognition (DepthAI 3.x)')
parser.add_argument('--log', action='store_true', help='Log detected text to file')
parser.add_argument('--discord', action='store_true',
                    help='Enable Discord notifications for text changes')
parser.add_argument('--discord-quiet', action='store_true',
                    help='Only send Discord notifications when new text appears (not when cleared)')
parser.add_argument('--display', action='store_true',
                    help='Show live detection window with text overlay')
parser.add_argument('--fps-limit', type=int, default=None,
                    help='FPS limit (default: 5 for RVC2, 30 for RVC4)')
parser.add_argument('--device', type=str, default=None,
                    help='Optional DeviceID or IP of the camera')
parser.add_argument('--confidence', type=float, default=0.25,
                    help='Minimum confidence threshold for text recognition (default: 0.25)')
args = parser.parse_args()

# Camera resolution (larger than model input to keep detail)
REQ_WIDTH, REQ_HEIGHT = 1152, 640

# Global state tracking
log_file = None
last_text_content = []  # List of detected text lines
last_text_detected = False
last_confirmed_text = []  # Last stable/confirmed text (after debouncing)

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

# History log for text tracking over time (JSONL format)
HISTORY_FILE = Path.home() / "oak-projects" / "whiteboard_history.jsonl"

# Config file for runtime reconfiguration via Discord bot
CONFIG_FILE = Path.home() / "oak-projects" / "whiteboard_config.json"
CONFIG_CHECK_INTERVAL = 2  # Check every 2 seconds
last_config_check = 0

# Smart feedback interval (don't spam suggestions)
FEEDBACK_INTERVAL = 30  # Seconds between feedback messages
last_feedback_time = 0


def log_event(message: str):
    """Print and optionally log an event."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'))

    if log_file:
        log_file.write(line + "\n")
        log_file.flush()


def load_config():
    """Read config file and return dict with defaults if file missing or invalid."""
    defaults = {
        "confidence": 0.25,
        "fps_limit": 5,
        "notifications_enabled": True
    }
    if not CONFIG_FILE.exists():
        return defaults
    try:
        config = json.loads(CONFIG_FILE.read_text())
        # Merge with defaults so missing keys get default values
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        return config
    except (json.JSONDecodeError, OSError) as e:
        log_event(f"WARNING: Could not read config file: {e}")
        return defaults


def write_default_config():
    """Write default config on startup if file doesn't exist."""
    if CONFIG_FILE.exists():
        return
    try:
        default_config = {
            "confidence": args.confidence,
            "fps_limit": args.fps_limit if args.fps_limit else 5,
            "notifications_enabled": True
        }
        CONFIG_FILE.write_text(json.dumps(default_config, indent=2))
        log_event(f"Created default config file: {CONFIG_FILE}")
    except OSError as e:
        log_event(f"WARNING: Could not write default config: {e}")


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


def log_text_history(text_lines: list, num_regions: int, avg_confidence: float = 0.0):
    """
    Log text detection to history file (JSONL format).

    Each line in the history file is a complete JSON object representing
    one detection event. This allows time-based analysis and tracking
    of text changes over time.

    Args:
        text_lines: List of recognized text strings
        num_regions: Number of text regions detected
        avg_confidence: Average confidence score across all recognitions
    """
    try:
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "text_lines": text_lines,
            "num_regions": num_regions,
            "avg_confidence": round(avg_confidence, 3)
        }

        # Append to JSONL file (one JSON object per line)
        with open(HISTORY_FILE, 'a') as f:
            f.write(json.dumps(history_entry) + '\n')

    except Exception as e:
        log_event(f"WARNING: Could not log text history: {e}")


def string_similarity(a: str, b: str) -> float:
    """
    Calculate similarity between two strings (0.0 to 1.0).
    Uses SequenceMatcher for fuzzy matching.
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def detect_text_changes(current_text: list, previous_text: list):
    """
    Detect what changed between two sets of text lines.

    Returns a dict with:
        - change_type: 'new', 'edited', 'removed', 'stable', 'camera_moved', or 'none'
        - new_lines: Lines that appeared
        - removed_lines: Lines that disappeared
        - edited_pairs: [(old, new)] pairs of similar lines
        - similarity: Overall similarity score (0.0-1.0)
    """
    if not current_text and not previous_text:
        return {'change_type': 'none', 'new_lines': [], 'removed_lines': [], 'edited_pairs': [], 'similarity': 1.0}

    if not previous_text:
        # First detection
        return {'change_type': 'new', 'new_lines': current_text, 'removed_lines': [], 'edited_pairs': [], 'similarity': 0.0}

    if not current_text:
        # Text disappeared
        return {'change_type': 'removed', 'new_lines': [], 'removed_lines': previous_text, 'edited_pairs': [], 'similarity': 0.0}

    # Calculate overall similarity
    current_set = set(current_text)
    previous_set = set(previous_text)

    # Jaccard similarity (set intersection / union)
    intersection = len(current_set & previous_set)
    union = len(current_set | previous_set)
    jaccard = intersection / union if union > 0 else 0.0

    # Detect specific changes
    new_lines = []
    removed_lines = list(previous_set - current_set)
    edited_pairs = []

    # Check for new or edited lines
    for curr_line in current_text:
        if curr_line in previous_text:
            continue  # Exact match - stable

        # Check if it's similar to any previous line (edit detection)
        best_match = None
        best_similarity = 0.0
        for prev_line in previous_text:
            sim = string_similarity(curr_line, prev_line)
            if sim > best_similarity:
                best_similarity = sim
                best_match = prev_line

        # Threshold: >0.6 similarity = edit, <0.6 = new
        if best_similarity > 0.6 and best_match:
            edited_pairs.append((best_match, curr_line))
            # Remove from removed_lines if it was there
            if best_match in removed_lines:
                removed_lines.remove(best_match)
        else:
            new_lines.append(curr_line)

    # Determine change type
    if jaccard < 0.3:
        # Very low similarity - camera likely moved
        change_type = 'camera_moved'
    elif new_lines and not removed_lines and not edited_pairs:
        change_type = 'new'
    elif edited_pairs and not new_lines and not removed_lines:
        change_type = 'edited'
    elif removed_lines and not new_lines and not edited_pairs:
        change_type = 'removed'
    elif current_text == previous_text:
        change_type = 'stable'
    elif new_lines or removed_lines or edited_pairs:
        change_type = 'mixed'  # Multiple changes
    else:
        change_type = 'none'

    return {
        'change_type': change_type,
        'new_lines': new_lines,
        'removed_lines': removed_lines,
        'edited_pairs': edited_pairs,
        'similarity': jaccard
    }


# ============================================================
# Feature 3: Conversational Messages
# ============================================================

def format_conversational_message(changes: dict, text_lines: list, for_discord: bool = False) -> str:
    """
    Turn change detection results into natural, conversational messages.

    Instead of "TEXT DETECTED (3 regions)", produces messages like:
    "I can see some new writing - looks like 'Project Due Monday'"
    """
    change_type = changes['change_type']
    bold = "**" if for_discord else ""

    if change_type == 'new':
        new_lines = changes['new_lines']
        if len(new_lines) == 1:
            openers = [
                f"I can see new text on the board: {bold}{new_lines[0]}{bold}",
                f"Someone just wrote: {bold}{new_lines[0]}{bold}",
                f"New writing appeared - it says {bold}{new_lines[0]}{bold}",
            ]
        else:
            preview = f"{bold}{new_lines[0]}{bold}"
            openers = [
                f"I see {len(new_lines)} new lines on the board. First one reads: {preview}",
                f"New text just appeared ({len(new_lines)} lines) - starts with: {preview}",
                f"Someone added {len(new_lines)} lines. The first says: {preview}",
            ]
        msg = random.choice(openers)
        if for_discord and len(new_lines) > 1:
            msg += "\n" + "\n".join(f"  {bold}{line}{bold}" for line in new_lines[1:3])
        return msg

    elif change_type == 'edited':
        pairs = changes['edited_pairs']
        if len(pairs) == 1:
            old, new = pairs[0]
            openers = [
                f"Looks like someone edited the board: {bold}{old}{bold} changed to {bold}{new}{bold}",
                f"I noticed an edit - {bold}{old}{bold} is now {bold}{new}{bold}",
                f"The text was updated: {bold}{old}{bold} became {bold}{new}{bold}",
            ]
        else:
            openers = [
                f"A few things were edited on the board ({len(pairs)} changes)",
                f"I see {len(pairs)} edits on the whiteboard",
                f"Someone made {len(pairs)} changes to the text",
            ]
        msg = random.choice(openers)
        if for_discord and len(pairs) > 1:
            for old, new in pairs[:2]:
                msg += f"\n  ~~{old}~~ -> {bold}{new}{bold}"
        return msg

    elif change_type == 'camera_moved':
        similarity = changes['similarity']
        preview = text_lines[0] if text_lines else "something new"
        openers = [
            f"Whoa, I think the camera moved! I'm looking at completely different text now ({similarity:.0%} match to before). I see: {bold}{preview}{bold}",
            f"The view changed dramatically - I'm now reading: {bold}{preview}{bold}",
            f"Looks like I'm pointed somewhere new. Now seeing: {bold}{preview}{bold}",
        ]
        return random.choice(openers)

    elif change_type == 'mixed':
        parts = []
        if changes['new_lines']:
            parts.append(f"added {bold}{changes['new_lines'][0]}{bold}")
        if changes['removed_lines']:
            parts.append(f"removed {bold}{changes['removed_lines'][0]}{bold}")
        if changes['edited_pairs']:
            old, new = changes['edited_pairs'][0]
            parts.append(f"changed {bold}{old}{bold} to {bold}{new}{bold}")
        detail = ", ".join(parts)
        openers = [
            f"The board got a makeover: {detail}",
            f"Several things changed - {detail}",
            f"Whiteboard update: {detail}",
        ]
        return random.choice(openers)

    elif change_type == 'removed':
        removed = changes['removed_lines']
        if len(removed) == 1:
            openers = [
                f"Someone erased: {bold}{removed[0]}{bold}",
                f"Text was removed: {bold}{removed[0]}{bold}",
            ]
        else:
            openers = [
                f"{len(removed)} lines were erased from the board",
                f"Looks like someone wiped {len(removed)} lines off the board",
            ]
        return random.choice(openers)

    elif change_type == 'stable':
        count = len(text_lines)
        openers = [
            f"Board looks the same - still reading {count} line{'s' if count != 1 else ''}",
            f"No changes - {count} line{'s' if count != 1 else ''} still there",
        ]
        return random.choice(openers)

    else:
        # Generic fallback
        if text_lines:
            return f"I can read {len(text_lines)} line{'s' if len(text_lines) != 1 else ''}: {bold}{text_lines[0]}{bold}"
        return "I don't see any text right now"


def format_cleared_message(previous_text: list, for_discord: bool = False) -> str:
    """Conversational message for when the whiteboard is cleared."""
    if previous_text:
        openers = [
            "Looks like the board was erased - it's blank now",
            "The whiteboard was cleared",
            "Everything was wiped off the board",
        ]
    else:
        openers = [
            "I don't see any text right now",
            "The board appears empty",
        ]
    return random.choice(openers)


# ============================================================
# Feature 4: Confidence Aggregation
# ============================================================

class ConfidenceAggregator:
    """
    Maintains a rolling buffer of recent OCR readings and finds consensus text.

    When the same text is read multiple times with variations:
        "LSREATEMACIK" (0.85)
        "LCREATE MACIK" (0.87)
        "CREATE MAGIK"  (0.92)

    Aggregates into: best guess "CREATE MAGIK" (seen 3x, best confidence 0.92)
    """

    def __init__(self, buffer_size: int = 10, similarity_threshold: float = 0.6):
        self.buffer = deque(maxlen=buffer_size)
        self.similarity_threshold = similarity_threshold
        self._consensus_text = []
        self._consensus_confidence = 0.0

    def add_reading(self, text_lines: list, confidence_scores: list):
        """Add a new OCR reading to the buffer."""
        if not text_lines:
            return
        self.buffer.append({
            'text_lines': text_lines,
            'confidences': confidence_scores,
            'timestamp': time.time()
        })
        self._update_consensus()

    def _update_consensus(self):
        """Recalculate consensus text from the buffer."""
        if not self.buffer:
            self._consensus_text = []
            self._consensus_confidence = 0.0
            return

        # Collect all text lines across the buffer, grouped by similarity
        clusters = []  # Each cluster: list of (text, confidence) tuples

        for reading in self.buffer:
            for i, line in enumerate(reading['text_lines']):
                conf = reading['confidences'][i] if i < len(reading['confidences']) else 0.5
                placed = False

                # Try to place in an existing cluster
                for cluster in clusters:
                    # Compare against the best candidate in the cluster
                    best_in_cluster = max(cluster, key=lambda x: x[1])
                    sim = SequenceMatcher(None, line.lower(), best_in_cluster[0].lower()).ratio()
                    if sim >= self.similarity_threshold:
                        cluster.append((line, conf))
                        placed = True
                        break

                if not placed:
                    clusters.append([(line, conf)])

        # For each cluster, pick the highest-confidence reading as the consensus
        consensus = []
        for cluster in clusters:
            best_text, best_conf = max(cluster, key=lambda x: x[1])
            times_seen = len(cluster)
            consensus.append({
                'text': best_text,
                'confidence': best_conf,
                'times_seen': times_seen,
                'avg_confidence': sum(c for _, c in cluster) / len(cluster)
            })

        # Sort by times_seen (most frequent first), then by confidence
        consensus.sort(key=lambda x: (x['times_seen'], x['confidence']), reverse=True)

        self._consensus_text = [c['text'] for c in consensus]
        self._consensus_confidence = (
            sum(c['avg_confidence'] for c in consensus) / len(consensus) if consensus else 0.0
        )
        self._consensus_details = consensus

    @property
    def consensus_text(self) -> list:
        """Best-guess text lines based on aggregated readings."""
        return self._consensus_text

    @property
    def consensus_confidence(self) -> float:
        """Overall aggregated confidence score."""
        return self._consensus_confidence

    @property
    def details(self) -> list:
        """Detailed consensus info: text, confidence, times_seen, avg_confidence."""
        return getattr(self, '_consensus_details', [])

    def format_consensus_summary(self) -> str:
        """Human-readable summary of the current consensus."""
        if not self.details:
            return "No readings yet"
        parts = []
        for d in self.details[:5]:
            seen = f"(seen {d['times_seen']}x, conf {d['avg_confidence']:.0%})"
            parts.append(f"  \"{d['text']}\" {seen}")
        return "Best reading:\n" + "\n".join(parts)

    def clear(self):
        """Reset the aggregator."""
        self.buffer.clear()
        self._consensus_text = []
        self._consensus_confidence = 0.0
        self._consensus_details = []


# ============================================================
# Feature 5: Smart Feedback
# ============================================================

def generate_smart_feedback(detections_list, recognitions_list, frame_shape,
                            avg_confidence: float, num_regions: int,
                            min_confidence: float = 0.25) -> list:
    """
    Analyze detection quality and provide actionable suggestions.

    Returns a list of feedback strings (may be empty if everything looks good).
    """
    feedback = []
    h, w = frame_shape[:2] if frame_shape is not None else (640, 1152)

    # Check overall confidence
    if avg_confidence > 0 and avg_confidence < 0.4:
        feedback.append(
            f"Confidence is low ({avg_confidence:.0%}) - try adjusting lighting or moving the camera closer"
        )
    elif avg_confidence >= 0.85:
        feedback.append(
            f"Getting clear reads! Confidence: {avg_confidence:.0%}"
        )

    # Check for text near frame edges (possibly cut off)
    if detections_list:
        edge_margin = 0.05  # 5% from edge
        left_cut = False
        right_cut = False
        top_cut = False
        bottom_cut = False

        for det in detections_list:
            xmin = getattr(det, 'xmin', 0)
            ymin = getattr(det, 'ymin', 0)
            xmax = getattr(det, 'xmax', 1)
            ymax = getattr(det, 'ymax', 1)

            if xmin < edge_margin:
                left_cut = True
            if xmax > (1.0 - edge_margin):
                right_cut = True
            if ymin < edge_margin:
                top_cut = True
            if ymax > (1.0 - edge_margin):
                bottom_cut = True

        cut_sides = []
        if left_cut:
            cut_sides.append("left")
        if right_cut:
            cut_sides.append("right")
        if top_cut:
            cut_sides.append("top")
        if bottom_cut:
            cut_sides.append("bottom")

        if cut_sides:
            sides_str = " and ".join(cut_sides)
            feedback.append(
                f"Text may be cut off on the {sides_str} - try panning the camera"
            )

    # Check for very small text regions (hard to recognize)
    if detections_list:
        tiny_count = 0
        for det in detections_list:
            det_w = getattr(det, 'xmax', 1) - getattr(det, 'xmin', 0)
            det_h = getattr(det, 'ymax', 1) - getattr(det, 'ymin', 0)
            # If region is less than 3% of frame in either dimension
            if det_w < 0.03 or det_h < 0.015:
                tiny_count += 1

        if tiny_count > 0:
            feedback.append(
                f"{tiny_count} text region{'s are' if tiny_count > 1 else ' is'} very small - moving closer might help"
            )

    # Check for no detections despite having regions (all filtered out)
    if num_regions > 0 and not recognitions_list:
        feedback.append(
            "Detected text regions but couldn't read them - the text may be too blurry or at an angle"
        )

    return feedback


def extract_text_from_recognition(recognition_result, min_confidence=0.25):
    """Extract text from PaddleOCR recognition result."""
    text_line = ""

    # Recognition results have 'classes' (text) and 'scores' (confidence)
    if hasattr(recognition_result, 'classes') and hasattr(recognition_result, 'scores'):
        for text, score in zip(recognition_result.classes, recognition_result.scores):
            # Filter out very short text and low confidence
            if len(text) <= 2 or score < min_confidence:
                continue

            text_line += text + " "

    return text_line.strip()


def draw_text_on_frame(frame, detections_list, recognitions_list, min_confidence=0.25):
    """Draw bounding boxes and recognized text on frame."""
    if not detections_list or not recognitions_list:
        return frame

    h, w = frame.shape[:2]

    for i, (detection, recognition) in enumerate(zip(detections_list, recognitions_list)):
        # Extract text from recognition
        text = extract_text_from_recognition(recognition, min_confidence)

        if not text:
            continue

        # Get bounding box coordinates
        if hasattr(detection, 'rotated_rect'):
            # Use rotated rectangle if available
            points = detection.rotated_rect.getPoints()
            pts = np.array([[int(p.x * w), int(p.y * h)] for p in points], np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            # Draw text near bottom-left corner of box
            text_x = int(points[3].x * w)
            text_y = int(points[3].y * h)
        else:
            # Fall back to regular bbox
            x1 = int(detection.xmin * w)
            y1 = int(detection.ymin * h)
            x2 = int(detection.xmax * w)
            y2 = int(detection.ymax * h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text_x = x1
            text_y = y2

        # Draw text with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Get text size for background
        (text_width, text_height), _ = cv2.getTextSize(text[:50], font, font_scale, thickness)

        # Draw background rectangle
        cv2.rectangle(frame, (text_x, text_y - text_height - 5),
                     (text_x + text_width, text_y + 5), (0, 255, 0), -1)

        # Draw text
        cv2.putText(frame, text[:50], (text_x, text_y),
                   font, font_scale, (0, 0, 0), thickness)

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
    """Main OCR detection loop using DepthAI 3.x with full text recognition."""
    global log_file, last_text_content, last_text_detected
    global last_confirmed_text
    global pending_state, pending_state_time
    global last_status_update_time, last_screenshot_time
    global last_feedback_time
    global last_config_check

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
        log_filename = f"whiteboard_ocr_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file = open(log_filename, 'w')
        log_event(f"Logging to {log_filename}")

    startup_msg = "Whiteboard OCR reader started (FULL - DepthAI 3.x with PaddlePaddle OCR)"
    log_event(startup_msg)
    log_event(f"Text confidence threshold: {args.confidence}")
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
        discord_startup = f"📋 **{username}** is now running whiteboard_reader_full.py on **{hostname}**"
        send_discord_notification(discord_startup)

    # Feature 4: Confidence aggregator for consensus text
    aggregator = ConfidenceAggregator(buffer_size=10, similarity_threshold=0.6)

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
            log_event("Creating full OCR pipeline (detection + recognition)...")

            # Load text detection model (Stage 1)
            det_model_path = Path(__file__).parent / "depthai_models" / f"paddle_text_detection.{platform}.yaml"
            if not det_model_path.exists():
                log_event(f"ERROR: Detection model not found at {det_model_path}")
                log_event("Please copy model files: scp -r depthai_models orbit:~/oak-projects/")
                return

            det_model_description = dai.NNModelDescription.fromYamlFile(str(det_model_path))
            det_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(det_model_description))
            det_model_w, det_model_h = det_model_nn_archive.getInputSize()

            # Load text recognition model (Stage 2)
            rec_model_path = Path(__file__).parent / "depthai_models" / f"paddle_text_recognition.{platform}.yaml"
            if not rec_model_path.exists():
                log_event(f"ERROR: Recognition model not found at {rec_model_path}")
                log_event("Please copy model files: scp -r depthai_models orbit:~/oak-projects/")
                return

            rec_model_description = dai.NNModelDescription.fromYamlFile(str(rec_model_path))
            rec_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(rec_model_description))
            rec_model_w, rec_model_h = rec_model_nn_archive.getInputSize()

            log_event(f"Detection model: {det_model_w}x{det_model_h}")
            log_event(f"Recognition model: {rec_model_w}x{rec_model_h}")

            # Camera input
            cam = pipeline.create(dai.node.Camera).build()
            cam_out = cam.requestOutput(
                size=(REQ_WIDTH, REQ_HEIGHT), type=frame_type, fps=fps_limit
            )

            # Resize to detection model input size
            resize_node = pipeline.create(dai.node.ImageManip)
            resize_node.initialConfig.setOutputSize(det_model_w, det_model_h)
            resize_node.initialConfig.setReusePreviousImage(False)
            resize_node.inputImage.setBlocking(True)
            cam_out.link(resize_node.inputImage)

            # Text detection neural network (Stage 1)
            det_nn = pipeline.create(ParsingNeuralNetwork).build(
                resize_node.out, det_model_nn_archive
            )
            det_nn.setNumPoolFrames(30)

            # Crop node for text regions (Stage 2 preparation)
            crop_node = pipeline.create(dai.node.ImageManip)
            crop_node.initialConfig.setReusePreviousImage(False)
            crop_node.inputConfig.setReusePreviousMessage(False)
            crop_node.inputImage.setReusePreviousMessage(True)
            crop_node.inputConfig.setMaxSize(30)
            crop_node.inputImage.setMaxSize(30)
            crop_node.setNumFramesPool(30)

            # Create host node for crop configuration (using oak-examples implementation)
            crop_config_creator = pipeline.create(CropConfigsCreator).build(
                det_nn.out,
                (REQ_WIDTH, REQ_HEIGHT),
                (rec_model_w, rec_model_h)
            )
            crop_config_creator.config_output.link(crop_node.inputConfig)
            cam_out.link(crop_node.inputImage)

            # Text recognition neural network (Stage 2)
            rec_nn = pipeline.create(ParsingNeuralNetwork).build(
                crop_node.out, rec_model_nn_archive
            )
            rec_nn.setNumPoolFrames(30)
            rec_nn.input.setMaxSize(30)

            # Sync detections with recognitions
            gather_data_node = pipeline.create(GatherData).build(fps_limit)
            crop_config_creator.detections_output.link(gather_data_node.input_reference)
            rec_nn.out.link(gather_data_node.input_data)

            # Get output queues (MUST be created before pipeline.start())
            q_gathered = gather_data_node.out.createOutputQueue(maxSize=4, blocking=False)
            q_preview = cam_out.createOutputQueue(maxSize=4, blocking=False)

            log_event("Pipeline created.")

            # Start pipeline
            pipeline.start()
            log_event("Full OCR started. Reading text from whiteboard...\n")

            # Ensure config file exists with defaults
            write_default_config()
            notifications_enabled = True

            # Create window if display is enabled
            if args.display:
                cv2.namedWindow("Whiteboard OCR - Full", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Whiteboard OCR - Full", 1152, 640)

            # Cache latest OCR results for screenshot annotation
            # (preview frames and gathered messages arrive at different rates)
            latest_gathered = None
            num_regions = 0
            text_lines = []
            avg_confidence = 0.0

            while pipeline.isRunning():
                # Get synced detection + recognition results
                gathered_msg = q_gathered.tryGet()

                # Get preview frame
                preview_frame = q_preview.tryGet()

                if gathered_msg is not None:
                    latest_gathered = gathered_msg
                    # Extract detections and recognitions
                    detections_msg = gathered_msg.reference_data
                    recognitions_list = gathered_msg.gathered

                    # Extract text from all recognitions
                    text_lines = []
                    confidence_scores = []
                    num_regions = 0

                    if hasattr(detections_msg, 'detections'):
                        num_regions = len(detections_msg.detections)

                        for i, recognition in enumerate(recognitions_list):
                            if i < len(detections_msg.detections):
                                text = extract_text_from_recognition(recognition, args.confidence)
                                if text:
                                    text_lines.append(text)

                                    # Collect confidence scores for averaging
                                    if hasattr(recognition, 'scores') and len(recognition.scores) > 0:
                                        avg_score = sum(recognition.scores) / len(recognition.scores)
                                        confidence_scores.append(avg_score)

                    # Calculate overall average confidence
                    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

                    text_detected = len(text_lines) > 0
                    text_detection_history.append(text_detected)

                    # Feature 4: Feed confidence aggregator
                    if text_lines:
                        aggregator.add_reading(text_lines, confidence_scores)

                    # Log to history file (every detection, not just changes)
                    if num_regions > 0:
                        log_text_history(text_lines, num_regions, avg_confidence)

                    # Smoothed detection (majority vote from last 5 frames)
                    smoothed_detection = sum(text_detection_history) >= len(text_detection_history) / 2

                    current_time = time.time()

                    # Update console status (show consensus text from aggregator)
                    consensus = aggregator.consensus_text
                    if consensus:
                        preview_text = consensus[0][:50] + ("..." if len(consensus[0]) > 50 else "")
                        conf_str = f"{aggregator.consensus_confidence:.0%}"
                        print(f"\r  Regions: {num_regions} | Best: \"{preview_text}\" ({conf_str})  ",
                              end="", flush=True)
                    elif text_lines:
                        preview_text = text_lines[0][:50] + ("..." if len(text_lines[0]) > 50 else "")
                        print(f"\r  Regions: {num_regions} | Text: \"{preview_text}\"  ",
                              end="", flush=True)
                    else:
                        print(f"\r  Regions: {num_regions} | Text: [none]  ",
                              end="", flush=True)

                    # Feature 5: Smart feedback (throttled)
                    if current_time - last_feedback_time >= FEEDBACK_INTERVAL and num_regions > 0:
                        detections_for_feedback = (
                            detections_msg.detections if hasattr(detections_msg, 'detections') else []
                        )
                        feedback = generate_smart_feedback(
                            detections_for_feedback, recognitions_list,
                            (REQ_HEIGHT, REQ_WIDTH), avg_confidence, num_regions,
                            args.confidence
                        )
                        if feedback:
                            for tip in feedback:
                                log_event(f"\n💡 {tip}")
                            last_feedback_time = current_time

                    # Debouncing logic for state changes
                    # Use consensus text when available for more stable change detection
                    effective_text = consensus if consensus else text_lines

                    if smoothed_detection != last_text_detected:
                        if pending_state == smoothed_detection:
                            if current_time - pending_state_time >= DEBOUNCE_SECONDS:
                                # Confirm state change - detect WHAT changed
                                if smoothed_detection:
                                    # Text is present - analyze what changed
                                    changes = detect_text_changes(effective_text, last_confirmed_text)

                                    # Feature 3: Conversational messages
                                    console_msg = format_conversational_message(changes, effective_text, for_discord=False)
                                    log_event(f"\n{console_msg}")

                                    # Show aggregator consensus details for multi-read text
                                    if aggregator.details:
                                        for d in aggregator.details[:3]:
                                            if d['times_seen'] > 1:
                                                log_event(f"  (seen {d['times_seen']}x, best confidence: {d['confidence']:.0%})")

                                    if args.discord and notifications_enabled:
                                        discord_msg = format_conversational_message(changes, effective_text, for_discord=True)
                                        send_discord_notification(discord_msg)

                                    # Update confirmed text
                                    last_confirmed_text = effective_text.copy()

                                else:
                                    # Text disappeared
                                    console_msg = format_cleared_message(last_confirmed_text, for_discord=False)
                                    log_event(f"\n{console_msg}")

                                    if args.discord and not args.discord_quiet and notifications_enabled:
                                        discord_msg = format_cleared_message(last_confirmed_text, for_discord=True)
                                        send_discord_notification(discord_msg)

                                    last_confirmed_text = []
                                    aggregator.clear()

                                last_text_detected = smoothed_detection
                                last_text_content = effective_text
                                pending_state = None
                                pending_state_time = None

                                # Update status file (include consensus text)
                                update_status_file(
                                    smoothed_detection, effective_text, num_regions,
                                    running=True, username=username, hostname=hostname)
                        else:
                            # New pending state
                            pending_state = smoothed_detection
                            pending_state_time = current_time
                    else:
                        # State matches - reset pending
                        pending_state = None
                        pending_state_time = None

                # Display frame with text overlay if enabled
                if args.display and preview_frame is not None:
                    frame = preview_frame.getCvFrame()

                    # Draw text boxes and recognized text
                    if gathered_msg is not None:
                        detections_list = gathered_msg.reference_data.detections if hasattr(gathered_msg.reference_data, 'detections') else []
                        recognitions_list = gathered_msg.gathered

                        frame = draw_text_on_frame(frame, detections_list, recognitions_list, args.confidence)

                    # Add status text
                    status_text = f"Text Lines: {len(text_lines) if gathered_msg else 0}"
                    cv2.putText(frame, status_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"User: {username}@{hostname}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    # Show frame
                    cv2.imshow("Whiteboard OCR - Full", frame)

                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        log_event("Display window closed by user")
                        break

                # Periodic status file update (always use latest OCR text)
                current_time = time.time()
                if current_time - last_status_update_time >= STATUS_UPDATE_INTERVAL:
                    detected = last_text_detected
                    # Use current frame's text, not stale last_text_content,
                    # so content changes are visible even without a
                    # presence transition (e.g. adding "SET 5 MIN TIMER"
                    # while text is already on the board).
                    consensus = aggregator.consensus_text
                    current_content = consensus if consensus else text_lines
                    regions = num_regions if num_regions else 0
                    update_status_file(detected, current_content, regions,
                                     running=True, username=username, hostname=hostname)
                    last_status_update_time = current_time

                # Periodic screenshot save (uses cached OCR results so it doesn't
                # depend on gathered_msg arriving in the same loop iteration)
                if preview_frame is not None and current_time - last_screenshot_time >= SCREENSHOT_UPDATE_INTERVAL:
                    try:
                        frame = preview_frame.getCvFrame()

                        # Draw text on screenshot using cached OCR results
                        if latest_gathered is not None:
                            detections_list = latest_gathered.reference_data.detections if hasattr(latest_gathered.reference_data, 'detections') else []
                            recognitions_list_cached = latest_gathered.gathered
                            frame = draw_text_on_frame(frame, detections_list, recognitions_list_cached, args.confidence)

                        # Add info banner
                        _num_regions = num_regions or 0
                        _text_lines = text_lines or []
                        _avg_confidence = avg_confidence or 0.0
                        info_items = [
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            f"{_num_regions} regions, {len(_text_lines)} lines",
                        ]
                        if _avg_confidence > 0:
                            info_items.append(f"Conf: {_avg_confidence:.0%}")
                        info_items.append(f"{username}@{hostname}")
                        frame = draw_info_banner(frame, info_items)

                        cv2.imwrite(str(SCREENSHOT_FILE), frame)
                        last_screenshot_time = current_time
                    except Exception as e:
                        log_event(f"WARNING: Could not save screenshot: {e}")

                # Periodic config file check
                if current_time - last_config_check >= CONFIG_CHECK_INTERVAL:
                    config = load_config()
                    new_confidence = config.get('confidence', args.confidence)
                    if new_confidence != args.confidence:
                        log_event(f"Config updated: confidence={new_confidence}")
                        args.confidence = new_confidence
                    new_fps = config.get('fps_limit', fps_limit)
                    if new_fps != fps_limit:
                        log_event(f"Config updated: fps_limit={new_fps} (requires pipeline restart to take effect)")
                    notifications_enabled = config.get('notifications_enabled', True)
                    last_config_check = current_time

                # Small sleep to prevent CPU spinning
                time.sleep(0.01)

    except KeyboardInterrupt:
        shutdown_msg = "Whiteboard OCR reader (full) stopped"
        log_event(f"\n{shutdown_msg}")
        if args.discord:
            discord_shutdown = f"📴 **{username}** stopped whiteboard_reader_full.py on **{hostname}** - camera is free"
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
