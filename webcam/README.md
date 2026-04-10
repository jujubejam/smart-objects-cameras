# Webcam Detector Suite

Run the Smart Objects camera detectors on your Mac laptop using the built-in webcam. No OAK-D camera or Raspberry Pi required.

These webcam versions write the **same status files** as the OAK-D versions, so the Discord bot (`discord_bot.py`) works with either setup.

## What's Included

| Detector | What It Does | Library |
|----------|-------------|---------|
| `webcam_person_detector.py` | Counts people in frame | YOLOv8-nano |
| `webcam_fatigue_detector.py` | Monitors eye closure + head tilt | MediaPipe Face Mesh |
| `webcam_gaze_detector.py` | Tracks where you're looking | MediaPipe Face Mesh (iris) |
| `webcam_whiteboard_reader.py` | Reads text from whiteboards | Tesseract OCR |

---

## Prerequisites

- **macOS** with a built-in or USB webcam
- **Python 3.9, 3.10, 3.11, 3.12, or 3.13**
- **Homebrew** ([install here](https://brew.sh) if you don't have it)

Check your Python version:

```bash
python3 --version
```

---

## Setup

### Step 1: Clone the repo (skip if you already have it)

```bash
git clone https://github.com/kandizzy/smart-objects-cameras.git
cd smart-objects-cameras
```

### Step 2: Create a virtual environment

```bash
python3 -m venv webcam-venv
source webcam-venv/bin/activate
```

You'll need to run `source webcam-venv/bin/activate` each time you open a new terminal.

### Step 3: Install Python dependencies

```bash
pip install -r webcam/requirements.txt
```

This downloads several packages including PyTorch (for YOLO). The first install may take a few minutes and ~500MB of disk space.

### Step 4: Install Tesseract (for whiteboard OCR only)

```bash
brew install tesseract
```

Skip this step if you don't plan to use the whiteboard reader.

### Step 5: Create the status file directory

```bash
mkdir -p ~/oak-projects
```

### Step 6: Set up Discord (optional)

If you want Discord notifications:

```bash
# Create the .env file
cat > ~/oak-projects/.env << 'EOF'
DISCORD_WEBHOOK_URL=your_webhook_url_here
DISCORD_BOT_TOKEN=your_bot_token_here
EOF
```

Replace the placeholder values with your actual webhook URL and bot token. See `docs/discord-integration.md` for how to get these.

---

## Running Detectors

**Important:** Only run one detector at a time (they all use the webcam).

### Person Detection

```bash
# Basic — just counts people
python3 webcam/webcam_person_detector.py

# With live video window
python3 webcam/webcam_person_detector.py --display

# With Discord notifications
python3 webcam/webcam_person_detector.py --discord

# Adjust detection sensitivity (0-1, higher = fewer false positives)
python3 webcam/webcam_person_detector.py --threshold 0.6

# Log events to a file
python3 webcam/webcam_person_detector.py --log

# Combine flags
python3 webcam/webcam_person_detector.py --display --discord --log
```

### Fatigue Detection

```bash
# Basic — monitors eye closure and head tilt
python3 webcam/webcam_fatigue_detector.py

# With live video showing face mesh overlay
python3 webcam/webcam_fatigue_detector.py --display

# Adjust thresholds
python3 webcam/webcam_fatigue_detector.py --ear-threshold 0.2 --pitch-threshold 25
```

The fatigue detector writes `~/oak-projects/fatigue_status.json`. If you're also running `discord_dm_notifier.py`, it will pick up fatigue alerts and send you DMs — no changes needed.

### Gaze Detection

```bash
# Basic — tracks where you're looking
python3 webcam/webcam_gaze_detector.py

# With live video showing gaze arrows from your eyes
python3 webcam/webcam_gaze_detector.py --display
```

### Whiteboard OCR

```bash
# Basic — reads text from whiteboard
python3 webcam/webcam_whiteboard_reader.py

# With live video showing detected text regions
python3 webcam/webcam_whiteboard_reader.py --display

# With Discord notifications when text changes
python3 webcam/webcam_whiteboard_reader.py --discord

# Adjust OCR confidence (0-100, higher = fewer false reads)
python3 webcam/webcam_whiteboard_reader.py --confidence 70
```

**Tip:** Point your webcam at a whiteboard, printed page, or your screen with large text. OCR works best with high contrast text.

---

## Selecting a Different Camera

All detectors support `--camera N` to pick a different camera:

```bash
# Use the second camera (e.g., an external USB webcam)
python3 webcam/webcam_person_detector.py --camera 1
```

Camera 0 is usually the built-in webcam.

---

## Controlling CPU Usage

Inference runs on your CPU, not a dedicated chip. If your laptop gets warm:

```bash
# Lower the FPS limit (default is 15 for most, 5 for whiteboard)
python3 webcam/webcam_person_detector.py --fps-limit 5
python3 webcam/webcam_fatigue_detector.py --fps-limit 10
```

---

## Status Files

Detectors write JSON status files to `~/oak-projects/` — the same paths and format as the OAK-D versions:

| File | Written By | Keys |
|------|-----------|------|
| `camera_status.json` | person detector | `detected`, `count`, `timestamp`, `running` |
| `fatigue_status.json` | fatigue detector | `fatigue_detected`, `eyes_closed`, `head_tilted`, `fatigue_percent` |
| `gaze_status.json` | gaze detector | `gaze_direction`, `gaze_x`, `gaze_y`, `head_yaw`, `head_pitch` |
| `whiteboard_status.json` | whiteboard reader | `text_detected`, `text_content`, `num_text_regions` |

Screenshots are also saved periodically:

| File | Updated Every |
|------|--------------|
| `latest_frame.jpg` | 5 seconds |
| `latest_fatigue_frame.jpg` | 5 seconds |
| `latest_gaze_frame.jpg` | 5 seconds |
| `latest_whiteboard_frame.jpg` | 5 seconds |

The Discord bot reads these files for `!status`, `!screenshot`, etc.

---

## Stopping a Detector

Press **Ctrl+C** in the terminal, or press **q** in the display window (if `--display` is on).

---

## Troubleshooting

### "No camera found" or black screen

- **macOS Sonoma/Sequoia:** The first time you run a detector, macOS will ask for camera permission. Click **Allow**.
- Check **System Settings > Privacy & Security > Camera** and make sure Terminal (or your IDE) has access.
- Try a different camera index: `--camera 1`

### mediapipe won't install (Python 3.13)

```bash
# Try upgrading pip first
pip install --upgrade pip

# Then try again
pip install mediapipe

# If it still fails, try the pre-release
pip install mediapipe --pre
```

If nothing works, use Python 3.11 or 3.12 instead (most compatible).

### "Tesseract not found"

```bash
brew install tesseract

# Verify it's installed
tesseract --version
```

### YOLO model download hangs

The first time you run the person detector, it downloads `yolov8n.pt` (~6MB). Make sure you have internet access. After the first download, it works offline.

### Slow performance

- Lower the FPS: `--fps-limit 5`
- Close other apps using the camera
- The whiteboard reader is the slowest (OCR is CPU-intensive). 2-5 FPS is normal.
- Person detection is the fastest — usually runs at 10-15 FPS on a modern MacBook.

### Import errors

Make sure your virtual environment is activated:

```bash
source webcam-venv/bin/activate
```

---

## Differences from the OAK-D Version

| | OAK-D (Raspberry Pi) | Webcam (Mac Laptop) |
|---|---|---|
| **Camera** | OAK-D via DepthAI | Built-in/USB webcam via OpenCV |
| **Inference** | On-device neural accelerator (VPU) | CPU (PyTorch/MediaPipe) |
| **Person model** | YOLOv6-nano | YOLOv8-nano |
| **Face analysis** | YuNet + MediaPipe on VPU | MediaPipe Face Mesh on CPU |
| **Gaze method** | 3-stage pipeline (YuNet + head pose + gaze ADAS) | Iris position relative to eye corners |
| **OCR engine** | PaddlePaddle (on VPU) | Tesseract (on CPU) |
| **Depth data** | Available (stereo cameras) | Not available |
| **Typical FPS** | 15-30 | 5-15 |
| **Status files** | Same | Same |
| **Discord bot** | Works | Works (same status files) |

The core detection logic (debouncing, notifications, status files) is identical. Only the camera input and inference backend differ.
