# PC Testing

Local Windows/Mac testing scripts for OAK-D cameras with GPU acceleration. This directory is separate from the Pi-based detector scripts used in class.

## Purpose

Offload heavy model processing (skeleton, hands, depth) to a PC GPU instead of running on the OAK camera or Raspberry Pi. Enables higher framerates for interactive installations.

## Setup

### Prerequisites

- **Python 3.10** recommended (tested with `py -3.10`)
- NVIDIA GPU with CUDA support (for GPU-accelerated scripts)

### 1. Install Dependencies

```bash
# From this directory (using Python 3.10)
py -3.10 -m pip install -r requirements-windows.txt

# For GPU-accelerated skeleton/hands (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics mediapipe
```

**Note for RTX 50 series (Blackwell):** Use nightly PyTorch with CUDA 12.8:
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 2. Claude Code Settings (Optional)

Create `.claude/settings.local.json` in the repo root for local Claude Code settings. This file is gitignored since settings are user-specific.

## Scripts

### Discovery & Diagnostics

| Script | Description |
|--------|-------------|
| `list_cameras.py` | Discover all connected OAK cameras (USB and PoE) |
| `diagnose.py` | Check USB speed, device health, troubleshoot connections |

### Camera Testing

| Script | Description |
|--------|-------------|
| `test_camera.py` | Basic camera test with live feed (640x480) |
| `test_person_detect.py` | YOLO person detection with bounding boxes |

### Skeleton + Depth (GPU Accelerated)

| Script | Description | FPS |
|--------|-------------|-----|
| `test_skeleton_depth.py` | YOLOv11-Pose skeleton + depth | ~30 |
| `test_skeleton_depth_mediapipe.py` | MediaPipe pose + hands (CPU) | ~20 |
| `test_skeleton_hands_depth.py` | YOLO skeleton + MediaPipe hands + gestures | ~25 |

## Usage

```bash
# USB camera (auto-detect)
python test_camera.py
python test_skeleton_depth.py

# PoE camera (specify IP)
python test_camera.py --ip <your-camera-ip>
python test_skeleton_depth.py --ip <your-camera-ip>
```

Press `q` to quit any script with a display window.

## Features

- **Skeleton tracking**: Full body pose estimation (17 keypoints)
- **Hand tracking**: Detailed finger landmarks (21 points per hand)
- **Gesture detection**: FIST, POINTING, OPEN palm recognition
- **Depth sensing**: Real-world distance measurements in meters
- **3D positioning**: Hand distance from camera, distance between hands

## Architecture

```
[OAK Camera] → RGB + Depth → [PC GPU] → YOLO/MediaPipe → [Display]
     │                          │
     └── Stereo depth ──────────┘
```

The OAK camera streams raw video and depth to the PC. Heavy inference runs on the GPU, enabling higher framerates than on-device or Pi processing.

## Troubleshooting

### Camera not found
- USB: Check cable, try different USB 3.0 port (blue interior)
- PoE: Ensure camera is on same network subnet as PC

### USB 2.0 crashes
- Run `diagnose.py` to check USB speed
- USB 2.0 causes bandwidth issues — use USB 3.0 port

### PyTorch CUDA errors
- RTX 50 series needs PyTorch nightly with CUDA 12.8
- Check GPU compatibility: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
