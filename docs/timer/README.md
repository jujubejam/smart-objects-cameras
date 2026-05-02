# Fiducial Timer

A physical timer controlled by ArUco marker tags held in front of an OAK-D camera. Rotate a tag to set the duration, then tap other tags to start, pause, or reset. A companion web page displays the live countdown on any device on the same network.

## Files

| File | Purpose |
|------|---------|
| `fiducial_sdk.py` | Reusable ArUco marker detection library built on DepthAI and OpenCV |
| `timer_app.py` | Main timer application — reads tags and runs the countdown |
| `timer_web.py` | Lightweight HTTP server that serves a live countdown page |

## Tag Assignments

| Tag ID | Action |
|--------|--------|
| 1 | On/off switch — present in frame = on |
| 3 | Set timer — rotate to dial in minutes (0–60) |
| 8 | Start countdown |
| 9 | Pause countdown |
| 10 | Reset to set time |

## Usage

**Terminal 1 — run the timer:**
```bash
python3 timer_app.py             # headless
python3 timer_app.py --display   # OpenCV window (requires X11/VNC)
```

**Terminal 2 — serve the web display:**
```bash
python3 timer_web.py             # http://0.0.0.0:8080
python3 timer_web.py --port 5000
```

Open `http://<pi-hostname>:8080` in any browser on the same network to see the countdown.

## How It Works

`timer_app.py` uses `FiducialSDK` to detect ArUco markers from the OAK-D camera feed each frame. It writes state to `timer_status.json` every second. `timer_web.py` reads that file and serves it at `/status`; the browser page polls it every second to update the display.

```
OAK-D camera
     │
fiducial_sdk.py  (detect markers, emit events)
     │
timer_app.py     (handle tag events → countdown logic → timer_status.json)
                                                               │
timer_web.py     (serve /status from timer_status.json)  ◄────┘
     │
browser          (poll /status every 1 s, render countdown)
```

## FiducialSDK API

```python
from fiducial_sdk import FiducialSDK

sdk = FiducialSDK(options={
    'smoothing': 0.35,      # position/size lerp factor (0–1)
    'rotation_step': 10,    # degrees to snap rotation to
    'exit_grace': 8,        # missed frames before MARKER_EXIT fires
    'width': 640,
    'height': 480,
})
sdk.init()

# Listen to any marker
sdk.on('MARKER_ENTER',  lambda m: print('entered', m['id']))
sdk.on('MARKER_EXIT',   lambda m: print('exited',  m['id']))
sdk.on('MARKER_MOVE',   lambda m: ...)
sdk.on('MARKER_ROTATE', lambda m: ...)
sdk.on('MARKER_UPDATE', lambda m: ...)

# Listen to a specific marker ID
sdk.on_marker(3, lambda m: print('tag 3 rotation:', m['rotation']))

sdk.run()   # blocking; pass on_frame=fn to receive raw BGR frames
```

Each marker event payload:

```python
{
    'id': 3,
    'present': True,
    'position': {'x': 320.0, 'y': 240.0},   # pixels
    'normalized': {'x': 0.5, 'y': 0.5},      # 0–1
    'size': 120.0,                            # diagonal pixels
    'distance': 0.45,                         # 0 = close, 1 = far
    'rotation': 90,                           # snapped degrees
    'raw_rotation': 88.3,
    'velocity': {'x': 0.0, 'y': 0.0},
    'corners': [{'x': ..., 'y': ...}, ...],  # 4 corners
    'timestamp': 1746200000.0,
}
```

## Requirements

- OAK-D camera on a Raspberry Pi 5 with the shared venv active (`activate-oak`)
- `depthai`, `opencv-contrib-python` (ArUco detector is in `contrib`)
- Printed ArUco markers from the `ARUCO_ORIGINAL` dictionary

## Online ArUco Marker Generator

https://fodi.github.io/arucosheetgen/