# V-JEPA 2 Windows Tools

Windows-specific tools for training and running V-JEPA activity classifiers with OAK-D cameras.

## What This Does

Train a custom activity classifier with minimal labeled data by leveraging V-JEPA 2's pre-trained video understanding. V-JEPA already knows how the visual world works - you just teach it your specific activities.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        YOUR WORKFLOW                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   1. RECORD          2. LABEL           3. TRAIN        4. INFER    │
│   ┌─────────┐       ┌─────────┐       ┌─────────┐     ┌─────────┐   │
│   │ Camera  │──────►│ Review  │──────►│ V-JEPA  │────►│  Live   │   │
│   │ Clips   │       │ & Sort  │       │ + Probe │     │ Classify│   │
│   └─────────┘       └─────────┘       └─────────┘     └─────────┘   │
│                                                                      │
│   ~20 clips/class    1-2 mins          ~30 secs        Real-time    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Conda environment** with V-JEPA dependencies:
   ```powershell
   conda activate vjepa
   ```

2. **V-JEPA server running** (in a separate terminal):
   ```powershell
   cd v-jepa
   python server.py
   ```
   First inference is slow (model warmup), then ~300ms per clip.

3. **OAK-D camera** connected via USB or Ethernet

## Windows Performance Note

**torch.compile() is disabled on Windows** because Triton (the compiler backend) doesn't support Windows. This results in ~20% slower inference compared to Linux.

| Platform | torch.compile | Inference time |
|----------|---------------|----------------|
| Linux | Enabled | ~250ms/clip |
| Windows | Disabled | ~300ms/clip |

For production classroom deployments, consider running the server on Linux for better performance. The Windows setup is ideal for development, testing, and training probes.

---

## Quick Start

### Option A: Batch Files (Simplest)

```powershell
# 1. Start recording clips (Ctrl+C to stop)
record.bat

# 2. Label the clips you recorded
label.bat

# 3. Train the probe (server.py must be running)
train.bat

# 4. Run live inference
infer.bat
```

### Option B: Command Line (More Control)

```powershell
# Record with countdown (for "acting out" activities)
python clip_recorder.py --output D:\ck_office --countdown 5 --duration 5

# Or auto-record on timer (natural behavior, label later)
python auto_recorder.py --output D:\ck_office --interval 10 --display

# Label recorded clips
python clip_labeler.py --input D:\ck_office --slow

# Train probe
python probe_trainer.py --clips-dir D:\ck_office

# Run inference
python probe_inference.py --probe C:\Users\carri\oak-projects\home_probe.pt --display
```

---

## Scripts Reference

### Recording

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `clip_recorder.py` | Manual recording with countdown | `--countdown 5` `--duration 5` `--output D:\clips` |
| `auto_recorder.py` | Automatic recording on timer | `--interval 10` `--display` `--max-clips 100` |

**clip_recorder.py** - Press 1/2/3 to record a clip for that class. 3-second countdown lets you get into position. Good for "acting out" specific activities.

**auto_recorder.py** - Records a clip every N seconds automatically. Good for capturing natural behavior over time. Label clips later.

### Labeling

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `clip_labeler.py` | Review and sort clips | `--slow` `--classes a,b,c` |

Controls:
- **1-9**: Label clip as that class
- **Space**: Skip (leave in unlabeled)
- **R**: Replay clip
- **Backspace**: Undo last label
- **Q**: Quit

### Training

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `probe_trainer.py` | Train classifier probe | `--clips-dir` `--epochs 100` `--no-cache` |

The probe is tiny (1024 -> 256 -> N classes). Training takes seconds on CPU after embeddings are extracted.

Embeddings are cached - re-training is instant. Use `--no-cache` to re-embed after adding new clips.

### Inference

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `probe_inference.py` | Live classification | `--display` `--interval 3` `--clip-secs 2` |
| `oak_client.py` | Anomaly detection (no probe) | `--display` `--threshold 0.2` |

### Multi-Camera

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `multi_camera_consensus.py` | Combine predictions | `--strategy average` `--cameras a,b` |

Strategies:
- `confidence` - Trust highest confidence camera
- `average` - Average class probabilities
- `agreement` - Only report if cameras agree

---

## Multi-Camera Setup

### Recording from Multiple Angles

More camera angles = more robust classifier.

```powershell
# USB camera
python clip_recorder.py --output D:\ck_office --countdown 5

# Ethernet camera
python clip_recorder.py --output D:\ck_office --ip 192.168.7.231 --countdown 5
```

Put all clips in the same class folders. The probe learns to recognize activities from multiple viewpoints.

### Running Multiple Inference Streams

```powershell
# Terminal 1 - Server (one instance, serves all cameras)
python server.py

# Terminal 2 - USB camera
python probe_inference.py --probe home_probe.pt --name usb-cam --display

# Terminal 3 - Ethernet camera
python probe_inference.py --probe home_probe.pt --ip 192.168.7.231 --name eth-cam --display

# Terminal 4 - Consensus (optional)
python multi_camera_consensus.py --cameras usb-cam,eth-cam --strategy average
```

**GPU usage**: The server loads the model once (~2.6 GB VRAM). Multiple cameras share the same model - requests queue up, not parallel. 2 cameras = 2x requests, same VRAM.

---

## Folder Structure

```
D:\ck_office\                    # Your clips directory
├── at_computer\                 # Class 1
│   ├── clip_001.mp4
│   ├── clip_002.mp4
│   └── ...
├── playing_keyboard\            # Class 2
│   └── ...
├── tending_plants\              # Class 3
│   └── ...
├── unlabeled\                   # Auto-recorded clips before labeling
│   └── ...
├── manifest.jsonl               # Auto-recorder metadata
└── home_probe.pt                # Trained probe (optional location)

C:\Users\you\oak-projects\       # Status files
├── probe_status_usb-cam.json    # Per-camera status
├── probe_status_eth-cam.json
├── consensus_status.json        # Multi-camera consensus
├── home_probe.pt                # Default probe location
└── .env                         # Discord webhook (optional)
```

---

## Design Considerations

### Latency

V-JEPA needs temporal context (video over time) to understand activities.

| Step | Time |
|------|------|
| Wait for interval | 2-5 sec |
| Capture clip | 1.5-3 sec |
| Server embedding | ~300ms |
| **Total response time** | **4-8 sec** |

Speed up with: `--interval 2 --clip-secs 1.5`

**Tradeoff**: Shorter clips = faster response but less motion to analyze.

**Good for**: Activities that last minutes (lecture, discussion, presentation)
**Not good for**: Instant gestures, rapid reactions

### When to Use V-JEPA vs Other Approaches

| Scenario | Best Tool | Why |
|----------|-----------|-----|
| "Is this a lecture or discussion?" | V-JEPA | Activity understanding over time |
| "Is someone at the whiteboard?" | V-JEPA | Spatial + temporal context |
| "Did someone just raise their hand?" | YOLO + Pose | Needs instant detection |
| "Is the room occupied?" | Motion sensor / YOLO | Simple presence, no video needed |
| "Count people in room" | YOLO | Single-frame detection |
| "Is the student engaged or distracted?" | V-JEPA | Subtle behavioral patterns |

### Hybrid Approaches for Faster Response

If you need faster reaction times, consider combining approaches:

**1. Motion trigger + V-JEPA classification**
```
Motion detected → Start V-JEPA analysis → Classify activity
```
Don't run V-JEPA constantly - only when something changes.

**2. YOLO presence + V-JEPA activity**
```
YOLO: "Person detected at whiteboard" (instant)
V-JEPA: "They are writing, not just standing" (5-10 sec later)
```
Layer fast detection with slower understanding.

**3. State machine with V-JEPA confirmation**
```
Fast: Detect major scene changes
Slow: V-JEPA confirms and names the new state
```
React quickly to change, classify accurately after.

### Realistic Classroom Timing

For classroom mode detection, 5-10 second latency is often fine:

| Transition | Real-world duration | V-JEPA detection |
|------------|---------------------|------------------|
| "Everyone break into groups" | 30-60 seconds of shuffling | Detected within 10 sec |
| Teacher walks to whiteboard | 5-10 seconds | Detected within 10 sec |
| Class ends, room empties | 2-3 minutes | Detected within 10 sec |

The system doesn't need to be instant - it needs to track the *current mode* reliably, not detect the exact moment of transition.

### Training Data

| Clips per class | Expected accuracy |
|-----------------|-------------------|
| 5-6 | ~50-60% (barely better than random) |
| 15-20 | ~70-80% |
| 30-50 | ~85-95% |

Tips:
- **Capture motion**, not static poses - V-JEPA needs to see you *doing* the activity
- **Include variation** - different postures, lighting, slight angle changes
- **Balance classes** - similar number of clips per class
- **Multiple camera angles** help generalization

### Camera Placement

- **Fixed position recommended** for training and inference
- If camera moves, train with clips from multiple positions
- Activities should be **visually distinct** from the camera's viewpoint
- Small rooms where everything is visible: focus on motion differences

### What V-JEPA Sees

V-JEPA captures:
- **Temporal patterns** - motion over time (transfers across viewpoints)
- **Spatial context** - what's in the scene (position-dependent)
- **Semantic content** - what's happening (mostly transfers)

Activities with distinctive motion (playing keyboard) classify better than similar postures (sitting at computer vs. sitting with plants).

---

## Troubleshooting

### "Server not reachable"
Start `server.py` in another terminal first.

### First inference very slow
Normal - model warmup/compilation. Subsequent clips are ~300ms.

### Unicode escape error
Windows paths with `\U` in docstrings. Fixed in current version.

### Empty/corrupted clips
Check camera connection. The labeler skips bad clips automatically.

### Low accuracy
- Add more clips (20+ per class)
- Ensure clips capture motion, not just poses
- Check class balance
- Make activities more visually distinct

### "unlabeled" showing as a class
The trainer now ignores the `unlabeled` folder. Retrain with `--no-cache`.

---

## Classroom Transfer

This setup mirrors the classroom Pi deployment:

```
┌─────────────────────────────────┐
│  Central PC (server.py)         │  ← One GPU, one model
│  RTX 5070 Ti / similar          │
└──────────────┬──────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼──┐  ┌───▼───┐  ┌───▼────┐
│orbit │  │gravity│  │horizon │  ← Raspberry Pis
│Pi 5  │  │Pi 5   │  │Pi 5    │     Each runs probe_inference.py
└──────┘  └───────┘  └────────┘
```

1. Train probe on PC with labeled clips from classroom cameras
2. Copy `probe.pt` to each Pi
3. Each Pi runs: `python probe_inference.py --server http://<pc-ip>:8765 --probe classroom_probe.pt`

---

## Files

| File | Description |
|------|-------------|
| `setup.bat` | Initial environment setup |
| `record.bat` | Start auto-recording |
| `label.bat` | Label recorded clips |
| `train.bat` | Train probe |
| `infer.bat` | Run inference |
| `auto_recorder.py` | Timer-based clip recording |
| `clip_recorder.py` | Manual recording with countdown |
| `clip_labeler.py` | Review and label clips |
| `probe_trainer.py` | Train classification probe |
| `probe_inference.py` | Live classification |
| `oak_client.py` | Anomaly detection client |
| `multi_camera_consensus.py` | Combine multi-camera predictions |
