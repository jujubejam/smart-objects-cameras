# V-JEPA 2 Integration Guide
## SVA Smart Objects Camera Project

---

## What you're building

```
┌────────────────────────────────┐
│  Your Home PC (RTX 5070 Ti)    │  ← V-JEPA 2 inference server
│  server.py  port 8765          │     16GB VRAM, Blackwell arch
└──────────────┬─────────────────┘
               │ local network (WiFi/Ethernet)
       ┌───────┼───────────────┐
       │       │               │
  ┌────▼───┐ ┌─▼──────┐ ┌─────▼──┐
  │ orbit  │ │gravity │ │horizon │  ← Raspberry Pi 5s
  │ OAK-D  │ │ OAK-D  │ │ OAK-D  │     pi_vjepa_client.py
  └────────┘ └────────┘ └────────┘
       │           │           │
       └───────────┴───────────┘
                   │
            vjepa_status.json
                   │
           discord_bot.py
                   │
            Discord (!worldmodel, !classify)
```

---

## Part 1: PC Setup (RTX 5070 Ti)

### Step 1.1: CUDA and PyTorch (Blackwell note)

The 5070 Ti uses NVIDIA's Blackwell architecture.
You need **CUDA 12.8** and **PyTorch 2.7+** for full support.

```bash
# Check your CUDA version
nvidia-smi

# If it shows < 12.8, update CUDA toolkit:
# https://developer.nvidia.com/cuda-downloads

# Then install PyTorch with CUDA 12.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

If the stable release doesn't work yet for your card, use nightly:
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Step 1.2: Run setup script

```bash
bash setup.sh
```

This clones the V-JEPA 2 repo, downloads ViT-L weights (~1.2GB), and
installs all dependencies.

### Step 1.3: Start the server

```bash
conda activate vjepa
python server.py
```

Expected output:
```
2026-xx-xx INFO Loading V-JEPA 2 ViT-L on cuda...
2026-xx-xx INFO   Weights dir: /home/you/vjepa2-weights/vitl
2026-xx-xx INFO   Loading checkpoint: model.safetensors
2026-xx-xx INFO   torch.compile() applied to encoder
2026-xx-xx INFO   Model ready. GPU: NVIDIA GeForce RTX 5070 Ti
2026-xx-xx INFO   VRAM used: 2.4 GB / 16.0 GB total
```

### Step 1.4: Verify it works

```bash
curl http://localhost:8765/health
# → {"status": "ok", "gpu": "NVIDIA GeForce RTX 5070 Ti", ...}
```

---

## Part 2: Make the server accessible from the Pis

The Pis need to reach your PC over the local network.

```bash
# Find your PC's local IP
ip addr show | grep "inet 192"
# or: hostname -I

# Example: <your-pc-ip>
```

Test from a Pi (or any machine on the same network):
```bash
curl http://<your-pc-ip>:8765/health
```

If it doesn't connect:
- Check your PC's firewall: `sudo ufw allow 8765`
- Or on Windows: add inbound rule for port 8765 in Windows Defender Firewall

---

## Part 3: Pi Setup

SSH into each Pi (orbit, gravity, horizon):

```bash
ssh orbit
source /opt/oak-shared/venv/bin/activate

# Install requests if not present
pip install requests

# Run the V-JEPA client
python3 pi_vjepa_client.py --server http://<your-pc-ip>:8765 --discord
```

This will:
1. Capture 2-second clips from the OAK-D camera every 10 seconds
2. Send them to your PC server
3. Get back an anomaly score (0.0 = normal, 1.0 = very surprising)
4. Write `~/oak-projects/vjepa_status.json`
5. Alert Discord when anomaly score exceeds 0.20

---

## Part 4: Understanding Anomaly Scores

V-JEPA's anomaly score is the **prediction error** of the world model.
- The model encodes the first ~75% of each clip (context)
- Predicts what the remaining 25% should look like in embedding space
- Compares the prediction to the actual observed frames
- High cosine distance → the world did something unexpected

**Calibration is important.** Run the client for 30 minutes during
normal classroom activity and note the typical score range.
Then adjust `--threshold` accordingly.

Rough starting point:
```
0.00 - 0.10 → very expected / routine
0.10 - 0.20 → mild variation
0.20 - 0.35 → notable change (new person, movement)
0.35+       → unusual event
```

---

## Part 5: Train a Classroom Probe

The probe lets you name specific activities rather than just "normal/anomaly".

### Step 5.1: Record labelled clips

Record short clips of each activity you want to recognise.
Aim for 20-50 clips per class, each 2-5 seconds long.

```
~/classroom-clips/
  at_whiteboard/
    clip_001.mp4   (someone writing on whiteboard)
    clip_002.mp4
    ...
  group_discussion/
    clip_001.mp4   (people talking at tables)
    ...
  empty_room/
    clip_001.mp4
    ...
  presentation/
    clip_001.mp4
    ...
```

You can record these from a phone or extract them from existing recordings.

### Step 5.2: Train the probe

Make sure `server.py` is running, then:

```bash
python3 probe_trainer.py \
    --clips-dir ~/classroom-clips \
    --server http://localhost:8765 \
    --output ~/oak-projects/classroom_probe.pt \
    --epochs 100
```

The embeddings are cached after the first run — re-training is instant.

Expected output:
```
Epoch  10/100  train_acc=0.823  val_acc=0.789
Epoch  20/100  train_acc=0.891  val_acc=0.847
...
Epoch 100/100  train_acc=0.967  val_acc=0.923
Best val accuracy: 0.923
Probe saved to ~/oak-projects/classroom_probe.pt
```

~90% accuracy with just 20 clips per class is typical.
This is the "label efficiency" advantage of JEPA representations.

### Step 5.3: Run live inference on Pis

```bash
python3 probe_inference.py \
    --server http://<your-pc-ip>:8765 \
    --probe ~/oak-projects/classroom_probe.pt \
    --discord
```

---

## Part 6: Discord Commands

Add to `discord_bot.py` (see `discord_vjepa_commands.py`):

| Command | What it shows |
|---------|---------------|
| `!worldmodel` | Current anomaly score + level for this camera |
| `!classify` | Probe-based activity label + confidence |
| `!surprise-history` | Sparkline of recent anomaly scores |

Example output:
```
🧠 World Model Status (orbit)
🟡 Level: UNUSUAL
Surprise score: 0.187 [████░░░░░░░░░░░░░░░░]
Server latency: 243ms
2026-03-16 14:23:01
```

---

## Architecture Notes for Students

### Why V-JEPA is different from YOLO

| YOLO (current detectors) | V-JEPA (world model) |
|---|---|
| Detects labelled categories | Learns any pattern from raw video |
| Needs annotated training data | Self-supervised — no labels needed |
| Fixed vocabulary (person, car, ...) | Open-ended representation |
| Can't reason about novelty | "Surprised" by unusual events |
| Single-frame (mostly) | Temporal / predictive |

### Why the probe trick works

V-JEPA's encoder compresses video into a 1024-d vector that captures
what matters semantically (motion, interaction, context) and discards
noise. A 2-layer MLP on top of that is enough to classify complex
activities because the hard work of understanding video was already done
by the world model's self-supervised pre-training on 1 million hours of
internet video.

The model never saw your classroom — but it learned general
physics and human motion, so your few labelled clips are enough.

---

## Troubleshooting

**"torch.compile() skipped"**
Blackwell support in torch.compile varies. It's optional — skip it and inference still works.

**Slow inference (>5 seconds per clip)**
Normal for the first few clips (model warmup). Should stabilise to ~500ms-2s on 5070 Ti.

**"No frames decoded"**
The Pi is sending empty video files. Check the OAK-D camera is running
and the preview queue is not empty before capturing.

**High anomaly scores on everything**
The model needs to be warmed up on your specific environment.
Run for 15+ minutes of normal activity before relying on scores.

**Connection refused from Pi**
Check PC firewall. On Linux: `sudo ufw allow 8765`.
On Windows: Windows Defender Firewall → Inbound Rules → New Rule → Port 8765.

---

## Windows-Specific Notes (Tested 2026-03-17)

### Setup for Windows

Use `v-jepa/windows/setup.bat` in Anaconda Prompt:
```powershell
cd v-jepa\windows
setup.bat
```

### Known Issues and Fixes Applied

**1. torch.compile() / Triton not supported on Windows**
- `server.py` automatically skips `torch.compile()` on Windows
- Inference is ~20% slower but works fine
- ~300ms per clip on RTX 5070 Ti

**2. Temp file path (Linux vs Windows)**
- Fixed `/tmp/vjepa_upload.mp4` → uses `tempfile.NamedTemporaryFile()`

**3. OpenCV GUI support**
- If you get "function not implemented" errors with `--display`:
  ```powershell
  pip uninstall opencv-python opencv-python-headless -y
  pip install opencv-python
  ```

**4. Type hints for older depthai**
- Removed `dai.DataOutputQueue` type hint that caused AttributeError

### Running on Windows (2 OAK-D cameras)

**Terminal 1 - Server:**
```powershell
conda activate vjepa
cd v-jepa
python server.py
```

**Terminal 2 - USB camera:**
```powershell
conda activate vjepa
cd v-jepa\windows
python oak_client.py --name usb-oak --display
```

**Terminal 3 - Ethernet camera:**
```powershell
conda activate vjepa
cd v-jepa\windows
python oak_client.py --ip 192.168.X.X --name eth-oak --display
```

### Current Anomaly Scoring (Simplified)

The current implementation uses **embedding variance** instead of true JEPA prediction:
- Measures how much V-JEPA embeddings change across frames in a clip
- Higher variance = more motion/change in scene
- Baseline for static scene: ~0.10-0.12 raw variance, ~0.20 score

**Observed behavior (2026-03-17 testing):**
- Scores remain fairly stable (~0.19-0.22) regardless of scene content
- Not sensitive enough to detect "anomalies" like person entering/leaving
- Better suited for the **probe training approach** (Part 5) than raw anomaly detection

**TODO:** Implement proper JEPA prediction-error scoring using the predictor model
(requires understanding the V-JEPA 2 predictor API)
