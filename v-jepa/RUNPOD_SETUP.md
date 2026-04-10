# RunPod V-JEPA Server Setup

Run the V-JEPA inference server in the cloud so you don't have to lug your PC to school.

---

## Overview

```
┌─────────────────────────────────────────────────────────┐
│  School Network                                          │
│                                                          │
│   ┌────────┐  ┌────────┐  ┌────────┐                    │
│   │ orbit  │  │gravity │  │horizon │  ← Raspberry Pis   │
│   │  Pi 5  │  │  Pi 5  │  │  Pi 5  │    running         │
│   └───┬────┘  └───┬────┘  └───┬────┘    probe_inference │
│       │           │           │                          │
│       └───────────┴───────────┘                          │
│                   │                                      │
└───────────────────┼──────────────────────────────────────┘
                    │ HTTPS
                    ▼
         ┌──────────────────────┐
         │  RunPod GPU Server   │  ← V-JEPA server.py
         │  RTX 4090 / A4000    │     in the cloud
         └──────────────────────┘
```

---

## Step 1: Create RunPod Account

1. Go to [runpod.io](https://runpod.io)
2. Create account
3. Add credits ($10-20 is plenty for testing)

---

## Step 2: Launch GPU Pod

### Recommended Specs

| Setting | Value | Notes |
|---------|-------|-------|
| GPU | RTX 4090 or RTX A4000 | 16-24GB VRAM, ~$0.40-0.75/hr |
| Template | RunPod PyTorch 2.1 | Has CUDA pre-installed |
| Volume | 20GB | For model weights |
| Expose Ports | 8765 | Our server port |

### Steps

1. Click **Deploy** → **GPU Pods**
2. Select a GPU (RTX 4090 is fast, A4000 is cheaper)
3. Choose **RunPod PyTorch 2.1** template
4. Under **Customize Deployment**:
   - Volume: 20GB persistent
   - Expose HTTP Ports: `8765`
5. Click **Deploy**

---

## Step 3: Connect to Pod

Once running, click **Connect** → **Web Terminal** (or use SSH).

---

## Step 4: Install V-JEPA

```bash
# Clone your repo (or upload files)
cd /workspace
git clone https://github.com/YOUR_USERNAME/smart-objects-cameras.git
cd smart-objects-cameras/v-jepa

# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn python-multipart opencv-python numpy timm huggingface_hub

# Download V-JEPA weights (first run does this automatically)
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/vjepa-2-vitl-fpc64-256', 'model.safetensors')"
```

---

## Step 5: Run Server

```bash
cd /workspace/smart-objects-cameras/v-jepa
source venv/bin/activate

# Run on all interfaces so it's accessible externally
python server.py --host 0.0.0.0 --port 8765
```

You should see:
```
INFO:     V-JEPA 2 loaded on cuda
INFO:     Uvicorn running on http://0.0.0.0:8765
```

---

## Step 6: Get Public URL

In RunPod dashboard:
1. Click on your pod
2. Find **Connect** → **HTTP Service [8765]**
3. Copy the URL (looks like `https://xxxx-8765.proxy.runpod.net`)

Test it:
```bash
curl https://xxxx-8765.proxy.runpod.net/health
```

---

## Step 7: Configure Pis

On each Raspberry Pi, update the server URL:

```bash
# Test connection
curl https://xxxx-8765.proxy.runpod.net/health

# Run inference pointing to RunPod
python probe_inference.py \
    --server https://xxxx-8765.proxy.runpod.net \
    --probe ~/oak-projects/classroom_probe.pt
```

---

## Step 8: Keep Server Running

### Option A: Screen/tmux (simple)

```bash
# Start a persistent session
tmux new -s vjepa
python server.py --host 0.0.0.0 --port 8765
# Ctrl+B, then D to detach

# Reattach later
tmux attach -t vjepa
```

### Option B: Systemd service (robust)

```bash
cat > /etc/systemd/system/vjepa.service << 'EOF'
[Unit]
Description=V-JEPA Inference Server

[Service]
WorkingDirectory=/workspace/smart-objects-cameras/v-jepa
ExecStart=/workspace/smart-objects-cameras/v-jepa/venv/bin/python server.py --host 0.0.0.0 --port 8765
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl enable vjepa
systemctl start vjepa
```

---

## Cost Estimate

| Usage | GPU | Cost |
|-------|-----|------|
| 3-hour class session | RTX A4000 | ~$1.20 |
| 3-hour class session | RTX 4090 | ~$2.25 |
| Full day (8 hrs) | RTX A4000 | ~$3.20 |

**Tip:** Stop the pod when not in use. Your volume persists, so you won't lose data.

---

## Saving Trained Probes

Probes are tiny (~1MB). Options:

1. **Train locally** (at home), copy probe to Pis via SCP
2. **Train on RunPod**, download probe file
3. **Store probes in repo** (they're small enough to commit)

```bash
# Download probe from RunPod to your PC
scp root@your-pod-ip:/workspace/classroom_probe.pt .

# Upload to Pi
scp classroom_probe.pt pi@orbit:~/oak-projects/
```

---

## Troubleshooting

### "Connection refused"
- Check pod is running
- Verify port 8765 is exposed
- Use HTTPS URL from dashboard, not raw IP

### "Model loading slow"
- First run downloads ~2GB weights
- Subsequent starts use cached weights

### Pod stopped unexpectedly
- Check RunPod dashboard for errors
- You may have run out of credits
- GPU may have been preempted (use on-demand, not spot)

### High latency
- RunPod servers are usually US-based
- Expect 100-300ms network overhead on top of inference time
- Still faster than not having a server at all

---

## Quick Reference

```bash
# SSH into pod
ssh root@your-pod-ip -i ~/.ssh/runpod_key

# Start server
cd /workspace/smart-objects-cameras/v-jepa
source venv/bin/activate
python server.py --host 0.0.0.0 --port 8765

# Health check
curl https://xxxx-8765.proxy.runpod.net/health

# Stop pod (from web dashboard)
# → saves money when not in use
```

---

## Alternative: Vast.ai

Similar service, sometimes cheaper:
- [vast.ai](https://vast.ai)
- Same general setup process
- Compare prices before choosing

---

*Your PC stays home. The cloud does the heavy lifting.*
