#!/bin/bash
# V-JEPA 2 Inference Server Setup
# RTX 5070 Ti (Blackwell / SM_100) — requires CUDA 12.8 + PyTorch 2.7+
# Run once on your home PC

set -e

echo "=== V-JEPA 2 Server Setup ==="
echo "Target: RTX 5070 Ti (Blackwell architecture)"
echo ""

# ── 1. Conda environment ──────────────────────────────────────────────────────
conda create -n vjepa python=3.11 -y
conda activate vjepa

# ── 2. PyTorch — Blackwell (SM_100) needs 2.7+ ───────────────────────────────
# Check: https://pytorch.org/get-started/locally/
# CUDA 12.8 is required for full Blackwell support.
# If pip install below fails, use the nightly:
#   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
echo "Installing PyTorch with CUDA 12.8 (Blackwell support)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# ── 3. V-JEPA 2 dependencies ──────────────────────────────────────────────────
pip install timm einops

# ── 4. Server dependencies ───────────────────────────────────────────────────
pip install fastapi uvicorn python-multipart opencv-python numpy pillow

# ── 5. Clone V-JEPA 2 repo (for model loading utilities) ─────────────────────
cd /opt || cd ~
if [ ! -d "vjepa2" ]; then
    git clone https://github.com/facebookresearch/vjepa2.git
    echo "Cloned facebookresearch/vjepa2"
else
    echo "vjepa2 repo already exists"
fi

# ── 6. Download model checkpoint ─────────────────────────────────────────────
# V-JEPA 2 ViT-L (recommended: good balance of speed vs quality on 16GB VRAM)
# Full list: https://huggingface.co/collections/facebook/v-jepa-2
echo ""
echo "Downloading V-JEPA 2 ViT-L checkpoint from HuggingFace..."
pip install huggingface_hub
python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="facebook/vjepa2-vitl-fpc64-256",
    local_dir=os.path.expanduser("~/vjepa2-weights/vitl"),
    ignore_patterns=["*.md", "*.txt"]
)
import os
print(f"Weights saved to ~/vjepa2-weights/vitl")
EOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start the server:"
echo "  conda activate vjepa"
echo "  python server.py"
echo ""
echo "Default endpoint: http://0.0.0.0:8765"
echo "  POST /embed   → extract V-JEPA 2 features from a video clip"
echo "  POST /anomaly → get prediction-error surprise score"
echo "  GET  /health  → confirm GPU and model status"
