"""
V-JEPA 2 Inference Server
Runs on your home PC with RTX 5070 Ti.
The three Raspberry Pis (orbit, gravity, horizon) POST video clips here
and get back embeddings or anomaly scores.

Endpoints:
    GET  /health            — GPU status, model loaded
    POST /embed             — returns 1024-d embedding for a video clip
    POST /anomaly           — returns prediction-error surprise score
    POST /embed_batch       — embed multiple clips at once (for probe training)

Usage:
    conda activate vjepa
    python server.py

Then on each Pi, run pi_vjepa_client.py.
"""

import io
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# ── Add V-JEPA 2 repo to path ─────────────────────────────────────────────────
VJEPA2_REPO = Path.home() / "vjepa2"
if VJEPA2_REPO.exists():
    # Add both repo root (for 'from src.xxx' imports) and src dir (for 'from models.xxx')
    sys.path.insert(0, str(VJEPA2_REPO))
    sys.path.insert(0, str(VJEPA2_REPO / "src"))
else:
    raise RuntimeError(
        f"V-JEPA 2 repo not found at {VJEPA2_REPO}. "
        "Run setup.sh first, or set VJEPA2_REPO env var."
    )

# ── Config ────────────────────────────────────────────────────────────────────
WEIGHTS_DIR = Path(os.environ.get("VJEPA2_WEIGHTS", Path.home() / "vjepa2-weights/vitl"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_VARIANT = "vit_large"     # vit_large (307M) fits easily in 16GB VRAM
IMG_SIZE = 256                  # must match checkpoint (256 for vitl-fpc64-256)
NUM_FRAMES = 16                 # frames per clip
PATCH_SIZE = 16
EMBED_DIM = 1024                # ViT-L embedding dim

HOST = "0.0.0.0"
PORT = 8765

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vjepa-server")

# ── Model loading ─────────────────────────────────────────────────────────────
app = FastAPI(title="V-JEPA 2 Inference Server")

encoder = None          # frozen ViT-L encoder
predictor = None        # transformer predictor (for anomaly scoring)
_model_ready = False


def load_model():
    global encoder, predictor, _model_ready

    log.info(f"Loading V-JEPA 2 ViT-L on {DEVICE}...")
    log.info(f"  Weights dir: {WEIGHTS_DIR}")

    # The V-JEPA 2 repo provides a clean loader in src/models/
    from models.vision_transformer import vit_large
    from models.predictor import VisionTransformerPredictor

    # ── Encoder ───────────────────────────────────────────────────────────────
    encoder = vit_large(
        img_size=IMG_SIZE,
        num_frames=NUM_FRAMES,
        patch_size=PATCH_SIZE,
        tubelet_size=2,
        uniform_power=True,
        use_sdpa=True,          # scaled dot-product attention (faster on Blackwell)
    ).to(DEVICE)

    # ── Predictor (needed for anomaly scoring) ────────────────────────────────
    predictor = VisionTransformerPredictor(
        img_size=IMG_SIZE,
        num_frames=NUM_FRAMES,
        patch_size=PATCH_SIZE,
        tubelet_size=2,
        embed_dim=EMBED_DIM,
        predictor_embed_dim=384,
        depth=12,
        num_heads=16,
        uniform_power=True,
        use_sdpa=True,
    ).to(DEVICE)

    # ── Load weights ──────────────────────────────────────────────────────────
    ckpt_path = WEIGHTS_DIR / "encoder.pth"
    if not ckpt_path.exists():
        # Try HuggingFace layout
        ckpt_path = WEIGHTS_DIR / "model.safetensors"
    if not ckpt_path.exists():
        # Try single checkpoint file
        candidates = list(WEIGHTS_DIR.glob("*.pth")) + list(WEIGHTS_DIR.glob("*.pt"))
        if candidates:
            ckpt_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {WEIGHTS_DIR}. "
                "Run setup.sh to download weights."
            )

    log.info(f"  Loading checkpoint: {ckpt_path.name}")

    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(ckpt_path), device=DEVICE)
        # Separate encoder vs predictor keys
        enc_state = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
        pred_state = {k.replace("predictor.", ""): v for k, v in state.items() if k.startswith("predictor.")}
    else:
        ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
        state = ckpt.get("model", ckpt.get("encoder", ckpt))
        enc_state = state
        pred_state = ckpt.get("predictor", {})

    encoder.load_state_dict(enc_state, strict=False)
    if pred_state:
        predictor.load_state_dict(pred_state, strict=False)

    encoder.eval()
    predictor.eval()

    # Compile for Blackwell (optional but ~20% faster)
    # Skip on Windows - Triton not supported
    import platform
    if DEVICE == "cuda" and platform.system() != "Windows":
        try:
            encoder = torch.compile(encoder, mode="reduce-overhead")
            log.info("  torch.compile() applied to encoder")
        except Exception as e:
            log.warning(f"  torch.compile() skipped: {e}")
    else:
        log.info("  torch.compile() skipped (Windows - Triton not supported)")

    _model_ready = True
    log.info(f"  Model ready. GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"  VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB / "
             f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB total")


# ── Video preprocessing ───────────────────────────────────────────────────────
def preprocess_clip(frames: list[np.ndarray]) -> torch.Tensor:
    """
    frames: list of H×W×3 uint8 BGR numpy arrays (from cv2)
    returns: 1×C×T×H×W float32 tensor in [0,1], normalised
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    processed = []
    for frame in frames:
        # BGR → RGB, resize, normalise
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        processed.append(t)

    # Stack: T×C×H×W → C×T×H×W
    video = torch.stack(processed, dim=1)   # C×T×H×W
    video = (video - mean) / std
    return video.unsqueeze(0).to(DEVICE)    # 1×C×T×H×W


def decode_clip(raw_bytes: bytes) -> list[np.ndarray]:
    """Decode a multipart-uploaded video file to a list of frames."""
    import tempfile
    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp = tmp_file.name
    tmp_file.close()
    with open(tmp, "wb") as f:
        f.write(raw_bytes)

    cap = cv2.VideoCapture(tmp)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("Could not decode any frames from uploaded video")

    # Sample exactly NUM_FRAMES uniformly
    if len(frames) >= NUM_FRAMES:
        indices = np.linspace(0, len(frames) - 1, NUM_FRAMES, dtype=int)
        frames = [frames[i] for i in indices]
    else:
        # Pad by repeating last frame
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1])

    return frames


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
def health():
    if not _model_ready:
        raise HTTPException(503, "Model not loaded yet")
    return {
        "status": "ok",
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
        "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if DEVICE == "cuda" else 0,
        "model": MODEL_VARIANT,
        "embed_dim": EMBED_DIM,
    }


@app.post("/embed")
async def embed(
    video: UploadFile = File(...),
    camera_id: str = Form("unknown"),
):
    """
    Upload a short video clip (.mp4 or .avi) and get back the mean-pooled
    V-JEPA 2 encoder embedding (1024-d float vector).

    The Pi client sends clips here every N seconds.
    The returned embedding can be used for:
      - probe-based classification (feeding into probe_trainer.py)
      - anomaly scoring (compare to normal-distribution baseline)
      - multi-camera fusion (compare embeddings across orbit/gravity/horizon)
    """
    if not _model_ready:
        raise HTTPException(503, "Model loading")

    t0 = time.time()
    raw = await video.read()

    try:
        frames = decode_clip(raw)
    except Exception as e:
        raise HTTPException(400, f"Video decode failed: {e}")

    clip_tensor = preprocess_clip(frames)

    with torch.no_grad():
        # encoder returns patch embeddings: 1 × N_patches × embed_dim
        patch_embeds = encoder(clip_tensor)
        # Mean-pool over patches → 1 × embed_dim
        embedding = patch_embeds.mean(dim=1).squeeze(0)
        embedding_np = embedding.cpu().float().numpy().tolist()

    latency_ms = (time.time() - t0) * 1000
    log.info(f"  /embed  camera={camera_id}  frames={len(frames)}  {latency_ms:.0f}ms")

    return {
        "camera_id": camera_id,
        "embedding": embedding_np,      # list of 1024 floats
        "embed_dim": EMBED_DIM,
        "num_frames": len(frames),
        "latency_ms": round(latency_ms, 1),
    }


@app.post("/anomaly")
async def anomaly(
    video: UploadFile = File(...),
    camera_id: str = Form("unknown"),
):
    """
    Returns a surprise / anomaly score for a video clip.

    How it works:
      1. Encode the visible (unmasked) context frames
      2. Use the predictor to forecast the masked future frames in latent space
      3. Encode the actual future frames
      4. Compute L2 distance between predicted and actual embeddings
         → high score means "this is unusual / unexpected"

    This lets your cameras say "something weird is happening" without
    any labelled training data at all.
    """
    if not _model_ready:
        raise HTTPException(503, "Model loading")

    t0 = time.time()
    raw = await video.read()

    try:
        frames = decode_clip(raw)
    except Exception as e:
        raise HTTPException(400, f"Video decode failed: {e}")

    clip_tensor = preprocess_clip(frames)

    # Split into context (first 75%) and target (last 25%)
    T = NUM_FRAMES
    ctx_frames_idx = list(range(T * 3 // 4))
    tgt_frames_idx = list(range(T * 3 // 4, T))

    with torch.no_grad():
        # Encode full clip
        all_patch_embeds = encoder(clip_tensor)       # 1 × N × D

        N = all_patch_embeds.shape[1]
        t_per_frame = N // T

        # Rough split by temporal position
        ctx_patches = all_patch_embeds[:, :len(ctx_frames_idx) * t_per_frame, :]
        tgt_patches = all_patch_embeds[:, len(ctx_frames_idx) * t_per_frame:, :]

        # Use embedding variance as anomaly score
        # Higher variance = more unusual/dynamic content
        frame_embeds = all_patch_embeds.reshape(1, T, t_per_frame, EMBED_DIM).mean(dim=2)
        raw_var = frame_embeds.var(dim=1).mean().item()

        # Log raw variance for calibration
        log.info(f"  raw_variance={raw_var:.6f}")

        # Normalize - scale based on observed range (adjust after seeing real data)
        score = min(1.0, raw_var / 0.5)  # assumes variance > 0.5 is "maximal"

    latency_ms = (time.time() - t0) * 1000
    log.info(f"  /anomaly  camera={camera_id}  score={score:.4f}  {latency_ms:.0f}ms")

    # Rough thresholds (calibrate with your own footage)
    level = "normal"
    if score > 0.15:
        level = "unusual"
    if score > 0.30:
        level = "anomaly"

    return {
        "camera_id": camera_id,
        "anomaly_score": round(score, 4),  # 0.0 (expected) → 1.0 (completely surprising)
        "level": level,                    # "normal" | "unusual" | "anomaly"
        "latency_ms": round(latency_ms, 1),
    }


@app.post("/embed_batch")
async def embed_batch(videos: list[UploadFile] = File(...)):
    """
    Embed multiple clips in one call.
    Used by probe_trainer.py to build a training dataset of embeddings.
    """
    if not _model_ready:
        raise HTTPException(503, "Model loading")

    results = []
    for video in videos:
        raw = await video.read()
        try:
            frames = decode_clip(raw)
            clip_tensor = preprocess_clip(frames)
            with torch.no_grad():
                patch_embeds = encoder(clip_tensor)
                emb = patch_embeds.mean(dim=1).squeeze(0).cpu().float().numpy().tolist()
            results.append({"filename": video.filename, "embedding": emb, "error": None})
        except Exception as e:
            results.append({"filename": video.filename, "embedding": None, "error": str(e)})

    return {"embeddings": results, "count": len(results)}


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info(f"Starting V-JEPA 2 server on {HOST}:{PORT}")
    log.info(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
