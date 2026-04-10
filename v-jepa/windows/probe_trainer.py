"""
probe_trainer.py (Windows version)
===================================
Train a lightweight probe on top of frozen V-JEPA 2 embeddings
to classify your custom activities.

Workflow:
  1. Record clips with clip_recorder.py (or manually)
  2. Run this script to embed all clips via the server, then train the probe
  3. Run probe_inference.py to classify live camera feeds

Usage:
    python probe_trainer.py --clips-dir C:/Users/you/home-clips

Clip directory structure:
    home-clips/
        at_computer/
            clip_001.mp4
            clip_002.mp4
            ...
        playing_keyboard/
            clip_001.mp4
            ...
        tending_plants/
            clip_001.mp4
            ...

The probe is tiny (1024 -> 256 -> N_classes), trains in seconds on CPU.
Embeddings are cached so re-training is instant.
"""

import argparse
import pickle
import logging
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("probe-trainer")


# ── Probe architecture ────────────────────────────────────────────────────────
class AttentiveProbe(nn.Module):
    """
    Small 2-layer MLP head on top of frozen V-JEPA embeddings.
    Intentionally tiny - the power is in the frozen V-JEPA features.
    """
    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Embedding extraction ──────────────────────────────────────────────────────
def embed_clip(server_url: str, clip_path: Path, camera_id: str = "trainer"):
    """Send a clip to the server and get back the 1024-d embedding."""
    url = f"{server_url.rstrip('/')}/embed"
    try:
        with open(clip_path, "rb") as f:
            r = requests.post(
                url,
                files={"video": (clip_path.name, f, "video/mp4")},
                data={"camera_id": camera_id},
                timeout=30,
            )
        r.raise_for_status()
        return r.json()["embedding"]
    except Exception as e:
        log.warning(f"  Embed failed for {clip_path.name}: {e}")
        return None


def extract_embeddings(
    clips_dir: Path,
    server_url: str,
    cache_path: Path,
    classes: list = None,
):
    """
    Walk clips_dir, embed every clip, return (X, y, class_names).
    Results are cached so you don't re-embed on every training run.
    """
    if cache_path.exists():
        log.info(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        return cache["X"], cache["y"], cache["class_names"]

    if classes:
        # Use provided classes list - only include dirs that exist
        existing = [c for c in classes if (clips_dir / c).is_dir()]
        if not existing:
            raise ValueError(f"None of the specified classes found in {clips_dir}: {classes}")
        class_names = existing
        class_dirs = [clips_dir / c for c in existing]
    else:
        # Auto-discover from subdirectories
        class_dirs = sorted([d for d in clips_dir.iterdir() if d.is_dir() and d.name != "unlabeled"])
        if not class_dirs:
            raise ValueError(f"No subdirectories found in {clips_dir}")
        class_names = [d.name for d in class_dirs]
    log.info(f"Classes: {class_names}")

    X, y = [], []
    for class_idx, class_dir in enumerate(class_dirs):
        clips = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))
        log.info(f"  [{class_dir.name}] {len(clips)} clips")
        for clip in clips:
            emb = embed_clip(server_url, clip)
            if emb is not None:
                X.append(emb)
                y.append(class_idx)
                print(".", end="", flush=True)
        print()

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    log.info(f"Total: {len(X)} embeddings, {len(class_names)} classes")
    log.info(f"Embedding shape: {X.shape}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"X": X, "y": y, "class_names": class_names}, f)
    log.info(f"Cached embeddings to {cache_path}")

    return X, y, class_names


# ── Training ──────────────────────────────────────────────────────────────────
def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list,
    epochs: int = 100,
    lr: float = 1e-3,
    hidden_dim: int = 256,
):
    embed_dim = X.shape[1]
    num_classes = len(class_names)

    # Train/val split (80/20)
    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    probe = AttentiveProbe(embed_dim, num_classes, hidden_dim)
    optimiser = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    criterion = nn.CrossEntropyLoss()

    log.info(f"Training probe: {embed_dim}d -> {hidden_dim} -> {num_classes} classes")
    log.info(f"  Train: {len(train_idx)} samples  Val: {len(val_idx)} samples")

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        probe.train()
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimiser.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            probe.eval()
            with torch.no_grad():
                val_logits = probe(X_val)
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == y_val).float().mean().item()
                train_logits = probe(X_train)
                train_preds = train_logits.argmax(dim=1)
                train_acc = (train_preds == y_train).float().mean().item()

            log.info(f"  Epoch {epoch:3d}/{epochs}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    if best_state:
        probe.load_state_dict(best_state)
    log.info(f"Best val accuracy: {best_val_acc:.3f}")
    return probe


# ── Save ──────────────────────────────────────────────────────────────────────
def save_probe(probe, class_names, output_path: Path):
    torch.save({
        "state_dict": probe.state_dict(),
        "class_names": class_names,
        "embed_dim": probe.net[1].in_features,
        "hidden_dim": probe.net[1].out_features,
    }, output_path)
    log.info(f"Probe saved to {output_path}")
    log.info(f"Classes: {class_names}")
    log.info(f"\nTo classify live video, run:")
    log.info(f"  python probe_inference.py --probe {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train V-JEPA attentive probe")
    parser.add_argument("--clips-dir", type=Path,
                       default=Path.home() / "home-clips",
                       help="Directory of labelled clips (one subdir per class)")
    parser.add_argument("--server", default="http://localhost:8765",
                       help="V-JEPA inference server URL")
    parser.add_argument("--output", type=Path,
                       default=Path.home() / "oak-projects" / "home_probe.pt",
                       help="Where to save the trained probe")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-cache", action="store_true",
                       help="Re-embed even if cache exists")
    parser.add_argument("--classes", type=str, default=None,
                       help="Comma-separated class names (e.g. empty_room,lecture,group_work)")
    args = parser.parse_args()

    # Validate clips directory
    if not args.clips_dir.exists():
        log.error(f"Clips directory not found: {args.clips_dir}")
        log.error("Run clip_recorder.py first to record training clips.")
        return

    cache_path = args.output.parent / f"{args.output.stem}_embed_cache.pkl"
    if args.no_cache and cache_path.exists():
        cache_path.unlink()

    # Health check
    try:
        r = requests.get(f"{args.server}/health", timeout=5)
        r.raise_for_status()
        info = r.json()
        log.info(f"Server: {info.get('gpu', 'ok')}  VRAM: {info.get('vram_used_gb', '?')} GB")
    except Exception as e:
        log.error(f"Server not reachable at {args.server}: {e}")
        log.error("Start server.py first.")
        return

    # Parse classes if provided
    classes = args.classes.split(",") if args.classes else None

    # Extract embeddings
    X, y, class_names = extract_embeddings(args.clips_dir, args.server, cache_path, classes)

    if len(X) < 6:
        log.error(f"Not enough clips! Got {len(X)}, need at least 6 (2 per class minimum)")
        return

    # Train
    probe = train_probe(X, y, class_names, epochs=args.epochs, lr=args.lr)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_probe(probe, class_names, args.output)


if __name__ == "__main__":
    main()
