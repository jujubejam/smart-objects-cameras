"""
probe_trainer.py
Train a lightweight "attentive probe" on top of frozen V-JEPA 2 embeddings
to classify classroom-specific behaviours without touching the world model.

This is the key pedagogical trick: V-JEPA learned general world representations,
and we just add a tiny trainable head for our specific context.

Workflow:
  1. Record labelled video clips of each class (e.g. "at_whiteboard", "discussion", "empty")
  2. Run this script to embed all clips via the server, then train the probe
  3. Run probe_inference.py to classify live camera feeds

Usage:
    # Point at your labelled clip directory and the inference server
    python3 probe_trainer.py \
        --clips-dir ~/classroom-clips \
        --server http://<your-pc-ip>:8765 \
        --output ~/oak-projects/classroom_probe.pt

Clip directory structure:
    classroom-clips/
        at_whiteboard/
            clip_001.mp4
            clip_002.mp4
            ...
        discussion/
            clip_001.mp4
            ...
        empty/
            clip_001.mp4
            ...

The probe itself is tiny (1024 → 256 → N_classes), trains in seconds on CPU
after the embeddings are extracted. Embeddings are cached to avoid re-running
the server on every training run.
"""

import argparse
import json
import pickle
import time
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
    Small 2-layer MLP head that goes on top of frozen V-JEPA embeddings.
    This is intentionally tiny — the power is in the frozen V-JEPA features.
    Train time: seconds. Matches the JEPA paper's evaluation protocol.
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
def embed_clip(server_url: str, clip_path: Path, camera_id: str = "trainer") -> list[float] | None:
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
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Walk clips_dir, embed every clip, return (X, y, class_names).
    Results are cached so you don't re-embed on every training run.
    """
    if cache_path.exists():
        log.info(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        return cache["X"], cache["y"], cache["class_names"]

    # Filter out 'unlabeled' folder - it's for staging, not training
    class_dirs = sorted([d for d in clips_dir.iterdir() if d.is_dir() and d.name != "unlabeled"])
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {clips_dir}. Create folders like 'lecture/', 'group_work/', etc.")

    class_names = [d.name for d in class_dirs]
    log.info(f"Classes: {class_names}")

    X, y = [], []
    for class_idx, class_dir in enumerate(class_dirs):
        clips = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))
        log.info(f"  [{class_dir.name}] {len(clips)} clips → embedding...")
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
    class_names: list[str],
    epochs: int = 100,
    lr: float = 1e-3,
    hidden_dim: int = 256,
) -> AttentiveProbe:
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

    log.info(f"Training probe: {embed_dim}d → {hidden_dim} → {num_classes} classes")
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
def save_probe(probe: AttentiveProbe, class_names: list[str], output_path: Path):
    torch.save({
        "state_dict": probe.state_dict(),
        "class_names": class_names,
        "embed_dim": probe.net[1].in_features,
        "hidden_dim": probe.net[1].out_features,
    }, output_path)
    log.info(f"Probe saved to {output_path}")
    log.info(f"Classes: {class_names}")
    log.info("To classify live video, run: python3 probe_inference.py --probe " + str(output_path))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train V-JEPA attentive probe")
    parser.add_argument("--clips-dir", type=Path, required=True,
                        help="Directory of labelled clips (one subdir per class)")
    parser.add_argument("--server", default="http://localhost:8765",
                        help="V-JEPA inference server URL")
    parser.add_argument("--output", type=Path, default=Path.home() / "oak-projects/classroom_probe.pt",
                        help="Where to save the trained probe")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-cache", action="store_true", help="Re-embed even if cache exists")
    args = parser.parse_args()

    cache_path = args.output.parent / f"{args.output.stem}_embed_cache.pkl"
    if args.no_cache and cache_path.exists():
        cache_path.unlink()

    # ── Health check ──────────────────────────────────────────────────────────
    try:
        r = requests.get(f"{args.server}/health", timeout=5)
        r.raise_for_status()
        log.info(f"Server: {r.json().get('gpu', 'ok')}")
    except Exception as e:
        log.error(f"Server not reachable: {e}")
        return

    # ── Extract embeddings ────────────────────────────────────────────────────
    X, y, class_names = extract_embeddings(args.clips_dir, args.server, cache_path)

    # ── Train ─────────────────────────────────────────────────────────────────
    probe = train_probe(X, y, class_names, epochs=args.epochs, lr=args.lr)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_probe(probe, class_names, args.output)


if __name__ == "__main__":
    main()
