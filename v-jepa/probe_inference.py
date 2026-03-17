"""
probe_inference.py
Live classification on a Raspberry Pi using a trained attentive probe.
Sends clips to the V-JEPA server, gets back embeddings, classifies them locally.
Writes probe_status.json for the Discord bot.

Usage:
    python3 probe_inference.py \
        --server http://<your-pc-ip>:8765 \
        --probe ~/oak-projects/classroom_probe.pt \
        --discord
"""

import argparse
import json
import os
import socket
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn

STATUS_FILE = Path.home() / "oak-projects" / "probe_status.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("probe-inference")


class AttentiveProbe(nn.Module):
    def __init__(self, embed_dim, num_classes, hidden_dim=256):
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


def load_probe(probe_path: Path):
    ckpt = torch.load(probe_path, map_location="cpu")
    probe = AttentiveProbe(
        embed_dim=ckpt["embed_dim"],
        num_classes=len(ckpt["class_names"]),
        hidden_dim=ckpt["hidden_dim"],
    )
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()
    return probe, ckpt["class_names"]


def classify_clip(server_url, clip_path, probe, class_names, camera_id="pi"):
    # Get embedding from server
    url = f"{server_url.rstrip('/')}/embed"
    with open(clip_path, "rb") as f:
        r = requests.post(
            url,
            files={"video": ("clip.mp4", f, "video/mp4")},
            data={"camera_id": camera_id},
            timeout=30,
        )
    r.raise_for_status()
    emb = r.json()["embedding"]

    # Classify with local probe (CPU, instant)
    x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = probe(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = probs.argmax().item()

    return {
        "predicted_class": class_names[pred_idx],
        "confidence": round(probs[pred_idx].item(), 3),
        "class_probs": {c: round(probs[i].item(), 3) for i, c in enumerate(class_names)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--clip-secs", type=float, default=2.0)
    parser.add_argument("--discord", action="store_true")
    args = parser.parse_args()

    probe, class_names = load_probe(args.probe)
    log.info(f"Probe loaded: {class_names}")

    camera_id = socket.gethostname()
    import depthai as dai, tempfile

    CAMERA_W, CAMERA_H = 640, 480
    FPS = 15
    num_frames = int(args.clip_secs * FPS)

    env_file = Path.home() / "oak-projects" / ".env"
    webhook_url = ""
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("DISCORD_WEBHOOK_URL="):
                webhook_url = line.split("=", 1)[1].strip().strip('"')

    with dai.Device() as device:
        with dai.Pipeline(device) as pipeline:
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setPreviewSize(CAMERA_W, CAMERA_H)
            cam.setInterleaved(False)
            cam.setFps(FPS)
            q = cam.preview.createOutputQueue(maxSize=FPS * 5, blocking=False)
            pipeline.start()
            log.info("Camera started")

            last_class = None
            loop = 0

            while True:
                loop += 1
                # Capture clip
                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                tmp.close()
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(tmp.name, fourcc, FPS, (CAMERA_W, CAMERA_H))
                collected = 0
                t0 = time.time()
                while collected < num_frames and time.time() - t0 < num_frames / FPS * 3:
                    pkt = q.tryGet()
                    if pkt:
                        writer.write(pkt.getCvFrame())
                        collected += 1
                    else:
                        time.sleep(0.005)
                writer.release()

                # Classify
                try:
                    result = classify_clip(args.server, tmp.name, probe, class_names, camera_id)
                    pred = result["predicted_class"]
                    conf = result["confidence"]
                    log.info(f"  [{loop:04d}] {pred} ({conf:.2%})  {result['class_probs']}")

                    status = {
                        "camera_id": camera_id,
                        "timestamp": datetime.now().isoformat(),
                        "predicted_class": pred,
                        "confidence": conf,
                        "class_probs": result["class_probs"],
                    }
                    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
                    STATUS_FILE.write_text(json.dumps(status, indent=2))

                    # Discord on class change
                    if args.discord and webhook_url and pred != last_class:
                        emoji = {"at_whiteboard": "✏️", "discussion": "💬", "empty": "🪑"}.get(pred, "🎥")
                        requests.post(webhook_url, json={
                            "content": f"{emoji} **{camera_id}** → `{pred}` ({conf:.0%} confidence)"
                        }, timeout=5)
                    last_class = pred
                except Exception as e:
                    log.error(f"  Classification error: {e}")
                finally:
                    import os; os.unlink(tmp.name)

                time.sleep(args.interval)


if __name__ == "__main__":
    main()
