"""
probe_inference.py (Windows version)
=====================================
Live classification using a trained probe on top of V-JEPA embeddings.

Workflow:
    1. Camera captures 3-second clip
    2. Clip sent to server -> V-JEPA embedding (1024-d)
    3. Local probe classifies embedding -> activity label
    4. Repeat every N seconds

Usage:
    python probe_inference.py --probe C:/Users/you/oak-projects/home_probe.pt

The probe runs locally (CPU, instant), only the embedding extraction
happens on the GPU server.
"""

import argparse
import json
import os
import tempfile
import time
import logging
from datetime import datetime
from pathlib import Path

import cv2
import requests
import torch
import torch.nn as nn
import depthai as dai

CAMERA_W, CAMERA_H = 640, 480
FPS = 30  # Match clip_recorder.py

OAK_PROJECTS = Path.home() / "oak-projects"

# ── Classroom API integration ────────────────────────────────────────────────
CLASSROOM_API_URL = os.getenv("CLASSROOM_API_URL", "")
CLASSROOM_API_KEY = os.getenv("CLASSROOM_API_KEY", "")


def push_to_classroom(camera_id: str, predicted_class: str, confidence: float,
                      class_probs: dict):
    """Push classification to the classroom API (fire-and-forget)."""
    if not CLASSROOM_API_URL:
        return
    try:
        requests.post(
            f"{CLASSROOM_API_URL.rstrip('/')}/push/state",
            json={
                "camera_id": camera_id,
                "predicted_class": predicted_class,
                "prediction_confidence": confidence,
                "class_probs": class_probs,
            },
            headers={"X-API-Key": CLASSROOM_API_KEY},
            timeout=2,
        )
    except Exception:
        pass  # Never block the detection loop


def get_status_file(camera_name: str) -> Path:
    return OAK_PROJECTS / f"probe_status_{camera_name}.json"

def get_history_file(camera_name: str) -> Path:
    return OAK_PROJECTS / f"probe_history_{camera_name}.jsonl"

def get_screenshot_file(camera_name: str) -> Path:
    return OAK_PROJECTS / f"latest_probe_frame_{camera_name}.jpg"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("probe-inference")


# ── Probe model ───────────────────────────────────────────────────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_env():
    env = {}
    env_file = OAK_PROJECTS / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def discord_notify(webhook_url: str, message: str):
    try:
        requests.post(webhook_url, json={"content": message}, timeout=5)
    except Exception as e:
        log.warning(f"Discord notify failed: {e}")


def capture_clip(queue, num_frames: int) -> tuple[str, int]:
    """Capture frames to a temp mp4 file. Returns (path, frame_count)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, FPS, (CAMERA_W, CAMERA_H))

    collected = 0
    deadline = time.time() + (num_frames / FPS) * 3

    while collected < num_frames and time.time() < deadline:
        pkt = queue.tryGet()
        if pkt is not None:
            frame = pkt.getCvFrame()
            if frame.shape[1] != CAMERA_W or frame.shape[0] != CAMERA_H:
                frame = cv2.resize(frame, (CAMERA_W, CAMERA_H))
            writer.write(frame)
            collected += 1
        else:
            time.sleep(0.005)

    writer.release()
    log.debug(f"Captured {collected}/{num_frames} frames")
    return tmp.name, collected


def get_embedding(server_url: str, clip_path: str, camera_id: str):
    """Send clip to server, get back 1024-d embedding."""
    url = f"{server_url.rstrip('/')}/embed"
    with open(clip_path, "rb") as f:
        r = requests.post(
            url,
            files={"video": ("clip.mp4", f, "video/mp4")},
            data={"camera_id": camera_id},
            timeout=30,
        )
    r.raise_for_status()
    return r.json()


def classify(probe, class_names, embedding):
    """Run local probe classification on embedding."""
    x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = probe(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = probs.argmax().item()

    return {
        "predicted_class": class_names[pred_idx],
        "confidence": round(probs[pred_idx].item(), 3),
        "class_probs": {c: round(probs[i].item(), 3) for i, c in enumerate(class_names)},
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Live probe classification (Windows)")
    parser.add_argument("--probe", type=Path,
                       default=Path.home() / "oak-projects" / "home_probe.pt",
                       help="Path to trained probe")
    parser.add_argument("--server", default="http://localhost:8765",
                       help="V-JEPA inference server URL")
    parser.add_argument("--ip", default=None,
                       help="Ethernet OAK-D IP (omit for USB)")
    parser.add_argument("--name", default=None,
                       help="Camera name (default: usb-oak or eth-oak)")
    parser.add_argument("--interval", type=float, default=5.0,
                       help="Seconds between classifications")
    parser.add_argument("--clip-secs", type=float, default=3.0,
                       help="Clip duration in seconds")
    parser.add_argument("--discord", action="store_true",
                       help="Notify Discord on class changes")
    parser.add_argument("--display", action="store_true",
                       help="Show live preview window")
    args = parser.parse_args()

    # Validate probe exists
    if not args.probe.exists():
        log.error(f"Probe not found: {args.probe}")
        log.error("Run probe_trainer.py first to train a probe.")
        return

    # Load probe
    probe, class_names = load_probe(args.probe)
    log.info(f"Probe loaded: {class_names}")

    # Camera name
    if args.name is None:
        args.name = "eth-oak" if args.ip else "usb-oak"

    num_frames = int(args.clip_secs * FPS)
    env = load_env()
    webhook_url = env.get("DISCORD_WEBHOOK_URL", "")

    log.info(f"Camera: {args.name}")
    log.info(f"Server: {args.server}")
    log.info(f"Interval: {args.interval}s  Clip: {args.clip_secs}s ({num_frames} frames)")

    # Server health check
    try:
        r = requests.get(f"{args.server}/health", timeout=5)
        r.raise_for_status()
        info = r.json()
        log.info(f"Server OK: {info.get('gpu', '?')}")
    except Exception as e:
        log.error(f"Server health check failed: {e}")
        return

    # Open device
    if args.ip:
        log.info(f"Connecting to Ethernet OAK-D at {args.ip}...")
        device_info = dai.DeviceInfo(args.ip)
        device = dai.Device(device_info)
    else:
        log.info("Connecting to USB OAK-D...")
        device = dai.Device()

    with device:
        log.info(f"Connected: {device.getDeviceId()}")

        with dai.Pipeline(device) as pipeline:
            # Use Camera node (depthai 3.x API)
            cam = pipeline.create(dai.node.Camera)
            cam.build(dai.CameraBoardSocket.CAM_A)
            cam_out = cam.requestOutput((CAMERA_W, CAMERA_H), dai.ImgFrame.Type.BGR888p)
            q_preview = cam_out.createOutputQueue(maxSize=FPS * 2, blocking=False)

            pipeline.start()
            log.info("Pipeline started, warming up camera...")

            # Warmup: wait for camera to start producing frames
            warmup_start = time.time()
            while time.time() - warmup_start < 2.0:
                pkt = q_preview.tryGet()
                if pkt is not None:
                    log.info("Camera ready\n")
                    break
                time.sleep(0.1)
            else:
                log.warning("Camera warmup timeout - continuing anyway\n")

            if args.discord and webhook_url:
                discord_notify(webhook_url,
                    f"🎯 Probe inference started: **{args.name}**\n"
                    f"Classes: {', '.join(class_names)}")

            last_class = None
            loop = 0

            try:
                while True:
                    loop += 1
                    t_start = time.time()
                    log.info(f"[{loop:04d}] Capturing {num_frames} frames...")

                    # Capture clip
                    clip_path, frame_count = capture_clip(q_preview, num_frames)
                    log.info(f"[{loop:04d}] Captured {frame_count} frames")

                    if frame_count == 0:
                        log.error("No frames captured! Camera issue?")
                        time.sleep(args.interval)
                        continue

                    if frame_count < num_frames // 2:
                        log.warning(f"Only captured {frame_count}/{num_frames} frames")

                    # Save screenshot (use LAST frame of clip for most recent view)
                    cap = cv2.VideoCapture(clip_path)
                    frame = None
                    ret = False
                    while True:
                        r, f = cap.read()
                        if not r:
                            break
                        ret, frame = r, f
                    cap.release()

                    if ret:
                        OAK_PROJECTS.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(get_screenshot_file(args.name)), frame)

                    # Debug: check video file
                    clip_size = os.path.getsize(clip_path)
                    log.info(f"  Clip size: {clip_size//1024}KB")

                    # Get embedding from server
                    try:
                        result = get_embedding(args.server, clip_path, args.name)
                        embedding = result["embedding"]
                        server_latency = result.get("latency_ms", 0)
                        # Debug: show first few embedding values to detect stuck values
                        emb_sample = embedding[:3]
                        log.info(f"  Embedding sample: [{emb_sample[0]:.3f}, {emb_sample[1]:.3f}, {emb_sample[2]:.3f}]")
                    except Exception as e:
                        log.error(f"Server error: {e}")
                        time.sleep(args.interval)
                        continue
                    finally:
                        try:
                            os.unlink(clip_path)
                        except Exception:
                            pass

                    # Classify locally
                    pred = classify(probe, class_names, embedding)
                    pred_class = pred["predicted_class"]
                    confidence = pred["confidence"]

                    # Log
                    probs_str = "  ".join(f"{c}:{p:.0%}" for c, p in pred["class_probs"].items())
                    log.info(f"[{loop:04d}] {pred_class} ({confidence:.0%})  |  {probs_str}")

                    # Write status
                    now = datetime.now().isoformat()
                    status = {
                        "camera_id": args.name,
                        "timestamp": now,
                        "predicted_class": pred_class,
                        "confidence": confidence,
                        "class_probs": pred["class_probs"],
                        "server_latency_ms": server_latency,
                        "loop_count": loop,
                    }
                    get_status_file(args.name).write_text(json.dumps(status, indent=2))

                    # Append history
                    with open(get_history_file(args.name), "a") as f:
                        f.write(json.dumps(status) + "\n")

                    # Push to classroom API
                    push_to_classroom(args.name, pred_class, confidence, pred["class_probs"])

                    # Display
                    if args.display and ret:
                        display = frame.copy()
                        # Draw prediction
                        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                        cv2.putText(display, f"{pred_class}: {confidence:.0%}",
                                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                        # Draw all class probabilities
                        y = 70
                        for cls, prob in pred["class_probs"].items():
                            bar_len = int(prob * 200)
                            cv2.rectangle(display, (10, y - 12), (10 + bar_len, y + 5), (100, 100, 100), -1)
                            cv2.putText(display, f"{cls}: {prob:.0%}", (10, y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            y += 25

                        cv2.imshow(f"Probe: {args.name}", display)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    # Discord on class change
                    if args.discord and webhook_url and pred_class != last_class:
                        emoji = {
                            "at_computer": "💻",
                            "playing_keyboard": "🎹",
                            "tending_plants": "🌱",
                        }.get(pred_class, "🎯")
                        discord_notify(webhook_url,
                            f"{emoji} **{args.name}**: `{pred_class}` ({confidence:.0%})")
                    last_class = pred_class

                    # Live preview during interval (instead of sleeping)
                    elapsed = time.time() - t_start
                    wait_until = t_start + args.interval

                    while time.time() < wait_until:
                        if args.display:
                            # Show live preview with current prediction overlay
                            pkt = q_preview.tryGet()
                            if pkt is not None:
                                live_frame = pkt.getCvFrame()
                                # Draw current prediction
                                color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                                cv2.putText(live_frame, f"{pred_class}: {confidence:.0%}",
                                           (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                                cv2.imshow(f"Probe: {args.name}", live_frame)
                            # waitKey also handles display refresh
                            if cv2.waitKey(30) & 0xFF == ord('q'):
                                raise KeyboardInterrupt
                        else:
                            time.sleep(0.1)

            except KeyboardInterrupt:
                log.info("\nStopped (Ctrl+C)")
                if args.discord and webhook_url:
                    discord_notify(webhook_url, f"⏹ Probe inference stopped: **{args.name}**")
            finally:
                if args.display:
                    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
