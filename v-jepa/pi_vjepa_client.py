"""
pi_vjepa_client.py
Runs on each Raspberry Pi (orbit / gravity / horizon).
Captures short video clips from the OAK-D camera and sends them to
the V-JEPA 2 inference server running on your home PC.
Writes vjepa_status.json for the Discord bot to read.

Usage:
    source /opt/oak-shared/venv/bin/activate
    python3 pi_vjepa_client.py --server http://<your-pc-ip>:8765
    python3 pi_vjepa_client.py --server http://<your-pc-ip>:8765 --discord --anomaly-only

Arguments:
    --server URL         V-JEPA inference server (required)
    --interval SECS      How often to send a clip (default: 10)
    --clip-secs SECS     Length of each captured clip (default: 2)
    --anomaly-only       Only alert Discord when anomaly detected (quieter)
    --discord            Post status updates to Discord webhook
    --threshold FLOAT    Anomaly threshold to trigger Discord alert (default: 0.20)
    --display            Show OpenCV preview window (requires X11)
"""

import argparse
import json
import os
import socket
import sys
import time
import tempfile
import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import requests
import depthai as dai

# ── Config ────────────────────────────────────────────────────────────────────
STATUS_FILE = Path.home() / "oak-projects" / "vjepa_status.json"
SCREENSHOT_FILE = Path.home() / "oak-projects" / "latest_vjepa_frame.jpg"
HISTORY_FILE = Path.home() / "oak-projects" / "vjepa_history.jsonl"
ENV_FILE = Path.home() / "oak-projects" / ".env"

CAMERA_W, CAMERA_H = 640, 480
CLIP_FPS = 15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("vjepa-client")


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_env():
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
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


def write_status(data: dict):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(data, indent=2))


def append_history(data: dict):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")


def capture_clip_to_tempfile(queue, num_frames: int, fps: int = CLIP_FPS) -> str:
    """
    Capture num_frames from the OAK-D preview queue and write to a temp .mp4 file.
    Returns the tempfile path (caller must delete it).
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, fps, (CAMERA_W, CAMERA_H))

    collected = 0
    t_start = time.time()
    timeout = num_frames / fps * 3  # 3x buffer

    while collected < num_frames and (time.time() - t_start) < timeout:
        pkt = queue.tryGet()
        if pkt is None:
            time.sleep(0.005)
            continue
        frame = pkt.getCvFrame()
        if frame.shape[1] != CAMERA_W or frame.shape[0] != CAMERA_H:
            frame = cv2.resize(frame, (CAMERA_W, CAMERA_H))
        writer.write(frame)
        collected += 1

    writer.release()

    if collected < num_frames:
        log.warning(f"  Only captured {collected}/{num_frames} frames")

    return tmp.name


def send_clip(server_url: str, clip_path: str, camera_id: str, endpoint: str = "anomaly") -> dict:
    """POST a video clip to the V-JEPA server. Returns the JSON response."""
    url = f"{server_url.rstrip('/')}/{endpoint}"
    try:
        with open(clip_path, "rb") as f:
            response = requests.post(
                url,
                files={"video": ("clip.mp4", f, "video/mp4")},
                data={"camera_id": camera_id},
                timeout=30,
            )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot reach V-JEPA server at {server_url}. Is it running?")
    except Exception as e:
        raise RuntimeError(f"Server request failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="V-JEPA Pi client")
    parser.add_argument("--server", required=True, help="V-JEPA server URL, e.g. http://<your-pc-ip>:8765")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between clips")
    parser.add_argument("--clip-secs", type=float, default=2.0, help="Clip duration in seconds")
    parser.add_argument("--anomaly-only", action="store_true", help="Only Discord-alert on anomaly")
    parser.add_argument("--discord", action="store_true", help="Send Discord notifications")
    parser.add_argument("--threshold", type=float, default=0.20, help="Anomaly threshold (0-1)")
    parser.add_argument("--display", action="store_true", help="Show preview window")
    args = parser.parse_args()

    env = load_env()
    webhook_url = env.get("DISCORD_WEBHOOK_URL", "")
    camera_id = socket.gethostname()  # orbit / gravity / horizon
    username = os.environ.get("USER", "unknown")

    num_clip_frames = int(args.clip_secs * CLIP_FPS)

    log.info(f"V-JEPA client starting on {camera_id} (user: {username})")
    log.info(f"  Server: {args.server}")
    log.info(f"  Clip: {args.clip_secs}s @ {CLIP_FPS}fps ({num_clip_frames} frames)")
    log.info(f"  Interval: {args.interval}s")
    log.info(f"  Anomaly threshold: {args.threshold}")

    # ── Health check ──────────────────────────────────────────────────────────
    try:
        r = requests.get(f"{args.server}/health", timeout=5)
        r.raise_for_status()
        info = r.json()
        log.info(f"  Server OK: {info.get('gpu', 'unknown GPU')}")
    except Exception as e:
        log.error(f"  Server health check failed: {e}")
        log.error("  Make sure server.py is running on your PC and the IP is correct.")
        sys.exit(1)

    if args.discord and webhook_url:
        discord_notify(webhook_url, f"🧠 V-JEPA client started on **{camera_id}** ({username})")

    # ── OAK-D pipeline ────────────────────────────────────────────────────────
    with dai.Device() as device:
        with dai.Pipeline(device) as pipeline:
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setPreviewSize(CAMERA_W, CAMERA_H)
            cam.setInterleaved(False)
            cam.setFps(CLIP_FPS)

            q_preview = cam.preview.createOutputQueue(maxSize=CLIP_FPS * 5, blocking=False)

            pipeline.start()
            log.info("OAK-D camera pipeline started")

            consecutive_anomalies = 0
            last_level = "normal"
            loop_count = 0

            try:
                while True:
                    loop_count += 1
                    t_loop = time.time()

                    # ── Capture a clip ────────────────────────────────────────
                    clip_path = capture_clip_to_tempfile(q_preview, num_clip_frames)

                    # Save a screenshot for Discord !screenshot command
                    cap = cv2.VideoCapture(clip_path)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        overlay = frame.copy()
                        ts = datetime.now().strftime("%H:%M:%S")
                        cv2.putText(overlay, f"VJEPA | {camera_id} | {ts}",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imwrite(str(SCREENSHOT_FILE), overlay)

                        if args.display:
                            cv2.imshow(f"V-JEPA {camera_id}", overlay)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break

                    # ── Send to server ────────────────────────────────────────
                    try:
                        result = send_clip(args.server, clip_path, camera_id, endpoint="anomaly")
                    except RuntimeError as e:
                        log.error(str(e))
                        time.sleep(args.interval)
                        continue
                    finally:
                        os.unlink(clip_path)

                    score = result.get("anomaly_score", 0.0)
                    level = result.get("level", "normal")
                    latency = result.get("latency_ms", 0)

                    log.info(f"  [{loop_count:04d}] score={score:.4f} ({level})  "
                             f"server_latency={latency:.0f}ms")

                    # ── Write status ──────────────────────────────────────────
                    now = datetime.now().isoformat()
                    status = {
                        "camera_id": camera_id,
                        "username": username,
                        "timestamp": now,
                        "anomaly_score": score,
                        "level": level,                # "normal" | "unusual" | "anomaly"
                        "threshold": args.threshold,
                        "consecutive_anomalies": consecutive_anomalies,
                        "server_latency_ms": latency,
                        "loop_count": loop_count,
                    }
                    write_status(status)
                    append_history({**status, "timestamp": now})

                    # ── Discord alerting ──────────────────────────────────────
                    if score >= args.threshold:
                        consecutive_anomalies += 1
                    else:
                        consecutive_anomalies = 0

                    level_changed = level != last_level
                    last_level = level

                    if args.discord and webhook_url:
                        if level == "anomaly" and consecutive_anomalies == 2:
                            # Alert on 2nd consecutive anomaly (debounce)
                            discord_notify(
                                webhook_url,
                                f"🚨 **{camera_id}** — unexpected activity detected!\n"
                                f"Anomaly score: `{score:.3f}` (threshold: {args.threshold})\n"
                                f"*V-JEPA world model says this is unusual*"
                            )
                        elif level_changed and level == "normal" and not args.anomaly_only:
                            discord_notify(
                                webhook_url,
                                f"✅ **{camera_id}** — back to normal  `score={score:.3f}`"
                            )

                    # ── Sleep for remainder of interval ───────────────────────
                    elapsed = time.time() - t_loop
                    sleep_time = max(0, args.interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except KeyboardInterrupt:
                log.info("Stopped by user")
                if args.discord and webhook_url:
                    discord_notify(webhook_url, f"⏹ V-JEPA client stopped on **{camera_id}**")

    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
