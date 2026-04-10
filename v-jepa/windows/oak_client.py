"""
oak_client.py  —  Windows OAK-D client for V-JEPA 2
=====================================================
Mirrors pi_vjepa_client.py exactly, but:
  - Uses Windows paths  (C:\\Users\\you\\oak-projects\\)
  - Accepts --ip for Ethernet OAK-D  (USB is auto-discovered)
  - Writes per-camera status files so both cameras are tracked independently

Run once per camera in separate Anaconda Prompt terminals:

  # USB OAK-D (auto-discovered, no --device needed)
  python oak_client.py --name usb-oak --display

  # Ethernet OAK-D (specify IP printed in Device Manager or PoE switch)
  python oak_client.py --ip <your-camera-ip> --name eth-oak --display

Both instances talk to the same server.py running on localhost.

Finding your Ethernet OAK-D IP:
  python -c "import depthai as dai; [print(d.deviceId, d.name) for d in dai.Device.getAllAvailableDevices()]"
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
import depthai as dai

# ── Paths — mirrors ~/oak-projects/ on Pi ─────────────────────────────────────
OAK_PROJECTS = Path.home() / "oak-projects"
ENV_FILE      = OAK_PROJECTS / ".env"

CAMERA_W, CAMERA_H = 640, 480
CLIP_FPS = 15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("oak-client")


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_env() -> dict:
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def status_file(camera_name: str) -> Path:
    """Each camera gets its own status file: vjepa_status_usb-oak.json etc."""
    return OAK_PROJECTS / f"vjepa_status_{camera_name}.json"


def history_file(camera_name: str) -> Path:
    return OAK_PROJECTS / f"vjepa_history_{camera_name}.jsonl"


def screenshot_file(camera_name: str) -> Path:
    return OAK_PROJECTS / f"latest_vjepa_frame_{camera_name}.jpg"


def write_status(camera_name: str, data: dict):
    OAK_PROJECTS.mkdir(parents=True, exist_ok=True)
    status_file(camera_name).write_text(json.dumps(data, indent=2))


def append_history(camera_name: str, data: dict):
    OAK_PROJECTS.mkdir(parents=True, exist_ok=True)
    with open(history_file(camera_name), "a") as f:
        f.write(json.dumps(data) + "\n")


def discord_notify(webhook_url: str, message: str):
    try:
        requests.post(webhook_url, json={"content": message}, timeout=5)
    except Exception as e:
        log.warning(f"Discord notify failed: {e}")


def open_device(ip: str | None) -> dai.Device:
    """
    Open either:
      - USB OAK-D   (ip is None → auto-discover, same as test_camera.py)
      - Ethernet/PoE OAK-D  (ip is an address like "<your-camera-ip>")
    """
    if ip:
        log.info(f"Connecting to PoE/Ethernet OAK-D at {ip}...")
        device_info = dai.DeviceInfo(ip)
        return dai.Device(device_info)
    else:
        log.info("Connecting to USB OAK-D (auto-discover)...")
        return dai.Device()


def capture_clip(queue, num_frames: int) -> str:
    """
    Capture num_frames from the OAK-D preview queue.
    Writes to a temp .mp4 and returns the path.
    Identical to Pi version.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, CLIP_FPS, (CAMERA_W, CAMERA_H))

    collected = 0
    deadline = time.time() + (num_frames / CLIP_FPS) * 3

    while collected < num_frames and time.time() < deadline:
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


def send_clip(server_url: str, clip_path: str, camera_id: str) -> dict:
    url = f"{server_url.rstrip('/')}/anomaly"
    try:
        with open(clip_path, "rb") as f:
            r = requests.post(
                url,
                files={"video": ("clip.mp4", f, "video/mp4")},
                data={"camera_id": camera_id},
                timeout=30,
            )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach server at {server_url}. "
            "Is server.py running in another terminal?"
        )
    except Exception as e:
        raise RuntimeError(f"Server error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="OAK-D → V-JEPA client (Windows)")
    parser.add_argument("--ip",           default=None,
                        help="Ethernet/PoE OAK-D IP address (e.g. <your-camera-ip>). "
                             "Omit for USB auto-discovery.")
    parser.add_argument("--name",         default=None,
                        help="Human-readable camera name for status files "
                             "(default: 'usb-oak' or 'eth-oak')")
    parser.add_argument("--server",       default="http://localhost:8765",
                        help="V-JEPA inference server URL")
    parser.add_argument("--interval",     type=float, default=10.0,
                        help="Seconds between clips (default: 10)")
    parser.add_argument("--clip-secs",    type=float, default=2.0,
                        help="Clip duration in seconds (default: 2)")
    parser.add_argument("--threshold",    type=float, default=0.20,
                        help="Anomaly score threshold for Discord alerts (default: 0.20)")
    parser.add_argument("--anomaly-only", action="store_true",
                        help="Only Discord-alert on anomaly, not on recovery")
    parser.add_argument("--discord",      action="store_true",
                        help="Enable Discord notifications (reads oak-projects/.env)")
    parser.add_argument("--display",      action="store_true",
                        help="Show live OpenCV preview window")
    args = parser.parse_args()

    # Derive a sensible name if not given
    if args.name is None:
        args.name = "eth-oak" if args.ip else "usb-oak"

    env = load_env()
    webhook_url = env.get("DISCORD_WEBHOOK_URL", "")
    username = os.environ.get("USERNAME", os.environ.get("USER", "unknown"))
    num_clip_frames = int(args.clip_secs * CLIP_FPS)

    log.info(f"OAK-D client  name={args.name}  user={username}")
    log.info(f"  Connection: {'Ethernet ' + args.ip if args.ip else 'USB (auto)'}")
    log.info(f"  Server:     {args.server}")
    log.info(f"  Clip:       {args.clip_secs}s @ {CLIP_FPS}fps ({num_clip_frames} frames)")
    log.info(f"  Interval:   {args.interval}s")
    log.info(f"  Threshold:  {args.threshold}")
    log.info(f"  Status:     {status_file(args.name)}")

    # ── Server health check ───────────────────────────────────────────────────
    try:
        r = requests.get(f"{args.server}/health", timeout=5)
        r.raise_for_status()
        info = r.json()
        log.info(f"  Server OK  gpu={info.get('gpu','?')}  "
                 f"vram={info.get('vram_used_gb','?')}GB used")
    except Exception as e:
        log.error(f"Server health check failed: {e}")
        log.error("Start server.py first, then run this script.")
        return

    if args.discord and webhook_url:
        discord_notify(webhook_url,
            f"🧠 V-JEPA client started  **{args.name}** "
            f"({'Ethernet ' + args.ip if args.ip else 'USB'})  user={username}")

    # ── OAK-D pipeline — uses new Camera API matching pc-testing scripts ─────
    with open_device(args.ip) as device:
        mxid = device.getDeviceId()
        log.info(f"  OAK-D connected  MxId={mxid}")
        log.info(f"  USB speed: {device.getUsbSpeed().name}")

        with dai.Pipeline(device) as pipeline:
            # New Camera API (same as test_camera.py / test_person_detect.py)
            cam = pipeline.create(dai.node.Camera)
            cam.build(dai.CameraBoardSocket.CAM_A)
            cam_out = cam.requestOutput(
                (CAMERA_W, CAMERA_H), dai.ImgFrame.Type.BGR888p
            )

            q_preview = cam_out.createOutputQueue(maxSize=CLIP_FPS * 5, blocking=False)

            pipeline.start()
            log.info("  Pipeline started — streaming")

            consecutive_anomalies = 0
            last_level = "normal"
            loop = 0

            try:
                while True:
                    loop += 1
                    t_loop = time.time()

                    # ── Capture clip ──────────────────────────────────────────
                    clip_path = capture_clip(q_preview, num_clip_frames)

                    # Save a screenshot
                    cap = cv2.VideoCapture(clip_path)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        ts_str = datetime.now().strftime("%H:%M:%S")
                        cv2.putText(frame,
                                    f"VJEPA | {args.name} | {ts_str}",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2)
                        OAK_PROJECTS.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(screenshot_file(args.name)), frame)

                        if args.display:
                            cv2.imshow(f"V-JEPA  {args.name}", frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break

                    # ── Send to server ────────────────────────────────────────
                    try:
                        result = send_clip(args.server, clip_path, args.name)
                    except RuntimeError as e:
                        log.error(str(e))
                        time.sleep(args.interval)
                        continue
                    finally:
                        try:
                            os.unlink(clip_path)
                        except Exception:
                            pass

                    score   = result.get("anomaly_score", 0.0)
                    level   = result.get("level", "normal")
                    latency = result.get("latency_ms", 0.0)

                    log.info(f"  [{loop:04d}] score={score:.4f} ({level:<8})  "
                             f"server={latency:.0f}ms")

                    # ── Write status (same schema as Pi) ──────────────────────
                    now = datetime.now().isoformat()
                    status = {
                        "camera_id":             args.name,
                        "mxid":                  mxid,
                        "connection":            "ethernet" if args.ip else "usb",
                        "username":              username,
                        "timestamp":             now,
                        "anomaly_score":         score,
                        "level":                 level,
                        "threshold":             args.threshold,
                        "consecutive_anomalies": consecutive_anomalies,
                        "server_latency_ms":     latency,
                        "loop_count":            loop,
                    }
                    write_status(args.name, status)
                    append_history(args.name, status)

                    # ── Discord alerting ──────────────────────────────────────
                    if score >= args.threshold:
                        consecutive_anomalies += 1
                    else:
                        consecutive_anomalies = 0

                    level_changed = (level != last_level)
                    last_level = level

                    if args.discord and webhook_url:
                        if level == "anomaly" and consecutive_anomalies == 2:
                            discord_notify(webhook_url,
                                f"🚨 **{args.name}** — unexpected activity!\n"
                                f"Score: `{score:.3f}`  (threshold {args.threshold})")
                        elif level_changed and level == "normal" and not args.anomaly_only:
                            discord_notify(webhook_url,
                                f"✅ **{args.name}** — back to normal  `{score:.3f}`")

                    # ── Sleep ─────────────────────────────────────────────────
                    sleep_for = max(0.0, args.interval - (time.time() - t_loop))
                    if sleep_for:
                        time.sleep(sleep_for)

            except KeyboardInterrupt:
                log.info("Stopped (Ctrl+C)")
                if args.discord and webhook_url:
                    discord_notify(webhook_url,
                        f"⏹ V-JEPA client stopped  **{args.name}**")
            finally:
                if args.display:
                    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
