"""
auto_recorder.py - Automated clip recording for later labeling
===============================================================
Records clips on a timer, saves with timestamps. Label them later.

Usage:
    python auto_recorder.py --output D:\vjepa-clips --interval 10

This creates:
    D:\vjepa-clips\
        unlabeled\
            2026-03-19_14-23-01.mp4
            2026-03-19_14-23-11.mp4
            ...
        manifest.jsonl   (metadata for each clip)

Then run clip_labeler.py to sort clips into class folders.
"""

import argparse
import json
import time
import logging
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai

CAMERA_W, CAMERA_H = 640, 480
FPS = 15
CLIP_SECONDS = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("auto-recorder")


def capture_clip(queue, output_path: Path, num_frames: int) -> dict:
    """Capture a clip and return metadata."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (CAMERA_W, CAMERA_H))

    collected = 0
    deadline = time.time() + (num_frames / FPS) * 3
    first_frame = None

    while collected < num_frames and time.time() < deadline:
        pkt = queue.tryGet()
        if pkt is not None:
            frame = pkt.getCvFrame()
            if frame.shape[1] != CAMERA_W or frame.shape[0] != CAMERA_H:
                frame = cv2.resize(frame, (CAMERA_W, CAMERA_H))
            writer.write(frame)
            if first_frame is None:
                first_frame = frame.copy()
            collected += 1
        else:
            time.sleep(0.005)

    writer.release()

    # Save thumbnail
    if first_frame is not None:
        thumb_path = output_path.with_suffix(".jpg")
        cv2.imwrite(str(thumb_path), first_frame)

    return {
        "filename": output_path.name,
        "timestamp": datetime.now().isoformat(),
        "frames": collected,
        "duration_sec": collected / FPS,
    }


def main():
    parser = argparse.ArgumentParser(description="Automated clip recorder")
    parser.add_argument("--output", type=Path, default=Path("D:/vjepa-clips"),
                       help="Output directory (default: D:/vjepa-clips)")
    parser.add_argument("--interval", type=float, default=10.0,
                       help="Seconds between clips (default: 10)")
    parser.add_argument("--duration", type=float, default=CLIP_SECONDS,
                       help="Clip duration in seconds (default: 3)")
    parser.add_argument("--ip", default=None,
                       help="Ethernet OAK-D IP (omit for USB)")
    parser.add_argument("--display", action="store_true",
                       help="Show live preview")
    parser.add_argument("--max-clips", type=int, default=0,
                       help="Stop after N clips (0 = unlimited)")
    args = parser.parse_args()

    # Setup output directory
    unlabeled_dir = args.output / "unlabeled"
    unlabeled_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.jsonl"

    num_frames = int(args.duration * FPS)

    print("\n" + "=" * 50)
    print("AUTO RECORDER")
    print("=" * 50)
    print(f"Output:    {unlabeled_dir}")
    print(f"Interval:  {args.interval}s")
    print(f"Duration:  {args.duration}s ({num_frames} frames)")
    print(f"Manifest:  {manifest_path}")
    print("=" * 50)
    print("\nPress Ctrl+C to stop recording\n")

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
            cam = pipeline.create(dai.node.Camera)
            cam.build(dai.CameraBoardSocket.CAM_A)
            cam_out = cam.requestOutput((CAMERA_W, CAMERA_H), dai.ImgFrame.Type.BGR888p)
            q_preview = cam_out.createOutputQueue(maxSize=FPS * 5, blocking=False)

            pipeline.start()
            log.info("Recording started\n")

            clip_count = 0

            try:
                while True:
                    t_start = time.time()

                    # Generate filename from timestamp
                    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    clip_path = unlabeled_dir / f"{ts}.mp4"

                    # Record
                    log.info(f"Recording clip {clip_count + 1}...")
                    meta = capture_clip(q_preview, clip_path, num_frames)
                    clip_count += 1

                    # Append to manifest
                    with open(manifest_path, "a") as f:
                        f.write(json.dumps(meta) + "\n")

                    log.info(f"  Saved: {clip_path.name} ({meta['frames']} frames)")

                    # Show preview if requested
                    if args.display:
                        thumb_path = clip_path.with_suffix(".jpg")
                        if thumb_path.exists():
                            img = cv2.imread(str(thumb_path))
                            cv2.putText(img, f"Clip {clip_count}: {ts}",
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                       0.6, (0, 255, 0), 2)
                            window_name = f"Auto Recorder - {args.ip}" if args.ip else "Auto Recorder - USB"
                            cv2.imshow(window_name, img)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                    # Check max clips
                    if args.max_clips > 0 and clip_count >= args.max_clips:
                        log.info(f"Reached max clips ({args.max_clips})")
                        break

                    # Wait for next interval
                    elapsed = time.time() - t_start
                    sleep_for = max(0, args.interval - elapsed)
                    if sleep_for > 0:
                        time.sleep(sleep_for)

            except KeyboardInterrupt:
                pass
            finally:
                if args.display:
                    cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    print(f"DONE - Recorded {clip_count} clips")
    print(f"Clips saved to: {unlabeled_dir}")
    print(f"\nNext step: python clip_labeler.py --input {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
