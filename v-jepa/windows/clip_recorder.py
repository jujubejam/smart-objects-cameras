"""
clip_recorder.py - Record training clips with countdown
========================================================
Press number keys to record clips for each activity class.
A 3-second countdown gives you time to get into position.

Usage:
    python clip_recorder.py --classes at_computer,playing_keyboard,tending_plants

Controls:
    1, 2, 3... = Start recording for that class (with countdown)
    s          = Show current clip counts
    q          = Quit

Each clip is 3 seconds at 15fps = 45 frames.
Saved to ~/home-clips/<class_name>/clip_XXX.mp4
"""

import argparse
import cv2
import time
from datetime import datetime
from pathlib import Path

import depthai as dai

CAMERA_W, CAMERA_H = 640, 480
FPS = 30  # Increased for smoother preview
CLIP_SECONDS = 3
COUNTDOWN_SECONDS = 3


def create_pipeline(device):
    """Create camera pipeline matching oak_client.py"""
    pipeline = dai.Pipeline(device)

    cam = pipeline.create(dai.node.Camera)
    cam.build(dai.CameraBoardSocket.CAM_A)
    cam_out = cam.requestOutput((CAMERA_W, CAMERA_H), dai.ImgFrame.Type.BGR888p)

    q_preview = cam_out.createOutputQueue(maxSize=FPS * 5, blocking=False)

    return pipeline, q_preview


def get_frame(queue):
    """Get a single frame from the queue."""
    deadline = time.time() + 0.1  # Faster timeout
    while time.time() < deadline:
        pkt = queue.tryGet()
        if pkt is not None:
            return pkt.getCvFrame()
        time.sleep(0.001)  # Faster polling
    return None


def record_clip(queue, output_path: Path, num_frames: int):
    """Record a clip to the output path."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (CAMERA_W, CAMERA_H))

    collected = 0
    deadline = time.time() + (num_frames / FPS) * 3  # generous timeout

    while collected < num_frames and time.time() < deadline:
        pkt = queue.tryGet()
        if pkt is not None:
            frame = pkt.getCvFrame()
            writer.write(frame)
            collected += 1

            # Show recording indicator
            display = frame.copy()
            cv2.circle(display, (30, 30), 15, (0, 0, 255), -1)  # Red dot
            cv2.putText(display, f"REC {collected}/{num_frames}", (55, 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Clip Recorder", display)
            cv2.waitKey(1)
        else:
            time.sleep(0.001)  # Faster polling

    writer.release()
    return collected


def countdown(queue, seconds: int, class_name: str):
    """Show countdown on screen before recording."""
    start = time.time()
    while time.time() - start < seconds:
        remaining = int(seconds - (time.time() - start)) + 1
        frame = get_frame(queue)
        if frame is not None:
            # Big countdown number
            cv2.putText(frame, str(remaining), (CAMERA_W // 2 - 50, CAMERA_H // 2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
            cv2.putText(frame, f"GET READY: {class_name}", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Clip Recorder", frame)
        cv2.waitKey(50)


def main():
    parser = argparse.ArgumentParser(description="Record training clips with countdown")
    parser.add_argument("--classes", default="at_computer,playing_keyboard,tending_plants",
                       help="Comma-separated class names")
    parser.add_argument("--output", type=Path, default=Path.home() / "home-clips",
                       help="Output directory for clips")
    parser.add_argument("--ip", default=None,
                       help="Ethernet OAK-D IP (omit for USB)")
    parser.add_argument("--countdown", type=int, default=COUNTDOWN_SECONDS,
                       help="Countdown seconds before recording")
    parser.add_argument("--duration", type=float, default=CLIP_SECONDS,
                       help="Clip duration in seconds")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]
    num_frames = int(args.duration * FPS)

    # Create output directories
    for cls in classes:
        (args.output / cls).mkdir(parents=True, exist_ok=True)

    # Count existing clips
    clip_counts = {}
    for cls in classes:
        existing = list((args.output / cls).glob("*.mp4"))
        clip_counts[cls] = len(existing)

    print("\n" + "=" * 50)
    print("CLIP RECORDER")
    print("=" * 50)
    print(f"\nOutput: {args.output}")
    print(f"Countdown: {args.countdown}s  |  Clip duration: {args.duration}s")
    print("\nClasses:")
    for i, cls in enumerate(classes, 1):
        print(f"  [{i}] {cls} ({clip_counts[cls]} clips)")
    print("\nControls:")
    print("  1-9 = Record clip for that class")
    print("  s   = Show clip counts")
    print("  q   = Quit")
    print("=" * 50 + "\n")

    # Open device
    if args.ip:
        print(f"Connecting to Ethernet OAK-D at {args.ip}...")
        device_info = dai.DeviceInfo(args.ip)
        device = dai.Device(device_info)
    else:
        print("Connecting to USB OAK-D...")
        device = dai.Device()

    with device:
        print(f"Connected: {device.getDeviceId()}")

        with dai.Pipeline(device) as pipeline:
            cam = pipeline.create(dai.node.Camera)
            cam.build(dai.CameraBoardSocket.CAM_A)
            cam_out = cam.requestOutput((CAMERA_W, CAMERA_H), dai.ImgFrame.Type.BGR888p)
            q_preview = cam_out.createOutputQueue(maxSize=FPS * 5, blocking=False)

            pipeline.start()
            print("Camera ready. Position yourself and press a number key.\n")

            cv2.namedWindow("Clip Recorder", cv2.WINDOW_NORMAL)

            while True:
                # Show live preview
                frame = get_frame(q_preview)
                if frame is not None:
                    # Draw class options
                    y = 30
                    for i, cls in enumerate(classes, 1):
                        text = f"[{i}] {cls}: {clip_counts[cls]}"
                        cv2.putText(frame, text, (10, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y += 25

                    cv2.putText(frame, "Press number to record, Q to quit",
                               (10, CAMERA_H - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    cv2.imshow("Clip Recorder", frame)

                key = cv2.waitKey(30) & 0xFF

                if key == ord('q'):
                    print("\nDone! Final counts:")
                    for cls in classes:
                        print(f"  {cls}: {clip_counts[cls]} clips")
                    break

                elif key == ord('s'):
                    print("\nCurrent counts:")
                    for cls in classes:
                        print(f"  {cls}: {clip_counts[cls]} clips")

                elif ord('1') <= key <= ord('9'):
                    class_idx = key - ord('1')
                    if class_idx < len(classes):
                        cls = classes[class_idx]
                        print(f"\n>>> Recording '{cls}' in {args.countdown} seconds...")

                        # Countdown
                        countdown(q_preview, args.countdown, cls)

                        # Drain queue - discard old frames from countdown period
                        while q_preview.tryGet() is not None:
                            pass

                        # Record
                        clip_num = clip_counts[cls] + 1
                        output_path = args.output / cls / f"clip_{clip_num:03d}.mp4"

                        print(f"    Recording {num_frames} frames...")
                        collected = record_clip(q_preview, output_path, num_frames)

                        if collected >= num_frames * 0.8:  # Allow 80% success
                            clip_counts[cls] += 1
                            print(f"    Saved: {output_path.name} ({collected} frames)")
                        else:
                            print(f"    Failed: only got {collected}/{num_frames} frames")
                            output_path.unlink(missing_ok=True)

            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
