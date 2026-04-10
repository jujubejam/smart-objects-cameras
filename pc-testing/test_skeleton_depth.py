#!/usr/bin/env python3
"""GPU-accelerated Skeleton + Depth tracking with YOLOv8-Pose.

Uses OAK camera for RGB + depth, YOLOv8-Pose on GPU for skeleton.
Much faster than MediaPipe CPU version.

Usage:
    python test_skeleton_depth.py              # USB camera
    python test_skeleton_depth.py --ip X.X.X.X # PoE camera
"""

import argparse
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import depthai as dai
import numpy as np
import torch
from ultralytics import YOLO

# Check GPU
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def create_pipeline(device):
    """Create OAK pipeline with RGB + stereo depth."""
    pipeline = dai.Pipeline(device)

    # RGB camera (new API)
    cam_rgb = pipeline.create(dai.node.Camera)
    cam_rgb.build(dai.CameraBoardSocket.CAM_A)
    rgb_out = cam_rgb.requestOutput((640, 480), dai.ImgFrame.Type.BGR888p)

    # Stereo depth (use deprecated MonoCamera - more reliable for stereo)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(640, 480)

    left.out.link(stereo.left)
    right.out.link(stereo.right)

    # Output queues
    q_rgb = rgb_out.createOutputQueue(maxSize=4, blocking=False)
    q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

    return pipeline, q_rgb, q_depth


# Skeleton connections for drawing
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]


def main():
    parser = argparse.ArgumentParser(description="GPU Skeleton + Depth")
    parser.add_argument("--ip", help="PoE camera IP address")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="YOLO pose model")
    args = parser.parse_args()

    # Load YOLO pose model on GPU
    print(f"Loading {args.model}...", flush=True)
    model = YOLO(args.model)
    if torch.cuda.is_available():
        model.to('cuda')
    print("Model loaded on GPU" if torch.cuda.is_available() else "Model on CPU", flush=True)

    # Connect to camera
    if args.ip:
        print(f"Connecting to PoE camera at {args.ip}...", flush=True)
        device_info = dai.DeviceInfo(args.ip)
        device = dai.Device(device_info)
    else:
        print("Searching for USB camera...", flush=True)
        device = dai.Device()

    with device:
        print(f"Connected: {device.getDeviceId()}", flush=True)

        pipeline, q_rgb, q_depth = create_pipeline(device)
        pipeline.start()
        print("Pipeline started! Running inference...\n", flush=True)

        fps_time = time.time()
        frame_count = 0
        fps = 0

        while True:
            rgb_msg = q_rgb.tryGet()
            depth_msg = q_depth.tryGet()

            if rgb_msg is not None:
                frame = rgb_msg.getCvFrame()
                frame_count += 1
                h, w = frame.shape[:2]

                # Get depth
                depth_frame = depth_msg.getFrame() if depth_msg else None

                # Run YOLO pose on GPU
                results = model(frame, verbose=False)

                # Process detections
                for result in results:
                    if result.keypoints is not None and len(result.keypoints) > 0:
                        keypoints = result.keypoints.xy.cpu().numpy()
                        confs = result.keypoints.conf
                        confidences = confs.cpu().numpy() if confs is not None else None

                        for person_idx, person_kpts in enumerate(keypoints):
                            # Draw skeleton
                            for i, j in SKELETON:
                                if i < len(person_kpts) and j < len(person_kpts):
                                    x1, y1 = person_kpts[i]
                                    x2, y2 = person_kpts[j]
                                    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                            # Draw keypoints and track wrists
                            left_wrist = right_wrist = None

                            for kpt_idx, (x, y) in enumerate(person_kpts):
                                if x > 0 and y > 0:
                                    conf = confidences[person_idx][kpt_idx] if confidences is not None else 1.0
                                    if conf > 0.5:
                                        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

                                        # Get wrist depths
                                        if depth_frame is not None:
                                            px = max(0, min(int(x), w - 1))
                                            py = max(0, min(int(y), h - 1))
                                            depth_mm = depth_frame[py, px]
                                            depth_m = depth_mm / 1000.0

                                            if kpt_idx == 9 and depth_m > 0:  # left_wrist
                                                left_wrist = {'x': x, 'y': y, 'd': depth_m}
                                            elif kpt_idx == 10 and depth_m > 0:  # right_wrist
                                                right_wrist = {'x': x, 'y': y, 'd': depth_m}

                            # Display info
                            y_off = 60
                            if left_wrist:
                                cv2.putText(frame, f"L wrist: {left_wrist['d']:.2f}m",
                                            (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                y_off += 25
                            if right_wrist:
                                cv2.putText(frame, f"R wrist: {right_wrist['d']:.2f}m",
                                            (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                y_off += 25

                            if left_wrist and right_wrist:
                                avg_d = (left_wrist['d'] + right_wrist['d']) / 2
                                dx = (right_wrist['x'] - left_wrist['x']) / w * avg_d
                                dy = (right_wrist['y'] - left_wrist['y']) / h * avg_d
                                dz = right_wrist['d'] - left_wrist['d']
                                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                                cv2.putText(frame, f"Apart: {dist:.2f}m",
                                            (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # FPS calculation
                if frame_count % 10 == 0:
                    fps = 10 / (time.time() - fps_time)
                    fps_time = time.time()
                    print(f"FPS: {fps:.1f}", flush=True)

                cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow("YOLOv11 Pose + Depth (GPU) - Q to quit", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()
        print(f"\nTotal frames: {frame_count}")


if __name__ == "__main__":
    main()
