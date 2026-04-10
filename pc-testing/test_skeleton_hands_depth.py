#!/usr/bin/env python3
"""GPU Skeleton + Detailed Hand Tracking + Depth.

Combines:
- YOLOv11-Pose on GPU for fast skeleton tracking
- MediaPipe Hands for detailed finger/gesture tracking
- OAK stereo depth for 3D positioning

Usage:
    python test_skeleton_hands_depth.py              # USB camera
    python test_skeleton_hands_depth.py --ip X.X.X.X # PoE camera
"""

import argparse
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import depthai as dai
import mediapipe as mp
import numpy as np
import torch
from ultralytics import YOLO

# MediaPipe setup
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Check GPU
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}", end="")
if torch.cuda.is_available():
    print(f" - {torch.cuda.get_device_name(0)}")
else:
    print()


def download_hand_model():
    if not HAND_MODEL.exists():
        print("Downloading hand model...")
        urllib.request.urlretrieve(HAND_URL, HAND_MODEL)


def create_pipeline(device):
    """Create OAK pipeline with RGB + stereo depth."""
    pipeline = dai.Pipeline(device)

    # RGB camera
    cam_rgb = pipeline.create(dai.node.Camera)
    cam_rgb.build(dai.CameraBoardSocket.CAM_A)
    rgb_out = cam_rgb.requestOutput((640, 480), dai.ImgFrame.Type.BGR888p)

    # Stereo depth
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

    q_rgb = rgb_out.createOutputQueue(maxSize=4, blocking=False)
    q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

    return pipeline, q_rgb, q_depth


# Skeleton connections
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

# Hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]


def is_fist(hand_landmarks):
    """Detect if hand is making a fist (fingers curled)."""
    # Check if fingertips are below their knuckles (y increases downward)
    tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
    knuckles = [6, 10, 14, 18]  # Corresponding knuckles

    curled = 0
    for tip, knuckle in zip(tips, knuckles):
        if hand_landmarks[tip].y > hand_landmarks[knuckle].y:
            curled += 1
    return curled >= 3


def is_open_palm(hand_landmarks):
    """Detect open palm (all fingers extended)."""
    tips = [8, 12, 16, 20]
    knuckles = [6, 10, 14, 18]

    extended = 0
    for tip, knuckle in zip(tips, knuckles):
        if hand_landmarks[tip].y < hand_landmarks[knuckle].y:
            extended += 1
    return extended >= 4


def is_pointing(hand_landmarks):
    """Detect pointing gesture (index extended, others curled)."""
    # Index tip above knuckle
    index_extended = hand_landmarks[8].y < hand_landmarks[6].y
    # Other fingers curled
    others_curled = sum(1 for tip, knuckle in [(12, 10), (16, 14), (20, 18)]
                       if hand_landmarks[tip].y > hand_landmarks[knuckle].y)
    return index_extended and others_curled >= 2


def get_gesture(hand_landmarks):
    """Identify hand gesture."""
    if is_fist(hand_landmarks):
        return "FIST"
    elif is_pointing(hand_landmarks):
        return "POINTING"
    elif is_open_palm(hand_landmarks):
        return "OPEN"
    return None


def main():
    parser = argparse.ArgumentParser(description="Skeleton + Hands + Depth")
    parser.add_argument("--ip", help="PoE camera IP address")
    args = parser.parse_args()

    download_hand_model()

    # Load models
    print("Loading YOLOv11-Pose...")
    pose_model = YOLO("yolo11n-pose.pt")
    if torch.cuda.is_available():
        pose_model.to('cuda')

    # MediaPipe hands
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Connect to camera
    if args.ip:
        print(f"Connecting to PoE camera at {args.ip}...")
        device_info = dai.DeviceInfo(args.ip)
        device = dai.Device(device_info)
    else:
        print("Searching for USB camera...")
        device = dai.Device()

    with device:
        print(f"Connected: {device.getDeviceId()}")

        pipeline, q_rgb, q_depth = create_pipeline(device)
        pipeline.start()
        print("Pipeline started!\n")

        with HandLandmarker.create_from_options(hand_options) as hand_landmarker:
            fps_time = time.time()
            frame_count = 0
            fps = 0
            timestamp_ms = 0

            while True:
                rgb_msg = q_rgb.tryGet()
                depth_msg = q_depth.tryGet()

                if rgb_msg is not None:
                    frame = rgb_msg.getCvFrame()
                    frame_count += 1
                    timestamp_ms += 33
                    h, w = frame.shape[:2]

                    depth_frame = depth_msg.getFrame() if depth_msg else None

                    # 1. YOLO Pose (GPU) - skeleton
                    pose_results = pose_model(frame, verbose=False)

                    for result in pose_results:
                        if result.keypoints is not None:
                            keypoints = result.keypoints.xy.cpu().numpy()
                            for person_kpts in keypoints:
                                for i, j in SKELETON:
                                    if i < len(person_kpts) and j < len(person_kpts):
                                        x1, y1 = person_kpts[i]
                                        x2, y2 = person_kpts[j]
                                        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                                            cv2.line(frame, (int(x1), int(y1)),
                                                    (int(x2), int(y2)), (0, 255, 0), 2)
                                for x, y in person_kpts:
                                    if x > 0 and y > 0:
                                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

                    # 2. MediaPipe Hands - detailed fingers
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
                    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

                    y_offset = 30
                    if hand_result.hand_landmarks:
                        for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                            # Get handedness
                            handedness = "?"
                            if hand_result.handedness and idx < len(hand_result.handedness):
                                handedness = hand_result.handedness[idx][0].category_name[0]  # L or R

                            color = (255, 100, 100) if handedness == "L" else (100, 100, 255)

                            # Draw hand skeleton
                            points = []
                            for lm in hand_landmarks:
                                px, py = int(lm.x * w), int(lm.y * h)
                                points.append((px, py))
                                cv2.circle(frame, (px, py), 3, color, -1)

                            for c1, c2 in HAND_CONNECTIONS:
                                cv2.line(frame, points[c1], points[c2], color, 2)

                            # Detect gesture
                            gesture = get_gesture(hand_landmarks)

                            # Get wrist depth
                            wrist = hand_landmarks[0]
                            px = max(0, min(int(wrist.x * w), w - 1))
                            py = max(0, min(int(wrist.y * h), h - 1))
                            depth_m = 0
                            if depth_frame is not None:
                                depth_m = depth_frame[py, px] / 1000.0

                            # Display hand info
                            label = f"{handedness}: {depth_m:.2f}m"
                            if gesture:
                                label += f" [{gesture}]"
                            cv2.putText(frame, label, (10, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            y_offset += 25

                    # FPS
                    if frame_count % 10 == 0:
                        fps = 10 / (time.time() - fps_time)
                        fps_time = time.time()
                        print(f"FPS: {fps:.1f}")

                    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    cv2.imshow("Skeleton + Hands + Gestures - Q to quit", frame)

                if cv2.waitKey(1) == ord('q'):
                    break

            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
