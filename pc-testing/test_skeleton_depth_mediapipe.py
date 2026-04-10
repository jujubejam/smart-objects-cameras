#!/usr/bin/env python3
"""Skeleton + Hands + Depth tracking with GPU acceleration.

Uses OAK camera for RGB + depth, MediaPipe Tasks for skeleton/hands.
Calculates 3D hand positions and distances.

Usage:
    python test_skeleton_depth.py              # USB camera
    python test_skeleton_depth.py --ip X.X.X.X # PoE camera
"""

import argparse
import time
import urllib.request
from pathlib import Path

import cv2
import depthai as dai
import mediapipe as mp
import numpy as np

# Download models if needed
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

POSE_MODEL = MODEL_DIR / "pose_landmarker_full.task"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"

POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def download_models():
    """Download MediaPipe models if not present."""
    if not POSE_MODEL.exists():
        print(f"Downloading pose model...")
        urllib.request.urlretrieve(POSE_URL, POSE_MODEL)
        print(f"  Saved to {POSE_MODEL}")

    if not HAND_MODEL.exists():
        print(f"Downloading hand model...")
        urllib.request.urlretrieve(HAND_URL, HAND_MODEL)
        print(f"  Saved to {HAND_MODEL}")


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
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align to RGB
    stereo.setOutputSize(640, 480)

    left.out.link(stereo.left)
    right.out.link(stereo.right)

    # Output queues
    q_rgb = rgb_out.createOutputQueue(maxSize=4, blocking=False)
    q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

    return pipeline, q_rgb, q_depth


def draw_landmarks(frame, landmarks, connections, color=(0, 255, 0)):
    """Draw landmarks and connections on frame."""
    h, w = frame.shape[:2]
    points = []

    for lm in landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        points.append((px, py))
        cv2.circle(frame, (px, py), 3, color, -1)

    if connections:
        for conn in connections:
            if conn[0] < len(points) and conn[1] < len(points):
                cv2.line(frame, points[conn[0]], points[conn[1]], color, 2)


def main():
    parser = argparse.ArgumentParser(description="Skeleton + Depth tracking")
    parser.add_argument("--ip", help="PoE camera IP address")
    args = parser.parse_args()

    # Download models
    download_models()

    # Setup MediaPipe
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Connect to device
    if args.ip:
        print(f"Connecting to PoE camera at {args.ip}...")
        device_info = dai.DeviceInfo(args.ip)
        device = dai.Device(device_info)
    else:
        print("Searching for USB camera...")
        device = dai.Device()

    with device:
        print(f"Connected: {device.getDeviceId()}")
        print(f"Platform: {device.getPlatformAsString()}")

        pipeline, q_rgb, q_depth = create_pipeline(device)
        pipeline.start()
        print("Pipeline started with RGB + Depth")

        with PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
             HandLandmarker.create_from_options(hand_options) as hand_landmarker:

            fps_time = time.time()
            frame_count = 0
            timestamp_ms = 0

            # Pose connections (simplified)
            POSE_CONNECTIONS = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
                (11, 23), (12, 24), (23, 24),  # Torso
                (23, 25), (25, 27), (24, 26), (26, 28)  # Legs
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

            while True:
                rgb_msg = q_rgb.tryGet()
                depth_msg = q_depth.tryGet()

                if rgb_msg is not None:
                    frame = rgb_msg.getCvFrame()
                    frame_count += 1
                    timestamp_ms += 33  # ~30fps

                    # Get depth frame
                    depth_frame = None
                    if depth_msg is not None:
                        depth_frame = depth_msg.getFrame()

                    # Convert to MediaPipe image
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # Detect pose
                    pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

                    # Detect hands
                    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

                    # Draw pose
                    if pose_result.pose_landmarks:
                        for pose_landmarks in pose_result.pose_landmarks:
                            draw_landmarks(frame, pose_landmarks, POSE_CONNECTIONS, (0, 255, 0))

                    # Process hands with depth
                    hand_positions = []
                    if hand_result.hand_landmarks:
                        for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                            # Determine handedness
                            handedness = "Unknown"
                            if hand_result.handedness and i < len(hand_result.handedness):
                                handedness = hand_result.handedness[i][0].category_name

                            color = (255, 0, 0) if handedness == "Left" else (0, 0, 255)
                            draw_landmarks(frame, hand_landmarks, HAND_CONNECTIONS, color)

                            # Get wrist position with depth
                            wrist = hand_landmarks[0]
                            h, w = frame.shape[:2]
                            px = int(wrist.x * w)
                            py = int(wrist.y * h)
                            px = max(0, min(px, w - 1))
                            py = max(0, min(py, h - 1))

                            depth_m = 0
                            if depth_frame is not None:
                                depth_mm = depth_frame[py, px]
                                depth_m = depth_mm / 1000.0

                            hand_positions.append({
                                'name': handedness,
                                'x': px, 'y': py,
                                'depth_m': depth_m
                            })

                    # Display hand info
                    y_offset = 30
                    for hand in hand_positions:
                        text = f"{hand['name']} hand: {hand['depth_m']:.2f}m"
                        color = (255, 0, 0) if hand['name'] == "Left" else (0, 0, 255)
                        cv2.putText(frame, text, (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y_offset += 25

                    # Calculate distance between hands
                    if len(hand_positions) == 2:
                        h1, h2 = hand_positions
                        # Simple 3D distance approximation
                        dx = (h2['x'] - h1['x']) / 640  # Normalized
                        dy = (h2['y'] - h1['y']) / 480
                        dz = h2['depth_m'] - h1['depth_m']
                        avg_depth = (h1['depth_m'] + h2['depth_m']) / 2
                        # Scale x/y by depth for approximate real-world distance
                        dx_m = dx * avg_depth * 1.2  # ~1.2 is rough FOV factor
                        dy_m = dy * avg_depth * 0.9
                        distance = np.sqrt(dx_m**2 + dy_m**2 + dz**2)

                        cv2.putText(frame, f"Hands apart: {distance:.2f}m", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        y_offset += 25

                    # FPS counter
                    if frame_count % 30 == 0:
                        fps = 30 / (time.time() - fps_time)
                        fps_time = time.time()
                        print(f"FPS: {fps:.1f}")

                    cv2.putText(frame, f"Frame: {frame_count}", (10, 470),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

                    cv2.imshow("Skeleton + Hands + Depth - Press Q to quit", frame)

                if cv2.waitKey(1) == ord('q'):
                    break

            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
