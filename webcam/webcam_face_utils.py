"""
Webcam Face Utilities — EAR, Head Pose, and Gaze Estimation
============================================================
Pure numpy/cv2 functions for fatigue and gaze detection.
No DepthAI dependency — works with MediaPipe FaceLandmarker (Tasks API).

Adapted from utils/face_landmarks.py (same math, different input format).
"""

from typing import Tuple
import cv2
import math
import os
import urllib.request
import numpy as np
from pathlib import Path


# ── MediaPipe landmark indices ──────────────────────────────────────────────
# Same indices used by the OAK-D fatigue detector's MediaPipe model.
LEFT_EYE_IDX = [33, 160, 158, 133, 144, 153]
RIGHT_EYE_IDX = [263, 387, 385, 362, 373, 380]
POSE_IDX = [199, 4, 33, 263, 61, 291]

# Iris landmarks (always included in FaceLandmarker output, 478 total)
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Model download URL and cache location
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
_MODEL_DIR = Path(__file__).parent / ".models"
_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"


def get_face_landmarker_model_path() -> str:
    """Download the FaceLandmarker model if not cached, return its path."""
    if _MODEL_PATH.exists():
        return str(_MODEL_PATH)

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face_landmarker model (~5MB)...")
    urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
    print(f"Model saved to {_MODEL_PATH}")
    return str(_MODEL_PATH)


def mediapipe_to_pixel_coords(face_landmarks, frame_w: int, frame_h: int) -> np.ndarray:
    """Convert MediaPipe landmark list to pixel-space numpy array.

    Args:
        face_landmarks: List of NormalizedLandmark objects (from Tasks API)
        frame_w: Frame width in pixels
        frame_h: Frame height in pixels

    Returns:
        numpy array of shape (N, 2) with pixel coordinates
    """
    return np.array(
        [[int(lm.x * frame_w), int(lm.y * frame_h)] for lm in face_landmarks]
    )


def calc_eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """Calculate Eye Aspect Ratio (EAR) from 6 eye landmark points.

    EAR = (|p2-p5| + |p3-p6|) / (2 * |p1-p4|)
    Values below ~0.15 indicate closed eyes.
    """
    A = np.linalg.norm(eye_points[1] - eye_points[4])
    B = np.linalg.norm(eye_points[2] - eye_points[5])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)


def get_pose_estimation(shape: Tuple[int, int], image_points: np.ndarray):
    """Estimate head pose using 6 facial landmarks and solvePnP.

    Args:
        shape: (height, width) of the frame
        image_points: 6x2 array of 2D landmark positions in pixel space

    Returns:
        (success, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    """
    model_points = np.array(
        [
            (0.0, -7.9422, 5.1812),    # Chin (index 199)
            (0.0, -0.4632, 7.5866),    # Nose tip (index 4)
            (-4.4459, 2.6640, 3.1734),  # Left eye corner (index 33)
            (4.4459, 2.6640, 3.1734),   # Right eye corner (index 263)
            (-2.4562, -4.3426, 4.2839), # Left mouth corner (index 61)
            (2.4562, -4.3426, 4.2839),  # Right mouth corner (index 291)
        ],
        dtype="double",
    )

    focal_length = shape[1]
    center = (shape[1] / 2, shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs


def get_euler_angles(rotation_vector) -> Tuple[float, float, float]:
    """Convert rotation vector to Euler angles (pitch, yaw, roll) in degrees."""
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0

    pitch_deg = pitch * 180 / math.pi
    yaw_deg = yaw * 180 / math.pi
    roll_deg = roll * 180 / math.pi

    return pitch_deg, yaw_deg, roll_deg


def determine_fatigue(
    frame_shape: Tuple[int, int],
    landmarks: np.ndarray,
    pitch_angle: int = 20,
    ear_threshold: float = 0.15,
) -> Tuple[bool, bool]:
    """Determine if a person is fatigued based on face landmarks.

    Args:
        frame_shape: (height, width) of the source frame
        landmarks: numpy array of shape (N, 2) with pixel coordinates
        pitch_angle: Head tilt threshold in degrees (default 20)
        ear_threshold: EAR threshold for eye closure (default 0.15)

    Returns:
        Tuple of (head_tilted: bool, eyes_closed: bool)
    """
    left_eye = landmarks[LEFT_EYE_IDX]
    right_eye = landmarks[RIGHT_EYE_IDX]

    image_points = landmarks[POSE_IDX].astype("double")

    # Head pose
    head_tilted = False
    success, rotation_vector, *_ = get_pose_estimation(frame_shape, image_points)
    if success:
        pitch, yaw, roll = get_euler_angles(rotation_vector)
        if pitch < -pitch_angle:
            head_tilted = True

    # Eye aspect ratio
    left_ear = calc_eye_aspect_ratio(left_eye)
    right_ear = calc_eye_aspect_ratio(right_eye)
    ear = (left_ear + right_ear) / 2.0
    eyes_closed = ear < ear_threshold

    return head_tilted, eyes_closed


def estimate_gaze_from_iris(
    landmarks: np.ndarray,
    frame_shape: Tuple[int, int],
) -> Tuple[float, float, float, float, float, float]:
    """Estimate gaze direction from iris landmarks relative to eye corners.

    Uses the position of each iris center within its eye opening to compute
    a normalized gaze vector. Also computes head pose via solvePnP.

    Args:
        landmarks: numpy array of shape (N, 2) with pixel coordinates
                   (must include iris landmarks, indices 468+)
        frame_shape: (height, width) of the source frame

    Returns:
        (gaze_x, gaze_y, gaze_z, head_yaw, head_pitch, head_roll)
        gaze_x: horizontal (-0.5=left, +0.5=right)
        gaze_y: vertical (-0.5=down, +0.5=up)
        gaze_z: placeholder (always ~1.0, for compatibility with OAK-D status format)
    """
    # Left eye: inner corner=33, outer corner=133
    left_inner = landmarks[33].astype(float)
    left_outer = landmarks[133].astype(float)
    left_iris = landmarks[LEFT_IRIS_CENTER].astype(float)

    # Right eye: inner corner=362, outer corner=263
    right_inner = landmarks[362].astype(float)
    right_outer = landmarks[263].astype(float)
    right_iris = landmarks[RIGHT_IRIS_CENTER].astype(float)

    # Horizontal: ratio of iris position within eye width (0=inner, 1=outer)
    left_eye_w = np.linalg.norm(left_outer - left_inner)
    right_eye_w = np.linalg.norm(right_outer - right_inner)

    if left_eye_w < 1 or right_eye_w < 1:
        # Eyes too small to measure
        return 0.0, 0.0, 1.0, 0.0, 0.0, 0.0

    # Project iris onto the inner-to-outer vector
    left_ratio = np.dot(left_iris - left_inner, left_outer - left_inner) / (left_eye_w ** 2)
    right_ratio = np.dot(right_iris - right_inner, right_outer - right_inner) / (right_eye_w ** 2)

    # Average and center: 0.5 = looking straight ahead
    avg_h = (left_ratio + right_ratio) / 2.0
    gaze_x = (avg_h - 0.5)  # negative=left, positive=right

    # Vertical: use upper/lower eyelid landmarks
    # Left eye: upper=159, lower=145; Right eye: upper=386, lower=374
    left_upper = landmarks[159].astype(float)
    left_lower = landmarks[145].astype(float)
    right_upper = landmarks[386].astype(float)
    right_lower = landmarks[374].astype(float)

    left_eye_h = np.linalg.norm(left_lower - left_upper)
    right_eye_h = np.linalg.norm(right_lower - right_upper)

    if left_eye_h < 1 or right_eye_h < 1:
        gaze_y = 0.0
    else:
        left_v_ratio = np.dot(left_iris - left_upper, left_lower - left_upper) / (left_eye_h ** 2)
        right_v_ratio = np.dot(right_iris - right_upper, right_lower - right_upper) / (right_eye_h ** 2)
        avg_v = (left_v_ratio + right_v_ratio) / 2.0
        gaze_y = -(avg_v - 0.5)  # negative=down, positive=up (invert because y increases downward)

    gaze_z = 1.0  # placeholder for compatibility

    # Head pose
    image_points = landmarks[POSE_IDX].astype("double")
    head_yaw, head_pitch, head_roll = 0.0, 0.0, 0.0
    success, rotation_vector, *_ = get_pose_estimation(frame_shape, image_points)
    if success:
        pitch_deg, yaw_deg, roll_deg = get_euler_angles(rotation_vector)
        head_yaw = yaw_deg
        head_pitch = pitch_deg
        head_roll = roll_deg

    return gaze_x, gaze_y, gaze_z, head_yaw, head_pitch, head_roll
