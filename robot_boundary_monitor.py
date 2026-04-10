"""
robot_boundary_monitor.py - Keep robots from falling off desks
==============================================================
Uses YOLO object detection to track a robot and warn/stop it
before it reaches the edge of the safe zone (desk boundary).

Latency: ~50-100ms (fast enough for robot safety)

Usage:
    # First, calibrate the safe zone (click corners of desk)
    python robot_boundary_monitor.py --calibrate

    # Then run monitoring
    python robot_boundary_monitor.py --display

    # With robot communication (HTTP endpoint)
    python robot_boundary_monitor.py --robot-url http://robot.local:8080/stop

Requirements:
    - OAK-D camera with view of the desk
    - Robot visible to camera (detected as an object)
"""

import argparse
import json
import time
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import depthai as dai
from depthai_nodes.parsing_neural_network import ParsingNeuralNetwork

# Config
CAMERA_W, CAMERA_H = 640, 480
FPS = 30
MODEL_REF = "luxonis/yolov6-nano:r2-coco-640x640"

# Safe zone config file
CONFIG_DIR = Path.home() / "oak-projects"
SAFE_ZONE_FILE = CONFIG_DIR / "robot_safe_zone.json"

# Detection settings
CONFIDENCE_THRESHOLD = 0.4
WARNING_MARGIN = 50  # pixels from edge to start warning
CRITICAL_MARGIN = 20  # pixels from edge to trigger stop

# What to track as "robot" - COCO class IDs
# Could be: 0=person (for testing), or custom trained model
# Common small objects: 39=bottle, 64=mouse, 73=book, 76=scissors
ROBOT_CLASSES = [0, 39, 64, 73, 76]  # Adjust based on your robot

# AprilTag settings
APRILTAG_FAMILY = "tag36h11"  # Most common family, good error correction
APRILTAG_MARKER_ID = 0  # Which tag ID to track as "the robot"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("boundary-monitor")


class SafeZone:
    """Polygon representing the safe area (desk surface)."""

    def __init__(self, points: list = None):
        self.points = points or []

    def contains(self, x: int, y: int) -> bool:
        """Check if point is inside the polygon."""
        if len(self.points) < 3:
            return True  # No zone defined, everything is safe

        # Ray casting algorithm
        n = len(self.points)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.points[i]
            xj, yj = self.points[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def distance_to_edge(self, x: int, y: int) -> float:
        """Approximate distance to nearest edge (negative if outside)."""
        if len(self.points) < 3:
            return float('inf')

        if not self.contains(x, y):
            return -1  # Outside

        # Simple approximation: distance to nearest edge line
        min_dist = float('inf')
        n = len(self.points)
        for i in range(n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % n]

            # Distance from point to line segment
            dist = self._point_to_segment_dist(x, y, x1, y1, x2, y2)
            min_dist = min(min_dist, dist)

        return min_dist

    def _point_to_segment_dist(self, px, py, x1, y1, x2, y2) -> float:
        """Distance from point to line segment."""
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def draw(self, frame: np.ndarray, color=(0, 255, 0), thickness=2):
        """Draw the safe zone on frame."""
        if len(self.points) >= 3:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, thickness)

            # Draw corner markers
            for i, (x, y) in enumerate(self.points):
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.putText(frame, str(i+1), (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def save(self, path: Path):
        """Save zone to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"points": self.points}, indent=2))
        log.info(f"Safe zone saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "SafeZone":
        """Load zone from JSON file."""
        if path.exists():
            data = json.loads(path.read_text())
            zone = cls(data.get("points", []))
            log.info(f"Loaded safe zone with {len(zone.points)} points")
            return zone
        return cls()


def calibrate_safe_zone():
    """Interactive calibration - click corners of desk to define safe zone."""
    log.info("CALIBRATION MODE")
    log.info("Click the corners of the desk (safe area) in order.")
    log.info("Press 'c' to clear, 's' to save, 'q' to quit.")

    zone = SafeZone()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            zone.points.append((x, y))
            log.info(f"  Added point {len(zone.points)}: ({x}, {y})")

    with dai.Device() as device:
        with dai.Pipeline(device) as pipeline:
            # Camera setup
            cam = pipeline.create(dai.node.Camera)
            cam.build(dai.CameraBoardSocket.CAM_A)
            cam_out = cam.requestOutput((CAMERA_W, CAMERA_H), dai.ImgFrame.Type.BGR888p)
            q_preview = cam_out.createOutputQueue(maxSize=4, blocking=False)

            pipeline.start()

            cv2.namedWindow("Calibration")
            cv2.setMouseCallback("Calibration", mouse_callback)

            while True:
                pkt = q_preview.tryGet()
                if pkt is not None:
                    frame = pkt.getCvFrame()

                    # Draw current zone
                    zone.draw(frame, color=(0, 255, 255), thickness=2)

                    # Instructions
                    cv2.putText(frame, "Click corners of safe zone (desk)",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Points: {len(zone.points)} | C=clear S=save Q=quit",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    cv2.imshow("Calibration", frame)

                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    zone.points = []
                    log.info("  Cleared all points")
                elif key == ord('s'):
                    if len(zone.points) >= 3:
                        zone.save(SAFE_ZONE_FILE)
                        break
                    else:
                        log.warning("  Need at least 3 points!")

            cv2.destroyAllWindows()


def send_robot_command(url: str, command: str):
    """Send command to robot via HTTP."""
    if not url:
        return
    try:
        import requests
        requests.post(f"{url}/{command}", timeout=0.5)
        log.info(f"  Sent {command} to robot")
    except Exception as e:
        log.warning(f"  Robot command failed: {e}")


class AprilTagTracker:
    """Track AprilTag markers for robot position."""

    def __init__(self, family=APRILTAG_FAMILY, marker_id=APRILTAG_MARKER_ID):
        try:
            from pupil_apriltags import Detector
            self.detector = Detector(families=family)
            self.marker_id = marker_id
            self.available = True
            log.info(f"AprilTag detector initialized (family: {family})")
        except ImportError:
            log.warning("pupil-apriltags not installed. Install with: pip install pupil-apriltags")
            self.detector = None
            self.available = False

    def detect(self, frame: np.ndarray) -> list:
        """
        Detect AprilTag markers in frame.
        Returns list of (tag_id, center_x, center_y, corners) tuples.
        """
        if not self.available:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray)

        results = []
        for tag in tags:
            cx, cy = int(tag.center[0]), int(tag.center[1])
            corners = tag.corners.astype(np.float32)
            results.append((tag.tag_id, cx, cy, corners))

        return results

    def draw(self, frame: np.ndarray, detections: list, highlight_id: int = None):
        """Draw detected tags on frame."""
        for tag_id, cx, cy, corners in detections:
            # Draw tag outline
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            color = (0, 255, 0) if tag_id == highlight_id else (255, 0, 0)
            cv2.polylines(frame, [pts], True, color, 2)

            # Draw center and ID
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, f"Tag:{tag_id}", (cx + 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def generate_apriltag_marker(tag_id: int = 0, family: str = "tag36h11"):
    """
    Instructions to generate AprilTag markers.
    Unlike ArUco, AprilTag markers are best generated from official sources.
    """
    log.info(f"To generate AprilTag {family} ID {tag_id}:")
    log.info("  1. Visit: https://github.com/AprilRobotics/apriltag-imgs")
    log.info(f"  2. Download: {family}/tag{family.replace('tag', '')}_{tag_id:05d}.png")
    log.info("  3. Print at ~5cm size and attach to robot")
    log.info("")
    log.info("Or generate with Python:")
    log.info("  pip install apriltag")
    log.info("  python -c \"import apriltag; apriltag.generate_tag(0, 'tag36h11', 200)\"")

    # Save a placeholder instruction file
    output = CONFIG_DIR / f"apriltag_instructions.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"""AprilTag Generation Instructions
================================

Download pre-made tags from:
https://github.com/AprilRobotics/apriltag-imgs/tree/master/{family}

For tag ID {tag_id}, download:
tag{family.replace('tag', '')}_{tag_id:05d}.png

Print at approximately 5cm x 5cm.
Attach to top of robot so camera can see it.
""")
    log.info(f"Instructions saved to {output}")


def run_monitor(args):
    """Main monitoring loop."""
    zone = SafeZone.load(SAFE_ZONE_FILE)
    if len(zone.points) < 3:
        log.error("No safe zone defined! Run with --calibrate first.")
        return

    log.info(f"Monitoring with {len(zone.points)}-point safe zone")
    log.info(f"Warning margin: {WARNING_MARGIN}px, Critical: {CRITICAL_MARGIN}px")

    # Initialize tracker
    apriltag_tracker = None
    if args.apriltag:
        apriltag_tracker = AprilTagTracker(marker_id=args.tag_id)
        if apriltag_tracker.available:
            log.info(f"Using AprilTag tracking (tag ID: {args.tag_id})")
        else:
            log.warning("Falling back to YOLO detection")
            apriltag_tracker = None

    if not apriltag_tracker:
        log.info(f"Using YOLO detection (COCO classes: {ROBOT_CLASSES})")

    # Discord webhook
    webhook_url = ""
    if args.discord:
        env_file = CONFIG_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("DISCORD_WEBHOOK_URL="):
                    webhook_url = line.split("=", 1)[1].strip().strip('"')

    last_alert_time = 0
    alert_cooldown = 3  # seconds between alerts

    with dai.Device() as device:
        platform = device.getPlatformAsString()
        log.info(f"Device: {device.getDeviceId()} ({platform})")

        with dai.Pipeline(device) as pipeline:
            # Camera
            cam = pipeline.create(dai.node.Camera)
            cam.build(dai.CameraBoardSocket.CAM_A)
            cam_out = cam.requestOutput((CAMERA_W, CAMERA_H), dai.ImgFrame.Type.BGR888p)

            # YOLO detector (only if not using AprilTag)
            q_det = None
            if not apriltag_tracker:
                model_desc = dai.NNModelDescription(MODEL_REF, platform=platform)
                nn_archive = dai.NNArchive(dai.getModelFromZoo(model_desc))
                nn = pipeline.create(ParsingNeuralNetwork).build(cam_out, nn_archive)
                q_det = nn.out.createOutputQueue(maxSize=4, blocking=False)

            # Queues
            q_preview = cam_out.createOutputQueue(maxSize=4, blocking=False)

            pipeline.start()
            log.info("Monitoring started - press 'q' to quit")

            frame_count = 0

            try:
                while True:
                    # Get frame
                    pkt = q_preview.tryGet()

                    if pkt is None:
                        time.sleep(0.001)
                        continue

                    frame = pkt.getCvFrame()
                    frame_count += 1

                    # Draw safe zone
                    zone.draw(frame, color=(0, 255, 0), thickness=2)

                    # Process detections
                    robot_status = "SAFE"
                    status_color = (0, 255, 0)
                    robot_positions = []  # List of (cx, cy) for detected robots

                    # --- AprilTag detection ---
                    if apriltag_tracker:
                        tags = apriltag_tracker.detect(frame)
                        apriltag_tracker.draw(frame, tags, highlight_id=args.tag_id)

                        for tag_id, cx, cy, corners in tags:
                            if tag_id == args.tag_id:
                                robot_positions.append((cx, cy, f"Tag:{tag_id}"))

                    # --- YOLO detection ---
                    elif q_det is not None:
                        det_pkt = q_det.tryGet()
                        if det_pkt is not None:
                            detections = det_pkt.detections if hasattr(det_pkt, 'detections') else []

                            for det in detections:
                                # Filter by class and confidence
                                label = int(det.label) if hasattr(det, 'label') else -1
                                conf = det.confidence if hasattr(det, 'confidence') else 0

                                if label not in ROBOT_CLASSES or conf < CONFIDENCE_THRESHOLD:
                                    continue

                                # Get bounding box
                                x1 = int(det.xmin * CAMERA_W)
                                y1 = int(det.ymin * CAMERA_H)
                                x2 = int(det.xmax * CAMERA_W)
                                y2 = int(det.ymax * CAMERA_H)

                                # Center point (robot position)
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                robot_positions.append((cx, cy, f"Class:{label}"))

                                # Draw YOLO detection box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # --- Check all robot positions against safe zone ---
                    for cx, cy, label in robot_positions:
                        # Check distance to edge
                        dist = zone.distance_to_edge(cx, cy)

                        # Determine status
                        if dist < 0:
                            robot_status = "OUTSIDE"
                            status_color = (0, 0, 255)
                            marker_color = (0, 0, 255)
                        elif dist < CRITICAL_MARGIN:
                            robot_status = "CRITICAL"
                            status_color = (0, 0, 255)
                            marker_color = (0, 0, 255)
                        elif dist < WARNING_MARGIN:
                            robot_status = "WARNING"
                            status_color = (0, 165, 255)
                            marker_color = (0, 165, 255)
                        else:
                            marker_color = (0, 255, 0)

                        # Draw robot marker
                        cv2.circle(frame, (cx, cy), 8, marker_color, -1)
                        cv2.putText(frame, f"{label} d={dist:.0f}px",
                                   (cx + 15, cy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.5, marker_color, 2)

                        # Take action if critical
                        if robot_status in ["CRITICAL", "OUTSIDE"]:
                            now = time.time()
                            if now - last_alert_time > alert_cooldown:
                                log.warning(f"ROBOT {robot_status}! {label} Distance: {dist:.0f}px")

                                # Send stop command
                                if args.robot_url:
                                    send_robot_command(args.robot_url, "stop")

                                # Discord alert
                                if webhook_url:
                                    import requests
                                    try:
                                        requests.post(webhook_url, json={
                                            "content": f"🚨 **Robot {robot_status}!** {label} - Distance to edge: {dist:.0f}px"
                                        }, timeout=2)
                                    except:
                                        pass

                                last_alert_time = now

                    # Status overlay
                    cv2.rectangle(frame, (0, 0), (200, 40), status_color, -1)
                    cv2.putText(frame, robot_status, (10, 28),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # Display
                    if args.display:
                        cv2.imshow("Robot Boundary Monitor", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    # Periodic log
                    if frame_count % 300 == 0:  # Every 10 sec at 30fps
                        log.info(f"  Frame {frame_count}, status: {robot_status}")

            except KeyboardInterrupt:
                log.info("Stopped")
            finally:
                if args.display:
                    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Robot boundary monitor")
    parser.add_argument("--calibrate", action="store_true",
                       help="Calibrate safe zone (click desk corners)")
    parser.add_argument("--display", action="store_true",
                       help="Show video display")
    parser.add_argument("--robot-url", default=None,
                       help="Robot HTTP endpoint for stop commands")
    parser.add_argument("--discord", action="store_true",
                       help="Send Discord alerts")
    parser.add_argument("--warning-margin", type=int, default=WARNING_MARGIN,
                       help="Pixels from edge to start warning")
    parser.add_argument("--critical-margin", type=int, default=CRITICAL_MARGIN,
                       help="Pixels from edge to trigger stop")
    parser.add_argument("--apriltag", action="store_true",
                       help="Use AprilTag tracking instead of YOLO")
    parser.add_argument("--tag-id", type=int, default=0,
                       help="AprilTag ID to track as the robot (default: 0)")
    parser.add_argument("--generate-tag", action="store_true",
                       help="Show instructions for generating AprilTag markers")
    args = parser.parse_args()

    global WARNING_MARGIN, CRITICAL_MARGIN
    WARNING_MARGIN = args.warning_margin
    CRITICAL_MARGIN = args.critical_margin

    if args.generate_tag:
        generate_apriltag_marker(args.tag_id)
    elif args.calibrate:
        calibrate_safe_zone()
    else:
        run_monitor(args)


if __name__ == "__main__":
    main()
