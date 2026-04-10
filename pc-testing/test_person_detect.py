#!/usr/bin/env python3
"""Minimal person detection test for Windows PC.

Simplified version of person_detector.py - no Discord, no status files.
Just camera + YOLO detection + display with temporal smoothing.

Usage:
    python test_person_detect.py              # USB camera (auto-detect)
    python test_person_detect.py --ip X.X.X.X # PoE camera
"""

import argparse
import time
from collections import deque

import cv2
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork


def main():
    parser = argparse.ArgumentParser(description="OAK-D Person Detection Test")
    parser.add_argument("--ip", help="PoE camera IP address")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Detection confidence (0-1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="luxonis/yolov6-nano:r2-coco-512x288",
        help="Model reference from Luxonis Hub",
    )
    args = parser.parse_args()

    # Connect to device
    if args.ip:
        print(f"Connecting to PoE camera at {args.ip}...")
        device_info = dai.DeviceInfo(args.ip)
        device = dai.Device(device_info)
    else:
        print("Searching for USB camera...")
        device = dai.Device()

    with device:
        platform = device.getPlatformAsString()
        print(f"Connected to: {device.getDeviceId()}")
        print(f"Platform: {platform}")
        print(f"USB Speed: {device.getUsbSpeed().name}")
        print(f"Model: {args.model}")
        print(f"Threshold: {args.threshold}")

        with dai.Pipeline(device) as pipeline:
            print("Creating pipeline...")

            # Load model from Luxonis Hub
            model_desc = dai.NNModelDescription(args.model, platform=platform)
            nn_archive = dai.NNArchive(dai.getModelFromZoo(model_desc))

            # Use new Camera API (not deprecated ColorCamera)
            cam = pipeline.create(dai.node.Camera)
            cam.build(dai.CameraBoardSocket.CAM_A)

            # Request output matching model input size (512x288)
            cam_out = cam.requestOutput((512, 288), dai.ImgFrame.Type.BGR888p)

            # Neural network with parser
            nn = pipeline.create(ParsingNeuralNetwork).build(cam_out, nn_archive)

            # Output queues
            q_det = nn.out.createOutputQueue(maxSize=4, blocking=False)
            q_preview = cam_out.createOutputQueue(maxSize=4, blocking=False)

            pipeline.start()
            print("\nDetection started. Press 'q' to quit.\n")

            # Temporal smoothing: track detections over time
            detection_history = deque(maxlen=5)  # Last 5 frames
            smoothed_boxes = []  # Persistent boxes
            last_detection_time = 0
            PERSISTENCE_TIME = 0.3  # Keep box visible for 300ms after losing detection

            while pipeline.isRunning():
                det_msg = q_det.tryGet()
                preview_msg = q_preview.tryGet()

                current_time = time.time()

                if preview_msg is not None:
                    frame = preview_msg.getCvFrame()
                    h, w = frame.shape[:2]

                    # Process new detections
                    current_detections = []
                    if det_msg is not None and hasattr(det_msg, "detections"):
                        for det in det_msg.detections:
                            if det.label == 0 and det.confidence >= args.threshold:
                                x1 = int(det.xmin * w)
                                y1 = int(det.ymin * h)
                                x2 = int(det.xmax * w)
                                y2 = int(det.ymax * h)
                                current_detections.append({
                                    'box': (x1, y1, x2, y2),
                                    'conf': det.confidence,
                                    'time': current_time
                                })

                    # Update detection history
                    if current_detections:
                        detection_history.append(current_detections)
                        last_detection_time = current_time
                        # Use current detections for smoothed output
                        smoothed_boxes = current_detections
                    elif current_time - last_detection_time < PERSISTENCE_TIME:
                        # Keep showing last detection briefly (reduces flicker)
                        pass
                    else:
                        # Clear after persistence time
                        smoothed_boxes = []
                        detection_history.clear()

                    # Draw smoothed boxes
                    for det in smoothed_boxes:
                        x1, y1, x2, y2 = det['box']
                        conf = det['conf']

                        # Smooth box color based on confidence
                        color = (0, int(255 * conf), 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        label = f"person {conf:.0%}"
                        # Background for text
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    # Show person count
                    count = len(smoothed_boxes)
                    cv2.putText(frame, f"People: {count}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv2.imshow("Person Detection - Press Q to quit", frame)

                if cv2.waitKey(1) == ord("q"):
                    break

            cv2.destroyAllWindows()
            print("Done.")


if __name__ == "__main__":
    main()
