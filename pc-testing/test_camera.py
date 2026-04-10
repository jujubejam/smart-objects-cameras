#!/usr/bin/env python3
"""Test OAK camera connectivity and display live feed.

Usage:
    python test_camera.py              # USB camera (auto-detect)
    python test_camera.py --ip X.X.X.X # PoE camera
"""

import argparse

import cv2
import depthai as dai


def main():
    parser = argparse.ArgumentParser(description="Test OAK camera connectivity")
    parser.add_argument("--ip", help="PoE camera IP address")
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
        print(f"Connected to: {device.getDeviceId()}")
        print(f"Platform: {device.getPlatformAsString()}")
        print(f"USB Speed: {device.getUsbSpeed().name}")

        with dai.Pipeline(device) as pipeline:
            # Use new Camera API (not deprecated ColorCamera)
            cam = pipeline.create(dai.node.Camera)
            cam.build(dai.CameraBoardSocket.CAM_A)

            # Request 640x480 BGR output
            out = cam.requestOutput((640, 480), dai.ImgFrame.Type.BGR888p)
            q = out.createOutputQueue(maxSize=4, blocking=False)

            pipeline.start()
            print("Pipeline started! Press Q to quit.")

            frame_count = 0
            while True:
                msg = q.tryGet()
                if msg is not None:
                    frame = msg.getCvFrame()
                    frame_count += 1
                    if frame_count == 1:
                        print(f"First frame: {frame.shape}")
                    cv2.imshow("Camera Test - Press Q to quit", frame)

                if cv2.waitKey(1) == ord("q"):
                    break

            print(f"Total frames: {frame_count}")
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
