#!/usr/bin/env python3
"""Test mono cameras - might be more stable."""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import depthai as dai

print("Connecting to device...", flush=True)
with dai.Device() as device:
    print(f"Connected: {device.getDeviceId()}", flush=True)
    print(f"USB Speed: {device.getUsbSpeed().name}", flush=True)

    with dai.Pipeline(device) as pipeline:
        # Try left mono camera
        mono = pipeline.create(dai.node.MonoCamera)
        mono.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono.setFps(10)

        q = mono.out.createOutputQueue(maxSize=4, blocking=False)

        pipeline.start()
        print("Mono camera pipeline started...", flush=True)

        import time
        start = time.time()
        frame_count = 0

        while time.time() - start < 15:
            msg = q.tryGet()
            if msg is not None:
                frame_count += 1
                frame = msg.getCvFrame()
                if frame_count == 1:
                    print(f"First frame: {frame.shape}", flush=True)
                if frame_count % 20 == 0:
                    print(f"Frames: {frame_count}", flush=True)
                cv2.imshow("Mono Camera", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        print(f"Total frames: {frame_count}", flush=True)
        cv2.destroyAllWindows()
