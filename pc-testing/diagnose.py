#!/usr/bin/env python3
"""Diagnose OAK camera connection issues.

Checks USB speed, device health, and lists all available cameras.
"""

import depthai as dai


def main():
    print("=" * 50)
    print("OAK Camera Diagnostics")
    print("=" * 50)

    # First, list all available devices
    print("\n[1] Scanning for all devices...")
    devices = dai.Device.getAllAvailableDevices()

    if not devices:
        print("    No devices found!")
        print("\n    Troubleshooting:")
        print("    - USB: Check cable and try different USB 3.0 port")
        print("    - PoE: Ensure camera is powered and on same subnet")
        return

    print(f"    Found {len(devices)} device(s):\n")
    for i, info in enumerate(devices):
        print(f"    [{i}] {info.name}")
        print(f"        Device ID: {info.deviceId}")
        print(f"        Platform: {info.platform.name}")
        print(f"        State: {info.state.name}")
        print(f"        Protocol: {info.protocol.name}")
        print()

    # Connect to first available device and check details
    print("[2] Connecting to first device for detailed check...")
    try:
        with dai.Device() as device:
            print(f"    Device ID: {device.getDeviceId()}")
            print(f"    Platform: {device.getPlatformAsString()}")

            # Check USB speed
            usb_speed = device.getUsbSpeed()
            print(f"    USB Speed: {usb_speed.name}")

            if usb_speed == dai.UsbSpeed.SUPER:
                print("    ✓ USB 3.0 detected - good!")
            elif usb_speed == dai.UsbSpeed.SUPER_PLUS:
                print("    ✓ USB 3.1 detected - excellent!")
            elif usb_speed == dai.UsbSpeed.HIGH:
                print("    ✗ USB 2.0 detected - THIS CAUSES CRASHES!")
                print("      → Try a different USB port (blue interior = USB 3.0)")
                print("      → Use a powered USB 3.0 hub")
            else:
                print(f"    ? Unknown USB speed: {usb_speed.name}")

            # Check bootloader (PoE cameras have firmware, USB cameras boot from host)
            try:
                bootloader = device.getBootloaderVersion()
                if bootloader:
                    print(f"    Bootloader: {bootloader}")
                else:
                    print("    Bootloader: None (normal for USB cameras)")
            except Exception:
                print("    Bootloader: Unable to check")

    except Exception as e:
        print(f"    Error connecting: {e}")

    print("\n" + "=" * 50)
    print("Diagnostics complete")
    print("=" * 50)


if __name__ == "__main__":
    main()
