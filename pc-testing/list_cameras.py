#!/usr/bin/env python3
"""Discover all connected OAK cameras (USB and PoE)."""

import depthai as dai


def main():
    print("Searching for DepthAI devices...")
    devices = dai.Device.getAllAvailableDevices()

    if not devices:
        print("No devices found!")
        print("\nTroubleshooting:")
        print("- USB: Check cable and try different USB 3.0 port")
        print("- PoE: Ensure camera is on same network/subnet")
        return

    print(f"\nFound {len(devices)} device(s):\n")
    for i, info in enumerate(devices):
        print(f"[{i}] {info.name}")
        print(f"    Device ID: {info.deviceId}")
        print(f"    Platform: {info.platform.name}")
        print(f"    State: {info.state.name}")
        print(f"    Protocol: {info.protocol.name}")
        print()


if __name__ == "__main__":
    main()
