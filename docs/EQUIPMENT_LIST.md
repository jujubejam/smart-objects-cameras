# Smart Objects Camera System - Equipment List

> **Prefer slides?** View the [Equipment List Slides](https://kandizzy.github.io/smart-objects-cameras/equipment-list-slides.html) for a visual overview.

## Per-Station Setup (x3 stations: Orbit, Gravity, Horizon)

| Item | Spec / Model | Qty | Est. Price | Notes |
|------|-------------|:---:|------------|-------|
| Raspberry Pi 5 | **16GB RAM** | 1 | ~$205 | 16GB recommended for CV workloads |
| OAK-D Camera | Luxonis OAK-D (USB) | 1 | ~$299 | Onboard Myriad X VPU for neural network inference |
| MicroSD Card | 64GB, Class 10 / A2 | 1 | ~$9 | Samsung 64GB Ultra microSDXC Class 10 |
| USB-C Power Supply | Official Raspberry Pi 5 27W PSU | 1 | ~$12 | 5.1V / 5A — required for stable operation |
| USB 3.0 Cable | USB-A to USB-C, short (~1m) | 1 | ~$5 | Connects OAK-D to Pi; must be USB 3.0 (blue port) |
| 5V DC Power Supply | 5V 2A, 5.5mm × 2.5mm barrel jack | 1 | ~$9 | Powers OAK-D via barrel jack |

**Per-station subtotal: ~$539**

---

## Full Set (3 Stations)

| Item | Qty | Est. Total |
|------|:---:|------------|
| Raspberry Pi 5 (16GB) | 3 | ~$615 |
| OAK-D Camera | 3 | ~$897 |
| MicroSD Card (64GB) | 3 | ~$27 |
| USB-C Power Supply (27W) | 3 | ~$36 |
| USB 3.0 Cable | 3 | ~$15 |
| 5V DC Power Supply (barrel jack) | 3 | ~$27 |

**3-station total: ~$1,617**

---

## Optional but Recommended

| Item | Spec / Model | Qty | Est. Price | Notes |
|------|-------------|:---:|------------|-------|
| Ethernet Cables | Cat6, various lengths | 3 | ~$15 | For reliable network during setup |
| Network Switch | 5-port Gigabit (TP-Link TL-SG105) | 1 | ~$20 | If classroom lacks enough Ethernet ports |
| USB Keyboard + Mouse | Logitech MK270 wireless combo | 1 | ~$25 | For initial Pi setup (shared across stations) |
| Micro-HDMI to HDMI Cable | For Pi 5, 6ft | 1 | ~$10 | For initial setup with a monitor (shared) |

### Portable Battery Kit (per station)

| Item | Spec / Model | Qty | Est. Price | Notes |
|------|-------------|:---:|------------|-------|
| Anker 737 Power Bank | 24,000mAh, 140W, 2×USB-C PD + 1×USB-A | 1 | ~$100 | ~3hr runtime. Pi 5 → USB-C PD port; OAK-D → USB-A port via barrel cable |
| USB-A to Barrel Jack Cable | USB-A to 5.5×2.5mm, 3ft | 1 | ~$9 | Connects OAK-D barrel jack to Anker USB-A port |

**Optional extras: ~$179**

---

## Optional: PoE Spatial Tracking Rig

A multi-camera spatial tracking setup using 3 OAK-D PoE cameras connected to a single Raspberry Pi over powered Ethernet. The PoE switch delivers both power and data to each camera — no USB cables or separate power supplies needed per camera.

**We already have:** 3x OAK-D PoE cameras

| Item | Spec / Model | Qty | Est. Price | Notes |
|------|-------------|:---:|------------|-------|
| OAK-D PoE Camera | Luxonis OAK-D PoE | 3 | *already owned* | PoE powers + connects each camera over one cable |
| PoE+ Network Switch | Ubiquiti UniFi Switch Pro Max 16 PoE | 1 | ~$399 | 180W PoE budget, 16 ports, 2×10G SFP+ uplinks — supports up to 12 cameras at 15W each |
| Ethernet Cables | Cat6, various lengths | 3 | ~$15 | One per camera — carries both power and data |
| Raspberry Pi 5 | **16GB RAM** | 1 | ~$205 | Single Pi manages all 3 PoE cameras over the network |
| MicroSD Card | 64GB, Class 10 / A2 | 1 | ~$12 | For Pi OS + project files |
| USB-C Power Supply | Official Raspberry Pi 5 27W PSU | 1 | ~$20 | Powers the Pi (cameras are powered by PoE switch) |
| Compact Switch (optional) | Ubiquiti UniFi Switch Flex Mini | 1 | ~$87 | Dedicated subnet for camera traffic |

**PoE rig subtotal: ~$738** (not counting cameras already owned)

**What spatial tracking enables:**
- Track people/objects across a room using triangulated positions from 3 camera angles
- Each OAK-D PoE runs its own neural network on-device and reports detections over the network
- The Pi fuses detection data from all 3 cameras into a unified spatial map
- Cleaner cable runs — one Ethernet cable per camera replaces USB + power

---

## Camera Upgrade Path: OAK 4 (RVC4)

The current OAK-D cameras use the **RVC2 (Myriad X)** chip from 2018 — capable but slow for multi-stage pipelines (2–8 FPS). The new **OAK 4** line uses the **RVC4** platform (Qualcomm QCS8550) with **52 TOPS** of inference — roughly **40x faster**. Multi-stage pipelines that crawl at 2–5 FPS on RVC2 run at 30 FPS on RVC4.

### OAK 4 Models

| Model | Price | Stereo Depth | Key Difference |
|-------|------:|:------------:|----------------|
| [OAK 4 S](https://shop.luxonis.com/products/oak-4-s-ea) | ~$749 | No | Single RGB camera only |
| [OAK 4 D](https://shop.luxonis.com/products/oak-4-d) | ~$849 | Yes | Direct OAK-D replacement — stereo depth + RGB |
| [OAK 4 D Pro](https://shop.luxonis.com/products/oak-4-d-ea) | ~$949 | Yes + laser dot projector | Best depth in low-light / textureless scenes |

All OAK 4 models: 8 GB RAM, 128 GB storage, 48 MP RGB sensor, standalone capable (runs apps on-device without a host computer). **Every OAK 4 has both USB and PoE built in** — no separate PoE variant needed.

> **Availability:** OAK 4 cameras ship starting **March 20, 2026** from [shop.luxonis.com](https://shop.luxonis.com/).

### Upgrade Cost: USB Stations (3-Station Set)

| | Current (OAK-D) | Upgrade (OAK 4 D) | Difference |
|--|----------------:|-----------------:|----------:|
| Cameras (×3) | ~$897 | ~$2,547 | +$1,650 |
| Pis + accessories | ~$720 | ~$720 | — |
| **Station total** | **~$1,617** | **~$3,267** | **+$1,650** |

### Upgrade Cost: PoE Spatial Tracking Rig

Since every OAK 4 has PoE built in, the same cameras replace the OAK-D PoE units. They plug into the same PoE switch with the same Ethernet cables.

| | Current (OAK-D PoE) | Upgrade (OAK 4 D) | Difference |
|--|--------------------:|-----------------:|----------:|
| Cameras (×3) | *already owned* | ~$2,547 | +$2,547 |
| PoE switch + Pi + accessories | ~$738 | ~$738 | — |
| **Rig total** | **~$738** | **~$3,285** | **+$2,547** |

> **Note:** The Raspberry Pis are still needed for Discord bot orchestration and status file management, even though the OAK 4 can run inference standalone. The Pi handles the conversational layer; the camera handles the vision.

### What the Upgrade Unlocks

- All [RVC4-only examples](https://github.com/luxonis/oak-examples) (YOLO-World open-vocabulary detection, DINO tracking, people demographics dashboard)
- Real-time multi-stage pipelines (gesture recognition, pose estimation, re-identification at 30 FPS instead of 2–5 FPS)
- On-device standalone apps — camera can process and act without streaming to a host
- Larger, more accurate models that won't fit on Myriad X
- **PoE + USB in one unit** — simplifies inventory (one camera model for both setups)

---

## Software (Free / No Cost)

- **Raspberry Pi OS** (Bookworm 64-bit) — free
- **DepthAI 3.x + depthai-nodes** — free, open source
- **Python 3 + OpenCV + NumPy** — free
- **Discord** (bot + webhooks) — free
- **VS Code + Remote SSH extension** — free
- **RealVNC Viewer** — free for non-commercial use

---

## What This System Does

Each station is a self-contained **smart object**: a Raspberry Pi with an AI camera that can:

- Detect people (or other objects) using YOLO neural networks running on the camera's dedicated VPU
- Send real-time alerts to a shared Discord channel
- Respond to interactive Discord commands (!status, !detect, !screenshot)
- Coordinate across multiple cameras as a networked sensing system

Students learn computer vision, hardware integration, networked systems, and Python development — all through hands-on experimentation with physical devices.

---

## Supplier Links

- **Raspberry Pi 5**: [raspberrypi.com](https://www.raspberrypi.com/products/raspberry-pi-5/)
- **OAK Cameras**: [shop.luxonis.com](https://shop.luxonis.com/)
- **MicroSD / Cables / Hubs**: Amazon, Adafruit, SparkFun

---

*Prices are estimates as of early 2026 and may vary by supplier.*
