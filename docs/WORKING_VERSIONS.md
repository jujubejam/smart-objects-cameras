# Working Package Versions - OAK-D Camera Setup

> **Prefer slides?** View the [Working Versions Slides](https://kandizzy.github.io/smart-objects-cameras/working-versions-slides.html) for a visual overview.

**Last Updated:** 2026-02-04
**Platform:** Raspberry Pi 5 (16GB)
**OS:** Debian GNU/Linux 13 (trixie) - aarch64
**Python:** 3.13.5

---

## Critical Package Versions

These versions are **confirmed working** together for OAK-D camera detection with GUI display support on Raspberry Pi 5:

```
depthai==3.3.0
depthai-nodes==0.3.7
depthai-pipeline-graph==0.0.5
opencv-contrib-python==4.10.0.84
numpy==1.26.4
scipy==1.17.0
matplotlib==3.10.8
```

### Additional Dependencies

```
requests==2.26.0
urllib3==1.26.20
charset-normalizer==2.0.12
Qt.py==2.0.2
contourpy==1.3.3
cycler==0.12.1
fonttools==4.61.1
kiwisolver==1.4.9
pillow==12.1.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
six==1.17.0
```

---

## Installation Instructions

### System Dependencies (Required for OpenCV GUI)

```bash
sudo apt-get update
sudo apt-get install -y libgtk-3-dev libgtk2.0-dev pkg-config
```

### Python Virtual Environment Setup

```bash
# Create shared virtual environment
sudo mkdir -p /opt/oak-shared
sudo python3 -m venv /opt/oak-shared/venv
sudo /opt/oak-shared/venv/bin/pip install --upgrade pip

# Install core packages (order matters!)
sudo /opt/oak-shared/venv/bin/pip install numpy==1.26.4
sudo /opt/oak-shared/venv/bin/pip install opencv-contrib-python==4.10.0.84
sudo /opt/oak-shared/venv/bin/pip install depthai==3.3.0
sudo /opt/oak-shared/venv/bin/pip install depthai-nodes==0.3.7
sudo /opt/oak-shared/venv/bin/pip install scipy==1.17.0
sudo /opt/oak-shared/venv/bin/pip install matplotlib==3.10.8
sudo /opt/oak-shared/venv/bin/pip install requests==2.26.0
sudo /opt/oak-shared/venv/bin/pip install Qt.py
```

### Shared Model Cache Setup

**Already configured on all three Pis!**

The shared model cache (`/opt/depthai-cache`) and environment variable (`DEPTHAI_ZOO_CACHE`) have been set up system-wide.

**For setup details** (instructors only), see `docs/archive/multi-user-setup.md` and `docs/archive/setup_shared_model_cache.sh`

### User Aliases (Optional but Recommended)

Add to all users' `~/.bashrc`:

```bash
# OAK-D shortcut to activate virtual environment
alias activate-oak='source /opt/oak-shared/venv/bin/activate'
```

---

## Key Version Compatibility Notes

### ✅ What Works

1. **opencv-contrib-python 4.10.0.84**
   - Has GUI support (cv2.namedWindow, cv2.imshow work)
   - Has all required attributes (cv2.COLORMAP_JET, etc.)
   - Compatible with depthai-nodes 0.3.7

2. **depthai 3.3.0 + depthai-nodes 0.3.7**
   - Use DepthAI 3.x API (new architecture)
   - Compatible with `ParsingNeuralNetwork` nodes
   - Works with Luxonis Hub models (e.g., `luxonis/yolov6-nano:r2-coco-512x288`)

3. **numpy 1.26.4**
   - Version constraint: `<2.0.0` required by depthai
   - Later numpy 2.x versions cause compatibility issues

### ❌ What Doesn't Work

1. **opencv-python-headless**
   - Missing GUI functions (no cv2.namedWindow, cv2.imshow)
   - Use `opencv-contrib-python` instead

2. **opencv-contrib-python 4.5.5.62** (older version)
   - Has GUI support BUT missing attributes like `cv2.COLORMAP_JET`
   - Causes `AttributeError` with depthai-nodes

3. **depthai 2.30.0** (from official depthai demo repo)
   - Incompatible with depthai-nodes 0.3.7
   - Missing `dai.Color` attribute
   - Use depthai 3.x instead

4. **Prebuilt ARM wheels without system GTK libraries**
   - OpenCV requires GTK dev packages installed via apt
   - Must install `libgtk-3-dev` and `libgtk2.0-dev` first

---

## Known Issues & Solutions

### Issue: "module 'cv2' has no attribute 'COLORMAP_JET'"
**Solution:** Upgrade to opencv-contrib-python 4.10.0.84 (not 4.5.5.62)

### Issue: "The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support"
**Solution:** Install system GTK libraries:
```bash
sudo apt-get install -y libgtk-3-dev libgtk2.0-dev pkg-config
```

### Issue: "module 'depthai' has no attribute 'Color'"
**Solution:** Upgrade to depthai 3.3.0 (not 2.30.0)

### Issue: Qt platform plugin warning
```
qt.qpa.plugin: Could not find the Qt platform plugin "wayland"
```
**Impact:** Harmless warning, GUI still works. OpenCV falls back to X11.

### Issue: Permission denied when accessing /opt/oak-shared/venv
**Solution:** Run pip commands with sudo:
```bash
sudo /opt/oak-shared/venv/bin/pip install <package>
```

---

## Hardware Configuration

### Raspberry Pi 5 Specifications
- **Model:** Raspberry Pi 5 (16GB RAM)
- **OS:** Raspberry Pi OS (Bookworm) 64-bit
- **USB:** Use USB 3.0 (blue) ports for OAK-D camera
- **Power:** 27W official adapter or powered USB hub recommended

### OAK-D Camera
- **Connection:** USB 3.0 (USB-C on camera side)
- **Platform Detected:** RVC2 (Luxonis Myriad X VPU)
- **Power Draw:** High during AI inference (use quality cable)

### Display Setup
- **VNC:** Works with wayvnc (shows desktop of logged-in user)
- **Physical Monitor:** Works with lightdm + lightdm-gtk-greeter
- **Headless:** Fully supported (no display needed for production)

---

## Testing the Setup

### Verify OpenCV GUI Support

```bash
source /opt/oak-shared/venv/bin/activate
python3 -c "import cv2; cv2.namedWindow('test'); cv2.destroyWindow('test'); print('GUI works')"
```

Expected output: `GUI works` (no errors)

### Verify DepthAI Camera Connection

```bash
source /opt/oak-shared/venv/bin/activate
python3 -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
```

Expected output: Device info with mxid (camera serial number)

### Test Detection with Display

```bash
source /opt/oak-shared/venv/bin/activate
cd ~/oak-projects
python3 person_detector_with_display.py --display
```

Expected behavior:
- Window opens showing live camera feed
- Green bounding boxes around detected people
- Status text showing detection count
- Press 'q' to quit

---

## Multi-User Setup Notes

### Shared Resources
- **Virtual Environment:** `/opt/oak-shared/venv/` (read-only for students)
- **Model Cache:** `/opt/depthai-cache/` (777 permissions, all users can write)
- **Demo Repository:** `/opt/oak-shared/depthai/` (optional, for reference)

### Per-User Resources
- **Project Directory:** `~/oak-projects/` (each student has their own)
- **Detection Scripts:** Copied to each user's directory
- **Logs & Screenshots:** Saved to user's own directory

### VNC Behavior
- Only ONE user can own the desktop session at a time
- That user can VNC in as themselves
- Other users must SSH or wait for desktop to be free
- Physical monitor login = VNC login as that same user

---

## Script Compatibility

### person_detector.py (DepthAI 3.x)
- ✅ Uses `depthai 3.3.0`
- ✅ Uses `ParsingNeuralNetwork` from depthai-nodes
- ✅ Console-only output (no GUI)
- ✅ Works headless

### person_detector_with_display.py (DepthAI 3.x + GUI)
- ✅ Uses `depthai 3.3.0`
- ✅ Uses `opencv-contrib-python 4.10.0.84` for GUI
- ✅ Works with `--display` flag for live visualization
- ✅ Works without `--display` flag (headless mode)
- ✅ Saves screenshots with detection boxes every 5 seconds

### discord_bot.py
- ✅ Uses Discord API via `discord.py`
- ✅ Reads `camera_status.json` for detection state
- ✅ Serves screenshots via `!screenshot` command
- ✅ Runs independently of detector scripts

---

## Future Considerations

### Alternative: Luxonis Official Image
- Available at: https://drive.google.com/drive/folders/1O50jPpGj_82jkAokdrsG--k9OBQfMXK5
- Pre-configured for OAK cameras
- Would require re-creating all user accounts and SSH keys
- Current setup is working well, so not recommended unless starting fresh

### Upgrading Packages
- **Be cautious** when upgrading depthai or opencv
- Test on one user account first
- Document any version changes
- Check compatibility with depthai-nodes

### Adding New Students
```bash
# Add user with home directory
sudo useradd -m -s /bin/bash newstudent

# Set password
echo "newstudent:oak2026" | sudo chpasswd

# Create oak-projects directory
sudo mkdir -p /home/newstudent/oak-projects
sudo cp /home/carrie/oak-projects/*.py /home/newstudent/oak-projects/
sudo chown -R newstudent:newstudent /home/newstudent/oak-projects

# Add activate-oak alias
echo 'alias activate-oak="source /opt/oak-shared/venv/bin/activate"' | sudo tee -a /home/newstudent/.bashrc
```

---

## Contact & Support

**Repository:** https://github.com/[your-username]/smart-objects-cameras
**Luxonis Docs:** https://docs.luxonis.com
**DepthAI Forum:** https://discuss.luxonis.com

---

## Acknowledgments

Setup developed and tested February 2026 for Smart Objects course.
Special thanks to Claude Code for debugging OpenCV GUI compatibility issues.
