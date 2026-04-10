# OAK-D + Raspberry Pi 5 — Smart Objects Template

This is a **template project** for building Discord bots that communicate with Luxonis OAK-D cameras. Use this as a starting point to create your own smart object systems with computer vision and interactive communication.

**What's Included:**
- 👁️ Real-time person detection using YOLO
- 😴 Fatigue detection (eye tracking + head pose)
- 👀 Gaze estimation (where someone is looking)
- 📝 Whiteboard OCR reader
- 🤖 Discord bot with commands (!status, !detect, !screenshot)
- 📢 Automatic webhook notifications
- 🎯 Temporal smoothing for stable detection
- 🖼️ Live camera screenshots on demand

**Your instructor has pre-configured the Raspberry Pis** — you can connect and start experimenting immediately! This guide shows you how to use the template, then extend it for your own creative projects.

| Camera  | RAM  | Configuration | Hostname | Access Method                |
| ------- | ---- | ------------- | -------- | ---------------------------- |
| Orbit   | 16GB | Desktop + VNC | orbit    | SSH (key-based) + VNC Viewer |
| Gravity | 16GB | Desktop + VNC | gravity  | SSH (key-based) + VNC Viewer |
| Horizon | 16GB | Desktop + VNC | horizon  | SSH (key-based) + VNC Viewer |

**Note:** All three Raspberry Pis have VNC enabled, but only one user can hold the VNC desktop seat at a time. Multiple users can SSH in simultaneously.

---

## 🎒 Important for Students

**The GitHub repository stays on YOUR LOCAL COMPUTER.** You do NOT clone it onto the Raspberry Pi.

Instead, you:
1. 📁 Clone and work with the repo on your laptop
2. 📤 Copy only the Python files you need to the Pi using `scp`
3. 🔐 Create your `.env` file with Discord tokens on the Pi
4. ▶️ Run the scripts on the Pi

**See [WORKFLOW.md](docs/WORKFLOW.md) for complete instructions and examples.**

---

**Note for instructors:**
- If you need to set up new Pis from scratch, see [INITIAL_SETUP.md](docs/INITIAL_SETUP.md)
- For camera bot token reference, see [CAMERA_BOT_TOKENS.md](docs/CAMERA_BOT_TOKENS.md) (instructor use only)

---

## 📚 Table of Contents

### For Students

- **[STUDENT_QUICKSTART.md](docs/STUDENT_QUICKSTART.md)** - Start here! Quick setup and common commands
- **[WORKFLOW.md](docs/WORKFLOW.md)** - How to copy files from your laptop to the Pi
- **[CHEATSHEET.md](docs/CHEATSHEET.md)** - Quick command reference (print this!)
- **Claude Code** - AI coding assistant (see installation in [STUDENT_QUICKSTART.md](docs/STUDENT_QUICKSTART.md#using-claude-code-for-help))

> **Prefer slides?** [View all documentation as slides](https://kandizzy.github.io/smart-objects-cameras/)

### Getting Started

- [Part 1: Connecting to the Pis](#part-1-connecting-to-the-pis)
- [Part 2: Person Detection Script](#part-2-person-detection-script)
- [Part 3: Auto-Start on Boot (Future)](#part-3-auto-start-on-boot-future)
- [Part 4: Quick Reference](#part-4-quick-reference)

### Troubleshooting

- [Troubleshooting](#troubleshooting)

### Additional Guides

- [WiFi Network Management](docs/wifi-management.md) - Switching between home and classroom networks
- [Multi-User Access](docs/multi-user-access.md) - Collaborative work on shared Pis
- [Discord Integration](docs/discord-integration.md) - Webhook setup (simple, one-way)
- [Discord Bot Setup](docs/DISCORD_BOT_PLAN.md) - Full bot with commands (advanced, two-way)
- [VS Code Remote Development](#vs-code-remote-development) - Professional development environment

### Additional Resources

- [Next Steps](#next-steps)
- [INITIAL_SETUP.md](docs/INITIAL_SETUP.md) - For instructors: setting up new Pis from scratch
- [CHEATSHEET.md](docs/CHEATSHEET.md) - Quick command reference (print this!)
- [WORKING_VERSIONS.md](docs/WORKING_VERSIONS.md) - Package version compatibility
- [NEXT_IDEAS.md](docs/NEXT_IDEAS.md) - Project extension ideas
- [EQUIPMENT_LIST.md](docs/EQUIPMENT_LIST.md) - Hardware reference
- [CLAUDE.md](CLAUDE.md) - For AI assistant context only

---

## Part 1: Connecting to the Pis

The Pis are already configured and ready to use! Here's how to connect.

### Prerequisites

**On your computer:**

- SSH client (built into Mac/Linux, use PowerShell on Windows)
- [RealVNC Viewer](https://www.realvnc.com/en/connect/download/viewer/) (optional, for desktop Pi GUI access)
- [VS Code](https://code.visualstudio.com/) with Remote-SSH extension (recommended - see [VS Code Remote Development](#vs-code-remote-development))

**Network:**

- Your computer and the Pis must be on the same network
- The Pis should power on and connect to WiFi automatically

### SSH Connection (Terminal Access)

```bash
# Connect to any of the three cameras
ssh orbit
ssh gravity
ssh horizon
```

**Your instructor has configured SSH key-based authentication** with SSH config files, so you should connect automatically without entering a password!

**First time connecting?** You'll see a fingerprint verification prompt:

```
The authenticity of host 'orbit' can't be established.
ED25519 key fingerprint is SHA256:...
Are you sure you want to continue connecting (yes/no)?
```

Type `yes` and press Enter.

**Troubleshooting:** If you get "Host not found" or connection fails:
- Make sure the Pi is powered on and connected to the network
- Check your SSH config file (`~/.ssh/config`) has the correct IP addresses
- Ask your instructor for the Pi's IP address if needed

**Example SSH config file (`~/.ssh/config`):**

```
# Orbit - 16GB Pi
Host orbit
    HostName 10.1.x.x
    User your_username
    IdentityFile ~/.ssh/id_ed25519_smartobjects

# Gravity - 16GB Pi
Host gravity
    HostName 10.1.x.x
    User your_username
    IdentityFile ~/.ssh/id_ed25519_smartobjects

# Horizon - 16GB Pi
Host horizon
    HostName 10.1.x.x
    User your_username
    IdentityFile ~/.ssh/id_ed25519_smartobjects
```

**Note:** Replace `10.1.x.x` with the actual IP addresses provided by your instructor. The `.local` hostnames don't work reliably on this network, so use IP addresses instead.

### VNC Connection (Optional)

For graphical desktop access to any of the Pis:

1. Open **RealVNC Viewer** on your computer
2. Enter the hostname: `orbit`, `gravity`, or `horizon`
3. Enter the username and password (ask your instructor)
4. You should see the Pi desktop

**Note:** All three Raspberry Pis have VNC enabled, but only one user can hold the VNC desktop seat at a time. Multiple users can SSH in simultaneously.

### VS Code Remote SSH (Recommended)

The best way to code on the Pi is using VS Code Remote-SSH extension. See **[VS Code Remote Development](#vs-code-remote-development)** for complete setup instructions.

**Quick start:**
1. Install VS Code and the "Remote - SSH" extension
2. **macOS users:** Grant VS Code "Local Network" permission (System Settings → Privacy & Security → Local Network)
3. Connect: `Ctrl+Shift+P` → "Remote-SSH: Connect to Host" → `orbit` (or `gravity`, `horizon`)

---

## Part 2: Person Detection Script

The person detector script is already installed on the Pis. Here's how to use it.

### Understanding the Project Structure

Once you're connected via SSH (or VS Code), navigate to the project directory:

```bash
cd ~/oak-projects
ls -la
```

You should see:
```
/opt/oak-shared/
└── venv/                    # Shared Python virtual environment (all users)

~/oak-projects/              # Your personal project directory
├── person_detector.py       # Person detection (YOLO)
├── fatigue_detector.py      # Fatigue detection (EAR + head pose)
├── gaze_detector.py         # Gaze direction estimation
├── whiteboard_reader.py     # OCR text detection
├── discord_notifier.py      # Discord webhook module
├── discord_bot.py           # Discord bot for commands
├── discord_dm_notifier.py   # Personal DM notifications
├── utils/                   # Helper modules
├── depthai_models/          # Model YAML configurations
├── camera_status.json       # Status for bot (auto-generated)
├── .env                     # Environment variables
└── *.log                    # Detection logs (if --log used)
```

### Activating the Virtual Environment

**Before running any Python scripts**, activate the shared virtual environment:

```bash
activate-oak
```

Your prompt should change to show `(venv)`:
```
(venv) username@orbit:~/oak-projects $
```

### Running the Person Detector

```bash
# Basic detection (console output only)
python3 person_detector.py

# With video display (requires VNC or X11)
python3 person_detector.py --display

# With file logging
python3 person_detector.py --log

# Adjust sensitivity (0.0 - 1.0, default 0.5)
python3 person_detector.py --threshold 0.7

# Combine options
python3 person_detector.py --log --threshold 0.6
```

**Stop the script with:** `Ctrl+C`

### Understanding the Detection Script

The script uses the OAK-D camera to detect people in real-time using a neural network that runs directly on the camera's processor.

**Key features:**
- Detects people using YOLO v6 (from Luxonis Hub)
- Runs at ~15 FPS on the OAK-D's onboard processor
- Temporal smoothing (1.5s debounce) to prevent detection flickering
- Only logs when detection status changes (person detected ↔ no person)
- Optional video display with bounding boxes
- Optional logging to timestamped files
- Adjustable confidence threshold
- Discord notifications (webhooks and bot support)

**Package versions:** See [WORKING_VERSIONS.md](docs/WORKING_VERSIONS.md) for tested compatible versions.

**To view the full script:**
```bash
cat ~/oak-projects/person_detector.py
# Or open in VS Code for syntax highlighting
```

**To modify the script:**
- Make your own copy first: `cp person_detector.py person_detector_yourname.py`
- Edit with VS Code (recommended) or nano: `nano person_detector_yourname.py`
- Run your version: `python3 person_detector_yourname.py`

### Viewing Detection Logs

If you ran with `--log`, check the logs:

```bash
# List all log files
ls -lh ~/oak-projects/*.log

# View the most recent log
tail -f ~/oak-projects/person_detection_*.log

# View specific log with line numbers
cat -n ~/oak-projects/person_detection_20260201_143052.log
```

---

## Part 3: Auto-Start on Boot (Future)

**Currently NOT implemented** - We're still in the exploratory phase!

Right now, you need to manually run `person_detector.py` each time you want to use the camera. This is intentional because we're still experimenting with different features and configurations.

### When Would You Use Auto-Start?

Once you're confident about what a camera should do permanently (for example, "Camera 1 should always detect people in the classroom entrance"), you could set it up to auto-start on boot using systemd services.

**Benefits of auto-start:**
- Camera begins detecting as soon as Pi powers on
- Automatically restarts if the script crashes
- Runs in background without keeping terminal open
- Useful for long-term deployments

**Why we're NOT using it now:**
- Still testing different detection settings
- Trying different models and thresholds
- Want flexibility to run different scripts
- Need to experiment without conflicting processes

### How to Set It Up (When Ready)

If you eventually want a camera to auto-start, see [INITIAL_SETUP.md](docs/INITIAL_SETUP.md) for systemd service configuration instructions.

**Basic concept:**
1. Create a systemd service file at `/etc/systemd/system/person-detector.service`
2. Configure it to run your script with specific flags
3. Enable it with `sudo systemctl enable person-detector`
4. Camera will auto-start on every boot

For now, just run scripts manually when you need them!

---

## WiFi & Multi-User Access

### Need to switch networks?

If you need to move your Pi between different WiFi networks (home ↔ classroom), see the complete guide:

**📡 [WiFi Network Management](docs/wifi-management.md)**

Quick command to add a new network:
```bash
ssh orbit
sudo nmtui  # Text menu to add WiFi networks
```

---

### Working with teammates?

Multiple students can access the same Pi simultaneously! The camera automatically announces who's using it via Discord.

**👥 [Multi-User Access Guide](docs/multi-user-access.md)**

Key points:
- Only one person runs `person_detector.py` at a time
- Everyone can SSH in and edit code simultaneously via VS Code
- Camera auto-announces via Discord when someone starts/stops

---

### Want Discord notifications?

Set up webhooks or an interactive bot to get camera alerts and control detection remotely.

**🤖 [Discord Integration Guide](docs/discord-integration.md)**

Features:
- Real-time detection alerts
- Interactive commands (!status, !screenshot, !detect)
- Multi-camera coordination

---

## Part 4: Quick Reference

### Useful Commands

```bash
# Activate shared virtual environment
activate-oak

# Check camera connection (depthai 3.x)
python3 -c "import depthai as dai; devices = dai.Device.getAllAvailableDevices(); print(f'Found {len(devices)} camera(s)')"

# Quick camera test with device info
python3 -c "import depthai as dai; device = dai.Device(); print(f'Device: {device.getDeviceId()}')"

# Monitor system resources
htop

# View detection logs (if using --log flag)
ls -la ~/oak-projects/*.log
tail -f ~/oak-projects/person_detection_*.log
```

### Network Access

| Camera  | Hostname | SSH Command  | VNC     |
| ------- | -------- | ------------ | ------- |
| Orbit   | orbit    | `ssh orbit`  | `orbit` |
| Gravity | gravity  | `ssh gravity`| `gravity` |
| Horizon | horizon  | `ssh horizon`| `horizon` |

**Note:** SSH uses key-based authentication via SSH config (no password needed if configured). All three Pis have VNC, but only one user can hold the desktop seat at a time.

### File Locations

```
/opt/oak-shared/
└── venv/                    # Shared Python virtual environment (all users)

~/oak-projects/              # Your personal project directory
├── person_detector.py       # Person detection (YOLO)
├── fatigue_detector.py      # Fatigue detection (EAR + head pose)
├── gaze_detector.py         # Gaze direction estimation
├── whiteboard_reader.py     # OCR text detection
├── discord_notifier.py      # Discord webhook module
├── discord_bot.py           # Discord bot for commands
├── discord_dm_notifier.py   # Personal DM notifications
├── utils/                   # Helper modules
├── depthai_models/          # Model YAML configurations
├── camera_status.json       # Status for bot (auto-generated)
├── .env                     # Environment variables
└── *.log                    # Detection logs (if --log used)
```

---

## Troubleshooting

### Camera Not Found

```bash
# Check USB connection
lsusb | grep Myriad

# Should show something like:
# Bus 001 Device 002: ID 03e7:2485 Intel Movidius MyriadX

# If not found, try:
# 1. Unplug and replug the camera
# 2. Try a different USB port (use USB 3.0 blue ports)
# 3. Use a powered USB hub
# 4. Check udev rules are set up correctly
```

### Update Camera Firmware

If you're experiencing camera issues or want the latest features, update the firmware:

```bash
# Activate venv
activate-oak

# Check current version
python3 -c "import depthai as dai; device = dai.Device(); print(f'Bootloader: {device.getBootloaderVersion()}')"
```

**Note for USB OAK-D cameras:** USB cameras boot from the host and typically show `None` for bootloader version - this is normal and expected. The depthai library manages bootloader compatibility automatically. Firmware updates are typically only needed for PoE cameras. See [INITIAL_SETUP.md](docs/INITIAL_SETUP.md) for detailed firmware information.

### Permission Denied

```bash
# Reapply udev rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Or add user to video group
sudo usermod -aG video $USER
# Then logout and back in
```

### VNC Black Screen

```bash
# Set a resolution in config
sudo raspi-config
# Display Options → VNC Resolution → 1920x1080

# Or edit config.txt directly
sudo nano /boot/firmware/config.txt
# Add:
# hdmi_force_hotplug=1
# hdmi_group=2
# hdmi_mode=82

sudo reboot
```

### Out of Memory (unlikely with your RAM)

```bash
# Check memory usage
free -h

# If needed, increase swap (not usually necessary with 8-16GB)
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Script Crashes on Startup

```bash
# Run manually to see error messages
activate-oak
python3 ~/oak-projects/person_detector.py

# If it crashes, check:
# 1. Is the camera connected? (lsusb | grep Myriad)
# 2. Is the virtual environment activated? (should see (venv) in prompt)
# 3. Are all dependencies installed?
```

---

## VS Code Remote Development

VS Code Remote SSH is the **recommended** way to develop on the Raspberry Pi. This gives you a professional development environment on your laptop while the code executes on the Pi.

### What You'll Get

- Full VS Code editor running on your laptop
- Files stored and code executed on the Pi
- Integrated terminal (no separate SSH window needed)
- IntelliSense, syntax highlighting, debugging
- Works great over slow WiFi

---

### Step 1: Install VS Code

Download and install VS Code on your computer:

👉 **https://code.visualstudio.com/download**

Available for Windows, macOS, and Linux.

---

### Step 2: Install the Remote SSH Extension

1. Open VS Code
2. Click the **Extensions** icon in the left sidebar (or press `Ctrl+Shift+X` / `Cmd+Shift+X`)
3. Search for: `Remote - SSH`
4. Click **Install** on "Remote - SSH" by Microsoft

---

### Step 3: Grant Network Permission (macOS Only)

**IMPORTANT for Mac users:** VS Code needs permission to access your local network.

1. Open **System Settings** (Apple menu → System Settings)
2. Go to **Privacy & Security**
3. Scroll down and click **Local Network**
4. Find **Visual Studio Code** in the list
5. **Toggle it ON** ✅

If you don't see Visual Studio Code:
- Try connecting once (it will fail)
- The app will appear in the list
- Enable it and try again

**This is required!** Without this permission, you'll get "No route to host" errors even though terminal SSH works fine.

---

### Step 4: Connect to the Raspberry Pi

#### Using the Command Palette

1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. Type: `Remote-SSH: Connect to Host`
3. If your SSH config is already set up, you should see `orbit`, `gravity`, and `horizon` in the list - just select one
4. Otherwise, click **+ Add New SSH Host** and enter:
   ```
   ssh orbit
   ```
   (or `gravity` or `horizon`)
5. Select your SSH config file (usually the first option)
6. Click **Connect** in the popup

#### First Connection

On first connect:

1. Select **Linux** when asked about the platform
2. Enter your password when prompted (if not using SSH keys)
3. Wait while VS Code installs its server component on the Pi (1-2 minutes)

You'll know you're connected when the bottom-left corner shows:

```
>< SSH: orbit
```
(or `gravity` or `horizon`, depending on which you connected to)

---

### Step 5: Open the Project Folder

1. Click **File** → **Open Folder** (or `Ctrl+K Ctrl+O`)
2. Navigate to `/home/[username]/oak-projects`
3. Click **OK**
4. Trust the folder when prompted

You should now see the project files in the Explorer sidebar.

---

### Step 6: Set Up the Python Environment

#### Install the Python Extension

1. Go to Extensions (`Ctrl+Shift+X`)
2. Search for: `Python`
3. Install "Python" by Microsoft

#### Select the Virtual Environment

1. Open any `.py` file
2. Look at the bottom status bar — click where it shows the Python version
3. Select: `/opt/oak-shared/venv/bin/python`

Or use Command Palette:

1. `Ctrl+Shift+P` → `Python: Select Interpreter`
2. Choose: `/opt/oak-shared/venv/bin/python`

---

### Step 7: Using the Integrated Terminal

Open a terminal inside VS Code:

- Press `` Ctrl+` `` (backtick), or
- **Terminal** → **New Terminal**

The terminal is already connected to the Pi! Activate the virtual environment:

```bash
activate-oak
```

Now you can run scripts:

```bash
python3 person_detector.py
```

---

### Step 8: Running and Debugging Code

#### Quick Run

- Open a Python file
- Click the **▶ Play** button in the top-right corner
- Or press `F5` to run with debugging

#### Run in Terminal

- Right-click in the editor
- Select **Run Python File in Terminal**

#### Debugging

1. Click the **Run and Debug** icon in the sidebar (or `Ctrl+Shift+D`)
2. Click **create a launch.json file**
3. Select **Python File**
4. Set breakpoints by clicking left of line numbers
5. Press `F5` to start debugging

---

### VS Code Tips and Tricks

#### Useful Keyboard Shortcuts

| Action          | Windows/Linux  | Mac           |
| --------------- | -------------- | ------------- |
| Command Palette | `Ctrl+Shift+P` | `Cmd+Shift+P` |
| Open Terminal   | `` Ctrl+` ``   | `` Cmd+` ``   |
| Open File       | `Ctrl+P`       | `Cmd+P`       |
| Find in Files   | `Ctrl+Shift+F` | `Cmd+Shift+F` |
| Toggle Sidebar  | `Ctrl+B`       | `Cmd+B`       |
| Run Code        | `F5`           | `F5`          |
| Save            | `Ctrl+S`       | `Cmd+S`       |

#### Recommended Extensions

Install these on the **remote** (they'll install on the Pi):

| Extension      | Purpose                    |
| -------------- | -------------------------- |
| Python         | Python language support    |
| Pylance        | Better Python IntelliSense |
| GitLens        | Git integration            |
| Error Lens     | Inline error highlighting  |
| indent-rainbow | Visualize indentation      |

---

### Working with Multiple Students

#### Avoid File Conflicts

If multiple people connect to the same Pi:

- **Don't edit the same file simultaneously** — VS Code doesn't merge changes
- Create your own branch or subfolder for experiments
- Communicate with your team!

#### Suggested Workflow

```bash
# Create your own working copy
cd ~/oak-projects
cp person_detector.py person_detector_yourname.py

# Edit your copy
code person_detector_yourname.py
```

---

### VS Code Troubleshooting

#### "No route to host" (macOS) - MOST COMMON

**Problem:** VS Code can't access local network even though terminal SSH works.

**Solution:** Grant VS Code Local Network permission:
1. Open **System Settings** → **Privacy & Security** → **Local Network**
2. Find **Visual Studio Code** and toggle it **ON** ✅
3. Restart VS Code completely
4. Try connecting again

This is **required on macOS** - VS Code needs explicit permission to access local IP addresses (192.168.x.x).

#### "Could not establish connection"

1. Make sure the Pi is powered on and booted
2. Check you're on the same network
3. Verify your SSH config has the correct IP address for the host
4. Ask your instructor if the IP address has changed

#### "Permission denied (publickey,password)"

- Double-check your password
- Make sure SSH is enabled on the Pi:
  ```bash
  sudo raspi-config
  # Interface Options → SSH → Enable
  ```

#### Extensions Not Working

Remote extensions install on the Pi, not your laptop. If an extension isn't working:

1. Open Extensions sidebar
2. Look for the extension
3. Check if it says "Install in SSH: orbit" (or gravity/horizon)
4. Click to install it on the remote

#### Terminal Shows Wrong Python

Make sure you:

1. Activated the venv: `activate-oak`
2. Selected the correct interpreter in VS Code (bottom status bar)

#### Slow Performance

- Use **wired Ethernet** if possible
- Close unnecessary VS Code extensions
- Reduce the number of open files/tabs

---

## Next Steps

Once basic detection is working, you might want to explore:

1. **Discord Integration** — Set up webhooks or interactive bot
   - See [Discord Integration Guide](docs/discord-integration.md) for complete setup
   - Get real-time detection alerts and control cameras with commands

2. **Different models** — Try other YOLO models from Luxonis Hub
   - Browse models at https://models.luxonis.com
   - Change model with `--model` argument
   - Example: `python3 person_detector.py --model luxonis/yolov8-nano:r2-coco-640x640`

3. **Depth integration** — Get distance to detected persons
   - Use `depthai-nodes` spatial detection features
   - See [DepthAI 3.x Documentation](https://docs.luxonis.com/software-v3/depthai/)

4. **Recording** — Save video clips when people are detected
   - Use DepthAI 3.x VideoEncoder node
   - See examples in depthai-python repository

5. **Web dashboard** — Flask/FastAPI server to view status remotely

6. **Multiple cameras** — Run detection on multiple OAK-D cameras
   - Coordinate them via Discord commands
   - See [Discord Integration Guide](docs/discord-integration.md) for multi-camera ideas

**Resources for learning more:**
- **DepthAI 3.x Documentation**: https://docs.luxonis.com/software-v3/depthai/
- **depthai-nodes GitHub**: https://github.com/luxonis/depthai-nodes
- **Luxonis Hub Models**: https://models.luxonis.com
- **OAK Examples & Experiments**: https://github.com/luxonis/depthai-experiments
- **Tutorials**: https://docs.luxonis.com/software-v3/tutorials/

---

## Contributors

- [Your name here]
- [Student names]

## License

MIT License - Feel free to use and modify for educational purposes.
