# Development Workflow - Smart Objects Cameras

> **Prefer slides?** View the [Workflow Slides](https://kandizzy.github.io/smart-objects-cameras/workflow-slides.html) for a visual walkthrough.

## Overview

**GitHub repo lives on your LOCAL computer. Copy files to Pi as needed.**

```
┌─────────────────────────────────┐
│   YOUR LOCAL COMPUTER           │
│                                 │
│  📁 smart-objects-cameras/      │
│     ├── person_detector.py      │
│     ├── fatigue_detector.py     │
│     ├── gaze_detector.py        │
│     ├── discord_bot.py          │
│     ├── utils/                  │
│     ├── docs/                   │
│     └── README.md               │
│                                 │
│  ✏️  Edit code here             │
│  🧪 Test logic here             │
│  📝 Commit to GitHub here       │
└─────────────────────────────────┘
           │
           │  scp (copy files)
           ▼
┌─────────────────────────────────┐
│   RASPBERRY PI (orbit/gravity/  │
│   horizon)                      │
│                                 │
│  📁 ~/oak-projects/             │
│     ├── person_detector.py ◄─── Only files you need
│     ├── discord_notifier.py     │
│     ├── .env ◄───────────────── Created on Pi, never in GitHub!
│     ├── latest_frame.jpg        │
│     └── camera_status.json      │
│                                 │
│  ▶️  Run code here              │
│  📸 Camera connected here       │
└─────────────────────────────────┘
```

---

## Step-by-Step Workflow

### 1. Initial Setup (Once)

**On your local computer:**
```bash
# Clone the repository
git clone https://github.com/[your-org]/smart-objects-cameras.git
cd smart-objects-cameras
```

**On the Raspberry Pi:**
```bash
# SSH into Pi
ssh orbit   # or gravity, or horizon

# Create project directory
mkdir -p ~/oak-projects
```

---

### 2. Copy Files to Pi

**From your local computer** (not on the Pi!):

```bash
# Navigate to repo
cd ~/path/to/smart-objects-cameras

# Copy the files you need
scp person_detector_with_display.py orbit:~/oak-projects/
scp discord_notifier.py orbit:~/oak-projects/

# For fatigue/gaze, also copy utils
scp -r utils orbit:~/oak-projects/
```

**Replace `orbit` with `gravity` or `horizon` for other Pis.**

---

### 3. Create .env File (Once per Pi)

**⚠️ Important:** The camera bot tokens (OrbitBot, GravityBot, HorizonBot) are **already configured** on the Pis! You only need to add your personal DM bot token if you want private notifications.

**On the Raspberry Pi:**
```bash
# SSH in
ssh orbit

# Check if .env already exists (it should!)
ls ~/oak-projects/.env

# Edit the existing .env file
nano ~/oak-projects/.env
```

**If .env already exists**, add ONLY your personal bot token at the bottom:
```bash
# Personal DM Bot (optional - for private notifications to you only)
DISCORD_USER_ID=your_discord_user_id_here
DISCORD_DM_BOT_TOKEN=your_dm_bot_token_here
```

**What's already configured (DO NOT MODIFY):**
```bash
# These are already configured on each Pi - don't change them!
DISCORD_WEBHOOK_URL=...          # Shared webhook for class
DISCORD_APPLICATION_ID=...        # Camera bot config
DISCORD_PUBLIC_KEY=...            # Camera bot config
DISCORD_BOT_TOKEN=...             # OrbitBot/GravityBot/HorizonBot token
```

**If .env doesn't exist** (unlikely), create it with all tokens:
```bash
# Discord Webhook (for person detection --discord notifications)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_URL_HERE

# Discord Bot Configuration (for public camera bot - ask instructor)
DISCORD_APPLICATION_ID=ask_instructor
DISCORD_PUBLIC_KEY=ask_instructor
DISCORD_BOT_TOKEN=ask_instructor

# Personal DM Bot (optional - for private notifications)
DISCORD_USER_ID=your_discord_user_id_here
DISCORD_DM_BOT_TOKEN=your_dm_bot_token_here
```

Save (`Ctrl+O`, Enter, `Ctrl+X`) and secure:
```bash
chmod 600 ~/oak-projects/.env
```

**⚠️ NEVER commit .env to GitHub!** It contains secret tokens.

---

### 4. Run Your Code

**On the Raspberry Pi:**
```bash
# SSH in
ssh orbit

# Activate environment
activate-oak

# Navigate to project
cd ~/oak-projects

# Run your script
python3 person_detector_with_display.py --display
```

---

### 5. Making Changes

**Edit code on your LOCAL computer:**
```bash
# On your laptop
cd ~/path/to/smart-objects-cameras
nano person_detector_with_display.py   # or use VS Code
# Make your changes...
```

**Copy updated file to Pi:**
```bash
# Still on your local computer
scp person_detector_with_display.py orbit:~/oak-projects/
```

**Run updated code on Pi:**
```bash
# SSH to Pi
ssh orbit

# Run updated script
activate-oak
cd ~/oak-projects
python3 person_detector_with_display.py --display
```

---

## File Dependencies

### Person Detection
**Copy these:**
- `person_detector_with_display.py` (or `person_detector.py`)
- `discord_notifier.py` (if using --discord flag)

### Fatigue Detection
**Copy these:**
- `fatigue_detector.py`
- `utils/` folder (entire directory)
- `discord_notifier.py` (if using Discord)

### Gaze Estimation
**Copy these:**
- `gaze_detector.py`
- `utils/` folder (entire directory)
- `discord_notifier.py` (if using Discord)

### Discord Bots
**Copy these:**
- `discord_bot.py` (public camera bot)
- `discord_dm_notifier.py` (personal DM bot)

---

## Common Mistakes

### ❌ Cloning repo on the Pi
```bash
# DON'T DO THIS ON THE PI:
git clone https://github.com/.../smart-objects-cameras.git
```
**Why:** Unnecessary files, risk of committing .env to GitHub

### ❌ Forgetting to copy updated files
```bash
# You edit on local computer...
# ... but forget to scp to Pi
# ... Pi still runs old version!
```
**Fix:** Always `scp` after editing

### ❌ Committing .env to GitHub
```bash
git add .env    # DON'T!
```
**Why:** Contains secret tokens that would be publicly visible

---

## Quick Command Reference

### Copy single file:
```bash
scp file.py orbit:~/oak-projects/
```

### Copy entire folder:
```bash
scp -r utils orbit:~/oak-projects/
```

### Copy multiple files at once:
```bash
scp file1.py file2.py file3.py orbit:~/oak-projects/
```

### Check what's on the Pi:
```bash
ssh orbit "ls ~/oak-projects/"
```

### Check if you're on local computer or Pi:
```bash
hostname    # Shows 'orbit'/'gravity'/'horizon' on Pi
            # Shows your laptop name on local computer
```

---

## Alternative: VS Code Remote SSH

Instead of using `scp`, you can use VS Code Remote SSH to work directly on the Pi:

### Method 1: Edit Files Directly on Pi
1. Install "Remote - SSH" extension in VS Code
2. Connect to Pi: `Ctrl+Shift+P` → "Remote-SSH: Connect to Host" → `orbit`
3. Open folder: `/home/yourusername/oak-projects`
4. Edit files directly on Pi (they save immediately to the Pi)

**Pros:** No need to `scp` after every change - edits save directly to Pi
**Cons:** None! This is actually the recommended workflow

### Method 2: Drag and Drop Between Windows
1. Open two VS Code windows:
   - Window 1: Local repo on your laptop
   - Window 2: Remote connection to Pi (`~/oak-projects/`)
2. Drag files from local window to remote window
3. Or copy/paste files between windows

**This is often easier than `scp` commands!**

---

## Summary

| Action | Where | Command |
|--------|-------|---------|
| Clone repo | **Local computer** | `git clone ...` |
| Edit code | **Local computer** | Use your favorite editor |
| Copy files | **Local computer** | `scp file.py orbit:~/oak-projects/` |
| Create .env | **Raspberry Pi** | `nano ~/oak-projects/.env` |
| Run scripts | **Raspberry Pi** | `python3 script.py` |
| Commit changes | **Local computer** | `git add`, `git commit`, `git push` |

**Remember:** Repo on laptop, files on Pi, run on Pi!
