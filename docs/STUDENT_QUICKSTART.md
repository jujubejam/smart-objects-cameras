# Student Quick Start Guide

> **Prefer slides?** [View the slide version](https://kandizzy.github.io/smart-objects-cameras/student-quickstart-slides.html)

## 🎯 Quick Start Checklist

Follow these steps in order. Check them off as you complete them!

- [ ] **Step 1:** Clone the GitHub repo to your laptop ([jump to section](#important-github-repo-location))
- [ ] **Step 2:** Set up SSH access to your Pi ([jump to section](#setting-up-ssh-access))
- [ ] **Step 3:** Copy Python files from your laptop to the Pi ([jump to section](#step-1-copy-python-files-from-your-computer-to-the-pi))
- [ ] **Step 4:** Check/create your `.env` file on the Pi ([jump to section](#step-2-create-your-environment-file-env))
- [ ] **Step 5:** Run your first detector! ([jump to section](#running-fatigue-detection))
- [ ] **Step 6 (Optional):** Set up personal DM bot for notifications ([jump to section](#running-multiple-things-at-once-important))

**Estimated time:** 20-30 minutes for first-time setup

**Most common starting points:**

- ⭐ **Fatigue Detection** - Eye tracking with personal DM notifications
- ⭐ **Gaze Estimation** - See where someone is looking

---

## Before You Start: Terminal Basics

### How to Tell Where You Are (Local Computer vs. Raspberry Pi)

**Look at your terminal prompt!** It tells you where you are:

**On your local computer:**

```
yourname@your-laptop ~ %
```

or

```
yourname@MacBook-Pro ~ $
```

**SSH'd into a Raspberry Pi:**

```
yourname@orbit ~ $
```

or

```
yourname@gravity ~ $
```

or

```
yourname@horizon ~ $
```

**The key difference:** The hostname (after the `@`) changes!

- `your-laptop` or `MacBook-Pro` = you're on your **local computer**
- `orbit`, `gravity`, or `horizon` = you're **SSH'd into a Pi**

**Pro tip:** If you're ever confused, type `hostname` and press Enter. It will tell you exactly which machine you're on.

---

### Common Typing Mistakes: File Paths

**❌ WRONG - Adding spaces:**

```bash
~/ .ssh/ config          # NO! This is THREE separate things
nano ~ / .ssh / config   # NO! Spaces break the path
```

**✅ CORRECT - No spaces:**

```bash
~/.ssh/config            # YES! This is ONE path
nano ~/.ssh/config       # YES! No spaces in the path
```

**Remember:**

- File paths are **one continuous string** with no spaces
- The `/` forward slash separates folders, not spaces
- `~` means "my home directory" and goes directly before the next `/`

**More examples:**

```bash
# CORRECT ✅
cd ~/oak-projects
nano ~/.bashrc
cat ~/.ssh/id_ed25519_smartobjects.pub
source /opt/oak-shared/venv/bin/activate

# WRONG ❌
cd ~ / oak-projects              # Has spaces!
nano ~ / .bashrc                 # Has spaces!
cat ~ / .ssh / id_ed25519_smartobjects.pub  # Has spaces!
```

**If you see an error like "No such file or directory"**, check your path for extra spaces!

---

## ⚙️ What's Already Configured vs. What You Need to Do

**Already set up on the Pis (you don't need to touch these):**

- ✅ Camera bot tokens (OrbitBot, GravityBot, HorizonBot)
- ✅ Discord webhooks for class notifications
- ✅ Shared Python environment (`/opt/oak-shared/venv/`)
- ✅ All required libraries (depthai, opencv, discord.py)

**What you need to do:**

- 📋 Copy Python files from your laptop to the Pi
- 🔐 (Optional) Add your personal DM bot token to `.env` if you want private notifications
- ▶️ Run the scripts!

---

## Important: GitHub Repo Location

**The GitHub repository stays on YOUR LOCAL COMPUTER.** You do NOT clone it onto the Raspberry Pi.

### The Workflow:

1. 📁 **GitHub repo on your laptop** - Clone and work with the repo here
2. 📤 **Copy files to Pi** - Use `scp` to copy only the Python files you need
3. 🔐 **Create .env on Pi** - Add your Discord tokens on the Pi
4. ▶️ **Run on Pi** - Execute the scripts on the Raspberry Pi

### Why This Way?

- **Keep secrets safe:** Your `.env` file with tokens stays only on the Pi, never in GitHub
- **Lightweight:** Pi only has the files it needs to run, not the whole repo with docs
- **Easy updates:** When you modify code, just `scp` the updated file to the Pi
- **Multiple projects:** You can work on different detectors locally, copy what you want to test

**Bottom line:** Edit code on your laptop, copy files to Pi, run on Pi.

---

## If You Already Have the Repo: Update It!

**⚠️ Run these commands on YOUR LOCAL COMPUTER**

If you already cloned this repository before, you need to get the latest changes! The instructor has updated documentation and code.

**Here's how to update:**

```bash
# Navigate to where you cloned the repo
cd ~/path/to/smart-objects-cameras

# Make sure you're on the main branch
git checkout main

# Get the latest changes from GitHub
git pull
```

**What this does:**

- `git checkout main` - Switches to the main branch (if you're not already there)
- `git pull` - Downloads the latest changes from GitHub and updates your files

**If you see errors:**

**Error: "You have local changes"**

```
error: Your local changes to the following files would be overwritten by merge:
    some_file.py
Please commit your changes or stash them before you merge.
```

**Solution:** Save your changes temporarily:

```bash
# Save your changes
git stash

# Now pull the updates
git pull

# Get your changes back (if you want them)
git stash pop
```

**Error: "fatal: not a git repository"**
This means you're not in the right folder. Use `cd` to navigate to where you cloned the repo, or clone it fresh if you deleted it:

```bash
git clone https://github.com/[your-org]/smart-objects-cameras.git
cd smart-objects-cameras
```

**Pro tip:** Update your repo before each class to get the latest code and documentation!

---

## Setting Up Your Pi Project Folder

### Step 1: Copy Python Files from Your Computer to the Pi

**Note:** The `~/oak-projects/` folder already exists on each Pi. You just need to copy your files into it!

You have two options for getting files onto the Pi:

#### Method 1: Using `scp` (Copy from your local computer)

**⚠️ Run these commands on YOUR LOCAL COMPUTER (not on the Pi!)**

From your LOCAL computer terminal (not from the Pi!), use `scp` to copy files:

**For Fatigue Detection:** ⭐ Recommended

```bash
# From your local computer terminal (check prompt shows your-laptop!)
cd ~/path/to/smart-objects-cameras    # Navigate to where you cloned the repo

# Copy fatigue detector and its dependencies
scp fatigue_detector.py orbit:~/oak-projects/
scp -r utils orbit:~/oak-projects/
scp -r depthai_models orbit:~/oak-projects/
scp discord_notifier.py orbit:~/oak-projects/
scp discord_dm_notifier.py orbit:~/oak-projects/

# Replace 'orbit' with 'gravity' or 'horizon' for other Pis
```

**For Gaze Estimation:** ⭐ Recommended

```bash
# From your local computer
cd ~/path/to/smart-objects-cameras

# Copy gaze detector and its dependencies
scp gaze_detector.py orbit:~/oak-projects/
scp -r utils orbit:~/oak-projects/
scp -r depthai_models orbit:~/oak-projects/

# Optional: Copy these if you want to add Discord notifications (bonus challenge)
scp discord_notifier.py orbit:~/oak-projects/
scp discord_dm_notifier.py orbit:~/oak-projects/
```

**For Person Detection:** (Optional)

```bash
# From your local computer
cd ~/path/to/smart-objects-cameras

# Copy person detection files
scp person_detector_with_display.py orbit:~/oak-projects/
scp discord_notifier.py orbit:~/oak-projects/
```

**For Discord Bots:**

```bash
# From your local computer
cd ~/path/to/smart-objects-cameras

# Public bot (posts to main Smart Objects channel)
scp discord_bot.py orbit:~/oak-projects/

# DM bot (sends you private messages)
scp discord_dm_notifier.py orbit:~/oak-projects/
```

**Pro tip:** Check which Pi you're copying to! Replace `orbit` with `gravity` or `horizon` as needed.

**Note:** Focus on fatigue and gaze detection - they use your personal DM bot for notifications!

---

#### Method 2: Using VS Code Remote SSH (Recommended)

If you have VS Code with Remote-SSH extension installed:

1. **Connect to the Pi** via VS Code Remote SSH (see main README for setup)
2. **Open your local repo folder** on your laptop in a separate VS Code window
3. **Copy files** from local window to remote window:
   - Select files in local VS Code Explorer
   - Right-click → Copy
   - Switch to remote VS Code window (connected to Pi)
   - Navigate to `~/oak-projects/`
   - Right-click → Paste

Or simply **drag and drop** files from your local VS Code to the remote VS Code window!

**Advantage:** No need to remember `scp` commands, and you can edit files directly on the Pi.

---

### Step 2: Create Your Environment File (.env)

**⚠️ Run these commands on the RASPBERRY PI**

First, SSH into the Pi from your local computer, then check if `.env` already exists:

```bash
# From your local computer - connect to a Pi:
ssh orbit
```

or

```bash
ssh gravity
```

or

```bash
ssh horizon
```

Now you're on the Pi! Check if .env exists:

```bash
ls ~/oak-projects/.env
```

---

#### If `.env` Already Exists:

Great! The camera bot tokens are already configured. You can optionally add your personal DM bot token:

**⚠️ On the RASPBERRY PI:**

```bash
# Edit the existing .env file
nano ~/oak-projects/.env
```

Scroll to the bottom and add your personal token (optional - only if you want private notifications):

```bash
# Personal DM Bot (optional - for private notifications to you only)
DISCORD_USER_ID=your_discord_user_id_here
DISCORD_DM_BOT_TOKEN=your_dm_bot_token_here
```

**Save:** Press `Ctrl+O` then `Enter`, then `Ctrl+X` to exit.

---

#### If `.env` Does NOT Exist:

No problem! Your instructor will provide the camera bot tokens via a 1Password secure link.

**⚠️ On the RASPBERRY PI:**

**Create the file:**

```bash
# Create new .env file
nano ~/oak-projects/.env
```

**Add the camera bot configuration** (get these tokens from the 1Password link your instructor sends):

```bash
# Discord Webhook (shared by all - for person detection --discord notifications)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_URL_HERE

# Discord Bot Configuration (camera bot - unique per Pi)
DISCORD_APPLICATION_ID=get_from_instructor
DISCORD_PUBLIC_KEY=get_from_instructor
DISCORD_BOT_TOKEN=get_from_instructor_for_your_pi

# Students add their personal DM bot tokens below (optional)
# DISCORD_USER_ID=your_discord_user_id_here
# DISCORD_DM_BOT_TOKEN=your_dm_bot_token_here
```

**Important notes:**

- Each Pi (orbit, gravity, horizon) has a **different** `DISCORD_BOT_TOKEN` - make sure you use the correct one for your Pi!
- OrbitBot token → orbit Pi
- GravityBot token → gravity Pi
- HorizonBot token → horizon Pi

**Save and secure:**

- Press `Ctrl+O` then `Enter` to save
- Press `Ctrl+X` to exit
- Run: `chmod 600 ~/oak-projects/.env` (keeps it private)

---

**⚠️ Important:** Never commit `.env` to GitHub! It contains secret tokens.

---

## Setting Up SSH Access

**⚠️ On YOUR LOCAL COMPUTER**

Before you can connect to the Pis, make sure your SSH config is set up. Test it:

```bash
# Connect to one of the three cameras
ssh orbit
```

or

```bash
ssh gravity
```

or

```bash
ssh horizon
```

**If it works:** Skip to [Getting Started](#getting-started).

**If you get "Host not found":** Your SSH config needs the Pi IP addresses (10.1.x.x). See the [main README - Part 1: Connecting to the Pis](../README.md#part-1-connecting-to-the-pis) for setup instructions, or ask your instructor for the IP addresses.

---

## Getting Started

### 1. Log In

- **Physical Monitor:** Log in with your username at the connected monitor
- **VNC:** Use VNC Viewer to connect to `orbit` (or `gravity`, `horizon`)
  - Username: your assigned username (e.g., `juju`, `akentou`)
  - Password: `oak2026`

> **Note:** Only one person can use VNC/monitor at a time. If someone else is logged in, you'll need to wait or use SSH.

### 2. Activate the Virtual Environment

**⚠️ On the RASPBERRY PI:**

Open Terminal and type:

```bash
activate-oak
```

You should see `(venv)` appear before your prompt. This means you're ready!

### 3. Navigate to Your Project Folder

**⚠️ On the RASPBERRY PI:**

```bash
cd ~/oak-projects
```

---

## Understanding Python Flags

### What Are Flags?

When running Python scripts, you'll see **flags** (also called **command-line arguments**) that start with `--`:

```bash
python3 gaze_detector.py --display --log
                         ^^^^^^^^^ ^^^^^
                         These are flags!
```

**Flags are optional settings** that change how the script runs. You can combine multiple flags!

### Common Flags in This Project

| Flag                  | What It Does                                 | Example                                                   |
| --------------------- | -------------------------------------------- | --------------------------------------------------------- |
| `--display`           | Shows live video window with detection boxes | `python3 gaze_detector.py --display`                      |
| `--log`               | Saves detection events to a log file         | `python3 fatigue_detector.py --log`                       |
| `--discord`           | Sends notifications to Discord class channel | `python3 person_detector_with_display.py --discord`       |
| `--threshold <value>` | Changes detection sensitivity (0.0-1.0)      | `python3 person_detector_with_display.py --threshold 0.7` |

### Important: When to Use `--display`

**The `--display` flag only works when you can see a graphical desktop:**

✅ **Use `--display` when:**

- Connected via **VNC** (you can see the desktop)
- Using a **physical monitor** connected to the Pi
- Working directly at the desktop

❌ **Don't use `--display` when:**

- SSH'd into the Pi from terminal (you won't see the window!)
- Running scripts remotely without VNC

**Example:**

```bash
# ✅ Good - you're using VNC or at the physical monitor
python3 gaze_detector.py --display

# ❌ Won't work - you're SSH'd in from terminal without display
# You'll get an error or just not see the window
```

**If you're SSH'd in without VNC:** Run the script **without** `--display`, and use Discord notifications or log files to monitor what's happening.

**Combining flags:**

```bash
# Multiple flags work together!
python3 fatigue_detector.py --display --log
python3 gaze_detector.py --display --log
```

---

## Running Fatigue Detection

Fatigue detection monitors eye aspect ratio (EAR) to detect if someone is getting drowsy. This uses your **personal DM bot** for notifications!

### With Visual Display (at desktop/VNC)

**⚠️ On the RASPBERRY PI:**

```bash
activate-oak
cd ~/oak-projects
python3 fatigue_detector.py --display
```

**What you'll see:**

- Live camera feed with face landmarks
- Eye aspect ratio (EAR) values displayed
- Green boxes around detected faces
- Alerts when fatigue is detected
- Press **'q'** to quit (or Ctrl+C in terminal)

### Without Display (SSH only)

**⚠️ On the RASPBERRY PI:**

```bash
activate-oak
cd ~/oak-projects
python3 fatigue_detector.py
```

**What happens:**

- Runs in the background
- Sends detection events to your personal DM bot
- Check your Discord DMs for notifications
- Press **Ctrl+C** to stop

### With Logging

**⚠️ On the RASPBERRY PI:**

```bash
python3 fatigue_detector.py --display --log
```

Creates a timestamped log file with all detection events.

---

## Running Gaze Estimation

Gaze estimation detects where someone is looking and outputs the direction to the terminal.

### With Visual Display (at desktop/VNC)

**⚠️ On the RASPBERRY PI:**

```bash
activate-oak
cd ~/oak-projects
python3 gaze_detector.py --display
```

**What you'll see:**

- Live camera feed with face detection
- Gaze vectors showing where person is looking
- Head pose estimation
- Press **'q'** to quit (or Ctrl+C in terminal)

### Without Display (SSH only)

**⚠️ On the RASPBERRY PI:**

```bash
activate-oak
cd ~/oak-projects
python3 gaze_detector.py
```

**What happens:**

- Runs in the background
- Outputs gaze direction to terminal (e.g., "looking left", "looking right")
- Press **Ctrl+C** to stop

### With Logging

**⚠️ On the RASPBERRY PI:**

```bash
python3 gaze_detector.py --display --log
```

Creates a log file with gaze direction data.

---

### 🎯 Bonus Challenge: Add Discord Notifications

Currently, gaze detection only outputs to the terminal. **Want a challenge?** Use Claude Code to add Discord notifications!

**Ideas to explore:**

- What gaze information is actually useful to send to Discord?
- Should it notify on every direction change, or only certain events?
- How can you filter out noise to avoid spamming notifications?
- Can you make it send a summary (e.g., "User looked left 15 times, right 8 times")?

**How to start:**

```bash
claude
> I want to add Discord notifications to gaze_detector.py.
  What information would be useful to send, and how should I implement it?
```

This is a great way to practice using Claude Code to extend existing code!

---

## Running Person Detection (Optional)

Person detection uses the **class camera bots** (OrbitBot, GravityBot, HorizonBot) for notifications, not your personal DM bot.

**⚠️ On the RASPBERRY PI:**

**Basic usage:**

```bash
activate-oak
cd ~/oak-projects

# With display (at desktop/VNC)
python3 person_detector_with_display.py --display

# Without display (SSH only)
python3 person_detector_with_display.py

# With Discord class notifications
python3 person_detector_with_display.py --discord
```

**Camera Bot Commands:**

- `!ping` - Check if bot is alive
- `!status` - Check if camera is running
- `!screenshot` - Get current camera image
- `!detect` - Get detection status

---

## Running Multiple Things at Once (Important!)

**Problem:** If you want your Discord DM bot to notify you while you run detection, you need **two terminal windows** - one for each program.

### Why Two Windows?

When you run a script (like `python3 fatigue_detector.py`), it takes over that terminal window. You can't type new commands until it stops. So if you also want to run your Discord DM bot, you need a second window!

### Common Scenarios

#### Scenario 1: Fatigue Detection + Your Personal DM Bot

**Window 1 - Run fatigue detector:**

**⚠️ On YOUR LOCAL COMPUTER, SSH to the Pi, then run commands on the Pi:**

```bash
# From local computer - connect to a Pi:
ssh orbit
```

or

```bash
ssh gravity
```

or

```bash
ssh horizon
```

Now you're on the Pi! Run the detector:

```bash
# Now on Pi - run the detector:
activate-oak
cd ~/oak-projects
python3 fatigue_detector.py --display
```

This window shows fatigue detection logs and stays open.

**Window 2 - Run your DM bot:**

**⚠️ On YOUR LOCAL COMPUTER, SSH to the Pi, then run commands on the Pi:**

```bash
# From local computer - connect to a Pi:
ssh orbit
```

or

```bash
ssh gravity
```

or

```bash
ssh horizon
```

Now you're on the Pi! Run the DM bot:

```bash
# Now on Pi - run the DM bot:
activate-oak
cd ~/oak-projects
python3 discord_dm_notifier.py
```

This window shows bot logs and stays open.

Now both are running! The detector sends fatigue alerts to your DMs.

---

#### Scenario 2: Gaze Detection (Single Window)

**Window 1 - Run gaze detector:**

**⚠️ On YOUR LOCAL COMPUTER, SSH to the Pi, then run commands on the Pi:**

```bash
# From local computer - connect to a Pi:
ssh orbit
```

or

```bash
ssh gravity
```

or

```bash
ssh horizon
```

Now you're on the Pi! Run the detector:

```bash
# Now on Pi - run the detector:
activate-oak
cd ~/oak-projects
python3 gaze_detector.py --display
```

**Note:** Gaze detection currently only outputs to the terminal. You only need one window!

**Bonus:** If you add Discord notifications (see the bonus challenge above), you'll need a second window for your DM bot.

---

#### Scenario 3: Person Detection + Your Personal DM Bot (Optional)

**Window 1 - Run the detector:**

**⚠️ On YOUR LOCAL COMPUTER, SSH to the Pi, then run commands on the Pi:**

```bash
# From local computer - connect to a Pi:
ssh orbit
```

or

```bash
ssh gravity
```

or

```bash
ssh horizon
```

Now you're on the Pi! Run the detector:

```bash
# Now on Pi - run the detector:
activate-oak
cd ~/oak-projects
python3 person_detector_with_display.py
```

**Window 2 - Run your DM bot:**

**⚠️ On YOUR LOCAL COMPUTER, SSH to the Pi, then run commands on the Pi:**

```bash
# From local computer - connect to a Pi:
ssh orbit
```

or

```bash
ssh gravity
```

or

```bash
ssh horizon
```

Now you're on the Pi! Run the DM bot:

```bash
# Now on Pi - run the DM bot:
activate-oak
cd ~/oak-projects
python3 discord_dm_notifier.py
```

---

### How to Open Multiple Terminal Windows

**On Mac:**

- Open Terminal
- Press `Cmd+T` for a new tab, OR
- Press `Cmd+N` for a new window
- SSH into the Pi in each tab/window

**On Windows:**

- Open Command Prompt or PowerShell
- Click the + button for a new tab, OR
- Open another window
- SSH into the Pi in each tab/window

**On Linux:**

- Open your terminal
- Press `Ctrl+Shift+T` for a new tab, OR
- Right-click terminal icon → New Window
- SSH into the Pi in each tab/window

**In VS Code:**

- Open Terminal panel (`` Ctrl+` ``)
- Click the `+` button in terminal panel for additional terminal
- OR click the split terminal button (icon looks like two panels)

---

### Stopping Both Programs

**Important:** Press `Ctrl+C` in EACH window to stop each program.

**Example:**

1. Go to Window 1 (detector) → Press `Ctrl+C`
2. Go to Window 2 (DM bot) → Press `Ctrl+C`

Both programs are now stopped.

---

### Do I Always Need Two Windows?

**No!** You only need two windows if you want TWO things running at the same time.

**One window is fine for:**

- ✅ Just running person detection
- ✅ Just running fatigue detection
- ✅ Just testing the camera
- ✅ Just running the public camera bot (OrbitBot/GravityBot/HorizonBot)

**Two windows needed for:**

- 🔄 Running a detector AND your personal DM bot
- 🔄 Running multiple test scripts simultaneously
- 🔄 Monitoring logs while running another command

---

### Quick Reference: What Runs Where?

| What You Want                                   | Window 1                                            | Window 2                                    |
| ----------------------------------------------- | --------------------------------------------------- | ------------------------------------------- |
| **Fatigue detection with your personal DMs** ⭐ | `python3 fatigue_detector.py --display`             | `python3 discord_dm_notifier.py`            |
| **Gaze detection** ⭐                           | `python3 gaze_detector.py --display`                | _(not needed - outputs to terminal)_        |
| **Person detection with your personal DMs**     | `python3 person_detector_with_display.py`           | `python3 discord_dm_notifier.py`            |
| **Person detection with class notifications**   | `python3 person_detector_with_display.py --discord` | _(not needed - uses class webhook)_         |
| **Public camera bot (OrbitBot, etc.)**          | `python3 discord_bot.py`                            | _(not needed unless also running detector)_ |

**Tips:**

- ⭐ **Fatigue detection** with your personal DM bot is the recommended starting point!
- **Gaze detection** currently only outputs to terminal (adding Discord notifications is a bonus challenge)
- The class webhook (used with `--discord` flag) doesn't need a separate bot window - it's built into the detector

---

### VS Code Split Terminal (Easy Way!)

If you're using VS Code Remote SSH, you can split the terminal:

1. Open terminal (`` Ctrl+` ``)
2. Click the **split terminal** icon (looks like two panels side by side)
3. Now you have two terminal panes in one window!
4. Run detector in left pane, bot in right pane

**Super convenient!** Both visible at once, no window switching needed.

---

## Common Commands

### Check if Camera is Connected

**⚠️ On the RASPBERRY PI:**

```bash
activate-oak
python3 -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
```

You should see your camera's serial number.

### Test GUI Support

**⚠️ On the RASPBERRY PI:**

```bash
activate-oak
python3 -c "import cv2; cv2.namedWindow('test'); cv2.destroyWindow('test'); print('GUI works')"
```

Should print: `GUI works`

### View Your Files

**⚠️ On the RASPBERRY PI:**

```bash
ls ~/oak-projects
```

You should see:

- `person_detector_with_display.py` - Your main script
- `discord_notifier.py` - Discord integration
- `latest_frame.jpg` - Most recent screenshot
- `camera_status.json` - Current detection status

---

## Troubleshooting

### "I don't know if I'm on my computer or the Pi!"

**Solution:** Look at your terminal prompt (the text before the `$` or `%`):

- If it says `yourname@orbit` or `yourname@gravity` or `yourname@horizon` → You're on a **Pi**
- If it says `yourname@your-laptop` or `yourname@MacBook-Pro` → You're on your **local computer**

**Quick check:** Type `hostname` and press Enter. It tells you exactly where you are!

### "No such file or directory" when typing paths

**Most common cause:** You added **spaces** in your file path!

**Check for spaces:**

```bash
# WRONG ❌ - has spaces
cd ~ / oak-projects

# CORRECT ✅ - no spaces
cd ~/oak-projects
```

**Remember:** File paths are one continuous string. The `/` separates folders, not spaces!

### "Command not found: activate-oak"

**⚠️ On the RASPBERRY PI:**

You need to reload your bash configuration:

```bash
source ~/.bashrc
```

Then try `activate-oak` again.

### "No module named 'depthai'"

**⚠️ On the RASPBERRY PI:**

Make sure the virtual environment is activated:

```bash
activate-oak
```

You should see `(venv)` in your prompt.

### "Permission denied" errors

You don't have sudo access. Ask the instructor for help.

### Camera not found

Check:

1. Is the USB cable plugged into a **blue** USB 3.0 port?
2. Is someone else using the camera right now?
3. Try unplugging and replugging the camera

### Black screen in display window

Wait a few seconds - frames take a moment to start flowing. If it stays black for more than 10 seconds, press Ctrl+C and try again.

### VNC shows grey screen or won't connect

Someone else is probably logged in at the monitor. Either:

- Wait for them to log out
- Use SSH instead: `ssh orbit` (if your SSH config is set up)
- Ask them to share the desktop

### "I tried to run the script but the file doesn't exist"

**Most likely:** You forgot to copy the file from your local computer to the Pi!

**Check:**

1. Are you on the Pi? (`hostname` should show `orbit`, `gravity`, or `horizon`)
2. Is the file on the Pi? (`ls ~/oak-projects/`)
3. Did you copy it from your local computer? (Use `scp` from your laptop)

**Remember:** The GitHub repo is on your LOCAL computer. You must `scp` files to the Pi to use them.

### "ImportError: No module named 'utils'"

**Problem:** You copied `fatigue_detector.py` or `gaze_detector.py` but forgot to copy the `utils/` folder.

**⚠️ On YOUR LOCAL COMPUTER:**

**Solution:** From your LOCAL computer:

```bash
cd ~/path/to/smart-objects-cameras
scp -r utils orbit:~/oak-projects/
```

The `-r` flag copies the entire folder.

### "Model YAML file not found" or "depthai_models not found"

**Problem:** You copied `fatigue_detector.py`, `gaze_detector.py`, or `whiteboard_reader*.py` but forgot to copy the `depthai_models/` folder.

**⚠️ On YOUR LOCAL COMPUTER:**

**Solution:** From your LOCAL computer:

```bash
cd ~/path/to/smart-objects-cameras
scp -r depthai_models orbit:~/oak-projects/
```

**Note:** The `depthai_models` folder contains YAML configuration files that tell the camera which AI models to use. Without these files, the scripts can't load the detection models.

### "I made changes to my code but nothing changed when I ran it"

**Problem:** You edited the file on your LOCAL computer, but the Pi is still running the old version.

**⚠️ On YOUR LOCAL COMPUTER:**

**Solution:** Copy the updated file to the Pi again:

```bash
# From your LOCAL computer
scp person_detector_with_display.py orbit:~/oak-projects/
```

**⚠️ Then on the RASPBERRY PI:**

Then run it again on the Pi.

---

## Tips for Development

### Testing Changes Quickly

1. Make your code changes **on your local computer**
2. Copy to Pi with `scp` (**from local computer**)
3. Run with `--display` to see results live (**on the Pi**)
4. Press 'q' to quit
5. Make more changes
6. Repeat!

### Running Headless

**⚠️ On the RASPBERRY PI:**

Once your code works:

1. Run without `--display` flag
2. Monitor via Discord (`!screenshot`)
3. Let it run in the background
4. Great for long-term deployments

### Adjusting Detection Sensitivity

**⚠️ On the RASPBERRY PI:**

```bash
# Lower threshold = more sensitive (may have false positives)
python3 person_detector_with_display.py --display --threshold 0.3

# Higher threshold = less sensitive (fewer false positives)
python3 person_detector_with_display.py --display --threshold 0.7

# Default is 0.5
```

### Logging to File

**⚠️ On the RASPBERRY PI:**

```bash
python3 person_detector_with_display.py --display --log
```

Creates a timestamped log file like `person_detection_20260204_151030.log`

---

## Your Project Directory

After copying files, your `~/oak-projects/` folder on the Pi will look like this:

### If You're Running Fatigue Detection: ⭐ Recommended

```
~/oak-projects/
├── fatigue_detector.py              # Fatigue detection script
├── utils/                           # Helper modules (folder)
│   └── face_landmarks.py
├── depthai_models/                  # Model YAML files (folder)
│   ├── yunet.RVC2.yaml
│   └── mediapipe_face_landmarker.RVC2.yaml
├── discord_notifier.py              # Discord webhook helper
├── discord_dm_notifier.py           # Your personal DM bot
└── .env                             # Your tokens
```

### If You're Running Gaze Estimation: ⭐ Recommended

```
~/oak-projects/
├── gaze_detector.py                 # Gaze estimation script
├── utils/                           # Helper modules (folder)
│   ├── process_keypoints.py
│   ├── node_creators.py
│   └── host_concatenate_head_pose.py
├── depthai_models/                  # Model YAML files (folder)
│   ├── gaze_estimation_adas.RVC2.yaml
│   └── head_pose_estimation.RVC2.yaml
└── .env                             # Your tokens (already configured)

# Optional (for bonus challenge - adding Discord notifications):
├── discord_notifier.py              # Discord webhook helper
└── discord_dm_notifier.py           # Your personal DM bot
```

### If You're Running Person Detection: (Optional)

```
~/oak-projects/
├── person_detector_with_display.py  # Detection script
├── discord_notifier.py              # Discord webhook helper
├── .env                             # Your tokens (already configured)
├── camera_status.json               # Status (auto-generated)
└── latest_frame.jpg                 # Screenshot (auto-generated)
```

### If You're Running Discord Bots:

```
~/oak-projects/
├── discord_bot.py                   # Public camera bot (OrbitBot, etc.)
├── discord_dm_notifier.py           # Personal DM bot
└── .env                             # Your bot tokens
```

**Note:** The actual Python packages are in `/opt/oak-shared/venv/` (shared by everyone).

---

## Quick Reference: What Files Do I Need?

| What You Want To Run     | Files to Copy                                                                                                          | Additional Notes                                                                    |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Fatigue Detection** ⭐ | `fatigue_detector.py`<br>`utils/` folder<br>`depthai_models/` folder<br>`discord_notifier.py`<br>`discord_dm_notifier.py` | Recommended! Uses personal DM bot                                                   |
| **Gaze Estimation** ⭐   | `gaze_detector.py`<br>`utils/` folder<br>`depthai_models/` folder                                                      | Recommended! Terminal output only<br>_(Discord files optional for bonus challenge)_ |
| **Personal DM Bot**      | `discord_dm_notifier.py`                                                                      | Add your DISCORD_DM_BOT_TOKEN to .env                                               |
| **Person Detection**     | `person_detector_with_display.py`<br>`discord_notifier.py`                                    | Uses class camera bots                                                              |
| **Public Camera Bot**    | `discord_bot.py`                                                                              | Already configured! Just run it                                                     |

**⭐ Best for learning:**

- **Fatigue Detection** - Full Discord integration with your personal DM bot
- **Gaze Detection** - Terminal output (bonus: add Discord notifications yourself!)

**Remember:**

- `.env` file already exists on each Pi with camera bot tokens configured
- Add your personal DM bot token if using fatigue detection or bonus gaze features

---

## Need Help?

1. Check the main README: `~/oak-projects/README.md`
2. Check version compatibility: `~/oak-projects/WORKING_VERSIONS.md`
3. Ask your instructor
4. Check Luxonis docs: https://docs.luxonis.com

---

## Working Versions (for reference)

Your system uses these specific versions (tested and working):

- Python 3.13.5
- depthai 3.3.0
- opencv-contrib-python 4.10.0.84
- numpy 1.26.4

See `WORKING_VERSIONS.md` for complete list.

---

## 🤖 Using Claude Code for Help

Your instructor has provided you with access to Claude Code! You can use it in two ways: in VS Code or in the terminal.

### Option 1: Using Claude Code in the Terminal (Recommended)

**Why use the terminal?** Claude Code has access to ALL your project files, including `CLAUDE.md`, `README.md`, and documentation. It understands your entire project context!

#### Install Claude Code CLI

**⚠️ Run these commands on YOUR LOCAL COMPUTER (your laptop)**

First, install Claude Code on your laptop (do this once):

**For Mac/Linux users (Recommended - Native Install):**

```bash
# Install using the native installer (auto-updates!)
curl -fsSL https://claude.ai/install.sh | bash

# Verify installation:
claude --version
```

**For Mac users (Alternative - Homebrew):**

```bash
# Install using Homebrew (does NOT auto-update)
brew install --cask claude-code

# Verify installation:
claude --version

# To update later: brew upgrade claude-code
```

**For Windows users:**

First, install [Git for Windows](https://git-scm.com/downloads/win) if you don't have it.

Then run one of these commands:

```powershell
# PowerShell (Recommended):
irm https://claude.ai/install.ps1 | iex

# Or Command Prompt:
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

```bash
# Verify installation:
claude --version
```

**Alternative for Windows (WinGet):**

```powershell
winget install Anthropic.ClaudeCode
```

**First time setup:**
When you run `claude` for the first time, it will ask you to authenticate with your Anthropic account (your instructor set this up for you).

#### Using Claude Code

1. **Navigate to your project directory:**

   ```bash
   # On your LOCAL laptop (not on the Pi!)
   cd ~/path/to/smart-objects-cameras
   ```

2. **Start Claude Code:**

   ```bash
   claude
   ```

3. **Ask questions!** Claude Code has access to all your files:

   ```
   You: How does the person detection pipeline work in person_detector_with_display.py?

   You: I want to modify the detector to also detect cars. Show me what changes to make.

   You: I'm getting an error when I run the script. Here's the error: [paste error]
   ```

4. **Claude Code can SSH into the Pi for you!**

   ```
   You: SSH into orbit and run the person detector with display mode

   You: Copy person_detector_with_display.py to the orbit Pi

   You: Check if the camera is connected on the Pi
   ```

**⚠️ IMPORTANT SAFETY RULE:**

**NEVER auto-accept changes!** Always review what Claude Code wants to do:

- ✅ **DO:** Read the proposed changes carefully
- ✅ **DO:** Make sure you understand what it's changing
- ✅ **DO:** Ask questions if something looks wrong
- ❌ **DON'T:** Just press "y" to accept everything automatically
- ❌ **DON'T:** Let it make changes you don't understand

**This is critical!** Claude Code is powerful but you need to understand what's happening to your code and your Pi.

---

### Option 2: Using Claude Code in VS Code

If you prefer using Claude Code inside VS Code:

1. **Open VS Code** and connect to the Pi via Remote SSH (or open your local repo)
2. **Open Claude Code** panel (usually in the sidebar or via command palette)
3. **Use the prompt template below** to get started

---

### Why Terminal Claude Code is Powerful

When you run `claude` from your `smart-objects-cameras` directory, it automatically:

- ✅ Reads `CLAUDE.md` (project instructions and context)
- ✅ Understands your project structure
- ✅ Knows about depthai 3.x API and compatibility
- ✅ Can read any file in your project
- ✅ Can SSH into the Pis and run commands for you
- ✅ Can copy files to/from the Pis
- ✅ Proposes changes and waits for your approval

**Example conversation:**

```
You: I want to modify person_detector_with_display.py to detect both people and cars

Claude Code: I'll help you modify the detection to include cars. First, let me read the current script...
[Reads person_detector_with_display.py]

I'll need to:
1. Update the detection filter to include car class (class 2 in COCO)
2. Add different colored bounding boxes for people vs cars
3. Update the logging to distinguish between detection types

Would you like me to make these changes?

You: Yes, show me the changes first

Claude Code: [Shows exact code changes with before/after]
Should I apply these changes? (y/n)

You: [Review carefully, then] y

Claude Code: Changes applied! Would you like me to copy the updated file to the orbit Pi?
```

---

### Prompt Template for VS Code Claude Code

```
I'm working on a Smart Objects camera project using a Raspberry Pi 5 with an OAK-D camera. The project uses:

- **Hardware:** Raspberry Pi 5 (16GB), Luxonis OAK-D camera
- **Python Environment:** Shared venv at /opt/oak-shared/venv/
- **Libraries:** depthai 3.3.0, depthai-nodes 0.3.7, opencv-contrib-python 4.10.0.84
- **Project Location:** ~/oak-projects/ on the Pi

**Available Scripts:**
- person_detector_with_display.py (person detection with YOLO)
- fatigue_detector.py (fatigue detection using face landmarks)
- gaze_detector.py (gaze estimation)
- discord_bot.py (public camera bot)
- discord_dm_notifier.py (personal DM bot)

**What I'm trying to do:**
[Describe your goal here - e.g., "modify the person detector to also detect cats", "add a new Discord command", "change the detection threshold", etc.]

**Current issue/question:**
[Describe what's not working or what you want to understand]

Please help me understand how to [your specific request]. The code should work with the depthai 3.x API and be compatible with the existing project structure.
```

### Example Prompts

**Understanding the code:**

```
I'm working on a Smart Objects camera project using a Raspberry Pi 5 with an OAK-D camera (see system details above).

Can you explain how the person_detector_with_display.py script works? Specifically:
1. How does it connect to the camera?
2. How does the YOLO detection work?
3. What is the debouncing logic for?

Please reference specific line numbers and functions in your explanation.
```

**Modifying detection:**

```
I'm working on a Smart Objects camera project (see system details above).

I want to create a modified version of person_detector_with_display.py that:
- Detects multiple object types (people, cars, and dogs)
- Display different colored bounding boxes for each type
- Log when each type is detected

Please create a new file called person_detector_multiclass_myname.py with these changes.
Keep the original file intact so I can refer back to it.
```

**Adding Discord features:**

```
I'm working on a Smart Objects camera project (see system details above).

I want to add a new Discord command to discord_bot.py:
- Command: !set-threshold <value>
- Purpose: Change the detection threshold without restarting
- The detector should read a config file to get the new threshold

Can you show me:
1. How to add the Discord command
2. How to write/read a config file
3. How to modify the detector to check the config periodically
```

**Debugging issues:**

```
I'm working on a Smart Objects camera project (see system details above).

I'm getting this error when running person_detector_with_display.py:

[Paste your error message here]

What's causing this error and how can I fix it?
```

### Tips for Working with Claude Code

- **Be specific:** Include the exact script name and what you're trying to do
- **Include error messages:** Copy and paste the full error output
- **Reference the API version:** We're using depthai 3.x (not 2.x), which has a different API
- **Ask for explanations:** Claude Code can help you understand how the code works
- **Request line numbers:** Ask for specific line references to make changes easier
- **Iterate:** Start with small changes, test them, then ask for help with the next step

---

### 🚀 Using Slash Commands (Research-First Workflow)

Your instructor has set up special **slash commands** that help you build features the right way - by researching existing examples first, then making decisions based on what you find.

**Available Commands:**

#### `/research-dev` - Full Research-First Workflow

Use this when building something new or modifying existing code. It guides Claude Code through a structured process:

```bash
claude
/research-dev
```

Then tell Claude what you want to build. For example:
```
I want to add a command to the Discord bot that tells me the temperature from a sensor.
```

**What it does:**
1. **Research Phase** - Checks existing examples, reads project docs, understands patterns
2. **Decision Phase** - Compares approaches and explains why one is better
3. **Design Phase** - Creates architecture that matches the project
4. **Implementation** - Builds the code
5. **Review** - Makes sure it follows project patterns

**Why this is helpful:**
- Prevents reinventing the wheel (checks oak-examples first!)
- Follows project patterns automatically
- Explains decisions so you learn
- Creates code that fits with existing scripts

#### `/help-me-code` - Student-Friendly Version

A simpler version that's great for quick tasks:

```bash
claude
/help-me-code
```

Then describe what you need. For example:
```
I want to modify person_detector.py to also count how many people it sees over time.
```

**What it does:**
- Looks at similar code first
- Explains the plan before building
- Shows you how to use what it creates
- Teaches as it goes

**Pro tip:** Start with `/help-me-code` for simple tasks, use `/research-dev` for bigger features or when you're unsure of the best approach.

#### Creating Your Own Slash Commands

Want to make your own? Create a file in `.claude/commands/`:

```bash
# In your local repo (not on the Pi!)
cd ~/path/to/smart-objects-cameras
nano .claude/commands/my-command.md
```

Write what you want Claude to do, save it, then use it:

```bash
claude
/my-command
```

**Example:** Create `.claude/commands/explain.md`:
```markdown
# Explain This Code

I'm a student and I don't understand this code. Please:
1. Read the file I'm asking about
2. Explain what it does in simple terms
3. Point out the important parts
4. Tell me how it connects to other scripts in this project
```

Then use it:
```bash
claude
/explain

> Can you explain fatigue_detector.py?
```

---

### Safety and Best Practices

**⚠️ CRITICAL: Always Review Changes Before Accepting**

Your instructor emphasized this in class, but it's worth repeating:

**💡 Best Practice: Make a New File for Your Changes**

When asking Claude Code to modify an existing file, ask it to create a **new file** with your name in it instead of changing the original:

```bash
claude
> I want to modify person_detector_with_display.py to also detect cars.
  Please create a new file called person_detector_with_display_alice.py with these changes,
  so I can keep the original file as reference.
```

**Why this is smart:**

- ✅ You keep the working original file intact
- ✅ You can always go back to the original if something breaks
- ✅ Easy to compare your changes to the original
- ✅ You can identify which files are yours at a glance
- ✅ Multiple students can have their own versions without conflicts

**Example naming:**

- `person_detector_with_display_alice.py`
- `fatigue_detector_bob.py`
- `gaze_detector_improved_charlie.py`

**What to watch for when Claude Code proposes changes:**

1. **File modifications:**

   ```
   Claude Code: I'll modify person_detector_with_display.py

   ✅ GOOD: Review the diff - what changed?
   ❌ BAD: Just accepting without reading
   ```

2. **SSH commands:**

   ```
   Claude Code: Should I run this on the Pi?
   > ssh orbit "sudo systemctl restart person-detector"

   ✅ GOOD: Understand what the command does first
   ❌ BAD: Auto-accepting "sudo" commands you don't understand
   ```

3. **File transfers:**

   ```
   Claude Code: Should I copy this to the Pi?
   > scp modified_file.py orbit:~/oak-projects/

   ✅ GOOD: Verify it's copying the right file to the right place
   ❌ BAD: Accepting without checking destination
   ```

4. **Installing packages:**

   ```
   Claude Code: Should I install this package?
   > pip install unknown-package

   ✅ GOOD: Ask "Why do I need this? What does it do?"
   ❌ BAD: Installing random packages
   ```

**Questions to ask yourself before accepting:**

- ✅ Do I understand what this change does?
- ✅ Is this what I asked for?
- ✅ Could this break something?
- ✅ Am I modifying the right file?
- ✅ Will this affect other students?

**If you're unsure:**

- Ask Claude Code to explain the changes in more detail
- Ask "What could go wrong with this change?"
- Test on a copy of the file first
- Ask your instructor

**Remember:** You're in control! Claude Code is a powerful tool, but **you** make the final decisions about your code and your Pi.

---

### Common Claude Code Commands

**Understanding code:**

```bash
claude
> Explain how the debouncing logic works in person_detector_with_display.py

> What does the ParsingNeuralNetwork node do?

> Walk me through the pipeline setup step by step
```

**Making changes:**

```bash
claude
> Create a new file called person_detector_myname.py that saves screenshots
  every 10 seconds instead of 5. Base it on person_detector_with_display.py

> I want to add a command line argument to change the model at runtime.
  Create a new file called person_detector_flexible_myname.py with this feature.

> Fix this error: [paste error message]
```

**Pro tip:** Always ask Claude to create a NEW file with your name when making changes, so you keep the original as a working reference!

**Working with the Pi:**

```bash
claude
> Copy person_detector_with_display.py to the orbit Pi

> SSH into gravity and check if the camera is connected

> Run the person detector on horizon with display mode
```

**Debugging:**

```bash
claude
> I'm getting "No module named 'depthai'" - what's wrong?

> The camera isn't detecting anything - help me debug

> Why is my Discord bot not responding to commands?
```

---

### Important Reminders

- Always test changes on the Pi (not just locally on your laptop)
- Remember to copy updated files to the Pi after editing (`scp` or VS Code Remote SSH)
- The GitHub repo stays on your local computer - never clone it on the Pi
- Keep your `.env` file secure and never commit it to GitHub
- **Review every change** before accepting - this protects your code and the Pi

**Need more help?** Check the main [README.md](../README.md) or [WORKFLOW.md](WORKFLOW.md) for additional guidance.
