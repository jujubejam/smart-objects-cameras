# Multi-User Access

> **Prefer slides?** View the [Multi-User Access Slides](https://kandizzy.github.io/smart-objects-cameras/multi-user-access-slides.html) for a visual walkthrough.

Multiple students can access the same Pi simultaneously for collaborative work.

---

## Important: One Camera = One Person Running the Script

**Key concept:** Each Pi has ONE camera. Only ONE person should run `person_detector.py` at a time.

**Smart object feature:** When you run the script with `--discord`, the camera **automatically announces** who's using it!

### Typical Classroom Workflow

```bash
# Student A runs the script:
ssh orbit
source /opt/oak-shared/venv/bin/activate
python3 person_detector.py --discord

# Discord automatically shows:
# 🎥 **alice** is now running person_detector.py on **orbit**

# Other students can simultaneously:
# - SSH in and view/edit code via VS Code Remote
# - Make their own copies: person_detector_bob.py
# - Prepare changes for when it's their turn
# - Watch the Discord channel to see who's using which camera

# When Student A stops (Ctrl+C):
# Discord automatically shows:
# 📴 **alice** stopped person_detector.py on **orbit** - camera is free
```

**No manual coordination needed!** The camera announces itself automatically.

---

## Best Practices for Shared Access

### 1. Check Discord Before Running

- The camera automatically announces who's using it
- **If no "camera is free" message recently**, check: `ps aux | grep person_detector.py`
- **Be considerate:** Don't run for hours - test and let others use it
- **Run with `--discord` flag** so others can see when you're done

### 2. Everyone Can Collaborate via Code Editing

- **Multiple people can SSH in simultaneously** to view/edit code
- **Use VS Code Remote SSH** - each person gets their own editing session
- **Create personal test scripts:**
  ```bash
  cd ~/oak-projects
  cp person_detector.py person_detector_alice.py
  cp person_detector.py person_detector_bob.py
  # Edit your copy, test when camera is free
  ```

### 3. Use Git for Collaboration

- Work on your own branch: `git checkout -b feature/alice-zone-detection`
- Commit changes when you've tested them
- See [GIT_COLLABORATION.md](../GIT_COLLABORATION.md) for strategies

### 4. Communication is Essential

- Let teammates know before you run the detector
- Don't stop someone else's running script
- Use a shared Discord/Slack channel to coordinate
- If unsure, ask: "Is anyone using Camera 1?"

---

## Adding Your SSH Key

If multiple people use the same Pi account, each person can add their own SSH key for passwordless access.

### On Mac/Linux:

```bash
# Generate key if you don't have one (with project-specific name)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_smartobjects -C "your_email@example.com"

# Copy to Pi (if SSH config is set up from Part 1)
ssh-copy-id orbit

# Or specify the key explicitly
ssh-copy-id -i ~/.ssh/id_ed25519_smartobjects.pub orbit
```

### On Windows (PowerShell):

```powershell
# Generate key if needed (with project-specific name)
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\id_ed25519_smartobjects -C "your_email@example.com"

# Copy to Pi manually
type $env:USERPROFILE\.ssh\id_ed25519_smartobjects.pub | ssh orbit "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

### Multiple Keys for Shared Account

Each person's key gets added to the same `authorized_keys` file:

```bash
# Person 1 adds their key
ssh-copy-id orbit

# Person 2 adds their key (doesn't overwrite Person 1's key)
ssh-copy-id orbit

# Or manually add keys
ssh orbit
nano ~/.ssh/authorized_keys
# Paste each person's public key on a new line
```

Now everyone can connect without a password!

---

## Adding Additional User Accounts (Optional)

If you want separate user accounts for each student:

```bash
# SSH into the Pi
ssh orbit

# Create new user
sudo adduser alice

# Add to required groups for camera access
sudo usermod -aG video,gpio,i2c,spi alice

# Copy project files to their home directory
sudo cp -r /home/pi/oak-projects /home/alice/
sudo chown -R alice:alice /home/alice/oak-projects

# The new user can now log in (update SSH config with new username)
ssh orbit  # (after updating SSH config User field)
```

---

## Shared Resources (Already Configured)

The following shared resources have already been set up on all three Pis:

### Shared Model Cache
- **Location**: `/opt/depthai-cache`
- **What it does**: DepthAI models download once and are accessible to all users
- **Why it matters**: Prevents permission errors when multiple students test their own script copies
- No action needed - this is already configured!

### Shared Oak Examples
- **Location**: `/opt/oak-shared/oak-examples/`
- **Symlink**: `~/oak-examples/` in each user's home directory
- **What it includes**: Luxonis example code for neural networks, depth, tutorials, etc.
- Browse examples whenever you want inspiration for new features!

**For setup details** (instructors only), see [docs/archive/multi-user-setup.md](archive/multi-user-setup.md)

---

## Related Documentation

- [README.md](../README.md) - Main documentation
- [WiFi Network Management](wifi-management.md) - Managing network connections
- [GIT_COLLABORATION.md](../GIT_COLLABORATION.md) - Git workflow strategies
