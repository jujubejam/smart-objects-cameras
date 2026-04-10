# Classroom Recording Setup

Quick reference for recording V-JEPA clips from the classroom PoE cameras.

## Camera IPs

| Camera | IP | Status |
|--------|-----|--------|
| Camera 1 | 169.254.1.10 | Working |
| Camera 2 | 169.254.1.11 | Working |
| Camera 3 | 169.254.1.222 | Needs reboot |

## Prerequisites

1. **Kill any scripts using cameras on orbit:**
   ```bash
   ssh carrie@orbit.local
   sudo pkill -f camera_preview.py
   ```

2. **Windows firewall** must allow Python (already configured)

## Recording Commands

### Manual Recording (with countdown)

Press 1/2/3/4 to record clips directly into class folders:

```powershell
python clip_recorder.py --ip 169.254.1.10 --output D:\classroom --classes empty_room,lecture,group_work,individual_work
```

### Auto Recording (label later)

Records every 30 seconds, label afterward:

```powershell
python auto_recorder.py --ip 169.254.1.10 --output D:\classroom --interval 30 --display
```

Then label:
```powershell
python clip_labeler.py --input D:\classroom
```

### Multi-Camera Recording

Run in separate terminals:

```powershell
# Terminal 1
python auto_recorder.py --ip 169.254.1.10 --output D:\classroom --interval 30

# Terminal 2
python auto_recorder.py --ip 169.254.1.11 --output D:\classroom --interval 30
```

## Classroom Classes

| Class | What to capture |
|-------|-----------------|
| `empty_room` | No people, various lighting |
| `lecture` | Teacher at front, students watching |
| `group_work` | Students in clusters, discussion |
| `individual_work` | Heads down, quiet, solo work |

See [CLASSROOM_CLIP_STRATEGY.md](../CLASSROOM_CLIP_STRATEGY.md) for full details.

## Training

After collecting 20-30 clips per class:

```powershell
python probe_trainer.py --clips-dir D:\classroom
```

## Troubleshooting

**"X_LINK_DEVICE_NOT_FOUND"**
- Another script is using the camera (check orbit)
- Run: `ssh carrie@orbit.local "pkill -f python"`

**Camera not responding**
- Ping test: `ping 169.254.1.10`
- Check you're on Ethernet, not just WiFi

**"Failed to connect"**
- Windows firewall blocking Python
- Run as admin: `netsh advfirewall firewall set rule name="python.exe" new action=allow`
