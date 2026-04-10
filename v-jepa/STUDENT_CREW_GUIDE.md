# Student Crew Guide

You're running video capture for our classroom AI. This doc tells you everything you need.

---

## What We're Doing

Recording short video clips of different classroom activities. An AI will learn to recognize what's happening based on these clips.

---

## The Classes

| # | Class | What It Looks Like |
|---|-------|-------------------|
| 1 | `empty_room` | Nobody in the room |
| 2 | `lecture` | Teacher at front, students facing forward |
| 3 | `individual_work` | Heads down, quiet, working alone |
| 4 | `group_work` | Clusters of students talking |
| 5 | `transition` | People moving around, packing up |

---

## Equipment

- Raspberry Pi (orbit/gravity/horizon)
- OAK-D camera (ethernet, mounted overhead)
- Your laptop for SSH

---

## Commands

### Connect to Pi
```bash
ssh orbit
# or: ssh gravity / ssh horizon
```

### Activate Environment
```bash
source /opt/oak-shared/venv/bin/activate
```

### Start Recording
```bash
cd ~/smart-objects-cameras/v-jepa/windows

# Replace IP with your camera's IP
python auto_recorder.py --output ~/classroom_clips --interval 5 --ip 192.168.7.231 --display
```

This captures a 3-second clip every 5 seconds.

### Stop Recording
```
Ctrl+C
```

### Check What You Got
```bash
ls ~/classroom_clips/unlabeled/ | wc -l
```

---

## Running a Capture Session

### Before
- [ ] Camera is mounted and powered
- [ ] You can SSH into the Pi
- [ ] Test the camera works (run command, see display, Ctrl+C)

### During
1. Announce the scene: "We're doing LECTURE for 3 minutes"
2. Start recording
3. Let it run for 3 minutes
4. Stop recording
5. Move to next scene

### After
Tell the instructor how many clips you captured:
```bash
ls ~/classroom_clips/unlabeled/*.mp4 | wc -l
```

---

## Troubleshooting

**"Camera not found"**
- Check ethernet cable
- Check camera power
- Try: `ping 192.168.7.231` (use your camera's IP)

**"Module not found"**
- Did you activate the environment? `source /opt/oak-shared/venv/bin/activate`

**Display doesn't show**
- Remove `--display` flag (still records, just no preview)
- Or use VNC instead of SSH

**Clips are empty/corrupted**
- Camera might be busy - wait and retry
- Check disk space: `df -h`

---

## Tips

- **3 minutes per scene** gives ~36 clips (plenty)
- **Don't record transitions** between scenes (stop first)
- **Natural behavior** is better than posed (let people be normal)
- **Variety helps** - different students, different spots in room

---

## File Transfer

When done, the instructor will pull clips to their PC:
```bash
# From instructor's PC:
scp -r pi@orbit:~/classroom_clips/unlabeled/* D:\classroom_clips\unlabeled\
```

Or you can use a USB drive:
```bash
cp -r ~/classroom_clips/unlabeled /media/usb/
```

---

## Quick Reference

| Task | Command |
|------|---------|
| SSH in | `ssh orbit` |
| Activate env | `source /opt/oak-shared/venv/bin/activate` |
| Start capture | `python auto_recorder.py --output ~/classroom_clips --interval 5 --ip <IP>` |
| Stop capture | `Ctrl+C` |
| Count clips | `ls ~/classroom_clips/unlabeled/*.mp4 \| wc -l` |

---

*Questions? Ask the instructor or check the main docs in the repo.*
