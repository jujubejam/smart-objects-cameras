# Capture Day Script

**A short play in five acts, starring your students.**

---

## Before Class

- [ ] Camera mounted (front corner, high, sees whole room)
- [ ] Server running: `python server.py`
- [ ] Test camera works: `python clip_recorder.py --display` (q to quit)
- [ ] Clear output folder: `D:\classroom_clips`
- [ ] This script printed

---

## Opening (2 min)

**[To students]**

> "Today you're actors. We're filming training data for an AI that will learn to recognize what's happening in our classroom. I need you to act naturally in five different scenes. Each scene is about 3 minutes. Ready?"

**[Start recording]**
```
python auto_recorder.py --output D:\classroom_clips --interval 5 --display
```

---

## Act 1: Empty Room

**Duration:** 2 minutes
**Clips captured:** ~24

**[To students]**
> "Everyone out. Take your stuff, go to the hallway. I'll call you back."

**[While empty]**
- Let it run with no one in frame
- Optional: turn lights off for 30 sec, then back on

**[Call them back in]**

---

## Act 2: Lecture

**Duration:** 3 minutes
**Clips captured:** ~36

**[To students]**
> "Take your seats. Face forward. I'm going to lecture about something boring. Look engaged anyway."

**[You]**
- Stand at front
- Talk, gesture naturally
- Point at board occasionally
- Walk a little, but stay "at front"

---

## Act 3: Individual Work

**Duration:** 3 minutes
**Clips captured:** ~36

**[To students]**
> "Heads down. Silent work time. Pretend you're taking a test. No talking, no looking around."

**[You]**
- Step to the side or sit down
- Let them be the scene

---

## Act 4: Group Work

**Duration:** 3 minutes
**Clips captured:** ~36

**[To students]**
> "Form groups of 3-4. Actually discuss something - your weekend, a project, whatever. Move chairs. Be natural."

**[You]**
- Walk between groups briefly
- Let the clusters form and talk

---

## Act 5: Transition

**Duration:** 2 minutes
**Clips captured:** ~24

**[To students]**
> "Class is over. Pack up. Switch seats with someone across the room. Walk around. Controlled chaos."

**[Let it be messy]**
- People standing, moving, getting bags
- Captures the "between modes" state

---

## Wrap

**[Stop recording]** Ctrl+C

**[To students]**
> "That's a wrap. We got about 150 clips. I'll train the AI tonight, and tomorrow we'll see if it can tell what we're doing in real-time."

---

## After Class

**1. Label the clips**
```
python clip_labeler.py --input D:\classroom_clips --slow
```
- 1 = empty_room
- 2 = lecture
- 3 = individual_work
- 4 = group_work
- 5 = transition
- Space = skip (ambiguous clips)

**2. Check counts**
```
ls D:\classroom_clips\*\*.mp4 | wc -l
```
Aim for 20+ per class.

**3. Train**
```
python probe_trainer.py --clips-dir D:\classroom_clips
```
Target: 80%+ validation accuracy

**4. Test**
```
python probe_inference.py --probe classroom_probe.pt --display
```

---

## Backup Plan

If something breaks during capture:
- Manual recording works: `python clip_recorder.py --output D:\classroom_clips`
- Press number keys to record into specific class folders
- Slower but more controlled

---

## Props

None required. The classroom is the set. The students are the cast.

---

*Total runtime: ~15 minutes of recording + 5 min setup/transitions*
