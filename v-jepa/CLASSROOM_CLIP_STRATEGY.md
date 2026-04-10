# Classroom Clip Strategy

A guide to recording and classifying video clips for V-JEPA classroom activity recognition.

## Core Concept

V-JEPA understands **temporal patterns** - how scenes evolve over 2-3 seconds. It doesn't need constant motion; it recognizes the *character* of different classroom modes including relative stillness.

Think in terms of **modes**, not micro-actions.

## Recommended Classes

Start with 3-4 clearly distinct classes. Add more once the basics work.

### Tier 1: Start Here

| Class | Visual Pattern | Example Clips |
|-------|---------------|---------------|
| `empty_room` | No people, static scene | Various times of day, lights on/off |
| `lecture` | One person at front, students facing forward, minimal motion | Teacher talking, pointing at board |
| `group_work` | Clusters of students, movement between groups, conversation gestures | Small group discussions, collaborative work |
| `individual_work` | Heads down, quiet, occasional small movements | Test-taking, silent reading, solo work |

### Tier 2: Add Later

| Class | Visual Pattern |
|-------|---------------|
| `transition` | People moving in/out, packing up, arriving |
| `presentation` | One student standing at front, others watching |
| `whiteboard_active` | Person writing/drawing on board, arm motion |
| `discussion` | Whole-class, people looking at each other, hand-raising |

### What Won't Work Well

These require detection approaches other than V-JEPA:

| Don't Try | Why |
|-----------|-----|
| "Hand raised" | Too brief, too small in frame |
| "Student confused" | Facial expression, not body pattern |
| "Teacher asking question" | Audio-dependent, not visual |
| "Taking notes" | Too subtle, looks like individual_work |
| "Student X specifically" | V-JEPA sees patterns, not identities |

## Recording Guidelines

### Clip Specifications

- **Duration**: 3 seconds (default, 90 frames at 30fps)
- **Resolution**: 640x480 minimum
- **Clips per class**: 20-30 for good accuracy, 50+ for robust model

### What Makes Good Training Clips

**DO capture:**
- The typical/representative state (not transitions between states)
- Variety within the class (different students, different days)
- Multiple camera angles if available
- Different lighting conditions
- Edge cases (nearly empty group_work, very quiet lecture)

**DON'T capture:**
- Transitions between modes (save for `transition` class)
- Unusual one-off events
- Clips where you're unsure of the label
- Moments that could be multiple classes

### Camera Placement

| Position | Good For | Notes |
|----------|----------|-------|
| Front corner (high) | Overall room view, lecture detection | Classic classroom camera spot |
| Back of room | Seeing teacher + students | Good for presentation/lecture |
| Side angle | Group formations | Helps distinguish clusters |

Using **multiple camera angles** during training improves robustness. The probe learns to recognize activities from different viewpoints.

## Recording Workflow

### Option A: Scheduled Auto-Recording

Best for natural, unposed behavior:

```powershell
# Record a clip every 30 seconds during class
python auto_recorder.py --output D:\classroom_clips --interval 30 --display
```

Then label afterward:
```powershell
python clip_labeler.py --input D:\classroom_clips --slow
```

### Option B: Manual Recording

Best for ensuring specific classes are captured:

```powershell
python clip_recorder.py --output D:\classroom_clips --countdown 3
```

Press 1/2/3/4 to record directly into class folders.

### Multi-Camera Recording

If you have multiple cameras (USB + Ethernet):

```powershell
# Terminal 1 - USB camera
python auto_recorder.py --output D:\classroom_clips --interval 30

# Terminal 2 - Ethernet camera
python auto_recorder.py --output D:\classroom_clips --interval 30 --ip 192.168.7.231
```

All clips go to the same folder. Label them together. The probe learns from multiple viewpoints.

## Folder Structure

```
classroom_clips/
  empty_room/
    clip_001.mp4
    clip_002.mp4
    ...
  lecture/
    clip_001.mp4
    ...
  group_work/
    ...
  individual_work/
    ...
  unlabeled/           # Auto-recorded clips before sorting
    ...
```

## Training

```powershell
# First time (extracts embeddings via server)
python probe_trainer.py --clips-dir D:\classroom_clips

# After adding new clips
python probe_trainer.py --clips-dir D:\classroom_clips --no-cache
```

### Expected Accuracy

| Clips per Class | Expected Val Accuracy |
|-----------------|----------------------|
| 5-10 | ~50-60% (barely better than random) |
| 15-20 | ~70-80% |
| 30-50 | ~85-95% |
| 50+ | ~95%+ |

If accuracy is low:
1. Check class balance (similar clip counts)
2. Review clips for mislabels
3. Ensure classes are visually distinct from camera's view
4. Add more clips with variety

## Avoiding Common Pitfalls

### Constant Motion Objects

Avoid having constantly moving objects (fans, clocks with second hands, screens with video) in the camera's view. They add noise to every clip and make all embeddings more similar.

### Overlapping Classes

If two classes look similar from the camera angle, they'll confuse the model. Solutions:
- Merge them into one class
- Add a distinguishing element (location, number of people)
- Try a different camera angle

### Imbalanced Data

Having 100 clips of `lecture` but only 10 of `group_work` can bias the model. Aim for roughly equal counts per class.

### Transition Moments

A clip captured during the shift from lecture to group_work is ambiguous. Either:
- Create a `transition` class for these
- Skip them during labeling (leave in unlabeled)

## Inference

Once trained:

```powershell
python probe_inference.py --probe classroom_probe.pt --display --interval 5
```

The system will classify the current mode every 5 seconds.

### Multi-Camera Consensus

With multiple cameras, combine their predictions:

```powershell
python multi_camera_consensus.py --cameras front-cam,back-cam --strategy average
```

Strategies:
- `confidence`: Trust highest-confidence camera
- `average`: Average probabilities across cameras
- `agreement`: Only report if cameras agree

## Latency Expectations

| Step | Time |
|------|------|
| Capture clip | ~3 seconds |
| Server embedding | ~300ms |
| Probe classification | ~1ms |
| **Total cycle** | **~5-8 seconds** |

This is fine for classroom modes that last minutes. Not suitable for detecting momentary events.

## Example: Minimal Viable Test

Get something working quickly:

1. **Record 10 clips each** of `empty_room` and `lecture`
2. **Train**: `python probe_trainer.py --clips-dir D:\classroom_clips`
3. **Test**: `python probe_inference.py --probe classroom_probe.pt --display`
4. Walk in/out of frame - does it switch between empty/lecture?

If this works, add more classes one at a time.
