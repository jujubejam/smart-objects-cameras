# What Else Can Your Camera See?

> **Prefer slides?** View the [Next Ideas Slides](https://kandizzy.github.io/smart-objects-cameras/next-ideas-slides.html) for a visual overview.

Five project ideas from the `oak-examples` repo that work on our OAK-D cameras (RVC2) today. Each one builds on the existing detector + Discord bot pattern.

---

## What You Already Have

| Detector | What It Sees | Pipeline |
|----------|-------------|----------|
| `person_detector.py` | People in frame, count | YOLO v6 (single stage) |
| `fatigue_detector.py` | Drowsiness, head tilt | YuNet → MediaPipe landmarks |
| `gaze_detector.py` | Where someone is looking | YuNet → Head pose → Gaze ADAS |
| `whiteboard_reader.py` | Text on a whiteboard | PaddlePaddle detect + recognize |

**The pattern:** every detector writes a JSON status file → the Discord bot reads it → you ask questions in chat and get answers.

---

## Idea 1: Hand Gesture Recognition

**Example:** `oak-examples/neural-networks/pose-estimation/hand-pose/`

**Pipeline:** MediaPipe Palm Detection → Hand Landmarker (21 keypoints) → Gesture classifier

**Built-in gestures:** FIST, OK, PEACE, ONE, TWO, THREE, FOUR, FIVE

**On our hardware:** Works on RVC2, ~8 FPS

### Discord interaction ideas

- `!gesture` — what gesture is the camera seeing right now?
- `!vote` — thumbs up / thumbs down to vote on something
- `!gesture-trigger add PEACE "lights on"` — bind a gesture to an action
- Raise a fist to pause notifications, open palm to resume
- Silent classroom polling — "Hold up 1, 2, or 3"
- Gesture-controlled slideshow — swipe to advance
- Multi-camera gesture relay — gesture on orbit triggers action on horizon
- Combine with gaze: gesture + looking at camera = confirmed command

---

## Idea 2: Full Body Pose Estimation

**Example:** `oak-examples/neural-networks/pose-estimation/human-pose/`

**Pipeline:** YOLOv6 nano → Lite-HRNet (17 body keypoints)

**Alternative:** YOLOv8 Pose (single-stage, 17 keypoints in one pass)

**On our hardware:** Works on RVC2, ~5 FPS

### Discord interaction ideas

- `!hand-raised` — is anyone raising their hand? (wrist above shoulder)
- `!posture` — standing, sitting, or slouching?
- `!activity` — classify what the person is doing based on keypoint positions
- Alert when someone stands up or sits down
- Classroom hand-raise queue — first hand up gets called on first
- Movement energy score — how active is the room?
- Pose mirroring game — match the skeleton on screen
- Combine with person detector: track individual skeletons over time

---

## Idea 3: Object Tracking with Persistent IDs

**Example:** `oak-examples/neural-networks/object-tracking/deepsort-tracking/`

**Pipeline:** YOLOv6 nano → OSNet embedding → DeepSORT tracker

**On our hardware:** Works on RVC2, ~5 FPS

### Discord interaction ideas

- `!track` — list everyone currently tracked with their ID
- `!who-left` — report IDs that disappeared in the last N minutes
- `!dwell-time` — how long has person #3 been in frame?
- Announce arrivals and departures to a Discord channel
- Trajectory heatmap — overlay paths people walked
- Traffic flow — how many people crossed left-to-right vs. right-to-left?
- Multi-camera handoff — person leaves orbit's view, appears in gravity's
- Track non-person objects too — YOLO detects 80 COCO classes

---

## Idea 4: Human Re-Identification

**Example:** `oak-examples/neural-networks/reidentification/human-reidentification/`

**Pipeline (pose mode):** SCRFD person detection → OSNet body embedding → cosine similarity matching

**Pipeline (face mode):** SCRFD/YuNet face detection → ArcFace embedding → cosine similarity matching

**On our hardware:** Works on RVC2, ~2 FPS (slow but fine for attendance-style use cases)

### Discord interaction ideas

- `!attendance` — who has the camera seen today?
- `!register "Alex"` — name the current face for future recognition
- `!seen "Alex"` — when was Alex last spotted?
- Auto-greet returning people in Discord
- Privacy-first: store only embeddings, never photos
- Anonymous re-id: "Person A has visited 3 times" without knowing who A is
- Pair with fatigue: "Alex looks tired today" (personalized observation)
- Opt-in system: only track people who register themselves

---

## Idea 5: Segmentation & Silhouettes

**Examples:**
- `oak-examples/neural-networks/segmentation/blur-background/` — DeepLab V3+, blur everything that isn't a person
- `oak-examples/neural-networks/segmentation/depth-crop/` — DeepLab + stereo depth, isolate people by distance *and* shape

**Pipeline:** DeepLab V3+ → per-pixel mask

**On our hardware:** Works on RVC2, ~4 FPS (blur) / ~10 FPS (depth crop)

### Discord interaction ideas

- `!silhouette` — screenshot showing only person outlines
- `!privacy-mode on` — switch from full frame to silhouette-only capture
- `!background` — extract and share just the background (people removed)
- `!depth-mask` — combine segmentation + depth to isolate by distance
- Shadow puppet theater — silhouettes as art output
- Anonymous occupancy — count body shapes, not faces
- Background timelapse — capture the room without people over hours
- Combine with depth: "how much of the room is occupied?"

---

## Combining Ideas

Each idea is useful alone. Together, they start to describe a room that *understands* what's happening inside it.

- **Tracking + Pose** — Person #3 raised their hand 12 seconds ago and is still waiting.
- **Re-id + Fatigue** — Alex looks tired today. Send a private DM instead of a public alert.
- **Gesture + Segmentation** — Privacy-safe voting: count raised fists from silhouettes, no faces stored.

**Implementation is easy:** each detector writes a JSON status file. A new script can read *multiple* status files and fuse the information — no need to modify existing detectors.

---

## FPS Reality Check

Our OAK-D cameras use the RVC2 chip (Myriad X). It works — but it's not fast.

| Example | FPS on RVC2 | Good For |
|---------|:-----------:|----------|
| Hand gestures | ~8 | Interactive commands, voting |
| Human pose | ~5 | Posture checks, hand-raise detection |
| DeepSORT tracking | ~5 | Arrivals/departures, dwell time |
| Re-identification | ~2 | Attendance, periodic check-ins |
| Segmentation (blur) | ~4 | Privacy screenshots, silhouettes |
| Segmentation (depth crop) | ~10 | Distance-based isolation |

**Key insight:** These frame rates are fine for conversational interaction. You're asking the camera a question via Discord and getting an answer — you're not streaming 60fps video. 2 FPS is plenty fast for "who's in the room?"

### Want it faster?

- **OAK 4 cameras (RVC4)** — 52 TOPS, ~40x faster. Ships March 20, 2026. USB + PoE built into every unit. See the [Equipment List](EQUIPMENT_LIST.md#camera-upgrade-path-oak-4-rvc4) for pricing.
- **Offload to a GPU** — Stream frames from the current OAK-D to your PC or a cloud GPU (RunPod) and run inference there. Same speed boost, no new hardware.

---

## Getting Started

All five examples live in the `oak-examples` repo and follow the same pattern:

```bash
ssh orbit
activate-oak
cd ~/oak-examples/neural-networks/pose-estimation/hand-pose/
pip install -r requirements.txt
python3 main.py
```

To make it a Smart Objects detector, follow the existing pattern:

1. Copy structure from the closest existing detector (e.g. `fatigue_detector.py`)
2. Write a JSON status file so the Discord bot can read it
3. Add `--discord` / `--log` / `--display` flags
4. Add new `!commands` to `discord_bot.py`
5. Announce startup and shutdown to Discord

---

## Quick Reference

| Idea | Path in oak-examples | Models |
|------|---------------------|--------|
| Hand gestures | `neural-networks/pose-estimation/hand-pose/` | MediaPipe Palm + Hand Landmarker |
| Human pose | `neural-networks/pose-estimation/human-pose/` | YOLOv6 + Lite-HRNet |
| Object tracking | `neural-networks/object-tracking/deepsort-tracking/` | YOLOv6 + OSNet + DeepSORT |
| Re-identification | `neural-networks/reidentification/human-reidentification/` | SCRFD/YuNet + OSNet/ArcFace |
| Segmentation | `neural-networks/segmentation/blur-background/` | DeepLab V3+ |
| Depth crop | `neural-networks/segmentation/depth-crop/` | DeepLab V3+ + StereoDepth |

Browse all models: [models.luxonis.com](https://models.luxonis.com) — filter by RVC2 to see what runs on our cameras.
