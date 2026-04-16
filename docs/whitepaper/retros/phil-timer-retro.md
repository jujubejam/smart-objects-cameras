# Project: Timer

**Student:** Phil Cote

**Camera used:** Horizon

**One-line pitch:** A 60-min timer for **whiteboard** | **group work** | **focus time** using magnetic fiducial markers (arUco) and CV. The timer's behavior depends on what horizon sees.

## What I tried

> *I wanted to create an intuitive, casual, and tactile timer experience for both teachers and students. I started by seeing if Horizon could see handwriting. I also tried using Teachable Machine image model training — first using hand gestures, then numbers. Next, I tried Hand Detection, which worked better than the model training, but still difficult to think of how it would be sensed by the classroom consistently. Finally, I arrived at arUco tags attached to magnetically reinforced foam core, which were cheap and easy to produce.*

## What worked

• The arUco tags where the most responsive of all the interactions I tested — code is open-source so it was easy for Claude to find it when errors came up.

• The hand detection iteration technically worked, however it felt too limiting for the purpose of the timer.

• The hand written commands were successful in capturing the feel of a non-screen interaction.

## What broke

• Image model training did not work as well as I'd hoped — I learned that it's better for predicting specific objects, not interpreting commands.

• I have yet to resolve the positioning and mounting of the camera in a way which disappears for people without losing sight.

## Screenshot

<img src="../artifacts/phil-timer-screenshot.png" width="400">

Caption: one sentence describing what we're looking at.

## If I had another week

One paragraph. What's the next move? What would you add, fix, or rethink? If a future student picked this up, what should they know first?

**Next Move:**

**Add:**

**Fix:**

**Rethink:**

---

*Submission checklist:*
- [x] File named `<firstname>-<project>-retro.md` and placed in `docs/whitepaper/retros/`
- [x] Screenshot added to `docs/whitepaper/artifacts/` and linked above
- [x] "What broke" section is honest and specific
- [x] No code snippets — this is a story, not a tutorial
- [ ] Opened as a pull request, not pushed to `main`
