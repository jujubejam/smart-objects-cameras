# Project: Focus Beam

**Student:** Feifey Wang
**Camera used:** Orbit
**One-line pitch:** A classroom(screen) spotlight that reads the instructor's speech and pointing gestures to automatically highlight the part of the projected slide.

---

## What I tried

I tried to use Tiny Language model for detacting the instructor's speaching, which for the machiene to read we need it to turn into text. At the same time, for the gesture reading, I had tried to work with MediaPipe (I tried to use webcame on my laptop for testing)for gesture reading. I haven't tried on the lighting focusing part yet.


## What worked

- The webcam successfully detected a pointing gesture (index finger extended, other fingers curled) and mapped it to one of nine screen regions in real time
- The dimming overlay window launched on the second screen and responded to region changes without blocking the rest of the system after fixing a threading issue with macOS and Tkinter
- *(to be updated after further testing)*

## What broke

- The first version of the overlay froze on macOS because Tkinter and OpenCV both require the main thread, I later solved it by moving all signal, combining logic to a background thread and keeping the Tkinter loop lightweight
- I haven't really tried on the dimming part, it crashed the first time when I am trying to test it (it might be nmy computer's problem)
- *(to be updated after further testing, especially OCR accuracy on projected slides and voice-to-slide word matching)*

## One screenshot

Put your screenshot in `docs/whitepaper/artifacts/` using the naming convention below, then link to it here:

![](../artifacts/feifey-focusbeam-screenshot.png)

Caption: *(one sentence describing what we're looking at)*

## If I had another week

If I have another week, I would be refining the gesture detecting, so the casual arm movements during speech don't trigger the highlight, right now the line between "pointing intentionally" and "gesturing naturally" is still unclear. Beyond that, I also need to start testing the OCR in the classroom enviornment and different kind of slides. 

If a future student picked this up, the first thing I'd tell them is to test the voice-to-slide matching early with real lecture content, because the fuzzy word matching works well on clean text but needs tuning once you introduce abbreviations, formulas, or diagrams with no readable text at all.

---

*Submission checklist:*
- [V] File named `feifey-focusbeam-retro.md` and placed in `docs/whitepaper/retros/`
- [ ] Screenshot added to `docs/whitepaper/artifacts/` and linked above
- [V] "What broke" section is honest and specific
- [V] No code snippets — this is a story, not a tutorial
- [V] Opened as a pull request, not pushed to `main`