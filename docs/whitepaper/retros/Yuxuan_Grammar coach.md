# Project: <Grammar coach>

**Student:** <Yuxuan>
**Camera used:** orbit / horizon 
**One-line pitch:** <A desk device that quietly listens while you speak in class and gives you private, gentle grammar suggestions on a small screen — so international students can build confidence without fear of embarrassment.>

---

## What I tried

I built a real-time English grammar feedback tool called Lumi. The idea is that international students are often afraid to speak in class because they worry about making grammar mistakes in front of others. I wanted to create something that sits quietly on their desk, listens when they speak, and shows a small private suggestion on a screen — like a supportive coach, not a red-pen corrector. I used a microphone to capture speech, ran it through an AI language model to detect grammar issues, and displayed color-coded feedback cards that disappear after a few seconds.



## What worked

- The AI feedback was accurate and genuinely encouraging — it detected issues like wrong verb tense, missing articles, and wordy phrasing, and suggested natural alternatives rather than just flagging errors
- The delayed feedback mode worked well: students could choose to receive suggestions after they finish speaking instead of during, which felt less disruptive and less stressful
- The post-class summary feature generated a useful "research list" of grammar patterns to review, a vocabulary upgrade list, and a practice assignment — students found this more valuable than real-time correction alone

-
-
-

## What broke

- The microphone cannot distinguish the student's voice from nearby classmates — in a real classroom with multiple people talking, it picks up everything within range, making the feedback unreliable
- Speech recognition requires Chrome and does not work on Safari or mobile browsers, which limits how and where students can use it
- There is a 1–2 second delay between speaking and receiving feedback, which means very short phrases sometimes get missed or cut off before analysis completes

-
-
-

## One screenshot

Put your screenshot in `docs/whitepaper/artifacts/` using the naming convention `<firstname>-<project>-screenshot.png`, then link to it here:

![](../artifacts/<firstname>-<project>-screenshot.png)

Caption: one sentence describing what we're looking at.

## If I had another week

I would focus on solving the voice isolation problem — either by integrating with iPhone's Voice Isolation microphone mode via a companion app, or by using a directional microphone that only captures sound from directly in front of the device. I would also explore building a small physical object (a Raspberry Pi with a 2-inch screen) so the feedback feels more like a personal desk object and less like a browser tab. The post-class summary is the feature students responded to most positively, so I would expand it to track patterns across multiple sessions over weeks, not just one class at a time.

---

*Submission checklist:*
- [ ] File named `<firstname>-<project>-retro.md` and placed in `docs/whitepaper/retros/`
- [ ] Screenshot added to `docs/whitepaper/artifacts/` and linked above
- [ ] "What broke" section is honest and specific
- [ ] No code snippets — this is a story, not a tutorial
- [ ] Opened as a pull request, not pushed to `main`
