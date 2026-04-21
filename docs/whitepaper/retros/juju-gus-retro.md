# Project: Gus Mode

**Student:** Juju **Camera used:** orbit / gravity / horizon & GusBot
**One-line pitch:** Gus Mode virtually brings our classmate Kathy's dog, Gus (the IxD mascot), into the classroom. First, we deploy a VIAM rover at Kathy's apartment, where Gus stays while Kathy is at the studio. The rover streams live video of Gus, from which we segment his figure. We then project a life-size image of Gus onto our classroom wall, enabling Kathy and our classmates to interact with virtual Gus from within the studio.

---

## What I tried

We tried connecting the Viam Rover camera to our laptop using its API while both the Viam and the laptop being at the studio. After fetching the real time camera footage, we used a dog segmentation model (YOLOv8) to segment the dog (this time we used a dog plushie to test it).


## What worked

- connect Viam Rover camera to laptop using the same network
- segment dog from the fetched background
- project the dog onto the studio wall

## What broke

Two or three bullet points. **Be honest.** False positives, weird edge cases, things you'd warn the next student about. This is the most valuable section of your retro — don't skip it.

- not yet tested while Viam Rover is at Kathy's house
- the projection seemed blinking, not sure if it's the dog plushie's issue or the issue of Viam's camera (further testing needed)

## One screenshot

We weren't able to take a screenshot because the Rover's battery died before we took a photo. We will update on Monday before class starts.


## If I had another week

One paragraph. What's the next move? What would you add, fix, or rethink? If a future student picked this up, what should they know first?

We will test the Rover at Kathy' house, so that it works remotely. Also, we will look into the blinking of the dog, and make fixes if needed to make the camera connection natural and soft. Additionally, we will design interactions to make the experience more playful and rich. (ie, UIs to control the Rover, direct interactions with Gus through connected toys) Finally, virtual Gus should be arranged so that it gets embedded into the smart classroom system.

---

*Submission checklist:*
- [ ] File named `<firstname>-<project>-retro.md` and placed in `docs/whitepaper/retros/`
- [ ] Screenshot added to `docs/whitepaper/artifacts/` and linked above
- [ ] "What broke" section is honest and specific
- [ ] No code snippets — this is a story, not a tutorial
- [ ] Opened as a pull request, not pushed to `main`
