# Interview: Timer

**Student / interviewer:** Phil

**Project:** Timer

**Camera used:** Horizon

**Interviewee:** Gwen, wife

**Their relationship to the class:** Gwen had seen some earlier versions of this, but hadn't interacted with the magnetic tags on the whiteboard yet.

**Date(s) of interview:** Sat May 1

**Format:** In-person

**Length:** ~20 minutes

---

## Why these people?

I chose Gwen to participate with the confidence I would get  **honest and unexpected** responses from her. I take for granted that the sequence and the interactions are intuitive — if unfamiliar with fiducial markers, the controls might be confusing.

## What you showed them

I had the arUco tags set up on a whiteboard and the camera facing it with the programs `timer_app.py` and  `timer_web.py` running. I started simply by asking Gwen what she thinks it does. I tried to leave it open ended — I mainly wanted to know how intuitive (or confusing) the arUco tags would be. I used a second screen to display the countdown. I left it up to her to decide how she would use it and in what sequence to manipulate the tags. 

## The conversation

### Before I tell you anything — what do you think this is doing?

> *“Looks like a clock or a timer...or a watch.”*
>
> *"They look like VCR controls — play, pause...maybe rewind."*

### What do you think it's *for*? Who would use it?

> *“For us.”* *(at home)*
>
**Uses for timer**

- Cooking (oven timing)
- Managing a child’s activities (countdowns for Henry)
- Visual cue for time
- Running group therapy sessions (tracking remaining time)

**Timer functions — while imbued with a sameness — change drastically with context.**

### What was confusing or unclear?

> *“There’s nothing that tells me that these have a front and a back to them.”*

- No indication that the tags have a front and back (flip function)
- Didn’t realize the tags themselves are interactive
- Assumed tags were objects placed onto an active area, not the input themselves
- Active space is unclear (no boundary/frame showing camera area)
- Thought the “timer off” area was the main interaction zone
- Icons/logos were interpreted as controls rather than markers

### What surprised you — good or bad?

>*“I guess that didn’t work...I would think like putting something on top of it.”*

- Realization that the tags work once they understand the objects limitations
- Initial confusion about how the tags operate (implicit negative surprise)

### What would make this feel creepy or invasive? What would make it feel useful?

> *“If I had a countdown...and then the lights on the unit change the color to let patients know that group was starting...that’d be great.”*

**Useful**
- Visual environmental cues (e.g., lights changing to signal time transitions)
- Coordinating group activities without verbal interruption
- Supporting structured environments like classrooms or hospital units
 
**Creepy/invasive** 
- *no response*

### What's missing? What did you expect it to do that it doesn't?

> *“I think having like a boundary...like active space...would make it more clear that the whole space is in play.”*

**Missing**

- Active interaction area (camera boundary)
- Tag orientation (front/back)
- That tags are part of the system input

**Expected**

- A more obvious “play/interaction” area
- More direct mapping between icons and behavior


### What else do you imagine this timer might control?

>*“It would be nice to just like maybe alert other staff like, oh—like when the unit turns...when the lights on the unit turn slightly blue, it means that, you know, a therapy group is about to start.”*

**Environmental signals**

- Lights changing at specific time intervals (classroom)
- Color changes in a hospital unit to signal group start times

**Potential coordination tool**

- Shared timing across groups or teams

## What I learned that I couldn't have learned from a classmate

On an intuitive level, interactions which use arUco tags and CV must visually communicate the affordances they offer; because each tag represents a specific function, they also need to be seen in a certain sequence in order to work. The tension between these two functionalities create an opportunity signal for further design investigation. I also was able to confirm that a timer like this would most likely be useful not just in the classroom, but in other spaces used by groups.

## One quote worth keeping

> *"Once I learned the role of each of the parts, it was very clear — but coming in raw...I had no idea that these cards have a function, a flip function.”*
>
>— *Gwen*

## What you'd change based on this

I'd like to rethink how the arUco tags are displayed, as well as to start thinking through some ways in which the timer can communicate to other systems. The physical installation of the camera still needs more consideration; a huge issue is that the camera still gets in the way of the board — affecting both the CV and the UX.

---

*Submission checklist:*
- [x] File named `<firstname>-<project>-interview.md` and placed in `docs/whitepaper/interviews/`
  - Example: `phil-timer-interview.md`
- [x] Interviewee is a genuine outsider — not another student in the class
- [x] At least 6 question/answer exchanges, with answers in the interviewee's voice
- [x] At least one question is your own, specific to your project
- [x] Direct quotes preserved (not paraphrased into engineering-speak)
- [x] "What I learned that I couldn't have learned from a classmate" section filled in honestly
- [x] One verbatim quote highlighted at the end
- [x] Your interviewee has read the writeup and is okay with how they're quoted
- [x] Your words, not AI-generated
- [x] Opened as a pull request, not pushed to `main`
