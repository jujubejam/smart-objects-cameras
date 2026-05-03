# Project: English Communication Coach / Ambient English Feedback Object

Student: Yuxuan Chen  
Project: English Communication Coach / Ambient English Feedback Object  
Camera used: none / laptop and browser prototype  
One-line pitch: A quiet classroom support tool that gives international students private, gentle English feedback so they can speak with more confidence in class.

---

## What I tried

I built a browser-based prototype called Lumi for my English Communication Coach project.

The original idea was a small smart object that sits on a student’s desk and listens when the student speaks English in class. If the student says something unclear or uses a repeated grammar pattern, the object gives a private suggestion. The goal is not to correct students publicly, but to help international students notice language patterns and build confidence over time.

During the process, the project shifted from “real-time grammar correction” to “private, delayed, and optional feedback.” This change came from class feedback. Kathy pointed out that if the system corrected someone every time, especially in a harsh or visible way, the student might become more afraid of speaking. Phil also reminded me that the most important part was not only how to build it, but how I wanted the student to interact with it.

The current prototype includes:

- Presenter Mode and Review Mode
- Type/Paste input as the stable demo method
- Speech input as an experimental method
- A transcript review step before analysis
- Demo Mode with rule-based pattern matching
- Private feedback cards
- After-session feedback by default
- Gentle language such as “Have you considered…” and “You might try…”
- A note explaining that future versions could connect to Claude or another language model through a secure backend

The prototype is not a full grammar checker. It does not understand every possible English sentence. It is a working interaction prototype that demonstrates the system logic: transcript in, private feedback out.

---

## What worked

The Type/Paste input worked well as the stable demo mode. It allowed me to clearly show the main interaction without depending on browser speech recognition.

The rule-based feedback engine can now catch common ESL patterns such as:

- “She go” → “She goes”
- “she want” → “she wants”
- “can helps” → “can help”
- “I am agree” → “I agree”
- “more good” → “better”
- “homeworks” → “homework”
- “informations” → “information”
- “researches” → “research”

The feedback cards also became more aligned with the project values. Instead of using harsh language like “wrong,” “mistake,” or “bad grammar,” the prototype uses softer wording. This makes the system feel more like support and less like grading.

The delayed feedback mode also worked conceptually. Suggestions can be saved privately and reviewed after the speaking moment. This supports the idea that students should not be interrupted while speaking.

Another thing that worked was the separation between Demo Mode and future AI mode. The interface clearly explains that the current prototype uses rule-based pattern matching, while a future version could connect to Claude or another language model through a secure backend. This makes the prototype more honest and easier to explain.

The speech mode also worked in limited cases. For example, when I slowly said a short sentence like “I am agree with this idea because it is more good for students,” the system could capture the transcript and show feedback. This helped demonstrate the future direction of the project.

---

## What broke

The biggest limitation was speech recognition.

At first, I imagined that the object could listen to the student speaking and immediately understand what they said. In practice, browser speech recognition is not stable enough. Sometimes Chrome misheard my sentence. For example, if the transcript is inaccurate, the feedback will also be inaccurate. This showed me that the first problem is not grammar analysis. The first problem is getting a reliable transcript.

This changed the interaction design. Instead of analyzing speech immediately, I added a review-before-analysis step. The student can review and edit the transcript before Lumi analyzes it. This keeps the student in control.

Another limitation is speaker separation. In a real classroom, many people may speak at the same time. A normal microphone may capture voices from nearby students. The current prototype does not solve speaker diarization or voice isolation. In the future, this may require iPhone Voice Isolation, AirPods, a directional microphone, or a more advanced classroom microphone setup.

The rule-based Demo Mode is also limited. It only catches selected patterns that I programmed into the prototype. It cannot check any random English sentence like a real language model. This is acceptable for the class demo because the goal is to show the interaction, but it is not a complete product.

There were also some smaller prototype issues that I had to fix:

- The first grammar rule generated “She gos” instead of “She goes.”
- The system originally showed only one suggestion even when the sentence had multiple issues.
- The suggestion count accumulated across tests, which made the demo confusing.
- A Flow feedback card was styled in red, which felt too much like a warning.
- Speech input was not reliable enough to use as the main demo method.

These problems helped me better understand the design. A language support object needs to be accurate, but it also needs to be calm, private, and non-judgmental.

---

## One screenshot

Screenshot: `docs/whitepaper/artifacts/yuxuan-english-communication-coach-screenshot.png`

Caption: The prototype shows Type/Paste input, Demo Mode, delayed feedback, and private suggestion cards inside the English Communication Coach interface.

---

## If I had another week

If I had another week, I would focus on three things.

First, I would improve the speech input. I would test iPhone Voice Isolation, AirPods, or a directional microphone to see whether the transcript becomes more accurate in a classroom setting.

Second, I would connect the prototype to a real language model through a secure backend. The current Demo Mode uses rule-based patterns, but a future version could use Claude or OpenAI to give more flexible feedback on grammar, vocabulary, clarity, wordiness, and presentation flow.

Third, I would test the prototype with more international students. I want to understand whether they prefer immediate feedback, after-session feedback, or both. My roommate said she would want both: gentle feedback in the moment and a full review after class. This made me realize that feedback timing should be flexible and controlled by the student.

In the longer term, I would also explore the physical smart object form. The current prototype runs on a laptop or browser, but the future version could become a small desk object that quietly signals when feedback is available.

---

## What future students should know

Do not start by trying to build a perfect AI grammar checker. Start by defining the interaction.

For this project, the most important question was not “Can the system detect grammar mistakes?” The more important question was: “How can feedback appear without making the student feel embarrassed or watched?”

The prototype does not need to be perfect to be useful. The broken parts, especially speech recognition and speaker separation, revealed the real design problem. A smart classroom object needs to support people socially and emotionally, not only technically.

The final direction of the project is:

English Communication Coach is not about correcting students in front of others. It is about creating a private, gentle support system that helps international students speak more confidently in class.
