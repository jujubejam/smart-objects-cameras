# Vignette: English Communication Coach

Student: Yuxuan Chen  
Project: English Communication Coach / Ambient English Feedback Object  
When it happened: Smart Objects class discussion, [04.13.2026]  
Who was in the room: Kathy, Phil, classmates, and me  

## The moment

During a Smart Objects class discussion, I shared my idea for an English Communication Coach. At first, I imagined it as a small classroom object that could listen when I speak English and give grammar feedback. My early idea was more direct: if I made a grammar issue while speaking, the object could notice it and show a suggestion.

Kathy’s response changed how I thought about the project. She said she did not want the feedback to feel like, “you made a mistake here. Fix it.” She suggested that the system should sound more encouraging, like, “this is not bad, but have you considered this?” In another discussion, she also pointed out that if the system corrected someone every time, especially with something like a red light, the student might become afraid of speaking.

That moment made me realize that the biggest design problem was not only grammar detection. The bigger problem was how the feedback feels in a classroom. A grammar suggestion can be useful, but if it appears at the wrong time or in the wrong tone, it can make the student feel watched, embarrassed, or less confident.

Phil also reminded me that the most important thing was not only how to build the system, but how I wanted to interact with it. After this conversation, I changed the project direction. Lumi should not act like a correction machine. It should be a quiet support object that saves gentle suggestions and lets the student review them privately.

## Why this moment mattered

This moment helped me understand that feedback is not neutral. In a classroom, timing, tone, and privacy matter as much as technical accuracy.

Before this discussion, I was thinking mostly about real-time grammar correction. After Kathy’s feedback, I shifted the interaction toward delayed, private, and encouraging feedback. The student should not be interrupted while speaking. They should have control over when to look at the feedback and whether to use it.

This also changed the language of the prototype. Instead of saying “wrong” or “mistake,” Lumi uses softer phrases like “Have you considered…” or “You might try…” The goal is not to grade the student. The goal is to help international students speak with more confidence in class.

## What this revealed about the system

This moment revealed a tension inside the project:

The system needs to notice language patterns, but it should not make the student feel monitored.

Because of that, the current prototype uses Type/Paste as the stable demo input and treats speech input as experimental. Speech recognition can be inaccurate, so the student can review and edit the transcript before analysis. This keeps the student in control.

For me, this moment became the center of the project. English Communication Coach is not about correcting students in front of others. It is about creating a small, private support system inside the classroom.
