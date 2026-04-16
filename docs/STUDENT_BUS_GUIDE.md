# Smart Classroom Bus — Student Guide

**This is your finals assignment.**
Your project must connect to the classroom event bus and leave evidence that it
can participate in the room.

---

## What you need to submit

| What | How to verify |
|------|---------------|
| Your project is registered on the bus | It appears on `/projects/readiness` |
| Your project sent a heartbeat | `has_heartbeat: true` in the readiness table |
| Your project emitted at least one event | `event_count > 0` in the readiness table |
| Your project consumed and responded to at least one event | Visible in `/showcase/report` |

**The bus records evidence even when hardware fails.**
A mock heartbeat + one mock event counts. Your grade is based on the interaction
design — did the room know your project existed and did your project respond to
the room?

---

## Scoring (5 points total)

| Score | What it means |
|-------|---------------|
| 1/5 | Project is registered (has declared events) |
| 2/5 | Contract exists (consumes and emits declared) |
| 3/5 | **Heartbeat sent** + at least one event in the bus |
| 4/5 | Contract + heartbeat + event (full loop demonstrated) |
| 5/5 | **Live at critique** + contract + event |

Minimum passing: **3/5**. Target: **5/5** — be live when Carrie opens the report.

---

## The server address

All examples use `http://localhost:8766`.
If you are connecting from your own laptop (not the machine running the server),
replace `localhost` with the classroom IP that Carrie gives you, e.g.
`http://192.168.1.55:8766`.

The API key is `testkey` for everyone.

---

## Step 1 — Find your project

Check the readiness table:

```
http://CLASSROOM_IP:8766/projects/readiness
```

Find your `project_id` in the list. If your project shows `needs-heartbeat`,
go to Step 2 now.

Your personal packet (with the exact command to run):

```
http://localhost:8766/projects/YOUR_PROJECT_ID/packet.md
```

---

## Step 2 — Send a heartbeat (pick one)

### Option A — Browser (no code, works on any computer)

Open:

```
http://CLASSROOM_IP:8766/heartbeat
```

Fill in your Project ID, select what your project Consumes and Emits, and click
**Send Heartbeat**. This is enough to move from `needs-heartbeat` to `stale`.

### Option B — Python script (recommended)

```bash
set CLASSROOM_API=http://CLASSROOM_IP:8766
set PROJECT_ID=YOUR_PROJECT_ID
set PROJECT_API_KEY=testkey
set CAPABILITIES=YOUR_CAPABILITY_1,YOUR_CAPABILITY_2
set CONSUMES=event_type_you_listen_for
set EMITS=event_type_you_send_back
python classroom-api/student_heartbeat.py
```

Leave this running. It sends a heartbeat every 30 seconds so you show as **live**
at critique time.

### Option C — curl (if Python isn't available)

```bash
curl -X POST http://localhost:8766/projects/YOUR_PROJECT_ID/heartbeat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: testkey" \
  -d "{\"status\":\"online\",\"capabilities\":[\"YOUR_CAPABILITY\"],\"consumes\":[\"room_mode_change\"],\"emits\":[\"your.event\"],\"message\":\"my project is alive\"}"
```

---

## Step 3 — Emit at least one event

Even if your hardware is not working, publish a mock event:

```bash
curl -X POST http://localhost:8766/projects/YOUR_PROJECT_ID/events \
  -H "Content-Type: application/json" \
  -H "X-API-Key: testkey" \
  -d "{\"event_type\":\"YOUR_EVENT\",\"payload\":{\"message\":\"mock event — hardware not ready\"}}"
```

Or use `student_template.py`:

```python
publish_event("your.event", {"status": "mock"})
```

---

## Step 4 — Verify you are done

Open the report:

```
http://CLASSROOM_IP:8766/showcase/report
```

Find your project in the table. You should see:

- `level`: `live` (if you're still running) or `stale` (if you ran earlier)
- `has_heartbeat`: `true`
- `event_count`: 1 or more

If you see that, you're done.

---

## Step 5 — Stay live at critique

Keep the heartbeat script running while the professor goes through the report.
The live window is **2 minutes** — the script re-sends every 30 seconds,
so you need to be running for the whole critique.

---

## Per-project starter prompts for Claude Code

Below are prompts you can paste directly into Claude Code to build your
bus integration. Each prompt produces a runnable Python script.

---

### seren-room — Yuxuan (Seren)

```
I'm building "A Room", a context-aware classroom ambience system.
My project ID on the classroom bus is "seren-room".
The bus API is at http://localhost:8766, API key is "testkey".

Using classroom-api/student_template.py as a base, write a Python script that:

1. Imports the heartbeat, publish_event, and subscribe_to_events functions
   from student_template.py (or copies them inline).
2. Sends a heartbeat on startup declaring:
   - capabilities: ["ambience", "ambience.changed", "room.adapted"]
   - consumes: ["person_change", "probe_classification", "room_mode_change"]
   - emits: ["ambience.changed", "room.adapted"]
3. Subscribes to SSE events at /subscribe/events?subscriber_id=seren-room
4. When a room_mode_change event arrives:
   - mode "lecture" or "focus"     → publish ambience.changed with {"mode": "focus", "intensity": 0.6}
   - mode "group" or "activity"    → publish ambience.changed with {"mode": "warm", "intensity": 0.8}
   - mode "empty"                  → publish ambience.changed with {"mode": "off", "intensity": 0.0}
   - any other mode                → publish room.adapted with {"adaptation": mode, "reason": "room shifted"}
5. Prints every event received and every event emitted with a timestamp.
6. Re-sends the heartbeat every 90 seconds in a background thread.

Run it with:
    set PROJECT_ID=seren-room
    set PROJECT_API_KEY=testkey
    set CLASSROOM_API=http://CLASSROOM_IP:8766
    python seren_room.py

Verify it works by running mock_classroom_driver.py in another terminal and
watching the printed output.
```

---

### forest-classroom — Sophie Lee

```
I'm building "Forest in the Classroom", a living forest projection that reacts
to who is in the room and how they are behaving.
My project ID is "forest-classroom".
The bus API is at http://localhost:8766, API key is "testkey".

Using classroom-api/student_template.py as a base, write a Python script that:

1. Sends a heartbeat on startup declaring:
   - capabilities: ["forest", "forest.mood_set", "forest.pulse"]
   - consumes: ["person_change", "room_mode_change"]
   - emits: ["forest.mood_set", "forest.pulse"]
2. Subscribes to SSE events at /subscribe/events?subscriber_id=forest-classroom
3. On person_change:
   - new_count > 10 → publish forest.mood_set with {"mood": "active", "intensity": 0.9}
   - new_count > 3  → publish forest.mood_set with {"mood": "present", "intensity": 0.7}
   - new_count == 0 → publish forest.mood_set with {"mood": "empty", "intensity": 0.1}
4. On room_mode_change:
   - mode "lecture"  → publish forest.mood_set with {"mood": "quiet", "intensity": 0.3}
   - mode "activity" → publish forest.pulse with {"count": payload.get("total_persons", 0)}
5. Prints each event received and emitted.
6. Re-sends heartbeat every 90 seconds.

Verify by running mock_classroom_driver.py and checking
http://CLASSROOM_IP:8766/showcase/report for forest-classroom events.
```

---

### focus-beam — Feifey

```
I'm building "Focus Beam", an assistive spotlight that follows the instructor
during presentations and turns off during group work.
My project ID is "focus-beam".
The bus API is at http://localhost:8766, API key is "testkey".

Using classroom-api/student_template.py as a base, write a Python script that:

1. Sends a heartbeat declaring:
   - capabilities: ["focus-beam", "beam.pointed", "beam.off"]
   - consumes: ["probe_classification", "room_mode_change"]
   - emits: ["beam.pointed", "beam.off"]
2. Subscribes to /subscribe/events?subscriber_id=focus-beam
3. On probe_classification:
   - new_class "presentation" → publish beam.pointed with {"x": 0.5, "y": 0.2, "focus": "instructor"}
   - new_class "discussion"   → publish beam.off with {"reason": "group mode"}
4. On room_mode_change:
   - mode "lecture"  → publish beam.pointed with {"x": 0.5, "y": 0.2, "focus": "whiteboard"}
   - mode "activity" → publish beam.off with {"reason": "activity mode"}
   - mode "empty"    → publish beam.off with {"reason": "room empty"}
5. Prints each event received and emitted.
6. Re-sends heartbeat every 90 seconds.

Verify by running mock_classroom_driver.py and watching beam events appear in
http://localhost:8766/projects/focus-beam/events
```

---

### calmball — Ramon Naula

```
I'm building "CalmBall", a stress regulation object that activates calming
sounds and haptics when the classroom energy gets high.
My project ID is "calmball".
The bus API is at http://localhost:8766, API key is "testkey".

Using classroom-api/student_template.py as a base, write a Python script that:

1. Sends a heartbeat declaring:
   - capabilities: ["calm", "calm.activated", "calm.deactivated"]
   - consumes: ["room_mode_change", "person_change"]
   - emits: ["calm.activated", "calm.deactivated"]
2. Subscribes to /subscribe/events?subscriber_id=calmball
3. On room_mode_change:
   - mode "activity" with total_persons > 8 → publish calm.activated with
     {"sound": "forest-ambience", "intensity": 0.7}
   - mode "lecture"  → publish calm.activated with {"sound": "silence", "intensity": 0.3}
   - mode "empty"    → publish calm.deactivated with {"reason": "room empty"}
4. On person_change where new_count drops below 3:
   → publish calm.deactivated with {"reason": "room quieting"}
5. Prints each event received and emitted.
6. Re-sends heartbeat every 90 seconds.
```

---

### echodesk — Kathy Choi

```
I'm building "EchoDesk", a shared question board that surfaces student questions
when the class is in discussion mode and accepts directed messages from the
orchestrator.
My project ID is "echodesk".
The bus API is at http://localhost:8766, API key is "testkey".

Using classroom-api/student_template.py as a base, write a Python script that:

1. Sends a heartbeat declaring:
   - capabilities: ["echodesk", "question.submitted", "question.displayed"]
   - consumes: ["room_mode_change", "agent.message"]
   - emits: ["question.submitted", "question.displayed"]
2. Subscribes to /subscribe/events?subscriber_id=echodesk
3. On room_mode_change to "activity" or "group":
   → publish question.displayed with {"text": "What does your project do?", "display_id": "q-auto"}
4. On agent.message (a directed message routed to echodesk):
   text = payload.get("text")
   → publish question.displayed with {"text": text, "display_id": "q-directed"}
5. Simulate a student submitting a question every 2 minutes:
   → publish question.submitted with {"text": "Can someone explain the orchestrator?", "student": "mock"}
6. Prints each event received and emitted.
7. Re-sends heartbeat every 90 seconds.

Test by sending a directed agent.message to echodesk:
    curl -X POST http://localhost:8766/projects/smart-stage/events \
      -H "Content-Type: application/json" -H "X-API-Key: testkey" \
      -d "{\"event_type\":\"agent.message\",\"target\":\"echodesk\",\"payload\":{\"text\":\"Hello EchoDesk\",\"tone\":\"helpful\"}}"
```

---

### imprint — Darren Chia

```
I'm building "Imprint", a system that reads handwriting from surfaces and saves
it as structured notes.
My project ID is "imprint".
The bus API is at http://localhost:8766, API key is "testkey".

Using classroom-api/student_template.py as a base, write a Python script that:

1. Sends a heartbeat declaring:
   - capabilities: ["imprint", "handwriting.captured", "note.saved"]
   - consumes: ["whiteboard_change", "fiducial.detected"]
   - emits: ["handwriting.captured", "note.saved"]
2. Subscribes to /subscribe/events?subscriber_id=imprint
3. On whiteboard_change where text_detected is True:
   text_list = payload.get("text", [])
   → publish handwriting.captured with {"text": ", ".join(text_list), "surface": "whiteboard"}
   → publish note.saved with {"text": ", ".join(text_list), "title": "Whiteboard capture"}
4. On fiducial.detected where zone is not None:
   → publish handwriting.captured with
     {"text": f"tag {payload['tag_id']} in zone {payload['zone']}", "surface": "fiducial"}
5. Prints each event received and emitted.
6. Re-sends heartbeat every 90 seconds.

Verify by running mock_classroom_driver.py — it triggers both whiteboard_change
and fiducial.detected events which imprint should respond to.
```

---

### gesture-timer

```
I'm building a "Gesture Classroom Timer" that responds to timer offers from the
room and starts/stops a visible countdown.
My project ID is "gesture-timer".
The bus API is at http://localhost:8766, API key is "testkey".

Using classroom-api/student_template.py as a base, write a Python script that:

1. Sends a heartbeat declaring:
   - capabilities: ["timer", "timer.offer", "timer.command"]
   - consumes: ["timer.offer", "timer.command"]
   - emits: ["timer.started", "timer.done", "timer.declined"]
2. Subscribes to /subscribe/events?subscriber_id=gesture-timer
3. On timer.offer (a directed event asking if we can run a timer):
   minutes = payload.get("minutes", 5)
   → publish timer.started with {"minutes": minutes, "started_by": "gesture",
     "ends_at": (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()}
   → after a simulated wait (or real timer): publish timer.done with
     {"minutes": minutes, "completed": True}
4. On timer.command where action == "stop":
   → publish timer.cancelled with {"reason": "manual stop"}
5. Prints each event.
6. Re-sends heartbeat every 90 seconds.

Test by sending a timer offer:
    curl -X POST http://localhost:8766/projects/smart-stage/events \
      -H "Content-Type: application/json" -H "X-API-Key: testkey" \
      -d "{\"event_type\":\"timer.offer\",\"target\":\"gesture-timer\",\"payload\":{\"minutes\":1,\"reason\":\"test\"}}"

Verify timer.started appears in the bus and timer.done appears 1 minute later.
```

---

### smart-stage — Gordon

```
I'm building "Smart Stage", a system that manages the presentation setup,
tracks fiducial markers (tagged objects), and offers timer control to the room.
My project ID is "smart-stage".
The bus API is at http://localhost:8766, API key is "testkey".

Using classroom-api/student_template.py as a base, write a Python script that:

1. Sends a heartbeat declaring:
   - capabilities: ["stage.ready", "stage.needs_setup", "timer.offer"]
   - consumes: ["person_change", "probe_classification", "room_mode_change", "fiducial.detected"]
   - emits: ["timer.offer", "fiducial.request", "stage.ready", "stage.needs_setup"]
2. Subscribes to /subscribe/events?subscriber_id=smart-stage
3. On probe_classification where new_class == "presentation":
   → publish stage.ready with {"message": "presentation mode detected"}
   → publish timer.offer directed to "gesture-timer" with
     {"minutes": 45, "reason": "class session started", "mode": "lecture"}
4. On room_mode_change to "activity":
   → publish timer.offer directed to "gesture-timer" with
     {"minutes": 5, "reason": "group activity started", "mode": "group"}
   → publish fiducial.request directed to "horizon" with
     {"marker_family": "tag36h11", "needed_for": "smart-stage",
      "zones": ["podium", "whiteboard", "collaboration"]}
5. On fiducial.detected:
   → print f"Marker {payload['tag_id']} seen at zone {payload.get('zone')}"
6. Re-sends heartbeat every 90 seconds.
```

---

### gus-mode — Juju

```
I'm building "Gus Mode" — a virtual presence of Gus the dog.
A VIAM rover at Kathy's apartment streams live video of Gus.
I segment his figure and project a life-size image onto the classroom wall.
My project ID is "gus-mode".
The bus API is at http://CLASSROOM_IP:8766, API key is "testkey".

Using classroom-api/student_template.py as a base, write a Python script that:

1. Sends a heartbeat declaring:
   - capabilities: ["gus-mode", "gus.present", "gus.excited", "gus.sleeping"]
   - consumes: ["person_change", "room_mode_change"]
   - emits: ["gus.present", "gus.excited", "gus.sleeping"]
2. Subscribes to /subscribe/events?subscriber_id=gus-mode
3. On person_change:
   - new_count > 0  → publish gus.present with {"activity": "watching", "energy": 0.6}
   - new_count == 0 → publish gus.sleeping with {"reason": "empty room"}
4. On room_mode_change:
   - mode "activity" or "group" → publish gus.excited with {"trigger": "group energy"}
   - mode "lecture"             → publish gus.sleeping with {"reason": "lecture in progress"}
   - mode "empty"               → publish gus.sleeping with {"reason": "everyone left"}
5. Prints each event received and emitted with a timestamp.
6. Re-sends heartbeat every 90 seconds.

Verify by running mock_classroom_driver.py and checking
http://CLASSROOM_IP:8766/projects/gus-mode/events — you should see gus.excited
appear during the activity phase and gus.sleeping during lecture.
```

---

### assignment-tracker — Shuyang Tian

```
I'm building "Assignment Progress Tracking", a system that reads whiteboard
content and suggests assignment reminders to students.
My project ID is "assignment-tracker".
The bus API is at http://localhost:8766, API key is "testkey".

Using classroom-api/student_template.py as a base, write a Python script that:

1. Sends a heartbeat declaring:
   - capabilities: ["assignment.suggested", "assignment.created", "assignment.reminder"]
   - consumes: ["whiteboard_change", "phase_change"]
   - emits: ["assignment.suggested", "assignment.created", "assignment.reminder"]
2. Subscribes to /subscribe/events?subscriber_id=assignment-tracker
3. On whiteboard_change where text_detected is True:
   text_list = payload.get("text", [])
   if any word in text_list looks like an assignment or deadline:
   → publish assignment.suggested with {"title": ", ".join(text_list), "due": "TBD"}
4. On phase_change to "conclude":
   → publish assignment.reminder with
     {"title": "Connect your project to the classroom bus", "due": "Monday"}
5. On phase_change to "activity":
   → publish assignment.created with
     {"title": "Group work session", "due": "end of class"}
6. Re-sends heartbeat every 90 seconds.
```

---

## How to run a full integration test

1. In terminal 1, start the API:

   ```bash
   CLASSROOM_API_KEY=testkey python classroom_api.py   # run from classroom-api/
   ```

2. In terminal 2, start your project script (or the heartbeat):

   ```bash
   set PROJECT_ID=seren-room
   set PROJECT_API_KEY=testkey
   python my_project.py
   ```

3. In terminal 3, run the mock classroom driver:

   ```bash
   CLASSROOM_API_URL=http://localhost:8766 CLASSROOM_API_KEY=testkey \
   python classroom-api/mock_classroom_driver.py --speed 3
   ```

4. Watch your project's output — it should print events it received
   and events it emitted.

5. Open the report:

   ```
   http://CLASSROOM_IP:8766/showcase/report
   ```

   Find your project. Check:
   - `level: live` (shows green)
   - `event_count: N` (shows your emitted events)

6. Open your project's packet for a summary:

   ```
   http://localhost:8766/projects/YOUR_PROJECT_ID/packet.md
   ```

---

## Troubleshooting

**"I get a connection error"**
The API isn't running. Start it first:
```bash
CLASSROOM_API_KEY=testkey python classroom_api.py   # run from classroom-api/
```

**"My heartbeat works but I'm still stale"**
Stale is fine — it means you checked in before. To show as `live`, keep the
heartbeat script running during critique (it refreshes every 30 seconds).

**"My project isn't showing up"**
Use the exact `project_id` from the roster above. Typos create new entries.
Check the readiness table: `http://CLASSROOM_IP:8766/projects/readiness`

**"The mock driver ran but I see no events for my project"**
Your project needs to subscribe to the SSE stream and respond.
Check that your `subscribe_to_events` callback is working.
Alternatively, publish a mock event manually:
```bash
curl -X POST http://localhost:8766/projects/YOUR_PROJECT_ID/events \
  -H "Content-Type: application/json" -H "X-API-Key: testkey" \
  -d "{\"event_type\":\"YOUR_EVENT\",\"payload\":{\"mock\":true}}"
```

**"I don't have working hardware"**
That's fine. Send mock events with `{"mock": true}` in the payload.
The contract is the interaction design. A mock event proving the loop is worth
the same grade points as real hardware.

**Validate your event payload before publishing:**
```bash
curl -X POST http://localhost:8766/contracts/validate \
  -H "Content-Type: application/json" \
  -d "{\"project_id\":\"YOUR_PROJECT_ID\",\"event_type\":\"YOUR_EVENT\",\"payload\":{...}}"
```

---

## What Carrie will see at critique

She will open `http://CLASSROOM_IP:8766/showcase/report`.

For each project she can see:
- Is it live right now? (has heartbeat in the last 2 minutes)
- Did it emit any events? (event_count)
- What did it claim to consume and emit? (contract)
- Did it leave evidence of a real consume → emit loop?

The demo script at `/showcase/demo-script` walks her through the run-of-show.
The roster CSV at `/projects/roster.csv` is the grade sheet export.

Your job is to make sure your row in that table shows:
- `has_heartbeat: true`
- `event_count > 0`
- `level: live` (at critique time)
