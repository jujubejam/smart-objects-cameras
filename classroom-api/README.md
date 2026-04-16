# Smart Classroom API

The event bus for the multi-agent smart classroom.
Cameras, detectors, and student projects all connect here.

---

## Instructor quick start

```bash
cd classroom-api
pip install -r requirements.txt

# Windows
set CLASSROOM_API_KEY=testkey
python classroom_api.py

# Mac / Linux
CLASSROOM_API_KEY=testkey python classroom_api.py
```

No Supabase required. State persists automatically to `.local/classroom-api/snapshot.json`.

> **Network IP note:** When students connect from their own laptops they need your
> machine's IP, not `localhost`. Find it with `ipconfig` (Windows) or `ifconfig`
> (Mac) and share it: `http://YOUR_IP:8766`. The server listens on `0.0.0.0`
> so it is reachable from any machine on the same network.

Open the console:

```
http://localhost:8766/console
```

---

## Browser shortcuts

| URL | Purpose |
|-----|---------|
| `/console` | Live room state, phase controls, event bus, mock demo button |
| `/heartbeat` | Student check-in page — no code required |
| `/showcase/report` | Evidence report for critique and portfolio |
| `/showcase/demo-script` | Run-of-show script for Monday |
| `/projects/readiness` | Which projects are live / stale / missing |
| `/projects/nudges.md` | Per-project next actions in plain text |
| `/projects/{id}/packet.md` | One student's full check-in packet |
| `/projects/roster.csv` | Roster export for grading |

---

## Running the mock demo

Press **Run full script** on the console, or run it from the terminal:

```bash
CLASSROOM_API_URL=http://localhost:8766 CLASSROOM_API_KEY=testkey python mock_classroom_driver.py --speed 3
```

The driver walks through arrival → lecture → activity → conclude.
All student project events are included in the script — you will see them in
the console event bus and in the showcase report.

---

## Project roster

| Project ID | Student | Connects via | Contract |
|---|---|---|---|
| `seren-room` | Yuxuan (Seren) | Python / p5.js | room_mode_change → ambience.changed |
| `forest-classroom` | Sophie Lee | Python / p5.js | person_change → forest.mood_set |
| `focus-beam` | Feifey | Python / Pi | probe_classification → beam.pointed |
| `calmball` | Ramon Naula | Python / hardware | room_mode_change → calm.activated |
| `echodesk` | Kathy Choi | Python / browser | agent.message → question.displayed |
| `imprint` | Darren Chia | Python / Pi | whiteboard_change → handwriting.captured |
| `gesture-timer` | Phil | Python / Pi | timer.offer → timer.started / timer.done |
| `smart-stage` | Gordon | Python / Pi | fiducial.detected → stage.ready |
| `assignment-tracker` | Shuyang Tian | Python / Pi | whiteboard_change → assignment.suggested |
| `gus-mode` | Juju | Python / VIAM | person_change → gus.present / gus.excited |
| `horizon` | System | Pi camera | fiducial.request → fiducial.detected |
| `gravity` | System | Pi camera | fiducial.request → fiducial.detected |

Student API key: `testkey` — same for everyone in local mode.

---

## Grading criteria (finals)

The showcase report at `/showcase/report` is the grade sheet.
Score is calculated automatically (0–5 points per project):

| Points | What it means |
|--------|---------------|
| 1 | Project is registered (declared what it listens for) |
| 2 | Contract exists (declared consumes and emits) |
| 3 | Heartbeat sent + at least one event in the bus |
| 4 | Contract + heartbeat + event (demonstrated the loop) |
| 5 | Live at critique time + contract + event |

**Minimum passing score for finals: 3/5**
(Heartbeat sent, at least one event proves the interaction happened)

**Full credit: 5/5**
(Live at critique time with a complete consume → emit loop on the bus)

The bus records evidence even when hardware fails.
A mock event with the right payload counts the same as real hardware —
the contract is the project, not the device.

---

## Sending a heartbeat for a student (if their code isn't working)

Open:

```
http://localhost:8766/heartbeat
```

Enter their project ID and capabilities. That counts as evidence.

Or run the heartbeat script directly:

```bash
set CLASSROOM_API=http://localhost:8766
set PROJECT_ID=seren-room
set PROJECT_API_KEY=testkey
set CAPABILITIES=ambience,ambience.changed,room.adapted
set CONSUMES=person_change,probe_classification,room_mode_change
set EMITS=ambience.changed,room.adapted
python student_heartbeat.py
```

---

## API key

The classroom API key for local demo is `testkey`.
Any student project that sends `X-API-Key: testkey` will be accepted.

For a real deployment, set `CLASSROOM_API_KEY=your-secret` in the environment
before starting the API.

---

## Resetting the room between demos

```bash
curl -X POST http://localhost:8766/mock/reset \
  -H "Content-Type: application/json" \
  -H "X-API-Key: testkey" \
  -d "{}"
```

Or press **Reset room** on the console.

---

## Checking what the room knows right now

```bash
curl http://localhost:8766/room/context
curl http://localhost:8766/projects/readiness
curl http://localhost:8766/capabilities
```

---

## File map

```
classroom-api/
  classroom_api.py        — the server (FastAPI + orchestrator + contracts)
  orchestrator.py         — phase state and salience routing logic
  mock_classroom_driver.py — demo script: runs a full class session
  student_heartbeat.py    — minimum heartbeat client (Python, no SSE)
  student_template.py     — full integration template with SSE listener
  student_heartbeat.html  — minimum heartbeat (browser, no code)
  student_template.html   — full integration template (browser, p5.js)
  console.html            — instructor console
  requirements.txt        — pip dependencies
  .local/classroom-api/   — local snapshot (auto-created, not committed)

docs/
  CLASSROOM_CONTRACTS.md  — event envelope spec and per-project contracts
  STUDENT_BUS_GUIDE.md    — student step-by-step + per-project Claude prompts
```

---

## Endpoints reference

```
GET  /health                          service status
GET  /state                           all camera states + room mode
GET  /mode                            room mode only
GET  /phase                           current session phase
GET  /room/context                    summarized room context for agents
GET  /contracts                       all project contracts
GET  /capabilities                    live capability index

GET  /projects                        project registry
GET  /projects/readiness              readiness table (use for grading)
GET  /projects/nudges.md              plain-text next actions per project
GET  /projects/roster.csv             roster CSV export
GET  /projects/{id}/packet.md         one student's check-in packet
GET  /projects/{id}/events            a project's event history

POST /projects/{id}/heartbeat         project check-in
POST /projects/{id}/events            project publishes an event
POST /contracts/validate              validate an event payload
POST /capabilities/route              resolve a capability to a project

GET  /subscribe/state                 SSE: room state changes
GET  /subscribe/events?subscriber_id= SSE: routed project events

GET  /showcase/report                 evidence report (Markdown)
GET  /showcase/demo-script            run-of-show script
GET  /showcase/report.json            evidence report (JSON)

POST /phase                           set session phase
POST /push/state                      detector pushes camera state
POST /mock/scenario                   run a built-in demo script
POST /mock/reset                      reset room to unknown phase
POST /labs/import                     import objects.json from labs repo

GET  /console                         instructor console (HTML)
GET  /heartbeat                       student check-in page (HTML)
```
