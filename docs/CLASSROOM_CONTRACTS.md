# Smart Classroom Contracts

This is the contract layer for the classroom bus. It is intentionally small:
students can mock it, projects can use it before hardware works, and working
projects can replace mocks without changing the rest of the system.

The design question from the FigJam board is the routing question:

> When does the room look at the room, and when does it output to an individual device?

The answer is a shared room state plus a live event bus:

- `GET /state` gives the current room snapshot.
- `GET /phase` gives the current class phase.
- `GET /subscribe/state` streams room state changes.
- `GET /subscribe/events?subscriber_id=<project_id>` streams routed events.
- `POST /projects/<project_id>/events` lets projects talk back.
- `GET /room/context` returns the shared room context object.
- `POST /contracts/validate` checks an event payload before publishing it.
- `POST /mock/scenario` runs a built-in mock path through the same API calls.
- `POST /projects/:id/heartbeat` lets a project announce it is alive and what it can handle.
- `GET /capabilities` lists available providers for things like `timer` or `fiducials`.
- `GET /labs/config` reads the companion labs/prototype repo object-zone config.
- `POST /labs/import` turns that object-zone config into classroom bus events.

The orchestrator sits in the middle. It decides whether events are ambient,
broadcast, or directed to a named subscriber.

## What To Run Right Now

Run the classroom API:

```bash
cd classroom-api
pip install fastapi uvicorn supabase sse-starlette python-dotenv requests sseclient-py
CLASSROOM_API_KEY=testkey python classroom_api.py
```

Run a browser/student project against it:

```bash
cd classroom-api
python -m http.server 8080
```

Open `http://localhost:8080/student_template.html` and set:

```js
const API_BASE = 'http://localhost:8766';
const PROJECT_ID = 'gesture-timer';
const API_KEY = 'any-key-in-local-mode';
```

For a backend/Pi project, copy `student_template.py`, set the same project ID,
and run it with `CLASSROOM_API=http://localhost:8766`.

Run the mock classroom script when hardware or student projects are not ready:

```bash
cd classroom-api
CLASSROOM_API_URL=http://localhost:8766 CLASSROOM_API_KEY=testkey python mock_classroom_driver.py --speed 2
```

It emits the same events real projects should emit: phase changes, detector
state, `timer.offer`, `fiducial.request`, `timer.started`, `fiducial.detected`,
and `timer.done`.

The frontend console at `http://localhost:8766/console` calls the same endpoint
when you press `Run full script`.

If the labs/prototype repo is on the same machine, import its object-zone
configuration into the bus:

```bash
curl -X POST http://localhost:8766/labs/import \
  -H "Content-Type: application/json" \
  -H "X-API-Key: testkey" \
  -d "{\"provider\":\"horizon\"}"
```

This emits `fiducial.detected`, `fiducial.zone_entered`, and
`labs.rule_triggered` events from the existing `server/objects.json` file.

## Event Envelope

Every event on `/subscribe/events` should look like this:

```json
{
  "event_type": "timer.offer",
  "source": "project",
  "project_id": "smart-stage",
  "target": "gesture-timer",
  "created_at": "2026-04-16T18:20:00Z",
  "payload": {
    "minutes": 5,
    "reason": "group activity started"
  }
}
```

Required:

- `event_type`: dotted name for project events, snake case for existing detector events.
- `payload`: event-specific data.

Usually present:

- `source`: `detector`, `project`, `orchestrator`, or `mock`.
- `project_id`: the student project that emitted the event.
- `target`: optional subscriber ID. If set, only that subscriber receives it.

The current detector event names stay as-is:

- `person_change`
- `probe_classification`
- `room_mode_change`
- `fatigue_change`
- `anomaly_change`
- `whiteboard_change`
- `phase_change`

New project events should use dotted names:

- `timer.offer`
- `timer.started`
- `timer.done`
- `fiducial.detected`
- `fiducial.zone_entered`
- `agent.message`

## Validation

Known contract events are validated by the API. Unknown custom events are still
accepted so students can experiment, but known events get fixable errors.

Validate without publishing:

```bash
curl -X POST http://localhost:8766/contracts/validate \
  -H "Content-Type: application/json" \
  -d "{\"project_id\":\"smart-stage\",\"event_type\":\"timer.offer\",\"target\":\"gesture-timer\",\"payload\":{\"minutes\":5,\"reason\":\"activity started\"}}"
```

If a required field is missing or has the wrong type, the API returns `422` from
`POST /projects/<project_id>/events` with:

```json
{
  "message": "timer.offer does not match the classroom contract. Fix the payload fields below and send it again.",
  "errors": [
    {"field": "minutes", "message": "missing required field 'minutes'", "expected": "number"}
  ],
  "example": {"minutes": 5, "reason": "activity started", "mode": "group"}
}
```

That error is meant to be pasted directly into Claude or fixed by comparing the
payload to the example.

## Room Context

`GET /room/context` is the shared state object for agents that do not want to
reconstruct the room from raw detector events:

```json
{
  "phase": "activity",
  "room_mode": "group",
  "feeling": "collaborative",
  "temporal": {"phase": "activity", "duration_sec": 31.2},
  "social": {"total_persons": 14, "group_present": true},
  "task": {"whiteboard_active": false, "probe_classes": ["discussion"]},
  "active_capabilities": ["presence", "jepa_probe", "timer", "fiducials", "directed_events"]
}
```

This is the object that answers "how does the room feel?" for project code.

## Heartbeats and Capabilities

The lowest acceptable integration is a heartbeat. It says:

- my project exists
- this is what I intend it to do
- this is what it consumes and emits
- this is the last time it was alive

Browser check-in:

```text
http://localhost:8766/heartbeat
```

Python check-in:

```bash
cd classroom-api
set CLASSROOM_API=http://localhost:8766
set PROJECT_ID=gesture-timer
set PROJECT_API_KEY=testkey
set CAPABILITIES=timer,timer.offer,timer.command
python student_heartbeat.py
```

Projects can announce what they can handle without changing their whole app:

```bash
curl -X POST http://localhost:8766/projects/gesture-timer/heartbeat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: testkey" \
  -d "{\"status\":\"online\",\"capabilities\":[\"timer\",\"timer.offer\",\"timer.command\"],\"message\":\"timer mock running\"}"
```

Resolve a capability into a target:

```bash
curl -X POST http://localhost:8766/capabilities/route \
  -H "Content-Type: application/json" \
  -d "{\"capability\":\"fiducials\",\"needed_by\":\"smart-stage\",\"reason\":\"stage setup\"}"
```

That returns the project ID that should receive the directed request. If a real
project is alive, it wins. If not, the registered mock contract can still carry
the interaction.

## Project Contract Shape

Each project should have a contract. This can live in `student_projects.config`
or in a local `contract.json` while mocking.

```json
{
  "project_id": "gesture-timer",
  "display_name": "Gesture Classroom Timer",
  "contract_version": "classroom-contracts/v0.1",
  "consumes": ["phase_change", "room_mode_change", "timer.offer", "timer.command"],
  "emits": ["timer.started", "timer.done", "timer.cancelled", "timer.declined"],
  "commands": {
    "timer.offer": {
      "payload": {
        "minutes": "number",
        "reason": "string",
        "mode": "string"
      }
    },
    "timer.command": {
      "payload": {
        "action": "start | stop | pause | resume",
        "minutes": "number?"
      }
    }
  },
  "mock": {
    "can_run_without_hardware": true,
    "default_response": "timer.started"
  }
}
```

This gives every student the same assignment boundary:

1. Listen for these events.
2. Send these events back.
3. If the real device is not working, run the mock path with the same event names.

## Timer Contract

The timer is the simplest example because the in/out surface is obvious.

Consumes:

```json
{
  "event_type": "timer.offer",
  "target": "gesture-timer",
  "payload": {
    "minutes": 5,
    "reason": "activity mode started",
    "mode": "group"
  }
}
```

```json
{
  "event_type": "timer.command",
  "target": "gesture-timer",
  "payload": {
    "action": "start",
    "minutes": 5
  }
}
```

Emits:

```json
{
  "event_type": "timer.started",
  "payload": {
    "minutes": 5,
    "started_by": "gesture",
    "ends_at": "2026-04-16T18:25:00Z"
  }
}
```

```json
{
  "event_type": "timer.done",
  "payload": {
    "minutes": 5,
    "completed": true
  }
}
```

If the gesture code fails, the mock timer should still accept `timer.offer` and
emit `timer.started` / `timer.done`.

## Fiducial Contract

This is how Phil's fiducials become meaningful.

Fiducials should not just be "more data." They matter when another agent needs
object identity or spatial position to make a decision.

Examples:

- The room enters `activity` phase and asks which table groups are active.
- Smart Stage needs to know whether the marker, laptop, or podium tag is in the stage zone.
- A student project asks, "Where is my tagged object right now?"
- Horizon or Gravity can say, "I can run the fiducial app, move me where you need me."

Fiducial agent consumes:

```json
{
  "event_type": "fiducial.request",
  "target": "horizon",
  "payload": {
    "marker_family": "tag36h11",
    "needed_for": "smart-stage",
    "zones": ["podium", "whiteboard", "collaboration"]
  }
}
```

Fiducial agent emits:

```json
{
  "event_type": "fiducial.detected",
  "project_id": "horizon",
  "payload": {
    "tag_id": 3,
    "name": "Whiteboard Marker",
    "x": 0.52,
    "y": 0.08,
    "zone": "whiteboard",
    "confidence": 0.92
  }
}
```

```json
{
  "event_type": "fiducial.zone_entered",
  "project_id": "horizon",
  "payload": {
    "tag_id": 7,
    "name": "Instructor Podium",
    "zone": "podium"
  }
}
```

The key is that consumers should ask for a capability, not a specific camera:

```json
{
  "event_type": "capability.request",
  "target": "orchestrator",
  "payload": {
    "capability": "fiducials",
    "needed_by": "smart-stage",
    "reason": "stage setup"
  }
}
```

The orchestrator can then direct `fiducial.request` to whichever camera/project
is available: `horizon`, `gravity`, or a mock fiducial agent.

## J-EPA / Room Mode Contract

V-JEPA can keep pushing through `/push/state`:

```json
{
  "camera_id": "orbit",
  "predicted_class": "discussion",
  "prediction_confidence": 0.86,
  "class_probs": {
    "lecture": 0.08,
    "discussion": 0.86,
    "presentation": 0.06
  }
}
```

The API turns that into:

- `probe_classification`
- possible `room_mode_change`
- updated `GET /state`

The important architectural move is: J-EPA does not command projects directly.
It tells the room what it thinks is happening. The orchestrator turns that into
offers, requests, or directed messages.

Example flow:

1. J-EPA emits `probe_classification` with `new_class=discussion`.
2. The room enters or suggests `activity`.
3. Orchestrator sends `timer.offer` to `gesture-timer`.
4. Orchestrator sends `fiducial.request` to `horizon` if spatial object tracking is useful.
5. Projects reply with `timer.started`, `timer.done`, `fiducial.detected`, or `fiducial.zone_entered`.

## Mock-First Rule

For Monday, every project should be accepted if it can do one of these:

- Real hardware listens and emits the contract events.
- Browser mock listens and emits the contract events.
- Python script listens and emits the contract events.
- Manual curl commands emit the contract events.

The system should not care which path they used. The contract is the project.

This is not a way to erase student work or rewrite their projects. It is the
opposite: the contract preserves the interaction design even when the technical
implementation fails. A student can still learn from a broken sensor, a fragile
browser sketch, or an incomplete Raspberry Pi script because the intended
conversation is explicit:

1. What does the project need from the room?
2. What can the project offer back?
3. What should happen when it works?
4. What should happen when it does not?

That makes failure usable. The live hardware is one implementation of the
interaction. The mock is another. The design can still be critiqued, filmed,
tested, and improved.

## Critique And Portfolio Evidence

The console now has a readiness and evidence layer:

- `GET /projects/readiness` shows whether each project has a contract,
  declared input/output, heartbeat, emitted events, and live status.
- `GET /projects/nudges.md` turns readiness into pasteable next actions.
- `GET /projects/:id/packet.md` creates a one-project student packet.
- `GET /projects/roster.csv` exports a roster/check-in CSV.
- `GET /showcase/report` returns a Markdown report for critique notes,
  portfolio process documentation, or a Monday demo recap.
- `GET /showcase/demo-script` gives the instructor a one-page run-of-show.
- `GET /showcase/report.json` returns the same evidence as structured JSON.

This lets critique focus on the actual interaction:

- What did the project claim it could do?
- What did it listen for?
- What did it emit?
- Was it live, stale, mocked, or missing?
- Which room events prove the interaction happened?

The point is not to punish a failed prototype. It is to make the state of the
prototype legible enough to discuss.

## Manual Test Events

Broadcast a timer offer to everyone:

```bash
curl -X POST http://localhost:8766/projects/smart-stage/events \
  -H "Content-Type: application/json" \
  -H "X-API-Key: any-key-in-local-mode" \
  -d "{\"event_type\":\"timer.offer\",\"payload\":{\"minutes\":5,\"reason\":\"activity started\"}}"
```

Send a timer offer only to `gesture-timer`:

```bash
curl -X POST http://localhost:8766/projects/smart-stage/events \
  -H "Content-Type: application/json" \
  -H "X-API-Key: any-key-in-local-mode" \
  -d "{\"event_type\":\"timer.offer\",\"target\":\"gesture-timer\",\"payload\":{\"minutes\":5,\"reason\":\"activity started\"}}"
```

Listen as `gesture-timer`:

```bash
curl -N "http://localhost:8766/subscribe/events?subscriber_id=gesture-timer"
```

Mock a fiducial detection:

```bash
curl -X POST http://localhost:8766/projects/horizon/events \
  -H "Content-Type: application/json" \
  -H "X-API-Key: any-key-in-local-mode" \
  -d "{\"event_type\":\"fiducial.detected\",\"payload\":{\"tag_id\":3,\"name\":\"Whiteboard Marker\",\"x\":0.52,\"y\":0.08,\"zone\":\"whiteboard\",\"confidence\":0.92}}"
```
