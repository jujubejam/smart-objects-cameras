# Smart Classroom API — Live Setup Guide

Do these steps in order during class. Each step builds on the last.

---

## 1. Create Supabase Project

1. Go to [supabase.com/dashboard](https://supabase.com/dashboard) → **New Project**
2. Name it `smart-classroom` (or whatever you want)
3. Set a database password (save it somewhere)
4. Wait for it to provision (~1 min)

Once ready, grab two things from **Settings → API**:
- **Project URL** — looks like `https://abc123.supabase.co`
- **service_role key** (under "Project API keys", the secret one, NOT the anon key)

---

## 2. Run the Schema SQL

1. Go to **SQL Editor** in the Supabase dashboard
2. Paste the contents of `supabase_schema.sql`
3. Click **Run**
4. Verify: go to **Table Editor** — you should see 4 tables:
   - `classroom_state` (empty)
   - `classroom_events` (empty)
   - `student_projects` (15 rows)
   - `project_events` (empty)

### Enable Realtime

Go to **Database → Replication** and enable Realtime for:
- [x] `classroom_state`
- [x] `classroom_events`
- [x] `project_events`

### Grab Student API Keys

```sql
SELECT project_id, display_name, api_key FROM student_projects;
```

Give each student/group their `api_key`. They'll need it to publish events.

---

## 3. Set Environment Variables on Your PC

Add to your `~/oak-projects/.env` (or wherever you run `classroom_api.py`):

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=eyJ...your-service-role-key...
CLASSROOM_API_KEY=pick-any-shared-secret
```

The `CLASSROOM_API_KEY` is a shared secret — Pis use it to authenticate when pushing detector data. Pick anything (e.g. `smart-classroom-2026`).

---

## 4. Install Dependencies & Start the API

```bash
pip install fastapi uvicorn supabase sse-starlette python-dotenv
python classroom_api.py
```

You should see:
```
13:54:00 [INFO] Supabase connected: https://abc123.supabase.co
13:54:00 [INFO] Starting Smart Classroom API on port 8766
```

### Quick Test

```bash
# Health check
curl http://localhost:8766/health

# Push a fake state to verify Supabase writes
curl -X POST http://localhost:8766/push/state \
  -H "Content-Type: application/json" \
  -H "X-API-Key: pick-any-shared-secret" \
  -d '{"camera_id":"test","person_count":3,"person_detected":true}'

# Read it back
curl http://localhost:8766/state
curl http://localhost:8766/mode
```

Check Supabase **Table Editor → classroom_state** — you should see the "test" row.

---

## 5. Set Environment Variables on Each Pi

SSH into each Pi and add to `~/oak-projects/.env`:

```bash
CLASSROOM_API_URL=http://YOUR_PC_IP:8766
CLASSROOM_API_KEY=pick-any-shared-secret
```

Replace `YOUR_PC_IP` with your actual PC IP on the network (e.g. `192.168.1.100`).

Now when you run any detector with the updated code, it automatically pushes to Supabase alongside the existing JSON files:

```bash
python3 person_detector.py --discord
python3 probe_inference.py --server http://YOUR_PC_IP:8765 --probe ~/oak-projects/classroom_probe.pt --discord
```

---

## 6. Test the SSE Stream

In a separate terminal:

```bash
curl -N http://localhost:8766/subscribe/state
```

This stays open. Now push another state update in a different terminal — you should see the SSE event appear in real time.

---

## 7. Give Students Their Templates

Students need:
- The **PC IP** running classroom_api.py (e.g. `http://192.168.1.100:8766`)
- Their **project API key** (from the `student_projects` table)

### Python projects (Pi-based)

Copy `templates/student_template.py`, set two env vars, run:

```bash
export CLASSROOM_API=http://192.168.1.100:8766
export PROJECT_API_KEY=their-key-here
python my_project.py
```

### Browser projects (p5.js, HTML)

Copy `templates/student_template.html`, edit the three config lines at the top:

```js
const API_BASE = 'http://192.168.1.100:8766';
const PROJECT_ID = 'gesture-timer';
const API_KEY = 'their-key-here';
```

Serve it locally (`python3 -m http.server 8080`) and open in a browser.

---

## 8. Discord Commands

Once `CLASSROOM_API_URL` is set in the Pi's `.env`, the Discord bot gets two new commands:

```
!classroom    — full state from all cameras + room mode
!mode         — just the current room mode
```

---

## Quick Reference

| What | Where |
|------|-------|
| API health check | `curl http://localhost:8766/health` |
| Current room mode | `curl http://localhost:8766/mode` |
| All camera states | `curl http://localhost:8766/state` |
| Event history | `curl http://localhost:8766/events?limit=10` |
| SSE live stream | `curl -N http://localhost:8766/subscribe/state` |
| Student project list | `curl http://localhost:8766/projects` |
| Supabase dashboard | `https://supabase.com/dashboard` |

| Port | Service |
|------|---------|
| 8765 | V-JEPA embedding server (GPU) |
| 8766 | Classroom API (Supabase bridge) |

| Env Var | Who needs it | What it is |
|---------|-------------|------------|
| `SUPABASE_URL` | PC only | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | PC only | Supabase service role key |
| `CLASSROOM_API_KEY` | PC + all Pis | Shared detector secret |
| `CLASSROOM_API_URL` | All Pis + Discord bot | `http://PC_IP:8766` |
| `PROJECT_API_KEY` | Each student | Per-project key from student_projects table |
