-- Smart Classroom: Supabase Schema
-- Run this in your Supabase SQL Editor to set up all tables.
-- After running, enable Realtime on: classroom_state, classroom_events, project_events
-- (Dashboard → Database → Replication → enable for those tables)

-- ============================================================
-- 1. classroom_state
--    Per-camera snapshot, upserted every ~10 seconds by detectors.
--    Primary key is camera_id (orbit, gravity, horizon).
-- ============================================================

CREATE TABLE IF NOT EXISTS classroom_state (
    camera_id               TEXT PRIMARY KEY,
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Person detection
    person_detected         BOOLEAN DEFAULT false,
    person_count            INTEGER DEFAULT 0,

    -- Fatigue detection
    fatigue_detected        BOOLEAN DEFAULT false,

    -- V-JEPA anomaly
    anomaly_score           REAL DEFAULT 0.0,
    anomaly_level           TEXT DEFAULT 'normal',

    -- Probe classification
    predicted_class         TEXT DEFAULT 'unknown',
    prediction_confidence   REAL DEFAULT 0.0,
    class_probs             JSONB DEFAULT '{}',

    -- Whiteboard OCR
    whiteboard_text         TEXT[] DEFAULT '{}',
    whiteboard_text_detected BOOLEAN DEFAULT false,

    -- Derived room mode (computed by classroom_api.py on each push)
    room_mode               TEXT DEFAULT 'unknown',

    -- Metadata
    detector_host           TEXT,
    detector_user           TEXT,
    running                 BOOLEAN DEFAULT true
);

CREATE INDEX IF NOT EXISTS idx_classroom_state_updated
    ON classroom_state (updated_at DESC);


-- ============================================================
-- 2. classroom_events
--    Append-only time-series log of state changes.
-- ============================================================

CREATE TABLE IF NOT EXISTS classroom_events (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    camera_id   TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    payload     JSONB NOT NULL DEFAULT '{}',
    source      TEXT NOT NULL DEFAULT 'detector'
);

CREATE INDEX IF NOT EXISTS idx_events_camera_time
    ON classroom_events (camera_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_events_type_time
    ON classroom_events (event_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_events_created
    ON classroom_events (created_at DESC);


-- ============================================================
-- 3. student_projects
--    Registry of student projects with config and API keys.
-- ============================================================

CREATE TABLE IF NOT EXISTS student_projects (
    project_id      TEXT PRIMARY KEY,
    display_name    TEXT NOT NULL,
    student_name    TEXT NOT NULL,
    description     TEXT,
    config          JSONB DEFAULT '{}',
    subscribed_events TEXT[] DEFAULT '{}',
    api_key         TEXT NOT NULL DEFAULT encode(gen_random_bytes(24), 'hex'),
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    is_active       BOOLEAN DEFAULT true
);


-- ============================================================
-- 4. project_events
--    Events published BY student projects.
-- ============================================================

CREATE TABLE IF NOT EXISTS project_events (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    project_id  TEXT NOT NULL REFERENCES student_projects(project_id),
    event_type  TEXT NOT NULL,
    payload     JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_project_events_project
    ON project_events (project_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_project_events_type
    ON project_events (event_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_project_events_created
    ON project_events (created_at DESC);


-- ============================================================
-- 5. Seed student projects
-- ============================================================

INSERT INTO student_projects (project_id, display_name, student_name, description, subscribed_events)
VALUES
    ('seren-room',
     'A Room (Context-Aware Classroom)',
     'Yuxuan (Seren)',
     'Classroom that reads the room and sets the vibe — adjusts music, lighting, screen content based on occupancy and activity mode (Solo/Duo/Group/Focus/Presentation).',
     ARRAY['person_change', 'probe_classification', 'room_mode_change']),

    ('echodesk',
     'EchoDesk',
     'TBD',
     'Conversational machine: students type a message in a mobile app, a Raspberry Pi speaker speaks it aloud. Shared question board for the professor.',
     ARRAY['room_mode_change']),

    ('ambient-english',
     'Ambient English Feedback Object',
     'TBD',
     'Desk object that listens when the student speaks English and provides gentle grammar suggestions on a small screen.',
     ARRAY['person_change']),

    ('calmball',
     'CalmBall',
     'TBD',
     'Stress-regulation squeeze ball that triggers calming sounds through a Pi speaker when squeezed.',
     ARRAY['room_mode_change']),

    ('gesture-timer',
     'Gesture Classroom Timer',
     'TBD',
     'Browser-based countdown timer controlled by hand gestures via p5.js and ml5.js Handpose.',
     ARRAY['room_mode_change']),

    ('smart-stage',
     'Smart Stage',
     'TBD',
     'Ambient intelligence: auto-lighting, auto-recording, live captioning, and AI lecture summaries when a speaker enters the stage area.',
     ARRAY['person_change', 'probe_classification', 'room_mode_change']),

    ('assignment-tracker',
     'Assignment Progress Tracking',
     'Shuyang Tian',
     'Ambient assignment tracking system — AI agent scans documents, extracts deadlines, surfaces reminders as a desktop companion.',
     ARRAY['person_change']),

    ('focus-beam',
     'Focus Beam',
     'Feifey',
     'Assistive classroom spotlight that follows the instructor''s pointing gestures on projected slides.',
     ARRAY['probe_classification', 'room_mode_change']),

    ('forest-classroom',
     'Forest in the Classroom',
     'Sophie',
     'Living forest projection on classroom wall inspired by Princess Mononoke — reacts to presence and spoken emotion.',
     ARRAY['person_change', 'room_mode_change']),

    ('sleep-detection',
     'Sleep Detection',
     'Kevin',
     'Camera detects sleeping students via posture and facial cues, triggers a public "Wake Up" projection on the classroom screen.',
     ARRAY['person_change']),

    ('smart-room-finder',
     'Smart Room Finder',
     'Mingyue Zhou',
     'Real-time map screen at studio entrance showing room availability, occupancy, and usage patterns.',
     ARRAY['person_change', 'room_mode_change']),

    ('nodcheck',
     'NodCheck',
     'Kathy',
     'Non-verbal comprehension feedback: webcam detects yes/no head nods when teacher asks "Do you understand?"',
     ARRAY['room_mode_change']),

    ('smart-surfaces',
     'SmartSurfaces / E-Wall',
     'Darren / Ramon',
     'Projection mapping with gesture-reactive visuals for dynamic presentations using TouchDesigner and depth cameras.',
     ARRAY['probe_classification', 'room_mode_change']),

    ('imprint',
     'Imprint',
     'TBD',
     'Camera reads handwriting off any surface (desk, napkin, hand) and saves it directly to notes. Can project notes back onto surfaces.',
     ARRAY['whiteboard_change']),

    ('virtual-gus',
     'JuJu: The Virtual GUS',
     'JuJu Kim / Kathy',
     'Pet telepresence — detect dog Gus on pet camera at home, segment and project him live in the studio. Two-way interaction.',
     ARRAY['person_change'])

ON CONFLICT (project_id) DO NOTHING;


-- ============================================================
-- 6. Verify
-- ============================================================

-- After running, check everything:
-- SELECT * FROM student_projects;
-- SELECT project_id, api_key FROM student_projects;
-- SELECT count(*) FROM classroom_state;
