import argparse
import json
import math
import threading
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from fiducial_sdk import FiducialSDK

parser = argparse.ArgumentParser(description='Fiducial Timer')
parser.add_argument('--display', action='store_true', help='Show live camera window')
args = parser.parse_args()

STATUS_FILE = Path(__file__).parent / 'timer_status.json'

# ── State ─────────────────────────────────────────────────────────────────────
set_minutes = 0
remaining_seconds = 0
timer_thread = None
status = 'stopped'   # running / paused / stopped / done
switch_on = False
tag1_in_frame = False
active_tags = set()
log_lines = []

# ── Helpers ───────────────────────────────────────────────────────────────────
def write_status(running=True):
    try:
        STATUS_FILE.write_text(json.dumps({
            'status': status,
            'set_minutes': set_minutes,
            'remaining_seconds': remaining_seconds,
            'switch_on': switch_on or tag1_in_frame,
            'timestamp': datetime.now().isoformat(),
            'running': running,
        }, indent=2))
    except Exception:
        pass

def log(msg):
    ts = time.strftime('%H:%M:%S')
    log_lines.insert(0, f'{ts}  {msg}')
    del log_lines[6:]
    print(msg)

def format_time(secs):
    s = int(secs)
    return f'{s // 60:02d}:{s % 60:02d}'

def rotation_to_minutes(deg):
    norm = ((-deg) % 360 + 360) % 360
    return round(norm / 360 * 60)

# ── Countdown ─────────────────────────────────────────────────────────────────
def tick():
    global remaining_seconds, timer_thread, status
    if status != 'running':
        return
    remaining_seconds -= 1
    if remaining_seconds <= 0:
        remaining_seconds = 0
        status = 'done'
        log('[TIMER] done')
        write_status()
        return
    timer_thread = threading.Timer(1.0, tick)
    timer_thread.daemon = True
    timer_thread.start()

def start_tick():
    global timer_thread
    stop_tick()
    timer_thread = threading.Timer(1.0, tick)
    timer_thread.daemon = True
    timer_thread.start()

def stop_tick():
    global timer_thread
    if timer_thread:
        timer_thread.cancel()
        timer_thread = None

# ── Drawing ───────────────────────────────────────────────────────────────────
PANEL_W = 220
CAM_W, CAM_H = 640, 480
BG      = (17,  17,  17)
DIM     = (85,  85,  85)
WHITE   = (230, 230, 230)
GREEN   = (76,  175, 80)
ORANGE  = (0,   152, 255)
RED     = (68,  68,  244)
DARK    = (42,  42,  42)

FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_dial(img, minutes, ox, oy, r=46):
    cx, cy = ox + r + 14, oy + r + 14
    cv2.circle(img, (cx, cy), r, DARK, 2)
    for t in range(0, 60, 5):
        a = -math.pi / 2 - (t / 60) * math.pi * 2
        major = (t % 15 == 0)
        inner = r - (8 if major else 4)
        p1 = (int(cx + math.cos(a) * inner), int(cy + math.sin(a) * inner))
        p2 = (int(cx + math.cos(a) * r),     int(cy + math.sin(a) * r))
        cv2.line(img, p1, p2, (85, 85, 85) if major else (51, 51, 51), 2 if major else 1)
    ha = -math.pi / 2 - (minutes / 60) * math.pi * 2
    hp = (int(cx + math.cos(ha) * r * 0.75), int(cy + math.sin(ha) * r * 0.75))
    cv2.line(img, (cx, cy), hp, WHITE, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, WHITE, -1)

def build_display(frame):
    mirrored = cv2.flip(frame, 1)
    panel = np.full((CAM_H, PANEL_W, 3), 17, dtype=np.uint8)
    px = 14  # left padding

    # set time
    cv2.putText(panel, 'SET TIME', (px, 28), FONT, 0.33, DIM, 1)
    cv2.putText(panel, f'{set_minutes} min', (px, 58), FONT, 0.85, (136, 136, 136), 1)

    cv2.line(panel, (px, 72), (PANEL_W - px, 72), (40, 40, 40), 1)

    # countdown
    cv2.putText(panel, 'COUNTDOWN', (px, 94), FONT, 0.33, DIM, 1)
    cv2.putText(panel, format_time(remaining_seconds), (px, 150), FONT, 1.9, WHITE, 2, cv2.LINE_AA)

    status_color = {'running': GREEN, 'paused': ORANGE, 'done': RED}.get(status, DIM)
    cv2.putText(panel, status, (px, 174), FONT, 0.4, status_color, 1)

    cv2.line(panel, (px, 192), (PANEL_W - px, 192), (40, 40, 40), 1)

    # dial
    draw_dial(panel, set_minutes, 0, 200)
    cv2.putText(panel, 'tag 3  set', (px, 318), FONT, 0.3, DIM, 1)

    # on/off indicator (tag 1)
    ind_color = GREEN if (switch_on or tag1_in_frame) else (51, 51, 51)
    cv2.circle(panel, (PANEL_W - 34, 232), 16, ind_color, -1)
    cv2.circle(panel, (PANEL_W - 34, 232), 16, (80, 80, 80), 1)
    cv2.putText(panel, 'tag 1', (PANEL_W - 52, 260), FONT, 0.28, DIM, 1)

    # tag hints
    cv2.putText(panel, 'tag 8   start', (px, 340), FONT, 0.3, DIM, 1)
    cv2.putText(panel, 'tag 9   pause', (px, 356), FONT, 0.3, DIM, 1)
    cv2.putText(panel, 'tag 10  reset', (px, 372), FONT, 0.3, DIM, 1)

    cv2.line(panel, (px, 384), (PANEL_W - px, 384), (40, 40, 40), 1)

    # active tags
    tag_str = f"detected: {', '.join(str(t) for t in sorted(active_tags))}" if active_tags else 'no tags'
    cv2.putText(panel, tag_str, (px, 400), FONT, 0.3, DIM, 1)

    # log
    for i, line in enumerate(log_lines[:4]):
        cv2.putText(panel, line[:30], (px, 420 + i * 15), FONT, 0.28, (60, 60, 60), 1)

    return np.hstack([mirrored, panel])


def main():
    global set_minutes, remaining_seconds, status, switch_on, tag1_in_frame

    sdk = FiducialSDK()
    sdk.init()

    # ── Tag 1: on/off ─────────────────────────────────────────────────────────
    def tag1_enter(m):
        global switch_on, tag1_in_frame
        tag1_in_frame = True
        switch_on = True
        log('[TAG 1] ON')
        write_status()

    def tag1_exit(m):
        global switch_on, tag1_in_frame
        tag1_in_frame = False
        switch_on = False
        log('[TAG 1] OFF')
        write_status()

    sdk.on('MARKER_ENTER', lambda m: tag1_enter(m) if m['id'] == 1 else None)
    sdk.on('MARKER_EXIT',  lambda m: tag1_exit(m)  if m['id'] == 1 else None)

    # ── Tag 3: set timer ──────────────────────────────────────────────────────
    def tag3_update(m):
        global set_minutes, remaining_seconds
        if status == 'running':
            return
        mins = rotation_to_minutes(m['rotation'])
        if mins == set_minutes:
            return
        set_minutes = mins
        remaining_seconds = mins * 60

    sdk.on_marker(3, tag3_update)

    # ── Tag 8: start ──────────────────────────────────────────────────────────
    def tag8_enter(m):
        global status
        if m['id'] != 8:
            return
        if status == 'running' or remaining_seconds == 0:
            return
        status = 'running'
        start_tick()
        log(f'[TAG 8] start - {format_time(remaining_seconds)}')
        write_status()

    sdk.on('MARKER_ENTER', tag8_enter)

    # ── Tag 9: pause ──────────────────────────────────────────────────────────
    def tag9_enter(m):
        global status
        if m['id'] != 9 or status != 'running':
            return
        status = 'paused'
        stop_tick()
        log(f'[TAG 9] paused - {format_time(remaining_seconds)}')
        write_status()

    sdk.on('MARKER_ENTER', tag9_enter)

    # ── Tag 10: reset ─────────────────────────────────────────────────────────
    def tag10_enter(m):
        global remaining_seconds, status
        if m['id'] != 10:
            return
        stop_tick()
        remaining_seconds = set_minutes * 60
        status = 'stopped'
        log(f'[TAG 10] reset - {set_minutes} min')
        write_status()

    sdk.on('MARKER_ENTER', tag10_enter)

    # ── Active tag tracking ───────────────────────────────────────────────────
    sdk.on('MARKER_ENTER', lambda m: (active_tags.add(m['id']),    log(f'--- ENTER tag:{m["id"]}')))
    sdk.on('MARKER_EXIT',  lambda m: (active_tags.discard(m['id']), log(f'--- EXIT  tag:{m["id"]}')))

    # ── Frame loop ────────────────────────────────────────────────────────────
    _last_status_write = [0.0]

    def on_frame(frame):
        if args.display:
            cv2.imshow('Fiducial Timer', build_display(frame))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sdk.stop()
        now = time.time()
        if now - _last_status_write[0] >= 1.0:
            write_status()
            _last_status_write[0] = now

    write_status()
    quit_hint = ' (press q to quit)' if args.display else ' (press Ctrl+C to quit)'
    print(f'[ready] starting camera{quit_hint}')
    try:
        sdk.run(on_frame=on_frame)
    finally:
        stop_tick()
        write_status(running=False)
        if args.display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
