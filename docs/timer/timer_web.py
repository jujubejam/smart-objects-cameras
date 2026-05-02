#!/usr/bin/env python3
"""
Web display for Fiducial Timer.
Reads timer_status.json and serves a live countdown page.

Usage:
    python3 timer_web.py               # serves on port 8080
    python3 timer_web.py --port 5000
"""

import argparse
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

STATUS_FILE = Path(__file__).parent / 'timer_status.json'

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--host', type=str, default='0.0.0.0')
args = parser.parse_args()

HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Timer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #111;
    color: #e6e6e6;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    gap: 24px;
  }

  #countdown {
    font-size: clamp(80px, 20vw, 180px);
    font-weight: 200;
    letter-spacing: 0.04em;
    font-variant-numeric: tabular-nums;
    line-height: 1;
    transition: color 0.3s;
  }

  #status-badge {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 20px;
    background: #2a2a2a;
    transition: background 0.3s, color 0.3s;
  }

  #meta {
    font-size: 20px;
    color: #555;
    display: flex;
    gap: 24px;
  }

  #switch {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #333;
    display: inline-block;
    margin-right: 6px;
    transition: background 0.3s;
    vertical-align: middle;
  }

  #offline {
    position: fixed;
    bottom: 16px;
    font-size: 11px;
    color: #444;
    display: none;
  }

  .state-running  { color: #4caf50; }
  .state-paused   { color: #ff9800; }
  .state-done     { color: #f44336; }
  .state-stopped  { color: #888; }

  .badge-running  { background: #1b3a1c; color: #4caf50; }
  .badge-paused   { background: #3a2a10; color: #ff9800; }
  .badge-done     { background: #3a1010; color: #f44336; }
  .badge-stopped  { background: #2a2a2a; color: #666; }

  #switch.switch-on  { background: #4caf50; }
  .switch-label-on { color: #4caf50; }
</style>
</head>
<body>

<div id="countdown">--:--</div>
<div id="status-badge">--</div>
<div id="meta">
  <span><span id="switch"></span><span id="switch-label">switch off</span></span>
  <span id="set-label">set: -- min</span>
</div>
<div id="offline">connection lost</div>

<script>
  const countdown   = document.getElementById('countdown');
  const badge       = document.getElementById('status-badge');
  const switchDot   = document.getElementById('switch');
  const switchLabel = document.getElementById('switch-label');
  const setLabel    = document.getElementById('set-label');
  const offline     = document.getElementById('offline');

  function pad(n) { return String(Math.floor(n)).padStart(2, '0'); }

  function fmt(secs) {
    const s = Math.max(0, Math.floor(secs));
    return pad(s / 60) + ':' + pad(s % 60);
  }

  let missedPolls = 0;

  async function poll() {
    try {
      const res = await fetch('/status');
      if (!res.ok) throw new Error();
      const d = await res.json();
      missedPolls = 0;
      offline.style.display = 'none';

      const state = d.status || 'stopped';

      countdown.textContent = fmt(d.remaining_seconds ?? 0);
      countdown.className   = 'state-' + state;

      badge.textContent = state.toUpperCase();
      badge.className   = 'state-badge badge-' + state;

      const on = d.switch_on;
      switchDot.className     = on ? 'switch-on' : '';
      switchLabel.className   = on ? 'switch-label-on' : '';
      switchLabel.textContent = on ? 'switch on' : 'switch off';

      setLabel.textContent = `set: ${d.set_minutes ?? 0} min`;

    } catch {
      missedPolls++;
      if (missedPolls >= 3) {
        offline.style.display = 'block';
        countdown.className = 'state-stopped';
      }
    }
  }

  poll();
  setInterval(poll, 1000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            body = HTML.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == '/status':
            try:
                body = STATUS_FILE.read_bytes()
            except FileNotFoundError:
                body = b'{}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_error(404)

    def log_message(self, fmt, *a):
        pass  # suppress per-request noise


def main():
    server = HTTPServer((args.host, args.port), Handler)
    print(f'Timer display running at http://{args.host}:{args.port}')
    print('Press Ctrl+C to stop')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped')


if __name__ == '__main__':
    main()
