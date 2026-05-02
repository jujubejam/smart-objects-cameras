import math
import time
import cv2
import depthai as dai

def _get_aruco_dict_id(name):
    mapping = {
        'ARUCO': cv2.aruco.DICT_ARUCO_ORIGINAL,
    }
    if hasattr(cv2.aruco, 'DICT_ARUCO_MIP_36h12'):
        mapping['ARUCO_MIP_36h12'] = cv2.aruco.DICT_ARUCO_MIP_36h12
    return mapping.get(name, cv2.aruco.DICT_ARUCO_ORIGINAL)


class FiducialSDK:
    def __init__(self, options=None):
        opts = options or {}
        self.smoothing = opts.get('smoothing', 0.35)
        self.rotation_step = opts.get('rotation_step', 10)
        self.min_size = opts.get('min_size', 30)
        self.max_size = opts.get('max_size', 400)
        self.exit_grace = opts.get('exit_grace', 8)
        self.width = opts.get('width', 640)
        self.height = opts.get('height', 480)
        self.dictionary_name = opts.get('dictionary', 'ARUCO')

        self._handlers = {}
        self._marker_handlers = {}
        self._state = {}
        self._missed_frames = {}
        self._running = False
        self._detector = None

    def init(self, dictionary_name=None):
        if dictionary_name:
            self.dictionary_name = dictionary_name
        dict_id = _get_aruco_dict_id(self.dictionary_name)
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return self

    def on(self, event_name, handler):
        self._handlers.setdefault(event_name, []).append(handler)
        return lambda: self._handlers[event_name].remove(handler)

    def on_marker(self, marker_id, handler):
        self._marker_handlers.setdefault(marker_id, []).append(handler)
        return lambda: self._marker_handlers[marker_id].remove(handler)

    def emit(self, event_name, payload):
        for h in self._handlers.get(event_name, []):
            h(payload)
        for h in self._marker_handlers.get(payload['id'], []):
            h(payload)
        for h in self._handlers.get('event', []):
            h(payload)

    def get_state(self):
        return list(self._state.values())

    def run(self, on_frame=None):
        """Blocking main loop. on_frame(frame) receives each raw BGR frame."""
        self._running = True

        device = dai.Device()
        platform = device.getPlatform().name
        frame_type = (
            dai.ImgFrame.Type.BGR888i if platform == 'RVC4'
            else dai.ImgFrame.Type.BGR888p
        )

        with dai.Pipeline(device) as pipeline:
            cam = pipeline.create(dai.node.Camera).build()
            cam_out = cam.requestOutput(
                size=(self.width, self.height), type=frame_type
            )
            q = cam_out.createOutputQueue(maxSize=4, blocking=False)

            pipeline.start()
            while self._running and pipeline.isRunning():
                packet = q.tryGet()
                if packet is None:
                    continue
                frame = packet.getCvFrame()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = self._detector.detectMarkers(gray)
                self.process_frame(self._parse_detections(corners, ids))
                if on_frame:
                    on_frame(frame)

    def stop(self):
        self._running = False

    def _parse_detections(self, corners, ids):
        if ids is None:
            return []
        markers = []
        for i, corner_set in enumerate(corners):
            pts = corner_set[0]
            markers.append({
                'id': int(ids[i][0]),
                'corners': [{'x': float(p[0]), 'y': float(p[1])} for p in pts],
            })
        return markers

    def process_frame(self, markers):
        now = time.time()
        seen = set()

        for marker in markers:
            next_state = self._to_marker_state(marker, now)
            mid = next_state['id']
            seen.add(mid)
            self._missed_frames[mid] = 0

            prev = self._state.get(mid)
            if prev is None:
                self._state[mid] = next_state
                self.emit('MARKER_ENTER', next_state)
                continue

            merged = self._merge_state(prev, next_state)
            self._state[mid] = merged

            if self._did_move(prev, merged):
                self.emit('MARKER_MOVE', merged)
            if self._did_rotate(prev, merged):
                self.emit('MARKER_ROTATE', merged)
            self.emit('MARKER_UPDATE', merged)

        for mid, prev in list(self._state.items()):
            if mid not in seen:
                missed = self._missed_frames.get(mid, 0) + 1
                self._missed_frames[mid] = missed
                if missed >= self.exit_grace:
                    self.emit('MARKER_EXIT', {**prev, 'present': False})
                    del self._state[mid]
                    self._missed_frames.pop(mid, None)

    def _to_marker_state(self, marker, timestamp):
        corners = marker['corners']
        cx = sum(c['x'] for c in corners) / 4
        cy = sum(c['y'] for c in corners) / 4
        dx = corners[2]['x'] - corners[0]['x']
        dy = corners[2]['y'] - corners[0]['y']
        size = math.sqrt(dx * dx + dy * dy)
        raw_rot = math.atan2(
            corners[1]['y'] - corners[0]['y'],
            corners[1]['x'] - corners[0]['x'],
        ) * (180 / math.pi)
        rotation = self._snap(raw_rot, self.rotation_step)

        prev = self._state.get(marker['id'])
        velocity = (
            {'x': cx - prev['position']['x'], 'y': cy - prev['position']['y']}
            if prev else {'x': 0.0, 'y': 0.0}
        )

        return {
            'id': marker['id'],
            'present': True,
            'timestamp': timestamp,
            'position': {'x': cx, 'y': cy},
            'normalized': {'x': cx / self.width, 'y': cy / self.height},
            'size': size,
            'distance': self._normalize(size, self.min_size, self.max_size),
            'rotation': rotation,
            'raw_rotation': raw_rot,
            'velocity': velocity,
            'corners': corners,
        }

    def _merge_state(self, prev, next_state):
        s = self.smoothing
        lp = self._lerp
        return {
            **next_state,
            'position': {
                'x': lp(prev['position']['x'], next_state['position']['x'], s),
                'y': lp(prev['position']['y'], next_state['position']['y'], s),
            },
            'normalized': {
                'x': lp(prev['normalized']['x'], next_state['normalized']['x'], s),
                'y': lp(prev['normalized']['y'], next_state['normalized']['y'], s),
            },
            'size': lp(prev['size'], next_state['size'], s),
            'distance': lp(prev['distance'], next_state['distance'], s),
            'velocity': {
                'x': lp(prev['velocity']['x'], next_state['velocity']['x'], s),
                'y': lp(prev['velocity']['y'], next_state['velocity']['y'], s),
            },
        }

    def _did_move(self, prev, next_state):
        return (
            abs(next_state['position']['x'] - prev['position']['x']) > 2 or
            abs(next_state['position']['y'] - prev['position']['y']) > 2 or
            abs(next_state['size'] - prev['size']) > 2
        )

    def _did_rotate(self, prev, next_state):
        return prev['rotation'] != next_state['rotation']

    def _snap(self, value, step):
        return round(value / step) * step

    def _normalize(self, value, mn, mx):
        return max(0.0, min(1.0, (value - mn) / (mx - mn)))

    def _lerp(self, a, b, t):
        return a + (b - a) * t
