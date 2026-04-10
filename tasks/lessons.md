# Lessons Learned

This file captures patterns and corrections to prevent repeated mistakes.

## Format

Each lesson follows this structure:
- **Pattern**: What went wrong
- **Rule**: How to prevent it
- **Date**: When learned

---

## Documentation

### Links must include docs/ prefix
- **Pattern**: README links to `INITIAL_SETUP.md` instead of `docs/INITIAL_SETUP.md`
- **Rule**: Always verify doc links include the full path from repo root
- **Date**: 2026-03-14

### Deprecated API calls
- **Pattern**: Used `getMxId()` which is deprecated in depthai 3.x
- **Rule**: Use `getDeviceId()` for device identification
- **Date**: 2026-03-14

### Python venv paths
- **Pattern**: Documented wrong venv path `~/oak-projects/venv/` instead of `/opt/oak-shared/venv/`
- **Rule**: The shared venv is at `/opt/oak-shared/venv/`, not in user directories
- **Date**: 2026-03-14

### Keep documentation tables complete
- **Pattern**: CLAUDE.md Key Documentation table was missing 5+ docs (WORKFLOW.md, WORKING_VERSIONS.md, NEXT_IDEAS.md, etc.)
- **Rule**: When adding new docs, update CLAUDE.md's Key Documentation table
- **Date**: 2026-03-14

### Reference slides when they exist
- **Pattern**: README and CLAUDE.md didn't mention slide versions even though 11 exist
- **Rule**: Add "Prefer slides?" links when slide versions are available
- **Date**: 2026-03-14

### Pin critical package versions
- **Pattern**: CLAUDE.md mentioned packages but not versions; numpy 2.x breaks depthai
- **Rule**: Always reference WORKING_VERSIONS.md and note numpy <2.0 constraint
- **Date**: 2026-03-14

---

## Code Patterns

### Use Camera node, not ColorCamera (depthai 3.x)

- **Pattern**: `ColorCamera` node crashes on Windows with depthai 3.x
- **Rule**: Use new `Camera` node API with `requestOutput()`
- **Example**:
  ```python
  # ✅ Works
  cam = pipeline.create(dai.node.Camera)
  cam.build(dai.CameraBoardSocket.CAM_A)
  out = cam.requestOutput((640, 480), dai.ImgFrame.Type.BGR888p)
  ```
- **Date**: 2026-03-14

### MediaPipe Tasks API (not solutions)

- **Pattern**: `mp.solutions.hands` doesn't exist in newer MediaPipe
- **Rule**: Use Tasks API with downloaded `.task` model files
- **Date**: 2026-03-14

---

## Hardware

### USB 2.0 causes OAK camera crashes

- **Pattern**: Camera connects but crashes with `RTEMS_FATAL_SOURCE_INVALID_HEAP_FREE`
- **Rule**: Always use USB 3.0 port (blue interior) and USB 3.0 cable
- **Verify**: Run `diagnose.py` — should show `USB Speed: SUPER`
- **Date**: 2026-03-14

### RTX 50 series needs PyTorch nightly

- **Pattern**: PyTorch stable doesn't support Blackwell GPUs (sm_120)
- **Rule**: Use `pip install --pre torch --index-url .../nightly/cu128`
- **Date**: 2026-03-14

---

## Testing

(Add lessons here as they're learned)
