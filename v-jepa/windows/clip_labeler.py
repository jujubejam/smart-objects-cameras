"""
clip_labeler.py - Sort recorded clips into class folders
==========================================================
Review clips and press number keys to label them.

Usage:
    python clip_labeler.py --input D:\vjepa-clips --classes at_computer,playing_keyboard,tending_plants

Controls:
    1, 2, 3...  = Move clip to that class folder
    Space       = Skip (leave in unlabeled)
    Backspace   = Undo last label
    R           = Replay current clip
    Q           = Quit (progress is saved)

Creates:
    D:\vjepa-clips\
        at_computer\
            2026-03-19_14-23-01.mp4
            ...
        playing_keyboard\
            ...
        tending_plants\
            ...
        unlabeled\
            (remaining clips)
"""

import argparse
import shutil
import json
from pathlib import Path
from datetime import datetime

import cv2


def play_clip(clip_path: Path, window_name: str, class_names: list, clip_idx: int, total: int, slow: bool = False):
    """Play a clip and wait for label input. Returns class index or -1 for skip."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"  Cannot open {clip_path.name}")
        return -1

    # Check if video has frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        print(f"  Empty clip: {clip_path.name}")
        cap.release()
        return -1

    # Test read first frame
    ret, test_frame = cap.read()
    if not ret or test_frame is None or test_frame.size == 0:
        print(f"  Cannot read frames: {clip_path.name}")
        cap.release()
        return -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start

    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    frame_delay = int(1000 / fps)
    if slow:
        frame_delay = frame_delay * 2  # Half speed
    frame = None
    h, w = test_frame.shape[:2]

    while True:
        # Reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Play through clip
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                break

            # Draw UI
            h, w = frame.shape[:2]

            # Header
            cv2.rectangle(frame, (0, 0), (w, 60), (40, 40, 40), -1)
            cv2.putText(frame, f"Clip {clip_idx}/{total}: {clip_path.name}",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "Press 1-9 to label, SPACE=skip, R=replay, Q=quit",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # Class options
            y = 80
            for i, cls in enumerate(class_names):
                cv2.putText(frame, f"[{i+1}] {cls}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y += 22

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(frame_delay) & 0xFF

            # Check for input during playback
            if key == ord('q'):
                cap.release()
                return -2  # Quit signal
            elif key == ord(' '):
                cap.release()
                return -1  # Skip
            elif key == ord('r'):
                break  # Replay
            elif key == 8:  # Backspace
                cap.release()
                return -3  # Undo
            elif ord('1') <= key <= ord('9'):
                class_idx = key - ord('1')
                if class_idx < len(class_names):
                    cap.release()
                    return class_idx

        # After clip ends, show clear prompt and wait for input
        if frame is not None and frame.size > 0:
            # Create a fresh display frame with clear instructions
            display = frame.copy()
            # Dark overlay for readability
            overlay = display.copy()
            cv2.rectangle(overlay, (0, h//2 - 60), (w, h//2 + 80), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

            cv2.putText(display, "CLIP ENDED - LABEL NOW:",
                       (w//2 - 150, h//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            y = h//2 + 20
            for i, cls in enumerate(class_names):
                cv2.putText(display, f"[{i+1}] {cls}",
                           (w//2 - 100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y += 25

            cv2.putText(display, "SPACE=skip  R=replay  Q=quit",
                       (w//2 - 130, h//2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow(window_name, display)
        else:
            # No valid frame to show
            cap.release()
            return -1

        # Wait indefinitely for user input (no timeout)
        while True:
            key = cv2.waitKey(0) & 0xFF  # Wait forever until keypress
            if key == ord('q'):
                cap.release()
                return -2
            elif key == ord(' '):
                cap.release()
                return -1
            elif key == ord('r'):
                break  # Replay
            elif key == 8:
                cap.release()
                return -3
            elif ord('1') <= key <= ord('9'):
                class_idx = key - ord('1')
                if class_idx < len(class_names):
                    cap.release()
                    return class_idx


def main():
    parser = argparse.ArgumentParser(description="Label recorded clips")
    parser.add_argument("--input", type=Path, default=Path("D:/vjepa-clips"),
                       help="Input directory with unlabeled/ folder")
    parser.add_argument("--classes", default="at_computer,playing_keyboard,tending_plants",
                       help="Comma-separated class names")
    parser.add_argument("--slow", action="store_true",
                       help="Play clips at half speed")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]
    unlabeled_dir = args.input / "unlabeled"

    if not unlabeled_dir.exists():
        print(f"No unlabeled directory found at {unlabeled_dir}")
        return

    # Create class directories
    for cls in classes:
        (args.input / cls).mkdir(exist_ok=True)

    # Get unlabeled clips
    clips = sorted(unlabeled_dir.glob("*.mp4"))
    if not clips:
        print("No unlabeled clips found!")
        return

    # Load progress if exists
    progress_file = args.input / ".labeling_progress.json"
    labeled_count = {cls: 0 for cls in classes}
    history = []  # For undo

    # Count existing labeled clips
    for cls in classes:
        labeled_count[cls] = len(list((args.input / cls).glob("*.mp4")))

    print("\n" + "=" * 50)
    print("CLIP LABELER")
    print("=" * 50)
    print(f"Unlabeled clips: {len(clips)}")
    print(f"Classes: {classes}")
    print("\nExisting labels:")
    for cls in classes:
        print(f"  {cls}: {labeled_count[cls]}")
    print("=" * 50)
    print("\nStarting labeling session...\n")

    cv2.namedWindow("Labeler", cv2.WINDOW_NORMAL)

    idx = 0
    skipped = 0
    labeled_this_session = 0

    try:
        while idx < len(clips):
            clip = clips[idx]
            try:
                result = play_clip(clip, "Labeler", classes, idx + 1, len(clips), slow=args.slow)
            except Exception as e:
                print(f"  Error playing {clip.name}: {e}")
                idx += 1
                skipped += 1
                continue

            if result == -2:  # Quit
                break
            elif result == -3:  # Undo
                if history:
                    last_clip, last_class, last_src = history.pop()
                    # Move back to unlabeled
                    shutil.move(str(args.input / last_class / last_clip.name), str(last_src))
                    clips.insert(idx, last_src)
                    labeled_count[last_class] -= 1
                    labeled_this_session -= 1
                    print(f"  Undo: {last_clip.name} -> unlabeled")
            elif result == -1:  # Skip
                idx += 1
                skipped += 1
                print(f"  Skipped: {clip.name}")
            else:  # Labeled
                cls = classes[result]
                dest = args.input / cls / clip.name

                # Also move thumbnail if exists
                thumb = clip.with_suffix(".jpg")
                thumb_dest = dest.with_suffix(".jpg")

                shutil.move(str(clip), str(dest))
                if thumb.exists():
                    shutil.move(str(thumb), str(thumb_dest))

                history.append((clip, cls, clip))
                labeled_count[cls] += 1
                labeled_this_session += 1
                idx += 1
                print(f"  {clip.name} -> {cls}")

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    print(f"Labeled this session: {labeled_this_session}")
    print(f"Skipped: {skipped}")
    print(f"Remaining unlabeled: {len(list(unlabeled_dir.glob('*.mp4')))}")
    print("\nTotal labeled clips:")
    for cls in classes:
        print(f"  {cls}: {labeled_count[cls]}")
    print("=" * 50)

    total_labeled = sum(labeled_count.values())
    if total_labeled >= 6:
        print(f"\nReady to train! Run:")
        print(f"  python probe_trainer.py --clips-dir {args.input}")
    else:
        print(f"\nNeed at least 6 labeled clips (have {total_labeled})")


if __name__ == "__main__":
    main()
