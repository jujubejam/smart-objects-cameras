"""
multi_camera_consensus.py - Combine predictions from multiple cameras
======================================================================
Reads probe_status files from multiple cameras and outputs a consensus.

Usage:
    python multi_camera_consensus.py --cameras usb-cam,eth-cam --strategy average

Strategies:
    confidence  - Trust the camera with highest confidence
    average     - Average class probabilities across cameras
    agreement   - Only report if cameras agree, else "uncertain"

Creates:
    ~/oak-projects/consensus_status.json
"""

import argparse
import json
import time
import logging
from datetime import datetime
from pathlib import Path

OAK_PROJECTS = Path.home() / "oak-projects"
CONSENSUS_STATUS = OAK_PROJECTS / "consensus_status.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("consensus")


def read_camera_status(camera_name: str) -> dict | None:
    """Read a camera's probe status file."""
    # Try different naming patterns
    patterns = [
        OAK_PROJECTS / f"probe_status_{camera_name}.json",
        OAK_PROJECTS / f"probe_status.json",  # Single camera fallback
    ]

    for path in patterns:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                # Check if it's recent (within 30 seconds)
                ts = datetime.fromisoformat(data.get("timestamp", "2000-01-01"))
                age = (datetime.now() - ts).total_seconds()
                if age < 30:
                    return data
                else:
                    log.warning(f"  {camera_name}: stale data ({age:.0f}s old)")
                    return None
            except Exception as e:
                log.warning(f"  {camera_name}: read error: {e}")
                return None

    return None


def strategy_confidence(statuses: list[dict]) -> dict:
    """Pick the prediction with highest confidence."""
    best = max(statuses, key=lambda s: s.get("confidence", 0))
    return {
        "predicted_class": best["predicted_class"],
        "confidence": best["confidence"],
        "method": "highest_confidence",
        "source_camera": best.get("camera_id", "unknown"),
    }


def strategy_average(statuses: list[dict]) -> dict:
    """Average class probabilities across cameras."""
    # Collect all class names
    all_classes = set()
    for s in statuses:
        all_classes.update(s.get("class_probs", {}).keys())

    # Average probabilities
    avg_probs = {}
    for cls in all_classes:
        probs = [s.get("class_probs", {}).get(cls, 0) for s in statuses]
        avg_probs[cls] = sum(probs) / len(probs)

    # Find winner
    if avg_probs:
        winner = max(avg_probs, key=avg_probs.get)
        return {
            "predicted_class": winner,
            "confidence": round(avg_probs[winner], 3),
            "class_probs": {k: round(v, 3) for k, v in avg_probs.items()},
            "method": "averaged",
            "num_cameras": len(statuses),
        }
    else:
        return {"predicted_class": "unknown", "confidence": 0, "method": "averaged"}


def strategy_agreement(statuses: list[dict], threshold: float = 0.6) -> dict:
    """Only report if cameras agree on the same class."""
    predictions = [s.get("predicted_class") for s in statuses]

    # Check if all cameras agree
    if len(set(predictions)) == 1:
        # All agree - average their confidences
        avg_conf = sum(s.get("confidence", 0) for s in statuses) / len(statuses)
        return {
            "predicted_class": predictions[0],
            "confidence": round(avg_conf, 3),
            "method": "unanimous",
            "num_cameras": len(statuses),
        }
    else:
        # Disagreement - report uncertain
        return {
            "predicted_class": "uncertain",
            "confidence": 0,
            "method": "disagreement",
            "camera_predictions": {s.get("camera_id", f"cam{i}"): s.get("predicted_class")
                                   for i, s in enumerate(statuses)},
        }


def main():
    parser = argparse.ArgumentParser(description="Multi-camera consensus")
    parser.add_argument("--cameras", default="usb-cam,eth-cam",
                       help="Comma-separated camera names")
    parser.add_argument("--strategy", default="average",
                       choices=["confidence", "average", "agreement"],
                       help="Consensus strategy")
    parser.add_argument("--interval", type=float, default=2.0,
                       help="Polling interval in seconds")
    parser.add_argument("--discord", action="store_true",
                       help="Notify Discord on class changes")
    args = parser.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",")]

    print("\n" + "=" * 50)
    print("MULTI-CAMERA CONSENSUS")
    print("=" * 50)
    print(f"Cameras:  {cameras}")
    print(f"Strategy: {args.strategy}")
    print(f"Interval: {args.interval}s")
    print(f"Output:   {CONSENSUS_STATUS}")
    print("=" * 50 + "\n")

    # Load Discord webhook if needed
    webhook_url = ""
    if args.discord:
        env_file = OAK_PROJECTS / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("DISCORD_WEBHOOK_URL="):
                    webhook_url = line.split("=", 1)[1].strip().strip('"')

    last_class = None

    try:
        while True:
            # Read all camera statuses
            statuses = []
            for cam in cameras:
                status = read_camera_status(cam)
                if status:
                    statuses.append(status)

            if not statuses:
                log.warning("No camera data available")
                time.sleep(args.interval)
                continue

            if len(statuses) < len(cameras):
                log.info(f"Only {len(statuses)}/{len(cameras)} cameras reporting")

            # Apply consensus strategy
            if args.strategy == "confidence":
                result = strategy_confidence(statuses)
            elif args.strategy == "average":
                result = strategy_average(statuses)
            elif args.strategy == "agreement":
                result = strategy_agreement(statuses)

            # Add metadata
            result["timestamp"] = datetime.now().isoformat()
            result["cameras_reporting"] = len(statuses)
            result["cameras_expected"] = len(cameras)

            # Log
            pred = result["predicted_class"]
            conf = result.get("confidence", 0)
            method = result.get("method", "?")

            if "class_probs" in result:
                probs_str = "  ".join(f"{c}:{p:.0%}" for c, p in result["class_probs"].items())
                log.info(f"{pred} ({conf:.0%}) [{method}]  |  {probs_str}")
            else:
                log.info(f"{pred} ({conf:.0%}) [{method}]")

            # Write status
            OAK_PROJECTS.mkdir(parents=True, exist_ok=True)
            CONSENSUS_STATUS.write_text(json.dumps(result, indent=2))

            # Discord on change
            if args.discord and webhook_url and pred != last_class and pred != "uncertain":
                import requests
                emoji = {"at_computer": "💻", "playing_keyboard": "🎹", "tending_plants": "🌱"}.get(pred, "🎯")
                try:
                    requests.post(webhook_url, json={
                        "content": f"{emoji} **Consensus**: `{pred}` ({conf:.0%}) - {len(statuses)} cameras"
                    }, timeout=5)
                except:
                    pass

            last_class = pred
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
