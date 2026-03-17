"""
discord_vjepa_commands.py
Add these commands to your existing discord_bot.py to expose
V-JEPA world model status via Discord.

Integration: paste the command functions into discord_bot.py and add
the STATUS paths to the existing status-file read pattern.
"""

import json
import time
from pathlib import Path

# ── Status file paths ─────────────────────────────────────────────────────────
HOME = Path.home() / "oak-projects"
VJEPA_STATUS  = HOME / "vjepa_status.json"       # anomaly scores (all cameras)
PROBE_STATUS  = HOME / "probe_status.json"        # probe classifications
VJEPA_HISTORY = HOME / "vjepa_history.jsonl"      # time-series anomaly scores

# Assumes 'bot' is the discord.ext.commands.Bot instance from discord_bot.py
# and 'CAMERA_IDS' = ["orbit", "gravity", "horizon"]


async def cmd_worldmodel(ctx):
    """
    !worldmodel — Show current V-JEPA anomaly scores for all cameras.
    """
    if not VJEPA_STATUS.exists():
        await ctx.send("❌ No V-JEPA status file found. Is `pi_vjepa_client.py` running?")
        return

    try:
        status = json.loads(VJEPA_STATUS.read_text())
    except Exception:
        await ctx.send("❌ Could not read V-JEPA status file.")
        return

    score = status.get("anomaly_score", 0)
    level = status.get("level", "unknown")
    camera = status.get("camera_id", "unknown")
    ts = status.get("timestamp", "")[:19].replace("T", " ")
    latency = status.get("server_latency_ms", 0)

    bar_len = 20
    filled = int(score * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    emoji = {"normal": "🟢", "unusual": "🟡", "anomaly": "🔴"}.get(level, "⚪")
    msg = (
        f"**🧠 World Model Status** ({camera})\n"
        f"{emoji} Level: `{level.upper()}`\n"
        f"Surprise score: `{score:.3f}` [{bar}]\n"
        f"Server latency: `{latency:.0f}ms`\n"
        f"*{ts}*"
    )
    await ctx.send(msg)


async def cmd_classify(ctx):
    """
    !classify — Show probe-based activity classification for this camera.
    """
    if not PROBE_STATUS.exists():
        await ctx.send("❌ No probe status found. Is `probe_inference.py` running?")
        return

    try:
        status = json.loads(PROBE_STATUS.read_text())
    except Exception:
        await ctx.send("❌ Could not read probe status.")
        return

    pred = status.get("predicted_class", "unknown")
    conf = status.get("confidence", 0)
    probs = status.get("class_probs", {})
    camera = status.get("camera_id", "unknown")
    ts = status.get("timestamp", "")[:19].replace("T", " ")

    prob_lines = "\n".join(
        f"  {'→' if c == pred else ' '} `{c}`: {p:.0%}"
        for c, p in sorted(probs.items(), key=lambda x: -x[1])
    )

    msg = (
        f"**🎓 Activity Classification** ({camera})\n"
        f"Detected: **{pred}** ({conf:.0%} confidence)\n\n"
        f"{prob_lines}\n\n"
        f"*{ts}*"
    )
    await ctx.send(msg)


async def cmd_surprise_history(ctx, n: int = 10):
    """
    !surprise-history [n] — Show last N anomaly scores as a sparkline.
    """
    if not VJEPA_HISTORY.exists():
        await ctx.send("❌ No V-JEPA history file.")
        return

    lines = VJEPA_HISTORY.read_text().strip().split("\n")
    recent = lines[-n:] if len(lines) >= n else lines

    scores = []
    for line in recent:
        try:
            scores.append(json.loads(line)["anomaly_score"])
        except Exception:
            pass

    if not scores:
        await ctx.send("❌ No valid history entries.")
        return

    # Sparkline using unicode blocks
    SPARK = " ▁▂▃▄▅▆▇█"
    mx = max(scores) if max(scores) > 0 else 1
    spark = "".join(SPARK[min(8, int(s / mx * 8))] for s in scores)

    avg = sum(scores) / len(scores)
    peak = max(scores)

    msg = (
        f"**📈 Surprise Score History** (last {len(scores)} readings)\n"
        f"`{spark}`\n"
        f"avg: `{avg:.3f}`  peak: `{peak:.3f}`"
    )
    await ctx.send(msg)


# ── Register commands in your discord_bot.py ──────────────────────────────────
# Add to your bot setup:
#
#   @bot.command(name="worldmodel")
#   async def worldmodel(ctx):
#       await cmd_worldmodel(ctx)
#
#   @bot.command(name="classify")
#   async def classify(ctx):
#       await cmd_classify(ctx)
#
#   @bot.command(name="surprise-history")
#   async def surprise_history(ctx, n: int = 10):
#       await cmd_surprise_history(ctx, n)
#
# Also update !help to include:
#   !worldmodel          Show V-JEPA anomaly score
#   !classify            Show probe-based activity classification
#   !surprise-history    Show recent surprise score sparkline
