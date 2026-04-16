#!/usr/bin/env python3
"""
Discord DM Bot for Whiteboard Reader
=====================================
Persistent bot that watches whiteboard_status.json and sends DMs
on state changes. Also accepts DM commands for two-way
conversation with the whiteboard camera system.

Run in a separate terminal alongside whiteboard_reader_full.py:
    Terminal 1: python3 whiteboard_reader_full.py --display
    Terminal 2: python3 whiteboard_dm_notifier.py

Note: The timer feature requires whiteboard_reader_full.py (not the basic
whiteboard_reader.py) because only the full version does text recognition
and writes actual text content to the status file.

DM Commands (send these to the bot via Discord DM):
    status     - Get current whiteboard status
    screenshot - Get latest camera frame
    timer      - Check if a timer is running
    cancel     - Cancel the active timer
    pause      - Pause DM notifications
    resume     - Resume DM notifications
    help       - Show available commands

Whiteboard Timer:
    Write "SET 5 MIN TIMER" on the whiteboard and the bot will
    start a countdown, DM you at halfway, 1-minute warning, and
    when time is up.

Setup:
    1. Create bot at https://discord.com/developers/applications
    2. Add DISCORD_DM_BOT_TOKEN and DISCORD_USER_ID to .env
    3. Invite bot to your server
    4. Send bot a DM to open the DM channel
"""

import os
import re
import json
import asyncio
from pathlib import Path
from datetime import datetime
import discord
from discord.ext import commands, tasks

# Load environment variables from ~/oak-projects/.env (per-user)
try:
    from dotenv import load_dotenv
    load_dotenv(Path.home() / "oak-projects" / ".env")
except ImportError:
    pass

# Configuration
BOT_TOKEN = os.getenv('DISCORD_DM_BOT_TOKEN')
USER_ID = os.getenv('DISCORD_USER_ID')
STATUS_FILE = Path.home() / "oak-projects" / "whiteboard_status.json"
SCREENSHOT_FILE = Path.home() / "oak-projects" / "latest_whiteboard_frame.jpg"

# State tracking for file watcher
_last_status = {}
_notifications_paused = False

# Timer state
_active_timer = None  # asyncio.Task for the running countdown
_active_timer_minutes = 0  # duration of the active timer
_timer_remaining = 0  # seconds remaining (updated every second)
_timer_armed = True  # re-arms when timer text is erased from the board

# Regex patterns for timer commands (tolerant of OCR noise)
# Pattern with number: SET 5 MIN TIMER, SET 10 MIN, SET5MINTIMER, etc.
TIMER_WITH_NUM = re.compile(
    r'SET\s*(\d+)\s*MIN(?:UTE)?S?\s*(?:TIMER)?',
    re.IGNORECASE
)
# Pattern without number: SET TIMER, SETTIMER
TIMER_NO_NUM = re.compile(
    r'SET\s*TIMER',
    re.IGNORECASE
)
# Standalone number (to find duration in a separate OCR region)
STANDALONE_NUM = re.compile(r'\b(\d{1,3})\b')

DEFAULT_TIMER_MINUTES = 5


def parse_timer_command(text_content):
    """Check text content for a timer command. Returns minutes or None.

    OCR may read "SET 5 MIN TIMER" in many ways:
      - Single line: "set 5 min timer"
      - Without number: "set timer" (OCR drops the number)
      - Split across regions: ["SET", "5 MIN", "TIMER"]
      - Number separate: ["set timer", "5"]

    If "SET TIMER" is found but no number, checks other text
    regions for a standalone number, then falls back to 5 minutes.
    """
    if not text_content:
        return None

    combined = " ".join(text_content)

    # First: try to find the full pattern with a number
    for text in list(text_content) + [combined]:
        match = TIMER_WITH_NUM.search(text)
        if match:
            return int(match.group(1))

    # Second: check for "SET TIMER" without a number
    if TIMER_NO_NUM.search(combined):
        # Look for a number in any text region (e.g. "5" on its own)
        for line in text_content:
            num_match = STANDALONE_NUM.search(line)
            if num_match:
                val = int(num_match.group(1))
                if 1 <= val <= 120:  # reasonable timer range
                    return val
        # No number found anywhere — use default
        return DEFAULT_TIMER_MINUTES

    return None


async def run_timer(user, minutes):
    """Run a countdown timer and DM the user at key moments."""
    global _active_timer, _active_timer_minutes, _timer_remaining

    total_seconds = minutes * 60
    _active_timer_minutes = minutes
    _timer_remaining = total_seconds

    await user.send(f"⏱️ **Timer set: {minutes} minute{'s' if minutes != 1 else ''}** (from whiteboard)")
    print(f"  Timer started: {minutes} min")

    # Send a halfway update for timers > 1 minute
    halfway = total_seconds // 2
    warned_halfway = False
    warned_one_min = False

    try:
        for remaining in range(total_seconds, 0, -1):
            _timer_remaining = remaining
            await asyncio.sleep(1)

            # Halfway notification (timers > 2 min)
            if not warned_halfway and minutes > 2 and remaining == halfway:
                mins_left = remaining // 60
                secs_left = remaining % 60
                time_str = f"{mins_left}m {secs_left}s" if secs_left else f"{mins_left}m"
                await user.send(f"⏱️ Halfway — **{time_str}** remaining")
                print(f"  Timer halfway: {time_str} left")
                warned_halfway = True

            # 1-minute warning (timers > 1 min)
            if not warned_one_min and minutes > 1 and remaining == 60:
                await user.send("⏱️ **1 minute** remaining!")
                print("  Timer: 1 min left")
                warned_one_min = True

        await user.send(f"🔔 **Time's up!** {minutes} minute{'s' if minutes != 1 else ''} elapsed.")
        print(f"  Timer finished: {minutes} min")

    except asyncio.CancelledError:
        await user.send(f"⏱️ Timer cancelled ({minutes} min timer).")
        print(f"  Timer cancelled: {minutes} min")

    finally:
        _active_timer = None
        _active_timer_minutes = 0
        _timer_remaining = 0


def read_status():
    """Read current whiteboard status from file."""
    try:
        if STATUS_FILE.exists():
            return json.loads(STATUS_FILE.read_text())
    except (json.JSONDecodeError, Exception):
        pass
    return None


def format_status(status):
    """Format status data into a readable message."""
    if not status:
        return "No whiteboard data available. Is whiteboard_reader.py running?"

    running = status.get('running', False)
    if not running:
        return "Whiteboard reader is not running."

    text_detected = status.get('text_detected', False)
    num_regions = status.get('num_text_regions', 0)
    text_content = status.get('text_content', [])
    ts = status.get('timestamp', 'unknown')
    username = status.get('username', 'unknown')
    hostname = status.get('hostname', 'unknown')

    if text_detected:
        icon = "📝"
        state = "TEXT DETECTED"
    else:
        icon = "⬜"
        state = "CLEAR"

    lines = [
        f"{icon} **{state}**",
        f"Text regions: {num_regions}",
        f"Running on: {username}@{hostname}",
        f"Last update: {ts}",
    ]

    if text_content:
        lines.append(f"Content: {', '.join(text_content[:5])}")

    return "\n".join(lines)


def main():
    if not BOT_TOKEN:
        print("Error: DISCORD_DM_BOT_TOKEN not set in .env file")
        print("  Add to .env: DISCORD_DM_BOT_TOKEN=your_token_here")
        return

    if not USER_ID:
        print("Error: DISCORD_USER_ID not set in .env file")
        print("  Add to .env: DISCORD_USER_ID=your_id_here")
        return

    user_id_int = int(USER_ID)

    # Create bot
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix='!', intents=intents)

    @bot.event
    async def on_ready():
        print(f"DM bot logged in as {bot.user.name}")
        print(f"Watching: {STATUS_FILE}")
        print(f"DM target: user {USER_ID}")
        print("Send 'help' to the bot via DM for commands\n")
        watch_status.start()

    @tasks.loop(seconds=1.0)
    async def watch_status():
        """Watch whiteboard_status.json for state changes and send DMs."""
        global _last_status, _notifications_paused, _active_timer, _timer_armed

        if _notifications_paused:
            return

        status = read_status()
        if not status or not status.get('running', False):
            return

        # First read — just store state, don't notify
        if not _last_status:
            _last_status = status.copy()
            return

        user = await bot.fetch_user(user_id_int)
        if not user:
            return

        # Check for state transitions
        prev_text = _last_status.get('text_detected', False)
        curr_text = status.get('text_detected', False)
        curr_regions = status.get('num_text_regions', 0)

        try:
            if curr_text and not prev_text:
                await user.send(f"📝 **TEXT DETECTED** on whiteboard ({curr_regions} regions)")
                print(f"  DM sent: TEXT DETECTED ({curr_regions} regions)")

            if not curr_text and prev_text:
                await user.send("⬜ Whiteboard cleared — no text detected")
                print("  DM sent: Whiteboard cleared")

            # Check for timer commands in text content
            text_content = status.get('text_content', [])
            timer_minutes = parse_timer_command(text_content)

            # Debug: show what the bot reads (visible in terminal)
            if text_content:
                print(f"\r  OCR text: {text_content} | timer={timer_minutes} armed={_timer_armed}  ",
                      end="", flush=True)

            if timer_minutes:
                timer_is_idle = (_active_timer is None or _active_timer.done())

                if _timer_armed and timer_is_idle:
                    # New timer command detected — start countdown
                    _active_timer = asyncio.create_task(
                        run_timer(user, timer_minutes))
                    _timer_armed = False  # Don't re-trigger until text is erased
                elif not timer_is_idle and timer_minutes != _active_timer_minutes:
                    # Different duration written while timer is running — restart
                    _active_timer.cancel()
                    _active_timer = asyncio.create_task(
                        run_timer(user, timer_minutes))
            else:
                # No timer text on board — re-arm so next write triggers
                _timer_armed = True

        except discord.Forbidden:
            print("ERROR: Can't send DMs. Send the bot a message first.")
        except Exception as e:
            print(f"ERROR sending DM: {e}")

        _last_status = status.copy()

    @bot.event
    async def on_message(message):
        """Handle DM commands from the user."""
        global _notifications_paused

        # Only respond to DMs from the configured user
        if message.author.id != user_id_int:
            return
        if message.author == bot.user:
            return
        if message.guild is not None:
            return  # Ignore server messages

        cmd = message.content.strip().lower()

        if cmd == "status":
            status = read_status()
            await message.channel.send(format_status(status))

        elif cmd == "screenshot":
            if SCREENSHOT_FILE.exists():
                age = datetime.now().timestamp() - SCREENSHOT_FILE.stat().st_mtime
                if age > 30:
                    await message.channel.send(
                        f"Screenshot is {age:.0f}s old — camera may not be running."
                    )
                else:
                    await message.channel.send(
                        f"Captured {age:.1f}s ago",
                        file=discord.File(str(SCREENSHOT_FILE))
                    )
            else:
                await message.channel.send(
                    "No screenshot available. Is whiteboard_reader.py running?"
                )

        elif cmd == "pause":
            _notifications_paused = True
            await message.channel.send("Notifications paused. Send 'resume' to restart.")
            print("  Notifications paused by user")

        elif cmd == "resume":
            _notifications_paused = False
            _last_status.clear()  # Reset to avoid stale transition alerts
            await message.channel.send("Notifications resumed.")
            print("  Notifications resumed by user")

        elif cmd == "timer":
            if _active_timer and not _active_timer.done():
                mins_left = _timer_remaining // 60
                secs_left = _timer_remaining % 60
                elapsed = (_active_timer_minutes * 60) - _timer_remaining
                elapsed_mins = elapsed // 60
                elapsed_secs = elapsed % 60
                await message.channel.send(
                    f"⏱️ **{mins_left}m {secs_left:02d}s remaining** "
                    f"({elapsed_mins}m {elapsed_secs:02d}s elapsed of {_active_timer_minutes} min timer)"
                )
            else:
                await message.channel.send("No timer is running.")

        elif cmd == "cancel":
            if _active_timer and not _active_timer.done():
                _active_timer.cancel()
                await message.channel.send("Timer cancelled.")
                print("  Timer cancelled by user command")
            else:
                await message.channel.send("No timer to cancel.")

        elif cmd == "help":
            help_text = (
                "**Whiteboard DM Bot Commands**\n"
                "`status` — Current whiteboard status\n"
                "`screenshot` — Latest camera frame\n"
                "`timer` — Check if a timer is running\n"
                "`cancel` — Cancel the active timer\n"
                "`pause` — Pause notifications\n"
                "`resume` — Resume notifications\n"
                "`help` — Show this message"
            )
            await message.channel.send(help_text)

        else:
            await message.channel.send(
                f"Unknown command: `{cmd}`\nSend `help` for available commands."
            )

    print("Starting Whiteboard DM bot...")
    print(f"Commands: status, screenshot, timer, cancel, pause, resume, help")
    print("Press Ctrl+C to stop\n")

    try:
        bot.run(BOT_TOKEN)
    except KeyboardInterrupt:
        print("\nDM bot stopped")


if __name__ == "__main__":
    main()
