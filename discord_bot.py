#!/usr/bin/env python3
"""
Simple Discord Bot for OAK-D Camera
====================================
Responds to commands and sends camera status updates.

Commands:
    !ping                - Test if bot is alive
    !status              - Check camera status
    !detect              - Get current detection info
    !screenshot          - Get live camera image
    !whiteboard          - Show current whiteboard text
    !whiteboard-status   - Full whiteboard status embed
    !whiteboard-history  - Show recent whiteboard readings
    !whiteboard-screenshot - Get whiteboard camera image
    !whiteboard-consensus  - Show aggregated reading
    !set-confidence      - Set OCR confidence threshold
    !set-fps             - Set camera FPS
    !toggle-notifications - Toggle Discord notifications
    !help                - Show available commands

Setup:
    1. Install discord.py: pip install discord.py
    2. Add DISCORD_BOT_TOKEN to .env file
    3. Run: python3 discord_bot.py
"""

import discord
from discord.ext import commands
import os
import json
import socket
from pathlib import Path
from datetime import datetime

# Load environment variables from ~/oak-projects/.env (per-user)
try:
    from dotenv import load_dotenv
    load_dotenv(Path.home() / "oak-projects" / ".env")
except ImportError:
    print("⚠️  python-dotenv not installed - make sure DISCORD_BOT_TOKEN is in environment")

# Configuration
BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
STATUS_FILE = Path.home() / "oak-projects" / "camera_status.json"
SCREENSHOT_FILE = Path.home() / "oak-projects" / "latest_frame.jpg"

# Whiteboard integration
WHITEBOARD_STATUS_FILE = Path.home() / "oak-projects" / "whiteboard_status.json"
WHITEBOARD_HISTORY_FILE = Path.home() / "oak-projects" / "whiteboard_history.jsonl"
WHITEBOARD_SCREENSHOT_FILE = Path.home() / "oak-projects" / "latest_whiteboard_frame.jpg"
WHITEBOARD_CONFIG_FILE = Path.home() / "oak-projects" / "whiteboard_config.json"

# Check token
if not BOT_TOKEN:
    print("❌ Error: DISCORD_BOT_TOKEN not set in .env file")
    print("   Add this line to ~/oak-projects/.env:")
    print("   DISCORD_BOT_TOKEN=your_token_here")
    exit(1)

# Create bot with command prefix
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content
bot = commands.Bot(command_prefix='!', intents=intents)

# Remove default help command (we'll make our own)
bot.remove_command('help')

# Camera identity from hostname (e.g., "orbit", "gravity", "horizon")
CAMERA_NAME = socket.gethostname().split('.')[0].lower()
KNOWN_CAMERAS = ["orbit", "gravity", "horizon"]


# --- Event Handlers ---

@bot.event
async def on_ready():
    """Called when bot successfully connects to Discord."""
    print(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    print(f'Camera: {CAMERA_NAME}')
    print('Bot is ready!')
    print('------')


@bot.event
async def on_message(message):
    """Called for every message sent in channels the bot can see."""
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Process commands
    await bot.process_commands(message)


# --- Commands ---

@bot.command(name='ping', help='Test if the bot is alive')
async def ping(ctx):
    """Simple ping command to test bot responsiveness."""
    latency = round(bot.latency * 1000)  # Convert to milliseconds
    await ctx.send(f'🏓 Pong! (Latency: {latency}ms)')


@bot.command(name='status', help='Check camera status')
async def status(ctx):
    """Check if camera system is running."""
    try:
        # Try to read status file
        if STATUS_FILE.exists():
            status_data = json.loads(STATUS_FILE.read_text())
            timestamp = status_data.get('timestamp', 'unknown')
            username = status_data.get('username', 'unknown')
            hostname = status_data.get('hostname', 'unknown')

            # Check if status is recent (within last 10 seconds)
            try:
                status_time = datetime.fromisoformat(timestamp)
                age = (datetime.now() - status_time).total_seconds()

                if age < 10:
                    user_info = f"👤 Running: **{username}** on **{hostname}**\n" if username != 'unknown' else ""
                    await ctx.send(f"✅ Camera is **ONLINE**\n{user_info}📊 Last update: {age:.1f}s ago")
                else:
                    await ctx.send(f"⚠️ Camera status is **STALE**\n📊 Last update: {age:.0f}s ago\nCamera may be offline.")
            except:
                await ctx.send(f"✅ Camera status file exists\n📊 Timestamp: {timestamp}")
        else:
            await ctx.send("❌ Camera is **OFFLINE**\n💡 Status file not found. Is person_detector.py running?")

    except Exception as e:
        await ctx.send(f"❌ Error checking status: {str(e)}")


@bot.command(name='detect', help='Get current detection status')
async def detect(ctx):
    """Show current person detection status."""
    try:
        if not STATUS_FILE.exists():
            await ctx.send("❌ No detection data available\n💡 Make sure person_detector.py is running")
            return

        # Read status
        status_data = json.loads(STATUS_FILE.read_text())
        detected = status_data.get('detected', False)
        count = status_data.get('count', 0)
        timestamp = status_data.get('timestamp', 'unknown')
        username = status_data.get('username', 'unknown')
        hostname = status_data.get('hostname', 'unknown')

        # Format response
        if detected:
            emoji = "🟢"
            status_text = f"**PERSON DETECTED**\n👥 Count: {count}"
        else:
            emoji = "⚪"
            status_text = "**No person detected**"

        # Add user info if available
        user_info = f"👤 Camera: **{username}** on **{hostname}**\n" if username != 'unknown' else ""

        await ctx.send(f"{emoji} {status_text}\n{user_info}🕐 Last update: {timestamp}")

    except Exception as e:
        await ctx.send(f"❌ Error reading detection data: {str(e)}")


@bot.command(name='screenshot', help='Get a screenshot from the camera')
async def screenshot(ctx):
    """Send the latest camera frame."""
    try:
        if not SCREENSHOT_FILE.exists():
            await ctx.send("❌ No screenshot available\n💡 Make sure person_detector.py is running")
            return

        # Check screenshot age
        file_age = datetime.now().timestamp() - SCREENSHOT_FILE.stat().st_mtime

        if file_age > 30:
            await ctx.send(f"⚠️ Screenshot is old ({file_age:.0f}s)\nCamera may not be running.")
            return

        # Send the screenshot
        await ctx.send(
            f"📸 **Camera Screenshot**\n🕐 Captured: {file_age:.1f}s ago",
            file=discord.File(str(SCREENSHOT_FILE))
        )

    except Exception as e:
        await ctx.send(f"❌ Error sending screenshot: {str(e)}")


# --- Camera Routing Commands ---

async def _dispatch_to_command(ctx, cmd_string):
    """
    Dispatch a command string to the appropriate handler.

    Rewrites ctx.message.content so discord.py handles argument parsing
    naturally (e.g., "whiteboard-history 10" passes 10 as the count param).
    """
    parts = cmd_string.strip().split(None, 1)
    cmd_name = parts[0]
    cmd_args = parts[1] if len(parts) > 1 else ""

    target_cmd = bot.get_command(cmd_name)
    if not target_cmd:
        await ctx.send(f"Unknown command: `{cmd_name}`\n💡 Use `!help` to see available commands")
        return

    # Rewrite the message content and re-process
    ctx.message.content = f"!{cmd_name} {cmd_args}".strip()
    await bot.process_commands(ctx.message)


@bot.command(name='orbit')
async def orbit_command(ctx, *, cmd):
    """Route command to Orbit camera only."""
    if CAMERA_NAME == 'orbit':
        await _dispatch_to_command(ctx, cmd)


@bot.command(name='gravity')
async def gravity_command(ctx, *, cmd):
    """Route command to Gravity camera only."""
    if CAMERA_NAME == 'gravity':
        await _dispatch_to_command(ctx, cmd)


@bot.command(name='horizon')
async def horizon_command(ctx, *, cmd):
    """Route command to Horizon camera only."""
    if CAMERA_NAME == 'horizon':
        await _dispatch_to_command(ctx, cmd)


@bot.command(name='all')
async def all_cameras_command(ctx, *, cmd):
    """Route command to all cameras (all bots respond)."""
    await _dispatch_to_command(ctx, cmd)


# --- Whiteboard Commands ---

@bot.command(name='whiteboard', aliases=['read-board'], help='Show current whiteboard text')
async def whiteboard(ctx):
    """Show current whiteboard text content."""
    try:
        if not WHITEBOARD_STATUS_FILE.exists():
            await ctx.send("❌ No whiteboard data available\n💡 Make sure whiteboard_reader_full.py is running")
            return

        status_data = json.loads(WHITEBOARD_STATUS_FILE.read_text())
        text_content = status_data.get('text_content', [])

        if not text_content:
            await ctx.send("📋 Whiteboard is empty - no text detected")
            return

        lines = "\n".join(f"  {line}" for line in text_content)
        await ctx.send(f"📋 **Whiteboard Text:**\n```\n{lines}\n```")

    except Exception as e:
        await ctx.send(f"❌ Error reading whiteboard: {str(e)}")


@bot.command(name='whiteboard-status', help='Full whiteboard status embed')
async def whiteboard_status(ctx):
    """Show detailed whiteboard status as a rich embed."""
    try:
        if not WHITEBOARD_STATUS_FILE.exists():
            await ctx.send("❌ No whiteboard data available\n💡 Make sure whiteboard_reader_full.py is running")
            return

        status_data = json.loads(WHITEBOARD_STATUS_FILE.read_text())
        text_detected = status_data.get('text_detected', False)
        text_content = status_data.get('text_content', [])
        num_regions = status_data.get('num_text_regions', 0)
        username = status_data.get('username', 'unknown')
        hostname = status_data.get('hostname', 'unknown')
        timestamp = status_data.get('timestamp', 'unknown')

        # Calculate timestamp age
        age_str = "unknown"
        try:
            status_time = datetime.fromisoformat(timestamp)
            age = (datetime.now() - status_time).total_seconds()
            if age < 60:
                age_str = f"{age:.0f}s ago"
            elif age < 3600:
                age_str = f"{age / 60:.0f}m ago"
            else:
                age_str = f"{age / 3600:.1f}h ago"
        except (ValueError, TypeError):
            pass

        color = discord.Color.green() if text_detected else discord.Color.light_grey()

        embed = discord.Embed(
            title="📋 Whiteboard Status",
            description="Real-time OCR status",
            color=color
        )
        embed.add_field(
            name="Text Detected",
            value="Yes" if text_detected else "No",
            inline=True
        )
        embed.add_field(name="Text Regions", value=str(num_regions), inline=True)
        embed.add_field(name="Last Update", value=age_str, inline=True)
        embed.add_field(
            name="Running On",
            value=f"{username}@{hostname}",
            inline=True
        )

        if text_content:
            content_preview = "\n".join(text_content[:5])
            if len(text_content) > 5:
                content_preview += f"\n... and {len(text_content) - 5} more lines"
            embed.add_field(name="Text Content", value=f"```{content_preview}```", inline=False)

        await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(f"❌ Error reading whiteboard status: {str(e)}")


@bot.command(name='whiteboard-history', help='Show recent whiteboard readings')
async def whiteboard_history(ctx, count: int = 5):
    """Show recent whiteboard text history."""
    try:
        if not WHITEBOARD_HISTORY_FILE.exists():
            await ctx.send("❌ No whiteboard history available\n💡 History is recorded when whiteboard_reader_full.py is running")
            return

        # Read last N lines from JSONL file
        lines = WHITEBOARD_HISTORY_FILE.read_text().strip().split('\n')
        recent = lines[-count:] if len(lines) >= count else lines

        if not recent or recent == ['']:
            await ctx.send("📋 Whiteboard history is empty")
            return

        entries = []
        for line in recent:
            try:
                entry = json.loads(line)
                ts = entry.get('timestamp', 'unknown')
                # Shorten timestamp for display
                try:
                    dt = datetime.fromisoformat(ts)
                    ts = dt.strftime("%H:%M:%S")
                except (ValueError, TypeError):
                    pass
                text_lines = entry.get('text_lines', [])
                avg_conf = entry.get('avg_confidence', 0.0)
                text_preview = ", ".join(text_lines[:2]) if text_lines else "[no text]"
                if len(text_preview) > 60:
                    text_preview = text_preview[:57] + "..."
                entries.append(f"[{ts}] conf={avg_conf:.0%} | {text_preview}")
            except json.JSONDecodeError:
                continue

        if not entries:
            await ctx.send("📋 No valid history entries found")
            return

        history_text = "\n".join(entries)
        await ctx.send(f"📋 **Whiteboard History** (last {len(entries)}):\n```\n{history_text}\n```")

    except Exception as e:
        await ctx.send(f"❌ Error reading whiteboard history: {str(e)}")


@bot.command(name='whiteboard-screenshot', help='Get a whiteboard screenshot')
async def whiteboard_screenshot(ctx):
    """Send the latest whiteboard camera frame."""
    try:
        if not WHITEBOARD_SCREENSHOT_FILE.exists():
            await ctx.send("❌ No whiteboard screenshot available\n💡 Make sure whiteboard_reader_full.py is running")
            return

        # Check screenshot age
        file_age = datetime.now().timestamp() - WHITEBOARD_SCREENSHOT_FILE.stat().st_mtime

        if file_age > 30:
            await ctx.send(f"⚠️ Whiteboard screenshot is old ({file_age:.0f}s)\nCamera may not be running.")
            return

        await ctx.send(
            f"📸 **Whiteboard Screenshot**\n🕐 Captured: {file_age:.1f}s ago",
            file=discord.File(str(WHITEBOARD_SCREENSHOT_FILE))
        )

    except Exception as e:
        await ctx.send(f"❌ Error sending whiteboard screenshot: {str(e)}")


@bot.command(name='whiteboard-consensus', help='Show aggregated whiteboard reading')
async def whiteboard_consensus(ctx):
    """Show aggregated consensus reading from recent whiteboard history."""
    try:
        # Get current text from status file
        current_text = []
        if WHITEBOARD_STATUS_FILE.exists():
            status_data = json.loads(WHITEBOARD_STATUS_FILE.read_text())
            current_text = status_data.get('text_content', [])

        # Read last 10 entries from history
        if not WHITEBOARD_HISTORY_FILE.exists():
            if current_text:
                await ctx.send(f"📋 **Current text:** {', '.join(current_text)}\n💡 No history available for consensus")
            else:
                await ctx.send("❌ No whiteboard data available\n💡 Make sure whiteboard_reader_full.py is running")
            return

        lines = WHITEBOARD_HISTORY_FILE.read_text().strip().split('\n')
        recent = lines[-10:] if len(lines) >= 10 else lines

        # Count text frequency and track confidence
        text_counts = {}
        text_confidences = {}
        for line in recent:
            try:
                entry = json.loads(line)
                for text in entry.get('text_lines', []):
                    text_lower = text.lower()
                    text_counts[text_lower] = text_counts.get(text_lower, 0) + 1
                    conf = entry.get('avg_confidence', 0.0)
                    if text_lower not in text_confidences:
                        text_confidences[text_lower] = []
                    text_confidences[text_lower].append(conf)
            except json.JSONDecodeError:
                continue

        if not text_counts:
            await ctx.send("📋 No text found in recent history")
            return

        # Sort by frequency
        sorted_texts = sorted(text_counts.items(), key=lambda x: x[1], reverse=True)

        result_lines = []
        for text, count in sorted_texts[:5]:
            confs = text_confidences.get(text, [0.0])
            min_conf = min(confs)
            max_conf = max(confs)
            conf_range = f"{min_conf:.0%}-{max_conf:.0%}" if min_conf != max_conf else f"{max_conf:.0%}"
            result_lines.append(f'  "{text}" - seen {count}x (confidence: {conf_range})')

        consensus_text = "\n".join(result_lines)
        await ctx.send(f"📋 **Whiteboard Consensus** (last {len(recent)} readings):\n```\n{consensus_text}\n```")

    except Exception as e:
        await ctx.send(f"❌ Error computing consensus: {str(e)}")


@bot.command(name='set-confidence', help='Set OCR confidence threshold (0.0-1.0)')
async def set_confidence(ctx, value: float):
    """Write confidence threshold to whiteboard config file."""
    try:
        if value < 0.0 or value > 1.0:
            await ctx.send("❌ Confidence must be between 0.0 and 1.0")
            return

        # Read existing config or start fresh
        config = {}
        if WHITEBOARD_CONFIG_FILE.exists():
            try:
                config = json.loads(WHITEBOARD_CONFIG_FILE.read_text())
            except json.JSONDecodeError:
                pass

        config['confidence'] = value
        WHITEBOARD_CONFIG_FILE.write_text(json.dumps(config, indent=2))
        await ctx.send(f"✅ Confidence threshold set to **{value}**\n💡 Whiteboard reader will pick this up within a few seconds")

    except Exception as e:
        await ctx.send(f"❌ Error setting confidence: {str(e)}")


@bot.command(name='set-fps', help='Set camera FPS (1-30)')
async def set_fps(ctx, value: int):
    """Write FPS to whiteboard config file."""
    try:
        if value < 1 or value > 30:
            await ctx.send("❌ FPS must be between 1 and 30")
            return

        # Read existing config or start fresh
        config = {}
        if WHITEBOARD_CONFIG_FILE.exists():
            try:
                config = json.loads(WHITEBOARD_CONFIG_FILE.read_text())
            except json.JSONDecodeError:
                pass

        config['fps_limit'] = value
        WHITEBOARD_CONFIG_FILE.write_text(json.dumps(config, indent=2))
        await ctx.send(f"✅ FPS limit set to **{value}**\n💡 Note: FPS changes require a pipeline restart to take effect")

    except Exception as e:
        await ctx.send(f"❌ Error setting FPS: {str(e)}")


@bot.command(name='toggle-notifications', help='Toggle Discord notifications')
async def toggle_notifications(ctx):
    """Toggle notifications_enabled in whiteboard config."""
    try:
        # Read existing config or start fresh
        config = {}
        if WHITEBOARD_CONFIG_FILE.exists():
            try:
                config = json.loads(WHITEBOARD_CONFIG_FILE.read_text())
            except json.JSONDecodeError:
                pass

        current = config.get('notifications_enabled', True)
        config['notifications_enabled'] = not current
        WHITEBOARD_CONFIG_FILE.write_text(json.dumps(config, indent=2))

        new_state = "ENABLED" if not current else "DISABLED"
        emoji = "🔔" if not current else "🔕"
        await ctx.send(f"{emoji} Notifications are now **{new_state}**")

    except Exception as e:
        await ctx.send(f"❌ Error toggling notifications: {str(e)}")


# --- Classroom API Commands (reads from Supabase via classroom_api.py) ---

CLASSROOM_API_URL = os.getenv("CLASSROOM_API_URL", "")


@bot.command(name='classroom', help='Show full classroom state from Supabase')
async def classroom_command(ctx):
    """Full classroom state from all cameras via the classroom API."""
    if not CLASSROOM_API_URL:
        await ctx.send("❌ `CLASSROOM_API_URL` not configured in .env")
        return

    try:
        import requests
        r = requests.get(f"{CLASSROOM_API_URL}/state", timeout=5)
        r.raise_for_status()
        state = r.json()

        mode = state.get("room_mode", "unknown")
        persons = state.get("total_persons", 0)
        whiteboard = "active" if state.get("whiteboard_active") else "idle"

        mode_emoji = {
            "solo": "🎉", "duo": "☕", "group": "🎊",
            "focus": "📝", "presentation": "🎤", "empty": "🪑",
        }.get(mode, "❓")

        lines = [
            f"{mode_emoji} **Room Mode: {mode.upper()}**",
            f"👥 People: **{persons}**  |  📋 Whiteboard: **{whiteboard}**",
            "",
        ]

        cameras = state.get("cameras", {})
        for cam_id, cam in cameras.items():
            probe = cam.get("predicted_class", "?")
            conf = cam.get("prediction_confidence", 0)
            count = cam.get("person_count", 0)
            lines.append(
                f"  **{cam_id}**: {count} people, "
                f"probe=`{probe}` ({conf:.0%})"
            )

        await ctx.send("\n".join(lines))

    except Exception as e:
        await ctx.send(f"❌ Classroom API error: {e}")


@bot.command(name='mode', help='Show current room mode')
async def mode_command(ctx):
    """Current room mode from the classroom API."""
    if not CLASSROOM_API_URL:
        await ctx.send("❌ `CLASSROOM_API_URL` not configured in .env")
        return

    try:
        import requests
        r = requests.get(f"{CLASSROOM_API_URL}/mode", timeout=5)
        r.raise_for_status()
        data = r.json()

        mode = data.get("room_mode", "unknown")
        persons = data.get("total_persons", 0)
        probe = data.get("probe_consensus") or "none"

        mode_emoji = {
            "solo": "🎉", "duo": "☕", "group": "🎊",
            "focus": "📝", "presentation": "🎤", "empty": "🪑",
        }.get(mode, "❓")

        msg = f"{mode_emoji} **{mode.upper()}** — {persons} people, probe consensus: `{probe}`"
        await ctx.send(msg)

    except Exception as e:
        await ctx.send(f"❌ Classroom API error: {e}")


@bot.command(name='help', help='Show available commands')
async def help_command(ctx):
    """Display help message with all available commands."""
    help_text = f"""
**🤖 OAK-D Camera Bot** (this is **{CAMERA_NAME}**)

**Camera:**
`!ping` - Test if bot is alive
`!status` - Check if camera is running
`!detect` - Get current detection status
`!screenshot` - Get a live image from camera

**Whiteboard:**
`!whiteboard` (or `!read-board`) - Show current whiteboard text
`!whiteboard-status` - Full whiteboard status embed
`!whiteboard-history [count]` - Show recent readings (default 5)
`!whiteboard-screenshot` - Get whiteboard camera image
`!whiteboard-consensus` - Show aggregated reading

**Whiteboard Config:**
`!set-confidence <0.0-1.0>` - Set OCR confidence threshold
`!set-fps <1-30>` - Set camera FPS
`!toggle-notifications` - Toggle Discord notifications

**Classroom (Supabase):**
`!classroom` - Full classroom state from all cameras
`!mode` - Current room mode (solo/duo/group/focus/presentation)

**Multi-Camera:**
`!orbit <command>` - Send command to Orbit only
`!gravity <command>` - Send command to Gravity only
`!horizon <command>` - Send command to Horizon only
`!all <command>` - Send command to all cameras

`!help` - Show this message

**💡 Tips:**
• Use `!orbit status` to target a specific camera
• Use `!all screenshot` to get images from all cameras
• Bare commands (e.g. `!status`) are answered by all bots
    """
    await ctx.send(help_text)


# --- Helper Function for Person Detector Integration ---

async def send_alert(message: str):
    """
    Send an alert to all channels the bot can access.
    Call this from person_detector.py to send notifications.

    Example:
        await bot.send_alert("🟢 Person detected!")
    """
    for guild in bot.guilds:
        for channel in guild.text_channels:
            if channel.permissions_for(guild.me).send_messages:
                try:
                    await channel.send(message)
                    break  # Only send to first available channel per server
                except:
                    continue


# Add send_alert to bot object so person_detector can access it
bot.send_alert = send_alert


# --- Main ---

if __name__ == '__main__':
    print(f"Starting Discord bot for camera: {CAMERA_NAME}")
    print(f"Command prefix: !")
    print(f"Camera routing: !{CAMERA_NAME} <command> (targeted), !all <command> (broadcast)")
    print("Commands: !ping, !status, !detect, !screenshot, !help")
    print("Whiteboard: !whiteboard, !whiteboard-status, !whiteboard-history,")
    print("           !whiteboard-screenshot, !whiteboard-consensus")
    print("Config: !set-confidence, !set-fps, !toggle-notifications")
    print("\nPress Ctrl+C to stop\n")

    try:
        # Start the bot
        bot.run(BOT_TOKEN)
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
