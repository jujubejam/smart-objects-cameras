@echo off
REM Run live probe inference
REM
REM Prerequisites:
REM   1. server.py must be running in another terminal
REM   2. Trained probe at D:\vjepa-clips\home_probe.pt

echo.
echo ========================================
echo V-JEPA Live Inference
echo ========================================
echo.
echo Make sure server.py is running!
echo.

call conda activate vjepa
cd /d "%~dp0"
python probe_inference.py --probe D:\ck_office\home_probe.pt --display

pause
