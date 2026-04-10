@echo off
REM Train V-JEPA probe on labeled clips
REM
REM Prerequisites:
REM   1. server.py must be running in another terminal
REM   2. Labeled clips in D:\vjepa-clips\{class_folders}

echo.
echo ========================================
echo V-JEPA Probe Trainer
echo ========================================
echo.
echo Make sure server.py is running!
echo.

call conda activate vjepa
cd /d "%~dp0"
python probe_trainer.py --clips-dir D:\ck_office --output D:\ck_office\home_probe.pt

pause
