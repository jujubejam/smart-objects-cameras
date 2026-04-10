@echo off
REM Auto-record clips for V-JEPA probe training
REM Saves to D:\vjepa-clips\unlabeled\
REM
REM Usage: Just double-click this file, or run from command prompt
REM        Press Ctrl+C to stop recording

echo.
echo ========================================
echo V-JEPA Auto Recorder
echo ========================================
echo.
echo Clips will be saved to: D:\ck_office\unlabeled\
echo Recording every 10 seconds, 3-second clips
echo.
echo Press Ctrl+C to stop
echo.

call conda activate vjepa
cd /d "%~dp0"
python auto_recorder.py --output D:\ck_office --interval 10 --display

echo.
echo ========================================
echo Recording complete!
echo.
echo Next step - label your clips:
echo   python clip_labeler.py --input D:\vjepa-clips
echo ========================================
pause
