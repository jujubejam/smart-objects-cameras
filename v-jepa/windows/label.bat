@echo off
REM Label recorded clips for V-JEPA probe training
REM
REM Usage: Just double-click this file after recording

echo.
echo ========================================
echo V-JEPA Clip Labeler
echo ========================================
echo.

call conda activate vjepa
cd /d "%~dp0"
python clip_labeler.py --input D:\ck_office --classes at_computer,playing_keyboard,tending_plants

pause
