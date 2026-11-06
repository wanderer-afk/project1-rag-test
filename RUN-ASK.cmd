@echo off
rem ---------------------------------------------------------------------------
rem Double-click me: ask the notes questions right here in this window.
rem Each answer comes only from the notes, with its sources [1][2][3].
rem A question the notes do not cover gets "Not in my notes."
rem   RUN-ASK.cmd          normal start
rem   RUN-ASK.cmd --dry    only run the checks, ask nothing (for testing)
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
title Project 1 - Ask my notes (in this window)
echo === Project 1 - Ask my notes: questions in this window ===
call "%~dp0scripts\checks.cmd"
if errorlevel 1 goto failed
if /i "%~1"=="--dry" goto dry

echo.
echo Type a question and press Enter. A blank line (just Enter) closes it.
echo The very first run reads and embeds the notes first (about a minute).
echo.
".venv\Scripts\python.exe" rag.py
if errorlevel 1 goto run_failed
echo.
echo Bye.
pause
exit /b 0

:dry
echo.
echo DRY RUN: every check passed. A normal double-click would now run:
echo   .venv\Scripts\python.exe rag.py   (asks for questions until a blank line)
exit /b 0

:run_failed
echo.
echo The assistant stopped with a problem - read the lines above (they say how to fix it).
pause
exit /b 1

:failed
echo.
echo Nothing was started. Fix the problem above, then double-click RUN-ASK.cmd again.
pause
exit /b 1
