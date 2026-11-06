@echo off
rem ---------------------------------------------------------------------------
rem Double-click me: score the assistant on YOUR OWN questions.
rem   1st time: makes eval\bailey_questions.json from the template and opens it
rem             in Notepad. Type 7 real questions + 3 traps, save, double-click again.
rem   then:     asks each question once and writes eval\my_scorecard_<date-time>.md
rem             for you to mark right / half / wrong. MIN_SCORE is never changed.
rem   RUN-MY-QUESTIONS.cmd --dry    only run the checks, ask nothing (for testing)
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
title Project 1 - Score my own questions
echo === Project 1 - Score my own questions ===
call "%~dp0scripts\checks.cmd"
if errorlevel 1 goto failed
if /i "%~1"=="--dry" goto dry

if exist "eval\bailey_questions.json" goto have_questions
copy /y "eval\bailey_questions.template.json" "eval\bailey_questions.json" >nul
if errorlevel 1 goto copy_failed
echo.
echo First run: made eval\bailey_questions.json and opened it in Notepad.
echo Type your 7 real questions and 3 traps between the quotes (read the lines at the top),
echo save it (Ctrl+S), close Notepad, then double-click RUN-MY-QUESTIONS.cmd again.
start "" notepad "eval\bailey_questions.json"
pause
exit /b 0

:have_questions
echo.
echo Asking your questions, one at a time (the first run embeds the notes first, about a minute).
echo.
".venv\Scripts\python.exe" run_my_questions.py
if errorlevel 1 goto run_failed
echo.
echo Done. Open the scorecard named above (in the eval folder) and mark each row.
pause
exit /b 0

:dry
echo.
if exist "eval\bailey_questions.json" (echo Your questions file: eval\bailey_questions.json is there.) else (echo Your questions file: not made yet - a normal double-click makes it from the template.)
echo DRY RUN: every check passed. A normal double-click would now run:
echo   .venv\Scripts\python.exe run_my_questions.py   (writes eval\my_scorecard_^<date-time^>.md)
exit /b 0

:copy_failed
echo.
echo PROBLEM: Could not copy eval\bailey_questions.template.json to eval\bailey_questions.json.
echo FIX:     Check that the eval folder and the template file are there (unzip the whole project), then try again.
pause
exit /b 1

:run_failed
echo.
echo The scorecard stopped with a problem - read the lines above (they say how to fix it).
pause
exit /b 1

:failed
echo.
echo Nothing was started. Fix the problem above, then double-click RUN-MY-QUESTIONS.cmd again.
pause
exit /b 1
