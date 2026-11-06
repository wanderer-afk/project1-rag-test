@echo off
rem ---------------------------------------------------------------------------
rem Double-click me: starts the "Ask my notes" chat page and opens it in the browser.
rem   RUN-CHAT.cmd          normal start
rem   RUN-CHAT.cmd --dry    only run the checks, start nothing (for testing)
rem The page listens on 127.0.0.1 only, so no other computer can reach it.
rem ---------------------------------------------------------------------------
setlocal
if /i "%~1"=="--open-when-ready" goto open_when_ready
cd /d "%~dp0"
set "PYTHONUTF8=1"
title Project 1 - Ask my notes (chat page)
echo === Project 1 - Ask my notes: chat page ===
set "P1_STEPS=4"
call "%~dp0scripts\checks.cmd"
if errorlevel 1 goto failed

echo [4/4] Is port 8599 free?
netstat -ano -p tcp | findstr /r /c:":8599 .*LISTENING" >nul
if errorlevel 1 goto port_free
curl -s -m 3 http://127.0.0.1:8599/_stcore/health 2>nul | findstr /x "ok" >nul
if errorlevel 1 goto port_busy
echo       The chat page is already running. Opening it: http://localhost:8599
if /i "%~1"=="--dry" exit /b 0
if not defined P1_NO_BROWSER start "" http://localhost:8599
ping -n 4 127.0.0.1 >nul
exit /b 0

:port_free
echo       yes
if /i "%~1"=="--dry" goto dry
echo.
echo Starting the chat page at http://localhost:8599 - the browser opens when it is ready.
echo Keep this window open while you use the page. To stop it: close this window or press Ctrl+C.
echo.
rem A second, hidden copy of this file waits for the page to answer, then opens the browser.
start "" /b cmd /c ""%~f0" --open-when-ready"
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8599 --server.headless true --browser.gatherUsageStats false
if errorlevel 1 goto stopped
exit /b 0

:open_when_ready
rem Ask the page "are you up?" once a second for up to 60 seconds, then open the browser.
for /l %%i in (1,1,60) do (
  curl -s -m 2 http://127.0.0.1:8599/_stcore/health 2>nul | findstr /x "ok" >nul && goto ready
  ping -n 2 127.0.0.1 >nul
)
:ready
echo Chat page is ready: http://localhost:8599
if not defined P1_NO_BROWSER start "" http://localhost:8599
exit /b 0

:dry
echo.
echo DRY RUN: every check passed. A normal double-click would now run:
echo   .venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8599
echo   and open http://localhost:8599 in the browser.
exit /b 0

:port_busy
echo.
echo PROBLEM: Port 8599 is used by another program, so the chat page cannot start there.
echo FIX:     Close the other window that is running a web page (often an older RUN-CHAT window),
echo          or restart the PC, then double-click this file again.
goto pause_fail

:stopped
echo.
echo The chat page stopped. If you did not stop it yourself, read the lines above for the reason.
goto pause_fail

:failed
echo.
echo Nothing was started. Fix the problem above, then double-click RUN-CHAT.cmd again.
:pause_fail
pause
exit /b 1
