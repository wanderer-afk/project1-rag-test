@echo off
rem ---------------------------------------------------------------------------
rem Shared checks for RUN-CHAT.cmd and RUN-ASK.cmd (they "call" this file).
rem The caller has already moved into the project folder (cd /d "%~dp0").
rem Exit code 0 = ready to run. 1 = a problem, already explained on screen.
rem Nothing here leaves the project folder: the only thing it creates is .venv\
rem ---------------------------------------------------------------------------

rem The two models rag.py uses. (P1_* can be set beforehand ONLY to test this
rem launcher's messages; rag.py itself always uses these two models.)
if not defined P1_EMBED_MODEL set "P1_EMBED_MODEL=nomic-embed-text:latest"
if not defined P1_CHAT_MODEL set "P1_CHAT_MODEL=gemma4:26b"
rem Oldest Python allowed is 3.%P1_MIN_MINOR%. 3.11 because numpy 2.3+ (the maths package in
rem requirements.txt) needs 3.11 or newer. Tested on 3.14 and 3.12.
if not defined P1_MIN_MINOR set "P1_MIN_MINOR=11"
rem How many steps the caller shows in total (RUN-CHAT adds a 4th: the port check).
if not defined P1_STEPS set "P1_STEPS=3"

echo [1/%P1_STEPS%] Is Ollama installed and running?
where ollama >nul 2>nul
if errorlevel 1 goto no_ollama
ollama list >nul 2>nul
if errorlevel 1 goto ollama_off
echo       yes

echo [2/%P1_STEPS%] Are the two models downloaded? (%P1_EMBED_MODEL% and %P1_CHAT_MODEL%)
set "MISSING="
ollama list 2>nul | findstr /b /i /c:"%P1_EMBED_MODEL% " >nul
if errorlevel 1 set "MISSING=%P1_EMBED_MODEL%"
if defined MISSING goto model_missing
ollama list 2>nul | findstr /b /i /c:"%P1_CHAT_MODEL% " >nul
if errorlevel 1 set "MISSING=%P1_CHAT_MODEL%"
if defined MISSING goto model_missing
echo       yes

echo [3/%P1_STEPS%] Is the Python environment (.venv, Python 3.%P1_MIN_MINOR% or newer) ready?
set "PYCHECK=import sys; sys.exit(0 if sys.version_info[:2] >= (3, %P1_MIN_MINOR%) else 1)"
if exist ".venv\Scripts\python.exe" goto venv_exists
rem Pick a Python: 3.14 first (what the project was built on), then the newest the
rem "py" launcher knows, then plain "python" - each only if it is 3.11 or newer.
set "P1_PY="
py -3.14 -c "%PYCHECK%" >nul 2>nul
if not errorlevel 1 set "P1_PY=py -3.14"
if defined P1_PY goto make_venv
py -3 -c "%PYCHECK%" >nul 2>nul
if not errorlevel 1 set "P1_PY=py -3"
if defined P1_PY goto make_venv
python -c "%PYCHECK%" >nul 2>nul
if not errorlevel 1 set "P1_PY=python"
if defined P1_PY goto make_venv
goto no_python
:make_venv
echo       First run: making the .venv folder with "%P1_PY%" (one time only)...
%P1_PY% -m venv .venv
if errorlevel 1 goto venv_failed
:venv_exists
".venv\Scripts\python.exe" -c "%PYCHECK%" >nul 2>nul
if errorlevel 1 goto wrong_python
".venv\Scripts\python.exe" -c "import ollama, numpy, streamlit" >nul 2>nul
if not errorlevel 1 goto deps_ok
echo       Installing the packages in requirements.txt (one time only, a few minutes)...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto pip_failed
:deps_ok
for /f "delims=" %%v in ('call ".venv\Scripts\python.exe" -c "import platform; print(platform.python_version())"') do echo       yes (Python %%v)
exit /b 0

rem ---------------------------- the fixes, in plain words ----------------------------
:no_ollama
echo.
echo PROBLEM: Ollama is not installed (the free program that runs the AI models on this PC).
echo FIX:     Install it from https://ollama.com/download , open it once, then double-click this file again.
exit /b 1

:ollama_off
echo.
echo PROBLEM: Ollama is installed but not running.
echo FIX:     Open the Ollama app from the Start menu (a llama icon appears near the clock),
echo          wait 10 seconds, then double-click this file again.
exit /b 1

:model_missing
echo.
echo PROBLEM: The model %MISSING% is not downloaded yet.
echo FIX:     Open a Command Prompt and type:   ollama pull %MISSING%
echo          (gemma4:26b is about 18 GB, nomic-embed-text about 0.3 GB.) Then double-click this file again.
exit /b 1

:no_python
echo.
echo PROBLEM: No Python 3.%P1_MIN_MINOR% or newer was found on this PC.
echo WHY:     The maths package this project uses (numpy 2.3 and later) needs Python 3.%P1_MIN_MINOR% or newer.
echo          This launcher looked for "py -3.14", then "py -3", then plain "python".
echo FIX:     Install Python 3.14 (any 3.%P1_MIN_MINOR% or newer works) from https://www.python.org/downloads/
echo          - keep "py launcher" ticked - then double-click this file again.
exit /b 1

:venv_failed
echo.
echo PROBLEM: Python could not make the .venv folder.
echo FIX:     Delete the half-made .venv folder in this project (if there is one) and double-click again.
exit /b 1

:wrong_python
echo.
echo PROBLEM: The .venv folder was made with a Python older than 3.%P1_MIN_MINOR%.
echo WHY:     The maths package this project uses (numpy 2.3 and later) needs Python 3.%P1_MIN_MINOR% or newer.
echo FIX:     Delete the .venv folder in this project, then double-click this file again;
echo          it rebuilds .venv with the newest Python it finds (3.14 if installed).
exit /b 1

:pip_failed
echo.
echo PROBLEM: Installing the packages failed (read the red lines above).
echo WHY:     Usually one of two things: (1) no internet, or (2) the project sits in a folder that
echo          OneDrive syncs, and OneDrive locks files while pip writes them.
echo FIX:     (1) Check the internet connection. (2) Move the whole project to a plain folder such as
echo          C:\project1-rag , delete its .venv folder, and double-click this file again.
exit /b 1
