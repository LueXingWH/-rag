@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ============================================================
echo   Campus RAG Demo  (Campus Document Q and A)
echo ============================================================
echo.
echo   [1] Offline  - no internet, no API key needed  (recommended)
echo   [2] LLM mode - needs DEEPSEEK_API_KEY, gives model-written answers
echo   [3] Run evaluation only  (show metrics)
echo.
set /p choice=Choose [1/2/3], press Enter for 1:
if "%choice%"=="" set choice=1
echo.

if "%choice%"=="3" goto eval
if "%choice%"=="2" goto llmon
goto offline

:offline
if not "%DEEPSEEK_API_KEY%"=="" echo   [note] DEEPSEEK_API_KEY is set, but OFFLINE mode ignores it.
if not "%DEEPSEEK_API_KEY%"=="" echo          To actually use the API, start again and choose 2.
if not "%DEEPSEEK_API_KEY%"=="" echo.
echo   [info] Starting in OFFLINE mode. Browser will open automatically...
echo   [tip ] Press Ctrl+C or close this window to stop the server.
echo.
py -3 ask.py --web
if errorlevel 1 goto fail
goto end

:llmon
if "%DEEPSEEK_API_KEY%"=="" goto nokey
echo   [info] Starting in LLM mode. Browser will open automatically...
echo.
py -3 ask.py --web --llm
goto end

:nokey
echo   [warn] DEEPSEEK_API_KEY is not set.
echo          To enable LLM mode, run this window first:
echo             set DEEPSEEK_API_KEY=sk-your-key
echo          then start this script again.
echo          Falling back to OFFLINE mode now.
echo.
goto offline

:eval
echo   Running offline evaluation (22 questions)...
echo.
py -3 ask.py --eval
goto end

:fail
echo.
echo   [ERROR] Failed to start.
echo           1. Is Python 3.8+ installed?  https://www.python.org/downloads/
echo           2. If installed, tick "Add Python to PATH" during setup.
echo           3. Try running this command manually to see the error:
echo                 py -3 ask.py --info
echo.

:end
echo.
pause
