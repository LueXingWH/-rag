@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ============================================================
echo   Campus RAG Demo  (Campus Document Q and A)
echo ============================================================
echo.

rem ---- 0) Can THIS window see the key? ----
rem IMPORTANT: this file must stay PURE ASCII and use CRLF line endings.
rem cmd.exe seeks through a batch file by BYTE offset, so non-ASCII characters
rem make it resume mid-word and report garbage like "'indow:' is not recognized".
rem Also: $env:XXX in PowerShell affects only that one window, and double-clicking
rem this file opens a NEW window which will not see it. Most "I set the key but it
rem still will not use the model" reports come down to exactly this.
if "%DEEPSEEK_API_KEY%"=="" goto key_missing
echo   [key ] DEEPSEEK_API_KEY detected in this window: ...%DEEPSEEK_API_KEY:~-4%
echo          The LLM switch in the web page can be turned on at any time.
echo.
goto menu

:key_missing
echo   [key ] DEEPSEEK_API_KEY NOT visible in this window - LLM is unavailable.
echo          Two ways to fix it:
echo            a) choose 2 below and paste the key when asked - this window only
echo            b) setx DEEPSEEK_API_KEY "sk-your-key"  then CLOSE and REOPEN this file
echo          Note: $env:DEEPSEEK_API_KEY in PowerShell affects ONLY that window.
echo                Double-clicking this file opens a NEW window and will not see
echo                it - run this file from that same PowerShell window instead.
echo.

:menu
echo   [1] Web demo  - LLM off by default  - the switch in the page still works
echo   [2] Web demo  - LLM on by default   - needs DEEPSEEK_API_KEY
echo   [3] Run evaluation only  - show metrics
echo   [4] Diagnose why LLM does not work  - full report, names the reason
echo.
set /p choice=Choose [1/2/3/4], press Enter for 1:
if "%choice%"=="" set choice=1
echo.

if "%choice%"=="4" goto doctor
if "%choice%"=="3" goto eval
if "%choice%"=="2" goto llmon
goto offline

:doctor
echo   Checking every step: env vars, config, network, /models, /chat/completions...
echo.
py -3 doctor.py
echo.
echo   Report also saved to data\logs\doctor_report.txt
echo.
pause
goto end

:offline
if not "%DEEPSEEK_API_KEY%"=="" echo   [note] Key IS set - offline is only the DEFAULT here.
if not "%DEEPSEEK_API_KEY%"=="" echo          Flip the "Use LLM" switch in the page to use it.
if not "%DEEPSEEK_API_KEY%"=="" echo.
echo   [info] Starting with LLM OFF by default. Browser will open automatically...
echo   [tip ] Press Ctrl+C or close this window to stop the server.
echo.
py -3 ask.py --web
if errorlevel 1 goto fail
goto end

:llmon
if "%DEEPSEEK_API_KEY%"=="" goto nokey
rem Check the API BEFORE starting the server: it is much easier to read a failure
rem here than to wonder why every answer in the browser looks like offline mode.
echo   [info] Checking the API connection first - this makes one real request...
echo.
py -3 ask.py --check-llm --llm
if errorlevel 1 goto llmfail
echo.
echo   [info] Starting with LLM ON by default. Browser will open automatically...
echo.
py -3 ask.py --web --llm
if errorlevel 1 goto fail
goto end

:llmfail
echo.
echo   [warn] The API check above FAILED - this is why LLM mode would not work.
echo          Read the reason printed above. Then:
echo            - key wrong or expired  -^> get a new one at platform.deepseek.com
echo            - balance / quota      -^> top up the account
echo            - cannot connect       -^> proxy, firewall or offline network
echo.
echo          Press Enter to start in OFFLINE mode anyway, or close this window.
echo          Tip: choose [4] for a full report of every step.
echo.
pause
goto offline

:nokey
rem The key prompt itself lives in Python (ask.py --ask-key), NOT here. Two reasons:
rem   1) this file must stay pure ASCII, so it cannot explain anything in Chinese -
rem      and "how do I copy the key" is exactly the step that needs clear wording;
rem   2) Python reads the key straight into its own process, so we do not depend on
rem      cmd's set / variable inheritance at all. That also removed a real bug:
rem      "%key:"=%" with an undefined key leaked a quote into the next "if" line.
echo   [info] No key in this window - switching to the guided setup in Chinese.
echo.
py -3 ask.py --web --llm --ask-key
if errorlevel 1 goto fail
goto end

:eval
echo   Running offline evaluation (22 questions)...
echo.
py -3 ask.py --eval
if errorlevel 1 goto fail
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
