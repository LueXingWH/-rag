@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ============================================================
echo   Demo Check - run the 10 nanhai demo cases, verify all pass
echo ============================================================
echo.

py -3 demo_check.py

echo.
pause
