@echo off
setlocal
set "ROOT=%~dp0"
where powershell >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%setup.ps1" %*
    exit /b %ERRORLEVEL%
)

echo [setup] PowerShell not found; using setup_venv.py directly.
python "%ROOT%setup_venv.py" %*
set "SETUP_STATUS=%ERRORLEVEL%"
if exist "%ROOT%.venv\Scripts\activate.bat" (
    call "%ROOT%.venv\Scripts\activate.bat"
    python "%ROOT%scripts\status_report.py"
) else (
    python "%ROOT%scripts\status_report.py"
)
if not "%SETUP_STATUS%"=="0" exit /b %SETUP_STATUS%
exit /b 0
