@echo off
rem Smart, re-runnable Windows setup entry point.
rem
rem Prefers setup.ps1 (which runs setup_venv.py, installs the Python + Node
rem dependencies and syncs the Capacitor Android project). If PowerShell is
rem unavailable it falls back to setup_venv.py directly.
rem
rem Usage:
rem   setup.bat                     Python + frontend + Capacitor sync
rem   setup.bat --cpu --no-verify   CPU PyTorch, skip model verification
rem   setup.bat --no-frontend       Python only
rem   setup.bat --no-android        frontend deps, no Capacitor sync
rem   setup.bat --android-build     also compile the debug APK
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
