# Smart, re-runnable Windows setup wrapper.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#   powershell -ExecutionPolicy Bypass -File setup.ps1 --cpu --no-verify
#
# Creates/reuses .venv, installs only missing/outdated packages, activates the
# venv in this session and runs the project status report at the end.
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) {
    $Python = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $Python) {
    Write-Error "No Python interpreter found. Install Python 3.10+ and re-run."
    exit 1
}

$SetupStatus = 0
Write-Host "[setup] Project: $Root"
Write-Host "[setup] Interpreter: $(& $Python.Source --version 2>&1)"

try {
    & $Python.Source "$Root\setup_venv.py" @args
    if ($LASTEXITCODE -ne 0) {
        $SetupStatus = $LASTEXITCODE
        Write-Host ""
        Write-Host "[setup] Setup reported an error (exit $SetupStatus). The venv was left intact."
        Write-Host "[setup] Re-run with --check to inspect packages, or --reinstall to repair PyTorch."
    }
} catch {
    $SetupStatus = 1
    Write-Host "[setup] Unexpected failure: $($_.Exception.Message)" -ForegroundColor Yellow
}

$Activate = Join-Path $Root ".venv\Scripts\Activate.ps1"
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $Activate) {
    . $Activate
    Write-Host "[setup] Activated virtual environment: $env:VIRTUAL_ENV"
} else {
    Write-Warning "No .venv activation script found; status will use the system Python."
}

$StatusScript = Join-Path $Root "scripts\status_report.py"
if (Test-Path $StatusScript) {
    if (Test-Path $VenvPython) {
        & $VenvPython $StatusScript
    } else {
        & $Python.Source $StatusScript
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Status report could not complete."
    }
} else {
    Write-Warning "Status report script missing: $StatusScript"
}

if ($SetupStatus -ne 0) {
    exit $SetupStatus
}
Write-Host ""
Write-Host "[setup] Next steps:"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "    python scripts\status_report.py"
Write-Host "    python -m src.verify"
exit 0
