# Quarter-data ablation sweep (Windows PowerShell). Same seed, distinct --run-tag.
#
#   powershell -File scripts\run_ablations.ps1
#   powershell -File scripts\run_ablations.ps1 -DryRun
#
# Compare afterwards:
#   python scripts/view_logs.py list
#   python scripts/diagnose_model.py --checkpoint <pt> --split test
#
# NOTE: the hard-pair *batch sampler* is not yet implemented; --hard-pairs only
# up-weights SupCon negatives.
param([switch]$DryRun)

Set-Location (Join-Path $PSScriptRoot "..")

$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

function Run-Ablation {
    param([string]$Tag, [string[]]$Flags)
    Write-Host ("=" * 67)
    Write-Host "  ABLATION $Tag : $($Flags -join ' ')"
    Write-Host ("=" * 67)
    if ($DryRun) {
        Write-Host "$py local_train.py --quarter-data --run-tag $Tag $($Flags -join ' ')"
    } else {
        & $py local_train.py --quarter-data --run-tag $Tag @Flags
    }
}

Run-Ablation R0 @("--no-augment")
Run-Ablation R1 @("--augment-preset", "light")
Run-Ablation R2 @("--augment-preset", "light", "--cosine-head")
Run-Ablation R3 @("--augment-preset", "light", "--cosine-head", "--trait-weight", "0.1")
Run-Ablation R4 @("--augment-preset", "light", "--cosine-head", "--breed-aug", "--trait-weight", "0.1")
Run-Ablation R5 @("--augment-preset", "light", "--cosine-head", "--breed-aug", "--trait-weight", "0.1",
                  "--hard-pairs", "outputs/metrics/confusion_pairs.json")
Run-Ablation R6 @("--augment-preset", "light", "--cosine-head", "--breed-aug", "--trait-weight", "0.1",
                  "--hard-pairs", "outputs/metrics/confusion_pairs.json", "--pad-to-square")
Run-Ablation R7 @("--augment-preset", "light", "--cosine-head", "--breed-aug", "--trait-weight", "0.1",
                  "--hard-pairs", "outputs/metrics/confusion_pairs.json", "--dedup-splits")

Write-Host ""
Write-Host "Done. Compare runs:  python scripts/view_logs.py list"
