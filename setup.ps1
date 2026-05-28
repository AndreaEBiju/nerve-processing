# Bootstrap a virtualenv for the vagus nerve cuff pipeline (Windows).
#
# Usage (from PowerShell, in the repo root):
#   .\setup.ps1                       # uses .venv inside the repo
#   $env:PY="py -3.11"; .\setup.ps1   # override interpreter
#
# If your repo sits on OneDrive / a network share and you see pip metadata
# errors, set $env:VENV_DIR to a path on your local C: drive first.

$ErrorActionPreference = "Stop"

$PY = if ($env:PY) { $env:PY } else { "python" }
$VENV_DIR = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".venv" }

if (-not (Test-Path $VENV_DIR)) {
    Write-Host ">>> Creating venv at $VENV_DIR (using $PY)"
    & $PY -m venv $VENV_DIR
} else {
    Write-Host ">>> Re-using existing venv at $VENV_DIR"
}

& "$VENV_DIR\Scripts\Activate.ps1"

Write-Host ">>> Upgrading pip / wheel"
python -m pip install --quiet --upgrade pip wheel

Write-Host ">>> Installing requirements (this can take several minutes)"
python -m pip install -r requirements.txt

Write-Host ""
Write-Host ">>> Done.  Activate this venv in new shells with:"
Write-Host "      .\$VENV_DIR\Scripts\Activate.ps1"
Write-Host ""
Write-Host ">>> Smoke-check (regenerates sample data + runs pytest):"
Write-Host "      python run.py --smoke"
