# Bootstrap a virtualenv for the vagus nerve cuff pipeline (Windows PowerShell).
#
# Usage (from PowerShell, in the repo root):
#   .\setup.ps1                          # uses .venv inside the repo, autodetects python
#   $env:VENV_DIR="C:\venvs\nerve"; .\setup.ps1
#   $env:PY="py -3.11"; .\setup.ps1      # force a specific interpreter
#
# If PowerShell refuses to run this script with "cannot be loaded because
# running scripts is disabled", you have three options (any one works):
#   1. Run setup.bat instead (does not need PowerShell at all).
#   2. Bypass policy for this one invocation:
#        powershell -ExecutionPolicy Bypass -File .\setup.ps1
#   3. Allow signed local scripts for the current user (one-time):
#        Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$ErrorActionPreference = "Stop"

function Resolve-Python {
    # Honor explicit override first.  Split on whitespace so "py -3.11" works.
    if ($env:PY) {
        $parts = $env:PY -split '\s+'
        return @{ Exe = $parts[0]; Args = @($parts | Select-Object -Skip 1) }
    }
    # Prefer the py launcher (ships with the python.org installer); fall
    # back to whichever ``python`` happens to be on PATH.
    foreach ($cand in @("py", "python", "python3")) {
        $cmd = Get-Command $cand -ErrorAction SilentlyContinue
        if ($cmd) {
            $extra = if ($cand -eq "py") { @("-3") } else { @() }
            return @{ Exe = $cmd.Path; Args = $extra }
        }
    }
    throw "Could not find a Python interpreter on PATH (tried py, python, python3). Install Python 3.10+ from https://www.python.org/downloads/windows/ and re-run."
}

$py = Resolve-Python
$VENV_DIR = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".venv" }

if (-not (Test-Path $VENV_DIR)) {
    Write-Host ">>> Creating venv at $VENV_DIR (using $($py.Exe) $($py.Args -join ' '))"
    $createArgs = @($py.Args) + @("-m", "venv", $VENV_DIR)
    & $py.Exe @createArgs
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed (exit $LASTEXITCODE)" }
} else {
    Write-Host ">>> Re-using existing venv at $VENV_DIR"
}

# Use the venv's python directly rather than activating it — avoids the
# Activate.ps1 execution-policy trap on locked-down machines.
$venvPython = Join-Path $VENV_DIR "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Venv was created but $venvPython is missing — is the venv corrupt?"
}

Write-Host ">>> Upgrading pip / wheel"
& $venvPython -m pip install --quiet --upgrade pip wheel
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit $LASTEXITCODE)" }

Write-Host ">>> Installing requirements (this can take several minutes the first time)"
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "requirements install failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host ">>> Done.  Activate this venv in new shells with:"
Write-Host "      $VENV_DIR\Scripts\Activate.ps1     # PowerShell"
Write-Host "      $VENV_DIR\Scripts\activate.bat     # cmd.exe"
Write-Host ""
Write-Host ">>> Smoke-check (regenerates sample data + runs pytest):"
Write-Host "      $venvPython run.py --smoke"
