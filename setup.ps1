# Bootstrap a Python 3.12 virtualenv for the vagus nerve cuff pipeline.
#
# The required Python version is pinned in .python-version and enforced
# here.  Set $env:PY to force a specific interpreter (it must still
# report Python 3.12.x).
#
# Usage (from PowerShell, in the repo root):
#   .\setup.ps1                          # auto-finds Python 3.12
#   $env:VENV_DIR="C:\venvs\nerve"; .\setup.ps1
#   $env:PY="C:\Python312\python.exe"; .\setup.ps1
#
# If PowerShell refuses to run this script with "cannot be loaded because
# running scripts is disabled", any one of the following works:
#   1. Run setup.bat instead (does not need PowerShell at all).
#   2. Bypass policy for this one invocation:
#        powershell -ExecutionPolicy Bypass -File .\setup.ps1
#   3. Allow signed local scripts for the current user (one-time):
#        Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$ErrorActionPreference = "Stop"
$Required = "3.12"

function Get-PythonVersion {
    param([string]$Exe, [string[]]$Args = @())
    try {
        $out = & $Exe @Args -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        return $out.Trim()
    } catch {
        return ""
    }
}

function Resolve-Python {
    # Honor explicit override first.
    if ($env:PY) {
        $parts = $env:PY -split '\s+'
        return @{ Exe = $parts[0]; Args = @($parts | Select-Object -Skip 1) }
    }
    # Prefer the py launcher with an explicit -3.12 selector; fall back to
    # python3.12 / python3 / python and verify version.
    $candidates = @(
        @{ Exe = "py";        Args = @("-$Required") },
        @{ Exe = "python$Required"; Args = @() },
        @{ Exe = "python3";   Args = @() },
        @{ Exe = "python";    Args = @() }
    )
    foreach ($c in $candidates) {
        $cmd = Get-Command $c.Exe -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        $v = Get-PythonVersion -Exe $cmd.Path -Args $c.Args
        if ($v -eq $Required) {
            return @{ Exe = $cmd.Path; Args = $c.Args }
        }
    }
    throw @"
Python $Required is required but was not found.

Install it from one of:
  python.org:   https://www.python.org/downloads/release/python-3120/
  Microsoft Store: search "Python 3.12"
  winget:       winget install Python.Python.3.12
  Chocolatey:   choco install python --version=3.12

Or point this script at an existing interpreter:
  `$env:PY="C:\path\to\python3.12.exe"
"@
}

$py = Resolve-Python
$actual = Get-PythonVersion -Exe $py.Exe -Args $py.Args
if ($actual -ne $Required) {
    throw "Selected interpreter ($($py.Exe) $($py.Args -join ' ')) reports Python $actual, need exactly $Required."
}
Write-Host ">>> Using Python $Required at $($py.Exe) $($py.Args -join ' ')"

$VENV_DIR = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".venv" }

if (Test-Path $VENV_DIR) {
    $existingPy = Join-Path $VENV_DIR "Scripts\python.exe"
    if (Test-Path $existingPy) {
        $existing = Get-PythonVersion -Exe $existingPy
        if ($existing -ne $Required) {
            throw "Existing venv at $VENV_DIR is Python $existing, expected $Required.  Delete it and re-run, or set `$env:VENV_DIR to a different path."
        }
    }
    Write-Host ">>> Re-using existing venv at $VENV_DIR"
} else {
    Write-Host ">>> Creating venv at $VENV_DIR"
    $createArgs = @($py.Args) + @("-m", "venv", $VENV_DIR)
    & $py.Exe @createArgs
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed (exit $LASTEXITCODE)" }
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
