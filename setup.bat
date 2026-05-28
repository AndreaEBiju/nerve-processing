@echo off
REM Bootstrap a virtualenv for the vagus nerve cuff pipeline (Windows cmd.exe).
REM
REM Usage (from cmd.exe, in the repo root):
REM   setup.bat
REM   set PY=py -3.11 && setup.bat
REM   set VENV_DIR=C:\venvs\nerve && setup.bat
REM
REM Unlike setup.ps1, this script does not need PowerShell at all and does
REM not touch PowerShell's execution policy.  Use this one if PowerShell
REM complains about "running scripts is disabled on this system".

setlocal

if "%VENV_DIR%"=="" set "VENV_DIR=.venv"

REM Pick an interpreter: honor %PY% if already set, else py launcher, else python, else python3.
if not "%PY%"=="" goto :have_py

where py >nul 2>&1
if not errorlevel 1 (
    set "PY=py -3"
    goto :have_py
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PY=python"
    goto :have_py
)

where python3 >nul 2>&1
if not errorlevel 1 (
    set "PY=python3"
    goto :have_py
)

echo Error: Could not find a Python interpreter on PATH (tried py, python, python3).
echo Install Python 3.10+ from https://www.python.org/downloads/windows/ and re-run.
exit /b 1

:have_py

if not exist "%VENV_DIR%" (
    echo ^>^>^> Creating venv at %VENV_DIR% using %PY%
    %PY% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Error: venv creation failed.
        exit /b 1
    )
) else (
    echo ^>^>^> Re-using existing venv at %VENV_DIR%
)

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo Error: %VENV_PY% missing -- is the venv corrupt?
    exit /b 1
)

echo ^>^>^> Upgrading pip / wheel
"%VENV_PY%" -m pip install --quiet --upgrade pip wheel
if errorlevel 1 (
    echo Error: pip upgrade failed.
    exit /b 1
)

echo ^>^>^> Installing requirements (this can take several minutes the first time)
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Error: requirements install failed.
    exit /b 1
)

echo.
echo ^>^>^> Done.  Activate this venv in new shells with:
echo       %VENV_DIR%\Scripts\activate.bat       (cmd)
echo       %VENV_DIR%\Scripts\Activate.ps1       (PowerShell)
echo.
echo ^>^>^> Smoke-check (regenerates sample data + runs pytest):
echo       "%VENV_PY%" run.py --smoke

endlocal
