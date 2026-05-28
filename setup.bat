@echo off
REM Bootstrap a Python 3.12 virtualenv for the vagus nerve cuff pipeline.
REM
REM The required Python version is pinned in .python-version and enforced
REM here.  Set PY=... to force a specific interpreter (it must still
REM report Python 3.12.x).
REM
REM Usage (from cmd.exe, in the repo root):
REM   setup.bat
REM   set PY=C:\Python312\python.exe && setup.bat
REM   set VENV_DIR=C:\venvs\nerve && setup.bat

setlocal enabledelayedexpansion
set "REQUIRED=3.12"

if "%VENV_DIR%"=="" set "VENV_DIR=.venv"

REM --- Resolve a Python 3.12 interpreter --------------------------------------
if not "%PY%"=="" goto :check_py

REM Try py launcher with -3.12 selector
where py >nul 2>&1
if not errorlevel 1 (
    py -%REQUIRED% -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PY=py -%REQUIRED%"
        goto :check_py
    )
)

REM Try python3.12 directly
where python%REQUIRED% >nul 2>&1
if not errorlevel 1 (
    set "PY=python%REQUIRED%"
    goto :check_py
)

REM Last-resort fall-back: whatever ``python`` happens to be on PATH; the
REM version check below will reject it if it's not 3.12.
where python >nul 2>&1
if not errorlevel 1 (
    set "PY=python"
    goto :check_py
)

echo Error: Python %REQUIRED% is required but was not found on PATH.
echo Install it from:
echo   python.org:        https://www.python.org/downloads/release/python-3120/
echo   Microsoft Store:   search "Python 3.12"
echo   winget:            winget install Python.Python.3.12
echo Or point this script at an existing interpreter:
echo   set PY=C:\path\to\python3.12.exe ^&^& setup.bat
exit /b 1

:check_py
for /f "delims=" %%v in ('%PY% -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set "PY_VER=%%v"
if not "%PY_VER%"=="%REQUIRED%" (
    echo Error: %PY% reports Python %PY_VER%, need exactly %REQUIRED%.
    echo Install Python %REQUIRED% or set PY to a 3.12 interpreter.
    exit /b 1
)
echo ^>^>^> Using Python %REQUIRED% via %PY%

REM --- Create / re-use venv ---------------------------------------------------
if exist "%VENV_DIR%\Scripts\python.exe" (
    for /f "delims=" %%v in ('"%VENV_DIR%\Scripts\python.exe" -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set "EXISTING_VER=%%v"
    if not "!EXISTING_VER!"=="%REQUIRED%" (
        echo Error: Existing venv at %VENV_DIR% is Python !EXISTING_VER!, expected %REQUIRED%.
        echo Delete it and re-run, or set VENV_DIR to a different path.
        exit /b 1
    )
    echo ^>^>^> Re-using existing venv at %VENV_DIR%
) else (
    echo ^>^>^> Creating venv at %VENV_DIR%
    %PY% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Error: venv creation failed.
        exit /b 1
    )
)

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo Error: %VENV_PY% missing -- is the venv corrupt?
    exit /b 1
)

echo ^>^>^> Upgrading pip / wheel
"%VENV_PY%" -m pip install --quiet --upgrade pip wheel
if errorlevel 1 exit /b 1

echo ^>^>^> Installing requirements (this can take several minutes the first time)
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo ^>^>^> Done.  Activate this venv in new shells with:
echo       %VENV_DIR%\Scripts\activate.bat       (cmd)
echo       %VENV_DIR%\Scripts\Activate.ps1       (PowerShell)
echo.
echo ^>^>^> Smoke-check (regenerates sample data + runs pytest):
echo       "%VENV_PY%" run.py --smoke

endlocal
