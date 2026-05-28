#!/usr/bin/env bash
# Bootstrap a Python 3.12 virtualenv for the vagus nerve cuff pipeline.
#
# The required Python version is pinned in .python-version and enforced
# here so the venv is reproducible across machines.  Set PY=... to force
# a specific interpreter (it must still report Python 3.12.x).
#
# Why the venv lives outside the repo by default
# ----------------------------------------------
# When the repo sits on a non-APFS/HFS+ filesystem (exFAT external drive,
# Google Drive File Stream, OneDrive, network share, ...), macOS sprinkles
# ``._*`` AppleDouble sidecar files into every directory.  pip's package-
# metadata reader treats those sidecars as real ``METADATA`` files and
# crashes with ``UnicodeDecodeError``.  Putting the venv on the user's
# main APFS disk side-steps the problem entirely.
#
# Usage:
#   ./setup.sh                       # auto-finds python3.12, creates venv
#   PY=/path/to/python3.12 ./setup.sh
#   VENV_DIR=.venv ./setup.sh        # force the venv inside the repo
#   VENV_DIR=/abs/path ./setup.sh    # any absolute path

set -euo pipefail

REQUIRED="3.12"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_NAME="$(basename "$REPO_DIR")"
DEFAULT_VENV="$HOME/.venvs/${REPO_NAME// /-}"
VENV_DIR="${VENV_DIR:-$DEFAULT_VENV}"
PY="${PY:-}"

# --- Resolve a Python 3.12 interpreter ---------------------------------------
py_version() {
    "$1" -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo ""
}

if [[ -z "$PY" ]]; then
    for cand in "python${REQUIRED}" "python3" "python"; do
        if command -v "$cand" >/dev/null 2>&1; then
            if [[ "$(py_version "$cand")" == "$REQUIRED" ]]; then
                PY="$cand"
                break
            fi
        fi
    done
fi

if [[ -z "$PY" ]]; then
    cat >&2 <<EOF
Error: Python ${REQUIRED} is required but was not found on PATH.

Install it (one of):
  macOS (Homebrew):   brew install python@${REQUIRED}
  macOS (pyenv):      pyenv install ${REQUIRED} && pyenv local ${REQUIRED}
  Ubuntu / Debian:    sudo apt install python${REQUIRED} python${REQUIRED}-venv
  python.org:         https://www.python.org/downloads/release/python-3120/

If you already have it under a non-standard name, point this script at it:
  PY=/full/path/to/python3.12 ./setup.sh
EOF
    exit 1
fi

actual="$(py_version "$PY")"
if [[ "$actual" != "$REQUIRED" ]]; then
    echo "Error: PY=$PY reports Python $actual, need exactly $REQUIRED." >&2
    exit 1
fi

PY_FULL="$(command -v "$PY")"
echo ">>> Using Python $REQUIRED at $PY_FULL"

# --- Create / re-use venv ----------------------------------------------------
mkdir -p "$(dirname "$VENV_DIR")"

if [[ -d "$VENV_DIR" ]]; then
    existing="$(py_version "$VENV_DIR/bin/python")"
    if [[ "$existing" != "$REQUIRED" ]]; then
        echo "!!! Existing venv at $VENV_DIR is Python $existing, expected $REQUIRED." >&2
        echo "!!! Delete it and re-run, or set VENV_DIR to a different path." >&2
        exit 1
    fi
    echo ">>> Re-using existing venv at $VENV_DIR"
else
    echo ">>> Creating venv at $VENV_DIR"
    "$PY" -m venv "$VENV_DIR"
fi

# Symlink ./.venv -> $VENV_DIR so editors and ``source .venv/bin/activate``
# still work even when the real venv lives outside the repo.
if [[ "$VENV_DIR" != "$REPO_DIR/.venv" ]]; then
    ln -sfn "$VENV_DIR" "$REPO_DIR/.venv"
    echo ">>> Symlinked $REPO_DIR/.venv -> $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo ">>> Upgrading pip / wheel"
python -m pip install --quiet --upgrade pip wheel

echo ">>> Installing core requirements (this can take several minutes the first time --"
echo "    spikeinterface + PySide6 are big)"
python -m pip install -r "$REPO_DIR/requirements.txt"

echo
echo ">>> Attempting to install optional dependency: mountainsort5"
echo "    (skipping is fine -- the pipeline falls back to a deterministic"
echo "     KMeans sorter automatically if mountainsort5 isn't importable)"
if python -m pip install -r "$REPO_DIR/requirements-optional.txt"; then
    echo ">>> mountainsort5 installed -- full spike sorting available."
else
    cat <<'WARN'
!!! mountainsort5 install failed.  This is usually because the C++ dependency
    isosplit6 has no pre-built wheel for this platform / Python version.
    The pipeline will fall back to the KMeans sorter automatically -- you
    can still process recordings end-to-end.

    If you need full MountainSort5 sorting:
      macOS / Linux: try installing in a Python 3.11 or 3.10 venv where
                     isosplit6 wheels are available, or
                     conda install -c conda-forge mountainsort5
      Windows:       install Visual Studio Build Tools
                     (https://visualstudio.microsoft.com/visual-cpp-build-tools/)
                     then re-run: pip install -r requirements-optional.txt
WARN
fi

echo
echo ">>> Done.  Activate this venv in new shells with:"
echo "      source $VENV_DIR/bin/activate"
echo "    or, from the repo directory:"
echo "      source .venv/bin/activate"
echo
echo ">>> Smoke-check (regenerates sample data + runs pytest):"
echo "      python run.py --smoke"
