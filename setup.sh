#!/usr/bin/env bash
# Bootstrap a virtualenv for the vagus nerve cuff pipeline.
#
# Why the venv lives outside the repo by default
# ----------------------------------------------
# When the repo sits on a non-APFS/HFS+ filesystem (exFAT external drive,
# Google Drive File Stream, OneDrive, network share, …), macOS sprinkles
# ``._*`` AppleDouble sidecar files into every directory.  pip's package-
# metadata reader treats those sidecars as real ``METADATA`` files and
# crashes with ``UnicodeDecodeError``.  Putting the venv on the user's
# main APFS disk side-steps the problem entirely.
#
# Usage:
#   ./setup.sh                       # creates ~/.venvs/<repo>, symlinks .venv -> there
#   VENV_DIR=.venv ./setup.sh        # force the venv inside the repo (APFS/ext4 only)
#   VENV_DIR=/abs/path ./setup.sh    # any absolute path
#   PY=python3.11 ./setup.sh         # override the interpreter
#
# Re-running is safe: pip reuses the existing venv and only installs what's
# missing / outdated.

set -euo pipefail

PY="${PY:-python3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_NAME="$(basename "$REPO_DIR")"
DEFAULT_VENV="$HOME/.venvs/${REPO_NAME// /-}"
VENV_DIR="${VENV_DIR:-$DEFAULT_VENV}"

mkdir -p "$(dirname "$VENV_DIR")"

if [[ ! -d "$VENV_DIR" ]]; then
    echo ">>> Creating venv at $VENV_DIR (using $PY)"
    "$PY" -m venv "$VENV_DIR"
else
    echo ">>> Re-using existing venv at $VENV_DIR"
fi

# Make ./.venv inside the repo point at the real venv so editors and the
# usual ``source .venv/bin/activate`` muscle memory still work.  Skip if the
# user asked for an in-repo venv to begin with.
if [[ "$VENV_DIR" != "$REPO_DIR/.venv" ]]; then
    ln -sfn "$VENV_DIR" "$REPO_DIR/.venv"
    echo ">>> Symlinked $REPO_DIR/.venv -> $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo ">>> Upgrading pip / wheel"
python -m pip install --quiet --upgrade pip wheel

echo ">>> Installing requirements (this can take several minutes the first time —"
echo "    spikeinterface + mountainsort5 + PySide6 are big)"
python -m pip install -r "$REPO_DIR/requirements.txt"

echo
echo ">>> Done.  Activate this venv in new shells with:"
echo "      source $VENV_DIR/bin/activate"
echo "    or, from the repo directory:"
echo "      source .venv/bin/activate"
echo
echo ">>> Smoke-check (regenerates sample data + runs pytest):"
echo "      python run.py --smoke"
