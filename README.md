# Vagus Nerve Cuff Recording Pipeline

Batch-processing tool that takes notch-filtered, motion-blanked vagal cuff
recordings (one channel per cuff), R-peak times, and an optional slow-wave
channel, runs a 14-step analysis pipeline, and writes one `.mat` metrics file
per recording pair next to its source files.

The scientific rationale for each step lives in the companion document
`vagus_nerve_cuff_recording_pipeline.docx`, Section 3. This README covers
only **how to use** the implementation; the build contract is `IMPLEMENTATION.md`.

## Installation

The pipeline depends on a fairly heavy scientific stack (numpy, scipy, sklearn,
spikeinterface + mountainsort5, PySide6, umap, hdbscan, etc.). Keep it isolated
from your system Python in a virtualenv.

**Python version is pinned to 3.12** (see `.python-version`). The setup
scripts find Python 3.12 on your system and create the venv from it — they
do **not** touch any global Python; your system Python keeps whatever
version it already had. If 3.12 isn't installed, the scripts error out with
install instructions instead of silently using the wrong version.

### Slow-wave input: continuous trace OR pre-detected peak times

Step 11 (slow-wave phase tagging) accepts two input shapes:

1. **Continuous trace** -- a 1-D numeric array sampled at any rate.
   Linearly resampled to the neural rate before Hilbert phase
   extraction.  Typical variable name: ``slowWaves.trace`` /
   ``slow_wave`` / ``LFP`` / ``filtered``.
2. **Cell array of peak sample indices** -- e.g.
   ``slowWavePeakLocs`` as a 1xK MATLAB cell where each cell holds
   the sample numbers of detected slow-wave peaks from one detector
   pass.  All cells are concatenated, sorted, and a synthetic
   ``sin(2*pi * (t - prev_peak) / (next_peak - prev_peak))`` trace is
   reconstructed at the neural sampling rate so the rest of Step 11
   (MRL, Rayleigh p, phase histograms) sees a clean sinusoid with
   true phase = 0 at every detected peak.  Mathematically identical
   to running Hilbert on a real sinusoid through the same peaks.

**Multi-channel slow-wave traces**: when the continuous-trace variable
is a 2-D matrix stacking several slow-wave channels (e.g. a 3xN array
of three slow-wave signals), pick which one to use with the
**slowwave channel idx** spinbox in the GUI -- 0-based, default 0.
Headless: ``--slowwave-channel 1`` (selects the second channel).
The pipeline orients the matrix automatically (time on the longer
axis) and slices the requested column.  Out-of-range indices error
out with a clear "channel N is out of range; set in [0..K-1]" message.

The autopop recognises both: it scores variable names containing
``peakloc`` / ``peaktimes`` / ``peaks_idx`` / ``sw_peaks`` as the
peak-times path (regardless of cell-array packaging), then falls back
to ``slow`` / ``wave`` / ``lfp`` / ``trace`` / ``filtered`` for the
continuous-trace path.  Either way the rest of the pipeline is unchanged.

### Excluding specific pairs after discovery

Discovery often picks up a few pairs you don't want to process this run
(a known-bad recording, a duplicate, a session you've already processed).
You don't have to rename or move the files -- just deselect them in the
GUI.

After **Discover pairs** fills the table, every row has an
**Include** checkbox in the first column (default on).  Three ways to
exclude rows:

1. **Click the checkbox** in the Include column to uncheck a single row.
2. **Select rows + right-click** → "Exclude selected rows" /
   "Include selected rows".  Standard table selection rules
   (Shift-click, Ctrl/Cmd-click) apply.
3. **Buttons below the table**: Exclude selected / Include selected /
   Exclude all / Include all.

When you click **Run batch**, only the rows whose Include box is still
checked are passed to the pipeline.  The log shows
``Running on N of M discovered pair(s) (the rest are excluded).``  If
every row is excluded, the run is blocked with a "Nothing to run"
dialog rather than failing silently.

Headless equivalent: there's no direct flag for excluding particular
pairs from the CLI -- if you need that, point ``--root`` at a
sub-directory that contains only the pairs you want, or move the
unwanted files to a sibling folder for the duration of the run.  The
GUI is the better tool for ad-hoc exclusion.

### Multi-channel acquisitions: picking which channels are cuffs

When the `data` variable inside your blanked `.mat` is a multi-channel
array (e.g. 5 channels: 2 nerve cuffs + 3 unrelated signals), tell the
pipeline which **0-based indices** to treat as cuffs.

In the GUI, fill in the **"cuff channel indices"** field (next to the
variable-mapping dropdowns) with a comma-separated list:

```
0,3            # use the 1st and 4th channels as cuffs, ignore the rest
0,2,4          # use the 1st, 3rd, and 5th
                # (leave blank to use ALL channels in the array)
```

Headless equivalent:

```bash
python run.py --no-ui --root ... --neural data --rpeak rpeak_samples \
    --channels 0,3
```

Validation: the pipeline rejects out-of-range indices with a clear error
(e.g. `channel_indices=[0,7] out of range for 5-channel array`), and
logs a warning if your neural array has 3+ channels but you didn't
specify which to use -- that's almost always a sign that some are not
cuff recordings.

The chosen channels are run independently through Steps 1-14 (no merging
across channels, per the spec).  In the resulting `.mat`, each one
appears as a separate `metrics.cuff(k)` entry.

### Split-machine workflow: prepass on Windows, MountainSort5 on Mac

The 14-step pipeline can be split across two machines so that the
cluster-heavy MountainSort5 step runs on a Mac/Linux box that has the
sorter installed, while a Windows box does the bulk preprocessing.

The split point is at Step 5/6.  Steps 1-5 are deterministic
preprocessing (bandpass, sigma, detection, waveforms, PCA features).
Step 6 is the MountainSort5 sort.  Steps 7-14 are downstream metrics
that need the sort labels.

```
Windows (prepass)                Google Drive                 Mac (resume)
-----------------                ------------                 ------------
discover pairs                                                discover *_checkpoint.npz
fit global PCA basis  ----->     batch_pca_basis.npz   ----> load PCA basis
per pair:                        <stem>_checkpoint.npz ----> per checkpoint:
  Steps 1-5                                                    Steps 6-14
  save checkpoint                                              save <stem>_metrics.mat
```

CLI:

```bash
# Step 1: on Windows, run prepass
python run.py --no-ui --mode prepass --root "C:\path\to\batch" \
    --neural data --rpeak rpeak_samples --slowwave slow_wave

# Step 2: copy <stem>_checkpoint.npz files (one at a time if your Mac
# storage is limited) plus batch_pca_basis.npz to the Mac

# Step 3: on Mac, run resume (no --neural / --rpeak needed -- the
# checkpoint already has the data)
python run.py --no-ui --mode resume --root /path/to/local/copies
```

In the GUI, pick the run mode from the dropdown next to the **Run batch**
button (`full` / `prepass` / `resume`).

Output identity: a prepass-then-resume run produces the same cluster
labels and the same per-cluster SNR as a single full-mode run, because
the global PCA basis is shared and the sort is deterministic.  This is
asserted by `tests/test_end_to_end.py::test_prepass_then_resume_matches_full`.

Checkpoint file size is dominated by the per-cuff filtered signal
(needed by MountainSort5).  Approximately:

| Recording length | Per-cuff checkpoint | Per-pair (2 cuffs) |
| --- | --- | --- |
| 2 min | ~5 MB | ~10 MB |
| 30 min | ~80-150 MB | ~150-300 MB |
| 60 min | ~150-300 MB | ~300-600 MB |

You can therefore select **one checkpoint at a time** from Google Drive
to bring across to your Mac, run resume on just that file, then move to
the next one -- never having to hold the full batch on the Mac at once.

### Optional dependency: MountainSort5

`mountainsort5` is the canonical sorter but its C++ dependency `isosplit6`
does not currently ship a pre-built wheel for Windows + Python 3.12.  The
setup scripts try to install it but **never fail the build if they can't**:
if `mountainsort5` isn't importable, the pipeline transparently falls back
to a deterministic KMeans + silhouette sorter and records `kmeans_fallback`
in `metrics.provenance.sorter`.  All other steps (quality, audit, cardiac,
respiration, slow-wave, responder, fibre-type, `.mat` output) work
identically.

To force MountainSort5 on Windows, install
[Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
(the "Desktop development with C++" workload) and then:

```cmd
.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

On macOS / Linux it usually just works — Linux/macOS wheels for `isosplit6`
exist on PyPI.

### Python 3.12 install

| OS | One-liner to install Python 3.12 (only if missing) |
| --- | --- |
| macOS | `brew install python@3.12` (or `pyenv install 3.12 && pyenv local 3.12`) |
| Windows | `winget install Python.Python.3.12` (or [python.org installer](https://www.python.org/downloads/release/python-3120/)) |
| Ubuntu / Debian | `sudo apt install python3.12 python3.12-venv` |

### macOS / Linux

```bash
git clone https://github.com/AndreaEBiju/nerve-processing.git
cd nerve-processing
./setup.sh                          # creates ~/.venvs/nerve-processing + .venv symlink
source .venv/bin/activate           # in any new shell
python run.py --smoke               # rebuilds sample data + runs pytest (~1 min)
```

`setup.sh` puts the venv outside the repo by default (at `~/.venvs/nerve-processing`)
and drops a symlink at `./.venv` so the usual `source .venv/bin/activate` muscle
memory still works. The reason for the indirection: when the repo lives on
exFAT, OneDrive, Google Drive File Stream, or a network share, macOS scatters
`._*` AppleDouble sidecar files into every directory — and pip's package
metadata reader chokes on them. Putting the venv on the user's main APFS disk
side-steps the problem.

Override the location if you want the venv inside the repo (only safe on
APFS / ext4 / NTFS):

```bash
VENV_DIR=.venv ./setup.sh
```

### Windows

Two setup entry points — pick whichever shell you use.

**Option 1: cmd.exe** (works on every Windows install, no execution-policy fiddling):

```cmd
git clone https://github.com/AndreaEBiju/nerve-processing.git
cd nerve-processing
setup.bat
.venv\Scripts\activate.bat
python run.py --smoke
```

**Option 2: PowerShell:**

```powershell
git clone https://github.com/AndreaEBiju/nerve-processing.git
cd nerve-processing
.\setup.ps1
.\.venv\Scripts\Activate.ps1
python run.py --smoke
```

If PowerShell rejects `setup.ps1` with *"running scripts is disabled on this system"*, you have three options:

| | |
| --- | --- |
| **Easiest** | Run `setup.bat` instead — it uses cmd.exe and never touches policy. |
| One-off bypass | `powershell -ExecutionPolicy Bypass -File .\setup.ps1` |
| Permanent (current user) | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (one time) |

The setup scripts auto-detect the Python interpreter — they try `py -3` first (the
launcher that ships with python.org installers), then `python`, then `python3`.
Override with `set PY=py -3.11` (cmd) or `$env:PY="py -3.11"` (PowerShell) before
running setup if you have multiple Pythons installed.

If your repo lives on OneDrive with **Files On-Demand** enabled and you see
metadata errors during install or the pipeline stalls on file reads, mark the
folder for offline availability via the OneDrive sync client (or move the venv
outside the synced folder: `set VENV_DIR=C:\venvs\nerve && setup.bat`).

### Manual install (any OS)

If you'd rather drive the venv yourself:

```bash
python3 -m venv .venv
source .venv/bin/activate           # (or .venv\Scripts\Activate.ps1 on Windows)
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
```

Key dependencies (full list in `requirements.txt`):

- `numpy`, `scipy`, `scikit-learn`
- `umap-learn`, `hdbscan`
- `spikeinterface` + `mountainsort5` (KMeans fallback if absent)
- `pymatreader` (robust `.mat` loading, including v7.3 / HDF5)
- `pingouin` (circular stats; falls back to `exp(-Z)` Rayleigh approximation)
- `PySide6` (GUI)

## Quick start

Activate the venv first (`source .venv/bin/activate` on macOS/Linux,
`.\.venv\Scripts\Activate.ps1` on Windows).

### GUI

```bash
python run.py
```

1. Pick the batch root folder.
2. Confirm/edit the blanked / R-peak / slow-wave filename patterns (defaults
   `*blank*`, `*HR*`, `*slow*`).
3. **Discover pairs** — review the table; any `ambiguous` rows are highlighted.
4. **Introspect variables** — autopopulates the variable mapping from a
   representative file. Editable per role.
5. Review the key config fields.
6. **Run batch** — progress bar + live log; on completion a dialog points at
   the summary CSV.

### Headless

```bash
python run.py --no-ui \
    --root /path/to/batch_root \
    --neural data \
    --rpeak rpeak_samples \
    --units sample \
    --slowwave slow_wave \
    --fs-var fs \
    --threshold-sigma 4.5 \
    --n-pca 8
```

All `PipelineConfig` defaults are exposed as `--<param-name>` flags. See
`python run.py --help` for the full list.

### Filename signature rule (which files are picked up)

Discovery is intentionally strict so stray copies, exports without the
pipeline-version stamp, or older lab files do not get processed by accident.
By default a file is only considered if its basename contains the version
tag matched by

```
_v\d+\.\d+\.\d+_
```

so e.g. `rat01_v0.1.0_blankmotion.mat` and `rat01_v0.1.0_recovery_HRBR.mat`
are eligible, while `rat01_blankmotion.mat` and `rat01_HRBR.mat` (no tag)
are silently skipped.

Within each directory, eligible files are paired by stripping the configured
**pair token** from each stem and matching the remainder:

| token | matches files like |
| --- | --- |
| `blankmotion` (blanked) | `<prefix>_v0.1.0_blankmotion.mat`, `<prefix>_v0.1.0_recovery_blankmotion.mat` |
| `HRBR` (R-peak) | `<prefix>_v0.1.0_HRBR.mat`, `<prefix>_v0.1.0_recovery_HRBR.mat` |

That keeps a regular pair and its `_recovery_` counterpart from getting
crossed up. All three knobs are configurable both in the GUI ("Required
filename regex", "Blanked pair token", "R-peak pair token") and on the CLI:

```bash
python run.py --no-ui --root ... \
    --required-regex '_v\d+\.\d+\.\d+_' \
    --blanked-token blankmotion \
    --rpeak-token HRBR
```

Pass `--required-regex ''` to disable the filter entirely (e.g. when running
against legacy data without the version stamp).

### Smoke test

```bash
python run.py --smoke
```

Regenerates the synthetic dataset under `tests/data/` and runs `pytest -q
tests/test_end_to_end.py`.

## Pipeline overview

| Step | Module | Description |
| --- | --- | --- |
| 1 | `preprocess.bandpass` | Butterworth bandpass (zero-phase filtfilt) |
| 2 | `preprocess.noise_sigma` | Sliding MAD-based σ track |
| 3 | `detect.detect_spikes` | σ-threshold crossings, polarity-aware, refractory pruning |
| 4 | `detect.extract_waveforms` | ±wf_pre/wf_post-ms windows |
| 5 | `features.scalar_features` + `features.project_pca` | Scalar features + projection on global PCA basis |
| 6 | `sort.run_mountainsort` | MountainSort5 via SpikeInterface (KMeans fallback) |
| 7 | `quality.cluster_metrics` | SNR, firing rate, mean/std waveform, **full ISIs**, violations |
| 8 | `audit.umap_hdbscan_audit` | UMAP + HDBSCAN audit, ARI, agreement matrix |
| 9 | `cardiac.peri_rwave` | PRWH + cardiac-locked flag + cleaned spike list |
| 10 | `respiration.verify` | Burst-rate vs RSA breathing rate, stability across stim |
| 11 | `slowwave.phase_tag` | SW-band phase per spike, MRL, Rayleigh p |
| 12 | `rates.firing_rates` | Per-cluster rate traces + correlations with resp/HR/SW |
| 13 | `responder.detect` | Pre/post-stim percentile responder test |
| 14 | `fibertype.tag` | A-like / C-like / ambiguous from trough-peak duration |

### Two-pass batch

`batch.run_batch` runs two passes for cross-condition comparability:

1. **Pass 1 (basis pass)** — Steps 1–4 on every recording, pooled waveforms,
   subsampled to `pca_pool_max_spikes`, then a single PCA basis is fit and
   saved to `batch_pca_basis.npz` at the batch root.
2. **Pass 2 (analysis pass)** — every recording is projected onto that same
   basis and Steps 5(project)–14 run.

If `run_batch` is given a single-pair batch the basis is fit on that pair's
own waveforms (a warning is logged).

## Output `.mat` schema

One `<common_stem>_metrics.mat` per recording pair, written next to the source
files. Top-level struct `metrics` with the fields described in
`IMPLEMENTATION.md` §8 (provenance, `fs`, `n_cuffs`, and a `cuff` struct
array). Re-load with `pymatreader.read_mat` (or `scipy.io.loadmat` with
`squeeze_me=True`).

Per-cluster ISI arrays are saved in full (per spec) so downstream analysis can
look at distributions, not just summary stats.

Provenance includes: software version, ISO timestamp, source file paths, the
resolved `var_map`, the full `PipelineConfig`, the seed, and the path to the
saved PCA basis.

At the batch root you also get:

- `batch_pca_basis.npz` — global PCA basis (components, mean, scale, n_pca)
- `batch_summary.csv` — one row per pair: n_cuffs, n_spikes, n_clusters,
  mean SNR, n_responders, output path, status, reason
- `batch_varmap.json` — the resolved variable mapping (for re-runs)

## Determinism

Every stochastic stage (UMAP, HDBSCAN, MountainSort, PCA subsampling,
KMeans fallback) is seeded from `PipelineConfig.seed`. Two runs of
`run_batch` with the same config produce identical labels and identical MRL
values; this is asserted by `tests/test_end_to_end.py::test_determinism`.

## Not in v1 (documented stubs)

- `detect.template_recovery` — Track-B template-matching recovery pass
- Conduction-velocity / cross-cuff velocity analysis (cuffs on separate branches)
- Pharmacological-validation hooks
- Cross-cuff coincidence QC

Each stub raises `NotImplementedError` and carries a docstring with the
trigger condition for implementing it, referencing the relevant docx section.
