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

### Windows (PowerShell)

```powershell
git clone https://github.com/AndreaEBiju/nerve-processing.git
cd nerve-processing
.\setup.ps1                         # creates .venv\ inside the repo
.\.venv\Scripts\Activate.ps1        # in any new shell
python run.py --smoke
```

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
