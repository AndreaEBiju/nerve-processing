# Vagus Nerve Cuff Recording — Pipeline Implementation Spec

**Audience:** Claude Code (autonomous build agent)
**Goal:** Build a batch-processing tool that takes notch-filtered, motion-blanked vagal cuff recordings (one channel per cuff) plus extracted R-peak times and a slow-wave channel, runs the 14-step analysis pipeline, and writes one `.mat` metrics file per recording pair into the folder that contained the source files. Include a UI for folder selection, recursive file discovery, file-pairing, and user assignment of variable names (with autopopulation). Ship with an end-to-end test on one sample dataset.

This spec is the build contract. The scientific rationale for every step lives in the companion `vagus_nerve_cuff_recording_pipeline.docx`, Section 3. Do not re-derive the science; implement what is written here.

---

## 0. Ground rules

- **Language:** Python 3.10+.
- **Core libraries:** `numpy`, `scipy`, `spikeinterface` (with `mountainsort5`), `scikit-learn`, `umap-learn`, `hdbscan`, `astropy` or `pingouin` or `scipy.stats` for circular stats, `matplotlib`, `scipy.io` (for `.mat` I/O), `pymatreader` (robust `.mat` loading incl. v7.3/HDF5).
- **UI:** A small desktop GUI. Use `PySide6` (preferred) or `tkinter` (fallback, zero extra deps). Keep UI logic separate from pipeline logic so the pipeline is callable headless.
- **Determinism:** Set and record all random seeds (UMAP, HDBSCAN, MountainSort, any subsampling). The same input must give the same output.
- **No silent failures:** Every stage validates its inputs and raises with an actionable message. Log to both console and a per-run log file.
- **Sampling rate:** 24414.0625 Hz (TDT default). Read from the file if present; otherwise accept as a config parameter defaulting to this value.
- **Per-cuff:** The pipeline runs independently per channel (cuff). A "recording" may contain one or two neural channels. Treat each neural channel as an independent unit of analysis; never merge clusters across channels.

---

## 1. Repository layout

```
vagus_pipeline/
  __init__.py
  config.py            # dataclasses for all parameters + defaults
  io_discovery.py      # recursive file finding, pairing, variable-name introspection
  io_load.py           # load blanked data, R-peaks, slow-wave; normalize to arrays
  preprocess.py        # Step 1 bandpass, Step 2 noise estimate
  detect.py            # Step 3 detection, Step 4 waveform extraction/alignment
  features.py          # Step 5 scalar features + global PCA basis fit/project
  sort.py              # Step 6 MountainSort via SpikeInterface
  quality.py           # Step 7 cluster QC metrics (incl. ISI distributions, saved)
  audit.py             # Step 8 UMAP + HDBSCAN audit + agreement stats
  cardiac.py           # Step 9 peri-R-wave histogram, cardiac-locked handling
  respiration.py       # Step 10 respiratory verification + cross-condition check
  slowwave.py          # Step 11 slow-wave phase tagging + circular stats
  rates.py             # Step 12 firing-rate series + physiological correlations
  responder.py         # Step 13 responder detection (Masi/Zanos criterion)
  fibertype.py         # Step 14 descriptive fibre-type tagging
  assemble.py          # collect all metrics -> single .mat per pair
  pipeline.py          # orchestrates Steps 1-14 for one recording pair
  batch.py             # iterate over discovered pairs, call pipeline, save outputs
  ui.py                # GUI: folder picker, file table, var-name assignment, run button
  logging_setup.py
tests/
  test_end_to_end.py   # runs full pipeline on sample dataset, asserts outputs
  make_sample_data.py  # synthesizes a labeled sample dataset (see Section 9)
  data/                # generated sample dataset lands here
run.py                 # entry point: launches UI, or headless batch with --no-ui
requirements.txt
README.md
```

---

## 2. Configuration (`config.py`)

Use a dataclass `PipelineConfig` with these fields and defaults. Every parameter that affects results must live here and be persisted into the output (Section 8, provenance).

```python
fs: float = 24414.0625            # Hz
# Step 1 bandpass
bp_low_hz: float = 100.0
bp_high_hz: float = 5000.0
bp_order: int = 4                 # Butterworth, zero-phase (filtfilt)
# Step 2 noise estimate
noise_window_s: float = 5.0       # sliding window for sigma
# Step 3 detection
threshold_sigma: float = 4.5      # 4-5 sigma
refractory_ms: float = 1.0
detect_polarity: str = "neg"      # align on negative peak
# Step 4 waveform window
wf_pre_ms: float = 1.0
wf_post_ms: float = 2.0           # total ~3 ms (see docx Step 4 justification)
# Step 5 features / PCA
n_pca: int = 8                    # GLOBAL fixed component count (see note below)
pca_pool_max_spikes: int = 50000  # cap on pooled waveforms for basis fit
# Step 8 audit
umap_n_neighbors: int = 15
umap_min_dist: float = 0.1
umap_n_components: int = 2
hdbscan_min_cluster_size: int = 50
# Step 9 cardiac
prwh_window_ms: float = 50.0      # +/- window for peri-R-wave histogram
cardiac_lock_window_ms: float = 7.5  # +/- narrow window for flagging (5-10 ms)
cardiac_peak_z: float = 3.0       # z-score over PRWH baseline to call "locked"
# Step 11 slow wave
sw_low_hz: float = 0.01
sw_high_hz: float = 0.2
# Step 12 rates
rate_bin_s: float = 2.0
# Step 13 responder
responder_pctile: float = 95.0
responder_frac_epoch: float = 1.0/3.0
seed: int = 0
```

### CRITICAL — global PCA basis

`n_pca` is **fixed and identical across all files** so feature spaces are comparable across stimulation conditions. The basis is **fit once on a pooled sample of waveforms drawn across all recordings in the batch** (capped at `pca_pool_max_spikes`), then **every recording is projected onto that same basis**. Do NOT fit a fresh PCA per file. The batch therefore has two passes:

1. **Pass 1 (basis pass):** run Steps 1–4 on every recording, collect waveforms, subsample to the cap, fit one PCA basis. Persist the basis (components + mean + scaler) to `batch_pca_basis.npz` at the batch root.
2. **Pass 2 (analysis pass):** for each recording, project its waveforms onto the persisted basis and run Steps 5(project)–14.

If the user runs a single file in isolation, fit the basis on that file's own waveforms but emit a warning that cross-condition comparability requires a batch-fit basis.

---

## 3. Input discovery & file pairing (`io_discovery.py`)

A "recording pair" = one **blanked neural data file** + one **R-peak (HR) location file**, residing in the **same directory**.

Requirements:
- `find_pairs(root_dir) -> list[RecordingPair]`: walk `root_dir` recursively (`os.walk`).
- The user supplies two filename patterns (glob or substring) — one identifying blanked-data files, one identifying R-peak files — via the UI (Section 7). Default guesses to autopopulate: blanked `*blank*`, R-peak `*HR*` / `*rpeak*` / `*Rpeak*`. Pairing rule: within each directory, match one blanked file to one R-peak file. If a directory has exactly one of each, pair them. If it has multiple, pair by longest common filename stem; if still ambiguous, surface to the UI for manual pairing and do not guess silently.
- The slow-wave channel may live inside the blanked data file (a separate variable) or as its own file. Support both: a user-assigned variable name within the blanked file (preferred), or a third filename pattern. If absent, Step 11 is skipped with a logged warning (do not crash).
- `RecordingPair` dataclass: `dir`, `blanked_path`, `rpeak_path`, optional `slowwave_path`, and a resolved `var_map` (Section 4).

### Variable-name introspection & autopopulation
- `.mat` files: load top-level variable names with `pymatreader` (handles v7.3). For `.npz`/`.npy`, use the array keys. List all candidate variable names and their shapes/dtypes.
- The UI prompts the user to assign each **logical role** to a **variable name**, autopopulated with the best guess:
  - `neural` (1×N or N×1 per cuff; if 2-D with 2 rows/cols, treat as two cuffs — ask user to confirm orientation and channel count)
  - `rpeak_times` (vector of sample indices or seconds — ask user which; autopopulate by range heuristic: if max value >> n_samples it's likely ms)
  - `slowwave` (optional)
  - `fs` (optional scalar; if present, override config default)
  - `stim_events` (optional; event onset times + a condition label, for Steps 12–13)
- Persist the chosen `var_map` per batch so re-runs don't re-prompt (store `batch_varmap.json` at root; UI offers "reuse previous mapping").

---

## 4. Loading & normalization (`io_load.py`)

- `load_recording(pair, config) -> Recording` returns a normalized structure:
  - `neural`: list of 1-D float arrays, one per cuff.
  - `rpeak_samples`: int array of R-peak sample indices (convert from seconds/ms if needed; conversion uses `fs`).
  - `slowwave`: 1-D float array or `None`.
  - `stim_events`: list of `(onset_sample, label)` or `None`.
  - `fs`: float.
- Validate: neural length > 0, rpeaks within `[0, n_samples)`, slow-wave length matches neural length (resample if a different `fs` is detected and logged).
- Assume the signal is **already notch-filtered and motion-blanked** (do not re-do these). Detect blanked regions if they are encoded as NaN or as runs of exact zeros, and carry a boolean `blanked_mask` forward so detection ignores those samples.

---

## 5. Pipeline steps (map to docx Section 3)

Implement each as a pure function taking arrays + config and returning a results dict. `pipeline.py` chains them per cuff.

| Step | Module.fn | Input | Output (keys) |
|---|---|---|---|
| 1 | `preprocess.bandpass` | neural, cfg | `filtered` |
| 2 | `preprocess.noise_sigma` | filtered, cfg | `sigma_track`, `sigma_times` |
| 3 | `detect.detect_spikes` | filtered, sigma_track, blanked_mask, cfg | `spike_samples` |
| 4 | `detect.extract_waveforms` | filtered, spike_samples, cfg | `waveforms` (n×L), `spike_samples` (re-aligned) |
| 5 | `features.scalar_features` + `features.project_pca` | waveforms, pca_basis, cfg | `scalar_feats` (dict of arrays), `pca_feats` (n×n_pca) |
| 6 | `sort.run_mountainsort` | waveforms or pca_feats, spike_samples, cfg | `labels` (n,) |
| 7 | `quality.cluster_metrics` | waveforms, labels, spike_samples, fs, cfg | per-cluster `snr`, `firing_rate`, `mean_wf`, `std_wf`, `isi` (full array), `isi_violation_rate` |
| 8 | `audit.umap_hdbscan_audit` | pca_feats, labels, cfg | `umap_xy`, `hdbscan_labels`, `adjusted_rand`, `agreement` |
| 9 | `cardiac.peri_rwave` | spike_samples, labels, rpeak_samples, fs, cfg | per-cluster `prwh`, `is_cardiac_locked`, `cleaned_spike_samples` |
| 10 | `respiration.verify` | spike_samples (or burst rate), resp_surrogate, stim_events, fs, cfg | `burst_rate`, `breathing_rate`, `rate_matches`, `breathing_rate_by_condition`, `stable_across_conditions` |
| 11 | `slowwave.phase_tag` | spike_samples, slowwave, fs, labels, cfg | per-spike `sw_phase`, per-cluster `phase_hist`, `mrl`, `rayleigh_p` |
| 12 | `rates.firing_rates` | spike_samples, labels, physio signals, cfg | per-cluster `rate_trace`, `corr_resp`, `corr_hr`, `corr_sw` |
| 13 | `responder.detect` | rate_trace, stim_events, cfg | per-cluster per-condition `is_responder` |
| 14 | `fibertype.tag` | scalar_feats, labels, cfg | per-cluster `type_tag` ("C-like"/"A-like"/"ambiguous") |

Notes:
- **Respiration surrogate:** derive from R-peak modulation (RSA: interpolate instantaneous HR from R-R intervals -> bandpass 0.1–3 Hz for rat resp band; and/or R-wave amplitude envelope). Implement in `respiration.py` as `resp_surrogate_from_rpeaks(rpeak_samples, neural_or_amplitude, fs)`. No separate EMG channel exists.
- **Step 9 deletes nothing by default:** produce `cleaned_spike_samples` (flagged events removed) AND keep the originals + flags. Downstream steps use originals unless a config flag `use_cardiac_cleaned=True`.
- **Circular stats (Step 11):** mean resultant vector length `R = |mean(exp(i*phase))|`; Rayleigh test p-value. Use `scipy.stats` or `pingouin.circ_rayleigh`.
- **Deferred:** template-matching second detection pass (Track B) is NOT implemented in v1. Leave a clearly marked stub `detect.template_recovery(...)` with a docstring explaining it is added only if the detected-amplitude histogram shows a hard cutoff at threshold.

---

## 6. Orchestration & batch (`pipeline.py`, `batch.py`)

- `run_pipeline_on_pair(pair, pca_basis, config) -> dict`: loads, then per cuff runs Steps 1–14, returns a nested results dict keyed by cuff index.
- `run_batch(root_dir, var_map, config)`:
  1. discover pairs;
  2. **Pass 1**: Steps 1–4 across all pairs/cuffs -> pooled waveforms -> fit global PCA -> save `batch_pca_basis.npz`;
  3. **Pass 2**: for each pair, `run_pipeline_on_pair`, then `assemble.save_mat(results, pair.dir)`;
  4. write a batch summary CSV (one row per cuff: n_spikes, n_clusters, mean SNR, n_responders, etc.) at root.
- Robustness: a failure on one pair logs the traceback and continues to the next; the failed pair is recorded in the summary with status="failed" and the reason.

---

## 7. UI (`ui.py`)

Minimal, functional. Flow:

1. **Folder picker** — choose the batch root directory.
2. **Pattern entry** — two text fields (blanked pattern, R-peak pattern), prefilled with default guesses (`*blank*`, `*HR*`). Optional slow-wave pattern field.
3. **Discovery preview** — a table listing every discovered pair (dir, blanked file, R-peak file, slow-wave source). Rows with ambiguous pairing are highlighted; user can fix pairing via dropdowns. "None of these / exclude" per row.
4. **Variable mapping** — for a representative file, show introspected variable names with shapes; provide a dropdown per logical role (`neural`, `rpeak_times`, units selector sample/sec/ms, `slowwave`, `fs`, `stim_events`), each **autopopulated** with the best guess. "Apply mapping to all files in batch" checkbox (default on). Option to reuse a saved `batch_varmap.json`.
5. **Config review** — expose key params (bandpass corners, threshold sigma, n_pca, bin size) with defaults from `PipelineConfig`; advanced params behind a collapsible.
6. **Run** — progress bar over pairs, live log pane. On completion, show the batch summary table and the output path of each `.mat`.

Keep all UI state serializable; "Run headless with these settings" button that prints the equivalent `run.py --no-ui ...` command for reproducibility.

---

## 8. Output `.mat` schema (`assemble.py`)

One file per recording pair, written into `pair.dir`, named `<common_stem>_metrics.mat`. Top-level struct `metrics` with:

```
metrics.provenance      : struct (software versions, full PipelineConfig as fields,
                                   source file paths, var_map, seed, datetime,
                                   pca_basis_path)
metrics.fs              : double
metrics.n_cuffs         : double
metrics.cuff(k)         : struct array, one per cuff, each with:
  .step1.bp_low, .bp_high, .bp_order
  .step2.sigma_track, .sigma_times
  .step3.spike_samples, .spike_times_s, .threshold_sigma, .amplitude_hist (for cutoff diagnostic)
  .step4.waveforms, .wf_pre_ms, .wf_post_ms, .wf_len_samples
  .step5.scalar_feats (struct: p2p, trough_peak_ms, halfwidth_ms, zc_slope),
         .pca_feats, .n_pca
  .step6.labels, .n_clusters, .sorter ("mountainsort5")
  .step7.cluster(c) : struct array per cluster:
            .snr, .firing_rate_hz, .mean_wf, .std_wf, .isi_s (FULL array), .isi_violation_rate, .n_spikes
  .step8.umap_xy, .hdbscan_labels, .adjusted_rand, .agreement
  .step9.cluster(c).prwh, .prwh_bin_ms, .is_cardiac_locked, .cleaned_spike_samples
  .step10.burst_rate_hz, .breathing_rate_hz, .rate_matches (bool),
          .breathing_rate_by_condition, .stable_across_conditions (bool)
  .step11.sw_phase_per_spike, .cluster(c).phase_hist, .mrl, .rayleigh_p   (empty if no slow-wave)
  .step12.cluster(c).rate_trace, .rate_bin_s, .corr_resp, .corr_hr, .corr_sw
  .step13.cluster(c).condition(j).label, .is_responder
  .step14.cluster(c).type_tag
```

Use `scipy.io.savemat` with `long_field_names=True, do_compression=True`. ISI arrays are saved in full (per the requirement to keep ISIs for later). Verify on save that the file re-loads with `pymatreader` and that all `cuff().step7.cluster().isi_s` arrays are present and non-empty for clusters with >1 spike.

---

## 9. Sample dataset (`tests/make_sample_data.py`)

Synthesize a labeled dataset so the test has ground truth:
- 120 s at `fs`, one or two cuffs.
- 2–3 synthetic "units" per cuff: each a fixed triphasic template (widths chosen so at least one is C-like ~1.5 ms and one A-like ~0.5 ms), Poisson firing at known rates, added at known times. Record ground-truth labels and times.
- Inject pink/Gaussian noise to a target SNR (~3–4).
- Inject a far-field ECG: QRS template at ~400 bpm (sub-band, so bandpass should remove it) AND a small cardiac-locked unit (in-band, to exercise Step 9).
- Inject respiratory modulation of one unit's rate at ~80 breaths/min.
- Inject a slow wave at ~0.05 Hz and phase-lock one unit to it (to exercise Step 11 with a known nonzero MRL).
- Add a couple of `stim_events` with pre/post rate changes in one unit (to exercise Step 13).
- Save as `.mat` (blanked data + slow-wave var) and a separate R-peak `.mat`, into `tests/data/`, in two subfolders to exercise recursive discovery and pairing.

---

## 10. End-to-end test (`tests/test_end_to_end.py`)

Acceptance criterion for the build. Using the generated sample dataset, assert:

1. **Discovery:** `find_pairs` finds exactly the expected pairs across subfolders.
2. **Var mapping:** introspection lists the known variable names; programmatic mapping resolves all roles.
3. **Run completes** headless on the batch with no exceptions; a `_metrics.mat` is written into each source subfolder; `batch_pca_basis.npz` exists at root and has `n_pca` components.
4. **Shapes/ranges per cuff:**
   - `step1.filtered` length == neural length; PSD shows ECG band attenuated (check power <0.1–40 Hz drops vs raw).
   - `step3.spike_samples` count is within ±30% of the ground-truth total spike count.
   - `step6.n_clusters` ≥ number of ground-truth in-band units (allow over-splitting; check not collapsed to 1).
   - `step7` every cluster with >1 spike has a non-empty `isi_s`; `snr` finite and >0.
   - `step8.adjusted_rand` between MountainSort and HDBSCAN > 0.4 on clean synthetic data.
   - `step9` the injected cardiac-locked unit is flagged `is_cardiac_locked == True`; a non-cardiac unit is `False`.
   - `step10.rate_matches == True` for the respiratory-modulated case; breathing rate within ±15% of injected 80 bpm.
   - `step11` the phase-locked unit has `rayleigh_p < 0.05` and `mrl` above an unlocked unit's `mrl`.
   - `step13` the stim-modulated unit is `is_responder == True` for the relevant condition; a flat unit is not.
5. **Determinism:** running twice produces identical `labels` and identical `mrl` values (seeded).
6. **Reload:** the `.mat` reloads via `pymatreader` and the `metrics.provenance` contains the full config and source paths.

Provide a `pytest` suite; `pytest -q` must pass. Also provide a `--smoke` flag on `run.py` that regenerates sample data and runs the test in one command.

---

## 11. Build order (phases for Claude Code)

- **Phase A — scaffolding:** repo layout, `config.py`, `logging_setup.py`, `requirements.txt`, `io_discovery.py` + `io_load.py` with `.mat`/`.npz` introspection. Unit-test discovery & loading on tiny fixtures.
- **Phase B — signal core:** `preprocess.py`, `detect.py`, `features.py` (incl. global PCA two-pass logic). Unit-test on synthetic single-unit traces.
- **Phase C — sorting & audit:** `sort.py` (SpikeInterface + MountainSort5), `quality.py`, `audit.py`. Verify on synthetic multi-unit data.
- **Phase D — physiology:** `cardiac.py`, `respiration.py`, `slowwave.py`, `rates.py`, `responder.py`, `fibertype.py`.
- **Phase E — assembly & batch:** `assemble.py` (.mat schema), `pipeline.py`, `batch.py` (two-pass PCA), batch summary CSV.
- **Phase F — UI:** `ui.py`, `run.py`.
- **Phase G — sample data + end-to-end test:** `tests/make_sample_data.py`, `tests/test_end_to_end.py`; iterate until `pytest -q` passes.
- **Phase H — README:** usage (UI and headless), requirements, the `.mat` schema, and a note that the science rationale lives in the companion docx.

Deliver each phase runnable; do not defer all testing to the end. The definition of done is Phase G passing on the sample dataset.

---

## 12. Things deliberately NOT in v1 (documented stubs)

- Track B template-matching recovery pass (`detect.template_recovery` stub).
- Conduction-velocity / cross-cuff velocity analysis (cuffs are on separate branches).
- Pharmacological-validation hooks (capsaicin/lidocaine).
- Cross-cuff coincidence QC may be added as a small `batch`-level extra (zero-lag cross-correlogram between the two cuffs' spike trains) — implement if cheap, else stub with a docstring.

Leave each stub with a clear docstring stating the trigger condition for implementing it, referencing the relevant docx section.
