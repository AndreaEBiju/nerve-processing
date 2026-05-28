# Vagus Nerve Cuff Recording — Pipeline Implementation Spec

**Audience:** Claude Code (autonomous build agent)
**Goal:** Build a batch-processing tool that takes notch-filtered, motion-blanked vagal cuff recordings (one channel per cuff) plus extracted R-peak times and **three antral mucosal full-thickness slow-wave channels** spaced 0.5 cm apart, runs the 15-step analysis pipeline, and writes one `.mat` metrics file per recording pair into the folder that contained the source files. Include a UI for folder selection, recursive file discovery, file-pairing, and user assignment of variable names (with autopopulation). Ship with an end-to-end test on one sample dataset.

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
  io_load.py           # load blanked data, R-peaks, 3 slow-wave channels; normalize to arrays
  preprocess.py        # Step 1 bandpass, Step 2 noise estimate
  detect.py            # Step 3 detection, Step 4 waveform extraction/alignment
  features.py          # Step 5 scalar features + global PCA basis fit/project
  sort.py              # Step 6 MountainSort via SpikeInterface
  quality.py           # Step 7 cluster QC metrics (incl. ISI distributions, saved)
  audit.py             # Step 8 UMAP + HDBSCAN audit + agreement stats
  cardiac.py           # Step 9 peri-R-wave histogram, cardiac-locked handling
  respiration.py       # Step 10 respiratory verification + cross-condition check
  slowwave_quality.py  # Step 11a: per-channel quality scoring, tiered inclusion, common-mode reference
  slowwave.py          # Step 11b: per-spike phase tagging, primary + per-channel stats, consistency check
  antral_bursts.py     # Step 12 antral spike-burst detection + spike-to-burst cross-correlogram
  rates.py             # Step 13 firing-rate series + physiological correlations
  responder.py         # Step 14 responder detection (Masi/Zanos criterion)
  fibertype.py         # Step 15 descriptive fibre-type tagging
  assemble.py          # collect all metrics -> single .mat per pair
  pipeline.py          # orchestrates Steps 1-15 for one recording pair
  batch.py             # iterate over discovered pairs, call pipeline, save outputs
  ui.py                # GUI: folder picker, file table, var-name assignment, run button
  plots.py             # per-cluster diagnostic plots (slow-wave traces + raster, etc.)
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
# Step 11 slow wave: three antral channels, 0.5 cm spacing, mucosal full-thickness
sw_low_hz: float = 0.01           # phase-extraction bandpass low
sw_high_hz: float = 0.2           # phase-extraction bandpass high
sw_inband_low_hz: float = 0.04    # in-band SNR numerator band
sw_inband_high_hz: float = 0.15
sw_outband_low1_hz: float = 0.005 # in-band SNR denominator band part 1
sw_outband_high1_hz: float = 0.04
sw_outband_low2_hz: float = 0.15  # in-band SNR denominator band part 2
sw_outband_high2_hz: float = 0.5
sw_quality_window_s: float = 60.0       # rolling window for per-channel quality
sw_quality_overlap_s: float = 30.0      # rolling window overlap
sw_quality_good_threshold: float = 0.7  # tier "good" >= this
sw_quality_marginal_threshold: float = 0.3  # tier "marginal" >= this, "bad" below
sw_coherence_min: float = 0.5     # below this with both other channels -> drop
# Per-channel quality normalization ranges (input -> [0,1])
sw_snr_norm_lo: float = 1.0       # below this -> score 0
sw_snr_norm_hi: float = 10.0      # at/above this -> score 1
sw_coherence_norm_lo: float = 0.3
sw_coherence_norm_hi: float = 0.9
sw_envcv_norm_lo: float = 1.0     # high CV -> bad
sw_envcv_norm_hi: float = 0.2     # low CV -> good (inverted)
# Cross-channel consistency check
sw_consistency_phase_spread_deg: float = 60.0  # circ. variance threshold (deg equivalent)
sw_per_channel_p_near: float = 0.10            # "near significant" threshold for per-channel Rayleigh
sw_primary_p: float = 0.05                     # significance threshold for common-mode Rayleigh
# Step 12 antral burst correlation
burst_band_low_hz: float = 1.0
burst_band_high_hz: float = 10.0
burst_threshold_sigma: float = 4.0
burst_min_duration_ms: float = 200.0
burst_min_channels: int = 2        # require burst on >= this many of 3 channels (high-confidence)
burst_xcorr_window_s: float = 10.0 # +/- lag range for spike-to-burst xcorr
burst_xcorr_bin_s: float = 0.05    # 50 ms bins
burst_peak_z_threshold: float = 3.0  # to call a cluster correlated with bursts
# Step 13 rates
rate_bin_s: float = 2.0
# Step 14 responder
responder_pctile: float = 95.0
responder_frac_epoch: float = 1.0/3.0
seed: int = 0
```

### CRITICAL — global PCA basis

`n_pca` is **fixed and identical across all files** so feature spaces are comparable across stimulation conditions. The basis is **fit once on a pooled sample of waveforms drawn across all recordings in the batch** (capped at `pca_pool_max_spikes`), then **every recording is projected onto that same basis**. Do NOT fit a fresh PCA per file. The batch therefore has two passes:

1. **Pass 1 (basis pass):** run Steps 1–4 on every recording, collect waveforms, subsample to the cap, fit one PCA basis. Persist the basis (components + mean + scaler) to `batch_pca_basis.npz` at the batch root.
2. **Pass 2 (analysis pass):** for each recording, project its waveforms onto the persisted basis and run Steps 5(project)–15.

If the user runs a single file in isolation, fit the basis on that file's own waveforms but emit a warning that cross-condition comparability requires a batch-fit basis.

---

## 3. Input discovery & file pairing (`io_discovery.py`)

A "recording pair" = one **blanked neural data file** + one **R-peak (HR) location file**, residing in the **same directory**.

Requirements:
- `find_pairs(root_dir) -> list[RecordingPair]`: walk `root_dir` recursively (`os.walk`).
- The user supplies two filename patterns (glob or substring) — one identifying blanked-data files, one identifying R-peak files — via the UI (Section 7). Default guesses to autopopulate: blanked `*blank*`, R-peak `*HR*` / `*rpeak*` / `*Rpeak*`. Pairing rule: within each directory, match one blanked file to one R-peak file. If a directory has exactly one of each, pair them. If it has multiple, pair by longest common filename stem; if still ambiguous, surface to the UI for manual pairing and do not guess silently.
- **Three slow-wave channels.** The slow-wave signal comprises three antral mucosal full-thickness electrode channels placed 0.5 cm apart. They may live inside the blanked data file as three separate variables, or as a single 2-D variable with three rows/columns, or as separate files. Support all three layouts; the UI asks the user to confirm the layout and assign the three channels in order along the gastric long axis (proximal → middle → distal). If only one or two channels exist (one or two electrodes failed at surgery), accept that gracefully and downgrade behaviour in Step 11 accordingly: with two channels, coherence is computed pairwise but cross-channel consistency for Step 11.4 requires manual qualification; with one channel, skip Step 11.1–11.2 (no common-mode), use that channel directly as the reference, and disable the consistency check. If zero usable channels, skip Steps 11 and 12 with a logged warning (do not crash).
- `RecordingPair` dataclass: `dir`, `blanked_path`, `rpeak_path`, optional `slowwave_paths` (list of up to three), and a resolved `var_map` (Section 4).

### Variable-name introspection & autopopulation
- `.mat` files: load top-level variable names with `pymatreader` (handles v7.3). For `.npz`/`.npy`, use the array keys. List all candidate variable names and their shapes/dtypes.
- The UI prompts the user to assign each **logical role** to a **variable name**, autopopulated with the best guess:
  - `neural` (1×N or N×1 per cuff; if 2-D with 2 rows/cols, treat as two cuffs — ask user to confirm orientation and channel count)
  - `rpeak_times` (vector of sample indices or seconds — ask user which; autopopulate by range heuristic: if max value >> n_samples it's likely ms)
  - `slowwave_ch1`, `slowwave_ch2`, `slowwave_ch3` (three antral electrodes, ordered proximal → middle → distal along the gastric long axis; the order matters for the propagation-lag estimate)
  - `fs` (optional scalar; if present, override config default)
  - `stim_events` (optional; event onset times + a condition label, for Steps 13–14)
- Persist the chosen `var_map` per batch so re-runs don't re-prompt (store `batch_varmap.json` at root; UI offers "reuse previous mapping").

---

## 4. Loading & normalization (`io_load.py`)

- `load_recording(pair, config) -> Recording` returns a normalized structure:
  - `neural`: list of 1-D float arrays, one per cuff.
  - `rpeak_samples`: int array of R-peak sample indices (convert from seconds/ms if needed; conversion uses `fs`).
  - `slowwave_channels`: list of 1-D float arrays, length 1, 2, or 3, ordered proximal → distal. Empty list if no slow-wave channel is available.
  - `stim_events`: list of `(onset_sample, label)` or `None`.
  - `fs`: float.
- Validate: neural length > 0, rpeaks within `[0, n_samples)`, slow-wave channels each match neural length (resample if a different `fs` is detected and logged). If slow-wave channels have different lengths to each other, raise with an actionable message; do not silently truncate.
- Assume the signal is **already notch-filtered and motion-blanked** (do not re-do these). Detect blanked regions if they are encoded as NaN or as runs of exact zeros, and carry a boolean `blanked_mask` forward so detection ignores those samples.

---

## 5. Pipeline steps (map to docx Section 3)

Implement each as a pure function taking arrays + config and returning a results dict. `pipeline.py` chains them per cuff (Steps 1–10, 13–15 are per-cuff; Steps 11–12 use the slow-wave channels which are shared across cuffs but are run within each cuff's analysis loop because their outputs are per-cluster).

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
| 11a | `slowwave_quality.score_channels` | slowwave_channels (list), fs, cfg | per-channel `snr_inband`, `peak_prominence`, `pairwise_coherence`, `envelope_cv`, combined `quality_score` (whole + rolling), `tier`, `dropped_with_reason` |
| 11a | `slowwave_quality.build_common_mode` | slowwave_channels, quality_scores_rolling, cfg | `common_mode` (1-D array), `rolling_weights` (3 × n_windows), `n_channels_contributing` (per timepoint), `propagation_lag_s` (between adjacent channels) |
| 11b | `slowwave.phase_tag` | spike_samples, common_mode, slowwave_channels, fs, labels, cfg | per-spike `sw_phase_common` (primary), `sw_phase_per_channel` (n_spikes × 3), per-cluster primary `phase_hist`, `mrl`, `rayleigh_p`, `preferred_phase` |
| 11b | `slowwave.consistency_check` | per-cluster per-channel stats, cfg | per-cluster `consistency_score` (circular variance of preferred phases), `robust_phase_locked` (bool) |
| 12 | `antral_bursts.detect_bursts` | slowwave_channels, quality_scores, fs, cfg | per-channel `burst_times_s`, high-confidence `consensus_burst_times_s` (on ≥ `burst_min_channels`) |
| 12 | `antral_bursts.xcorr` | spike_samples per cluster, consensus_burst_times_s, fs, cfg | per-cluster `xcorr`, `lag_axis_s`, `peak_lag_s`, `peak_z`, `direction_tag` ("efferent-like"/"afferent-like"/"none") |
| 13 | `rates.firing_rates` | spike_samples, labels, physio signals, cfg | per-cluster `rate_trace`, `corr_resp`, `corr_hr`, `corr_sw` |
| 14 | `responder.detect` | rate_trace, stim_events, cfg | per-cluster per-condition `is_responder` |
| 15 | `fibertype.tag` | scalar_feats, labels, cfg | per-cluster `type_tag` ("C-like"/"A-like"/"ambiguous") |

Notes:
- **Respiration surrogate:** derive from R-peak modulation (RSA: interpolate instantaneous HR from R-R intervals -> bandpass 0.1–3 Hz for rat resp band; and/or R-wave amplitude envelope). Implement in `respiration.py` as `resp_surrogate_from_rpeaks(rpeak_samples, neural_or_amplitude, fs)`. No separate EMG channel exists.
- **Step 9 deletes nothing by default:** produce `cleaned_spike_samples` (flagged events removed) AND keep the originals + flags. Downstream steps use originals unless a config flag `use_cardiac_cleaned=True`.
- **Step 11a quality scoring (`slowwave_quality.py`):** Compute the four diagnostics (in-band SNR, peak prominence, pairwise coherence at dominant frequency, envelope CV) per channel, both whole-recording and in rolling windows (`sw_quality_window_s`, `sw_quality_overlap_s`). Normalize each diagnostic to [0,1] using the `*_norm_lo`/`*_norm_hi` config ranges. Combine via geometric mean (use `np.exp(np.mean(np.log(scores + eps)))` for stability). Apply the tiered rule (`sw_quality_good_threshold`, `sw_quality_marginal_threshold`) per window per channel. If no channel reaches marginal in a window, that window contributes NaN to the common-mode. If no channel reaches marginal anywhere in the recording, raise a `SlowWaveUnusable` exception that the orchestrator catches to skip Steps 11–12 for that recording (everything else still runs).
- **Step 11a common-mode reference:** Within each window, estimate the inter-channel propagation phase lag (mean phase difference at the dominant slow-wave frequency between each channel and the middle channel via Welch cross-spectrum). Phase-align each channel by applying a fractional-sample delay (use `scipy.signal.resample` or FFT-based fractional shift). Sum the aligned surviving channels weighted by their window quality scores normalized to sum to 1 across surviving channels. Interpolate the resulting per-window common-mode pieces back to the full sample rate (linear interp on the analytic signal, then take the imaginary part — or simpler: assemble the time-domain common-mode at the slow-wave sample rate after first downsampling to e.g. 10 Hz, then upsample to neural fs only when extracting per-spike phase).
- **Step 11b per-spike phase:** Apply zero-phase Butterworth bandpass (`sw_low_hz` to `sw_high_hz`) to both the common-mode reference and each individual channel; Hilbert-transform to get instantaneous phase; sample at each spike time. Per-spike output has 4 columns: `[common, ch1, ch2, ch3]`. For dropped channels, the corresponding column is NaN.
- **Step 11b circular stats:** Mean resultant vector length `R = |mean(exp(i*phase))|`. Rayleigh test p-value via `scipy.stats` (compute z = n*R², p ≈ exp(-z)) or `pingouin.circ_rayleigh`. Preferred phase = `angle(mean(exp(i*phase)))`.
- **Step 11b consistency check:** Per cluster, take the three per-channel preferred phases (after the propagation lag is subtracted out, so all three are referenced to the middle channel). Compute the circular variance `1 - |mean(exp(i*phi_i))|` across the three preferred phases. Convert the threshold `sw_consistency_phase_spread_deg` (degrees) into a circular-variance threshold by noting that for three equally-spaced phases spanning ±X degrees, circular variance ≈ `1 - sinc(X*pi/180)` is a usable approximation; precompute and document. Set `robust_phase_locked = True` iff primary Rayleigh p < `sw_primary_p` AND circular variance below threshold AND ≥2 of 3 per-channel Rayleigh p-values are < `sw_per_channel_p_near`.
- **Step 12 burst detection (`antral_bursts.py`):** Bandpass each channel at (`burst_band_low_hz`, `burst_band_high_hz`), compute the Hilbert envelope, threshold at `burst_threshold_sigma` × MAD of envelope, enforce `burst_min_duration_ms`. Extract burst onset times per channel. Consensus burst = any burst onset confirmed within a small tolerance window (e.g. 100 ms) on at least `burst_min_channels` channels — use the median onset time across confirming channels as the consensus time. Only use bursts from channels with quality score ≥ marginal in the corresponding window.
- **Step 12 cross-correlogram:** Per cluster, bin spike train at `burst_xcorr_bin_s`; for each consensus burst onset, sum the spike-count vector in a window of ±`burst_xcorr_window_s` around the burst; average across bursts. Compute baseline as the mean count in the outer 50% of the lag window (i.e. far from lag 0). Peak lag = argmax of the xcorr; peak z = (peak - baseline_mean) / baseline_std. Direction tag: peak_lag > 0.5 s with z > threshold → "efferent-like" (spikes precede burst); peak_lag < -0.5 s with z > threshold → "afferent-like"; otherwise "none".
- **Deferred:** template-matching second detection pass (Track B) is NOT implemented in v1. Leave a clearly marked stub `detect.template_recovery(...)` with a docstring explaining it is added only if the detected-amplitude histogram shows a hard cutoff at threshold.

---

## 6. Orchestration & batch (`pipeline.py`, `batch.py`)

- `run_pipeline_on_pair(pair, pca_basis, config) -> dict`: loads, then per cuff runs Steps 1–15, returns a nested results dict keyed by cuff index. Catches `SlowWaveUnusable` and proceeds with NaN-filled slow-wave outputs while completing everything else.
- `run_batch(root_dir, var_map, config)`:
  1. discover pairs;
  2. **Pass 1**: Steps 1–4 across all pairs/cuffs -> pooled waveforms -> fit global PCA -> save `batch_pca_basis.npz`;
  3. **Pass 2**: for each pair, `run_pipeline_on_pair`, then `assemble.save_mat(results, pair.dir)`;
  4. write a batch summary CSV (one row per cuff: n_spikes, n_clusters, mean SNR, n_responders, n_robust_phase_locked, n_burst_correlated, slow-wave channels usable y/n, etc.) at root.
- Robustness: a failure on one pair logs the traceback and continues to the next; the failed pair is recorded in the summary with status="failed" and the reason.

---

## 7. UI (`ui.py`)

Minimal, functional. Flow:

1. **Folder picker** — choose the batch root directory.
2. **Pattern entry** — text fields for blanked-data pattern, R-peak pattern, and slow-wave pattern, prefilled with default guesses (`*blank*`, `*HR*`, `*slow*` or `*SW*`).
3. **Discovery preview** — a table listing every discovered pair (dir, blanked file, R-peak file, slow-wave source(s)). Rows with ambiguous pairing are highlighted; user can fix pairing via dropdowns. "None of these / exclude" per row.
4. **Variable mapping** — for a representative file, show introspected variable names with shapes; provide a dropdown per logical role (`neural`, `rpeak_times`, units selector sample/sec/ms, `slowwave_ch1`/`ch2`/`ch3`, `fs`, `stim_events`), each **autopopulated** with the best guess.
5. **Slow-wave channel ordering** — a dedicated step where the user confirms which of the three assigned channels is proximal, middle, and distal along the gastric long axis. This is critical because the propagation lag estimate and the consistency check both depend on knowing the spatial order. Autopopulate by variable-name heuristics (ch1/ch2/ch3, prox/mid/dist) but always require explicit user confirmation.
6. **Apply mapping to all files in batch** checkbox (default on). Option to reuse a saved `batch_varmap.json`.
7. **Config review** — expose key params (bandpass corners, threshold sigma, n_pca, bin size, slow-wave quality thresholds, burst detection parameters) with defaults from `PipelineConfig`; advanced params behind a collapsible.
8. **Run** — progress bar over pairs, live log pane. On completion, show the batch summary table and the output path of each `.mat`.

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
  .step11a (slow-wave channel quality and common-mode):
          .channels_present (1, 2, or 3),
          .channel(k=1..n).snr_inband, .peak_prominence, .pairwise_coherence_with_others,
                          .envelope_cv, .quality_score_whole,
                          .quality_score_rolling, .quality_window_times_s,
                          .tier (string), .reason_if_excluded (string),
          .rolling_weights (n_channels × n_windows),
          .n_channels_contributing (per timepoint, downsampled),
          .propagation_lag_s_between_adjacent (length n_channels-1),
          .common_mode (downsampled time series),
          .common_mode_fs_hz,
          .summary_text (machine-readable JSON string of channel status)
  .step11b (per-spike phase and per-cluster phase-locking):
          .sw_phase_per_spike (n_spikes × 4: [common, ch1, ch2, ch3]; NaN for absent/dropped),
          .cluster(c).phase_hist_common,
                    .mrl_common, .rayleigh_p_common, .preferred_phase_common,
                    .mrl_per_channel (3,), .rayleigh_p_per_channel (3,),
                                  .preferred_phase_per_channel (3,),
                    .consistency_score (circular variance across channels),
                    .robust_phase_locked (bool)
  .step12 (antral burst correlation):
          .burst_times_s_per_channel (cell array, length n_channels),
          .consensus_burst_times_s,
          .cluster(c).xcorr, .lag_axis_s, .peak_lag_s, .peak_z, .direction_tag (string)
  .step13.cluster(c).rate_trace, .rate_bin_s, .corr_resp, .corr_hr, .corr_sw_common
  .step14.cluster(c).condition(j).label, .is_responder
  .step15.cluster(c).type_tag
```

Use `scipy.io.savemat` with `long_field_names=True, do_compression=True`. ISI arrays are saved in full (per the requirement to keep ISIs for later). Verify on save that the file re-loads with `pymatreader` and that all `cuff().step7.cluster().isi_s` arrays are present and non-empty for clusters with >1 spike. If Steps 11–12 were skipped (no usable slow-wave channels), the corresponding fields are present but contain a single `.skipped = True` flag plus the reason; downstream MATLAB/Python code should check this before reading other fields.

---

## 9. Sample dataset (`tests/make_sample_data.py`)

Synthesize a labeled dataset so the test has ground truth. Generate **two** datasets to exercise the full set of code paths:

**Dataset A — all three slow-wave channels good:**
- 180 s at `fs`, one or two cuffs.
- 2–3 synthetic "units" per cuff: each a fixed triphasic template (widths chosen so at least one is C-like ~1.5 ms and one A-like ~0.5 ms), Poisson firing at known rates, added at known times. Record ground-truth labels and times.
- Inject pink/Gaussian noise to a target SNR (~3–4).
- Inject a far-field ECG: QRS template at ~400 bpm (sub-band, so bandpass should remove it) AND a small cardiac-locked unit (in-band, to exercise Step 9).
- Inject respiratory modulation of one unit's rate at ~80 breaths/min.
- Generate **three** slow-wave channels: a base sinusoid at ~0.08 Hz with ~50 mV amplitude, copied to three channels with inter-channel time delays corresponding to ~7 mm/s propagation (~0.7 s lag for 0.5 cm spacing → modest but detectable). Add channel-specific noise so each has a slightly different envelope but the underlying wave is shared.
- Phase-lock one unit to the slow wave at a known preferred phase (e.g. 90°) with a known concentration (so MRL is around 0.4). Phase-lock against the **middle channel's phase** so the consistency check should pass.
- Inject antral spike bursts at a rate of ~2 per slow-wave cycle, on top of the slow waves in all three channels, with 1–5 Hz oscillatory content during each burst lasting ~500 ms. Phase-couple one unit (different from the slow-wave-locked one) to occur ~2 s **before** each burst (efferent-like) and another unit ~1 s **after** each burst (afferent-like).
- Add a couple of `stim_events` with pre/post rate changes in one unit (to exercise Step 14).
- Save as `.mat` (blanked data + three slow-wave variables `sw_ch1`, `sw_ch2`, `sw_ch3`) and a separate R-peak `.mat`, into `tests/data/datasetA/`.

**Dataset B — one slow-wave channel deliberately corrupted:**
- Same structure as A but in a second subfolder `tests/data/datasetB/`.
- Channel 2 is corrupted: replace 60–120 s with pure Gaussian noise (mimicking a dropout), and add ~3× the noise amplitude throughout. The quality scorer must drop this channel in the middle window range and downweight it overall; the common-mode must use channels 1 and 3 during the dropout and all three (with 2 downweighted) elsewhere.
- Verifies the rolling quality machinery and the tiered weighting.

The two-folder layout exercises recursive discovery and pairing.

---

## 10. End-to-end test (`tests/test_end_to_end.py`)

Acceptance criterion for the build. Using both generated sample datasets (A: all three slow-wave channels good; B: channel 2 corrupted with a mid-recording dropout), assert:

1. **Discovery:** `find_pairs` finds exactly the expected pairs across both subfolders.
2. **Var mapping:** introspection lists the known variable names; programmatic mapping resolves all roles including the three slow-wave channels with their assigned order.
3. **Run completes** headless on the batch with no exceptions; a `_metrics.mat` is written into each source subfolder; `batch_pca_basis.npz` exists at root and has `n_pca` components.
4. **Shapes/ranges per cuff (both datasets):**
   - `step1.filtered` length == neural length; PSD shows ECG band attenuated (check power <0.1–40 Hz drops vs raw).
   - `step3.spike_samples` count is within ±30% of the ground-truth total spike count.
   - `step6.n_clusters` ≥ number of ground-truth in-band units (allow over-splitting; check not collapsed to 1).
   - `step7` every cluster with >1 spike has a non-empty `isi_s`; `snr` finite and >0.
   - `step8.adjusted_rand` between MountainSort and HDBSCAN > 0.4 on clean synthetic data.
   - `step9` the injected cardiac-locked unit is flagged `is_cardiac_locked == True`; a non-cardiac unit is `False`.
   - `step10.rate_matches == True` for the respiratory-modulated case; breathing rate within ±15% of injected 80 bpm.
5. **Step 11 (Dataset A, all channels good):**
   - All three channels have whole-recording `quality_score` in the "good" tier (≥ 0.7).
   - The common-mode reference has higher in-band SNR than the worst of the three input channels.
   - For the phase-locked unit: primary `rayleigh_p_common < 0.05`, `mrl_common` above an unlocked unit's MRL, `preferred_phase_common` within 30° of the injected phase, `robust_phase_locked == True`, `consistency_score` low (all three per-channel preferred phases within ~30° of each other after propagation-lag correction).
   - For an unlocked unit: `rayleigh_p_common >= 0.05`, `robust_phase_locked == False`.
6. **Step 11 (Dataset B, channel 2 dropout):**
   - Channel 2's `quality_score_whole` is below 0.5 and below channels 1 and 3.
   - Channel 2's `quality_score_rolling` in the 60–120 s window is below `sw_quality_marginal_threshold` (0.3); `rolling_weights` for channel 2 in those windows is 0 (excluded) or very low (downweighted).
   - The common-mode reference still produces a usable phase signal during the dropout (channels 1 and 3 carry it).
   - The phase-locked unit still passes `robust_phase_locked == True`, demonstrating that the pipeline is robust to partial channel failure.
7. **Step 12 (both datasets):**
   - `consensus_burst_times_s` has count within ±20% of injected burst count.
   - The efferent-injected unit has `direction_tag == "efferent-like"` with `peak_lag_s` in [-3.0, -1.0] s ± 0.5 s of injected lag (peak_lag sign convention: positive = spike precedes burst — verify your convention in the test).
   - The afferent-injected unit has `direction_tag == "afferent-like"` with `peak_lag_s` consistent with injection.
   - A non-correlated unit has `direction_tag == "none"` and `peak_z < burst_peak_z_threshold`.
8. **Step 14:** the stim-modulated unit is `is_responder == True` for the relevant condition; a flat unit is not.
9. **Determinism:** running the full batch twice produces identical `labels`, identical `mrl_common`, identical `consistency_score`, and identical `peak_lag_s` values (seeded).
10. **Reload:** the `.mat` reloads via `pymatreader`; `metrics.provenance` contains the full config (including all `sw_*` and `burst_*` parameters) and source paths; all step 11/12 fields are present and non-empty for Dataset A.

Provide a `pytest` suite; `pytest -q` must pass. Also provide a `--smoke` flag on `run.py` that regenerates both sample datasets and runs the tests in one command.

---

## 11. Build order (phases for Claude Code)

- **Phase A — scaffolding:** repo layout, `config.py`, `logging_setup.py`, `requirements.txt`, `io_discovery.py` + `io_load.py` with `.mat`/`.npz` introspection. Unit-test discovery & loading on tiny fixtures.
- **Phase B — signal core:** `preprocess.py`, `detect.py`, `features.py` (incl. global PCA two-pass logic). Unit-test on synthetic single-unit traces.
- **Phase C — sorting & audit:** `sort.py` (SpikeInterface + MountainSort5), `quality.py`, `audit.py`. Verify on synthetic multi-unit data.
- **Phase D — physiology core:** `cardiac.py`, `respiration.py`, `rates.py`, `responder.py`, `fibertype.py`.
- **Phase E — slow-wave + burst:** `slowwave_quality.py` (the quality-scoring, tiered inclusion, rolling weights, common-mode construction), `slowwave.py` (per-spike phase tagging, primary stats, consistency check), `antral_bursts.py` (burst detection + spike-to-burst xcorr). Unit-test each module on synthetic 3-channel data including a deliberately corrupted channel.
- **Phase F — assembly & batch:** `assemble.py` (.mat schema), `plots.py`, `pipeline.py`, `batch.py` (two-pass PCA), batch summary CSV.
- **Phase G — UI:** `ui.py`, `run.py`.
- **Phase H — sample data + end-to-end test:** `tests/make_sample_data.py` (both datasets A and B), `tests/test_end_to_end.py`; iterate until `pytest -q` passes.
- **Phase I — README:** usage (UI and headless), requirements, the `.mat` schema, and a note that the science rationale lives in the companion docx.

Deliver each phase runnable; do not defer all testing to the end. The definition of done is Phase H passing on both sample datasets.

---

## 12. Things deliberately NOT in v1 (documented stubs)

- Track B template-matching recovery pass (`detect.template_recovery` stub).
- Conduction-velocity / cross-cuff velocity analysis (cuffs are on separate branches).
- Pharmacological-validation hooks (capsaicin/lidocaine).
- Cross-cuff coincidence QC may be added as a small `batch`-level extra (zero-lag cross-correlogram between the two cuffs' spike trains) — implement if cheap, else stub with a docstring.

Leave each stub with a clear docstring stating the trigger condition for implementing it, referencing the relevant docx section.
