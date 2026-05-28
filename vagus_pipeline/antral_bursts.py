"""Step 12 -- antral spike-burst detection + spike-to-burst cross-correlogram.

Per spec §5 Step 12:

1. **Burst detection per channel**: bandpass each slow-wave channel at
   (``burst_band_low_hz``, ``burst_band_high_hz``) -- 1-10 Hz by default, the
   antral burst band sitting above the slow-wave frequency -- compute the
   Hilbert envelope, threshold at ``burst_threshold_sigma * MAD(envelope)``,
   enforce ``burst_min_duration_ms`` minimum, extract burst onset times.

2. **Consensus bursts**: an onset is "consensus" when at least
   ``burst_min_channels`` of the surviving channels show an onset within
   ``burst_consensus_window_ms`` of each other; the consensus time is the
   median of the contributing onsets.  Only channels whose quality tier was
   at least ``marginal`` in the corresponding rolling window contribute.

3. **Spike-to-burst cross-correlogram**: per cluster, bin the spike train at
   ``burst_xcorr_bin_s``; for each consensus burst onset, sum spike counts in
   a window of +/- ``burst_xcorr_window_s`` around the onset; average across
   bursts.  Baseline = mean count in the outer half of the lag window.
   ``peak_lag_s`` = lag of the maximum.  ``peak_z = (peak - baseline) / std``.
   Direction tag:
     * ``peak_lag_s < -burst_direction_min_lag_s`` and ``peak_z > threshold``
        -> "efferent-like" (spike precedes burst)
     * ``peak_lag_s > burst_direction_min_lag_s`` and ``peak_z > threshold``
        -> "afferent-like" (spike follows burst)
     * otherwise -> "none"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from .config import PipelineConfig

log = logging.getLogger("vagus.antral")


# ---------------------------------------------------------------------------
# Burst detection
# ---------------------------------------------------------------------------


def detect_bursts(
    slowwave_channels: list[np.ndarray],
    rolling_weights: np.ndarray | None,
    window_centres_s: np.ndarray | None,
    fs: float,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Detect burst onsets per channel, then build the consensus list.

    ``rolling_weights`` and ``window_centres_s`` come from
    :func:`slowwave_quality.build_common_mode` and are used to gate each
    channel's bursts by its rolling-window quality tier.  Pass ``None`` for
    both to skip gating (e.g. when only one channel is present).

    Returns
    -------
    dict with keys ``burst_times_s_per_channel`` (list of 1-D arrays in
    seconds) and ``consensus_burst_times_s`` (1-D array).
    """
    if not slowwave_channels:
        return {
            "burst_times_s_per_channel": [],
            "consensus_burst_times_s": np.zeros(0, dtype=np.float32),
            "n_channels": 0,
        }

    n_samples = slowwave_channels[0].size
    duration_s = n_samples / fs

    per_channel: list[np.ndarray] = []
    for k, ch in enumerate(slowwave_channels):
        onsets = _detect_channel_bursts(ch, fs, cfg)
        # Gate by quality if rolling weights provided.
        if rolling_weights is not None and window_centres_s is not None and rolling_weights.shape[0] > k:
            onsets = _gate_by_quality(onsets, rolling_weights[k], window_centres_s, cfg)
        per_channel.append(onsets)
        log.info("burst detection: channel %d -> %d onsets after quality gating.", k + 1, onsets.size)

    consensus = _consensus_bursts(per_channel, cfg, duration_s)
    log.info("consensus bursts (>= %d channels within %.0f ms): %d.",
             cfg.burst_min_channels, cfg.burst_consensus_window_ms, consensus.size)

    return {
        "burst_times_s_per_channel": per_channel,
        "consensus_burst_times_s": consensus,
        "n_channels": len(slowwave_channels),
    }


def _detect_channel_bursts(ch: np.ndarray, fs: float, cfg: PipelineConfig) -> np.ndarray:
    """Returns burst onset times (seconds) for one channel."""
    if ch.size < int(2 * fs):
        return np.zeros(0, dtype=np.float32)
    nyq = fs / 2.0
    low = max(cfg.burst_band_low_hz / nyq, 1e-6)
    high = min(cfg.burst_band_high_hz / nyq, 0.99)
    if not (0 < low < high < 1):
        return np.zeros(0, dtype=np.float32)
    b, a = butter(2, [low, high], btype="band")
    bp = filtfilt(b, a, ch.astype(np.float64))
    env = np.abs(hilbert(bp))
    mad = np.median(np.abs(env - np.median(env))) / 0.6745
    if mad <= 0:
        mad = np.std(env) + 1e-9
    thr = float(cfg.burst_threshold_sigma * mad + np.median(env))
    above = env > thr
    if not above.any():
        return np.zeros(0, dtype=np.float32)

    # Run-length encode `above`.
    d = np.diff(np.concatenate(([0], above.view(np.int8), [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    min_samples = int(round(cfg.burst_min_duration_ms * 1e-3 * fs))
    valid = (ends - starts) >= min_samples
    onsets_s = (starts[valid] / fs).astype(np.float32)
    return onsets_s


def _gate_by_quality(
    onsets_s: np.ndarray,
    weights_for_channel: np.ndarray,
    window_centres_s: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    """Drop bursts whose nearest rolling window had a zero weight (= bad tier)."""
    if onsets_s.size == 0 or window_centres_s.size == 0:
        return onsets_s
    keep = []
    for t in onsets_s:
        i = int(np.argmin(np.abs(window_centres_s - t)))
        w = float(weights_for_channel[i])
        if w > 0:
            keep.append(t)
    return np.asarray(keep, dtype=np.float32)


def _consensus_bursts(
    per_channel: list[np.ndarray],
    cfg: PipelineConfig,
    duration_s: float,
) -> np.ndarray:
    """For each onset on any channel, check whether at least
    ``burst_min_channels`` channels have an onset within
    ``burst_consensus_window_ms``.  Consensus time = median of contributors.
    """
    all_t = np.concatenate(per_channel) if per_channel else np.zeros(0, dtype=np.float32)
    if all_t.size == 0:
        return all_t
    all_t = np.sort(all_t)
    tol_s = cfg.burst_consensus_window_ms / 1000.0

    # Cluster onsets into groups within tol_s of each other.
    consensus = []
    i = 0
    while i < all_t.size:
        j = i
        while j + 1 < all_t.size and (all_t[j + 1] - all_t[i]) < tol_s:
            j += 1
        group = all_t[i : j + 1]
        # Count how many channels contributed (each onset belongs to exactly one channel).
        contributing_channels = 0
        for ch_onsets in per_channel:
            if ch_onsets.size and np.any((ch_onsets >= group.min()) & (ch_onsets <= group.max())):
                contributing_channels += 1
        if contributing_channels >= cfg.burst_min_channels:
            consensus.append(float(np.median(group)))
        i = j + 1
    consensus_arr = np.asarray(consensus, dtype=np.float32)
    # Enforce a minimum spacing between consecutive consensus times so a single
    # long burst doesn't produce overlapping entries.
    if consensus_arr.size > 1:
        keep = [consensus_arr[0]]
        for t in consensus_arr[1:]:
            if t - keep[-1] >= tol_s:
                keep.append(t)
        consensus_arr = np.asarray(keep, dtype=np.float32)
    return consensus_arr


# ---------------------------------------------------------------------------
# Spike-to-burst cross-correlogram
# ---------------------------------------------------------------------------


def xcorr(
    spike_samples: np.ndarray,
    labels: np.ndarray,
    consensus_burst_times_s: np.ndarray,
    fs: float,
    cfg: PipelineConfig,
) -> list[dict[str, Any]]:
    """Per cluster, compute the spike-to-burst cross-correlogram and tag direction.

    Returns one dict per cluster.
    """
    out: list[dict[str, Any]] = []
    if spike_samples.size == 0 or consensus_burst_times_s.size == 0:
        return out

    bin_s = cfg.burst_xcorr_bin_s
    half_w = cfg.burst_xcorr_window_s
    n_bins_half = int(round(half_w / bin_s))
    lags = np.arange(-n_bins_half, n_bins_half + 1) * bin_s

    unique = sorted({int(l) for l in labels if l >= 0})
    for c in unique:
        sp_s = (spike_samples[labels == c].astype(np.float64) / fs)
        if sp_s.size == 0:
            continue
        # Accumulate around each consensus burst.
        accum = np.zeros(lags.size, dtype=np.float64)
        n_used = 0
        for t_b in consensus_burst_times_s:
            window = (sp_s >= t_b - half_w - bin_s / 2) & (sp_s <= t_b + half_w + bin_s / 2)
            rel = sp_s[window] - t_b
            if rel.size == 0:
                continue
            edges = (lags[:1] - bin_s / 2).tolist() + (lags + bin_s / 2).tolist()
            counts, _ = np.histogram(rel, bins=np.asarray(edges))
            accum += counts
            n_used += 1
        if n_used == 0:
            continue
        xcorr = accum / n_used  # average count per burst per bin
        # Baseline = mean in the outer 50% of the window.
        edge_bins = max(int(0.25 * xcorr.size), 1)
        baseline = np.concatenate([xcorr[:edge_bins], xcorr[-edge_bins:]])
        base_mean = float(np.mean(baseline))
        base_std = float(np.std(baseline)) if baseline.size > 1 else 1e-9
        base_std = max(base_std, 1e-9)
        peak_idx = int(np.argmax(xcorr))
        peak_val = float(xcorr[peak_idx])
        peak_z = (peak_val - base_mean) / base_std
        peak_lag_s = float(lags[peak_idx])

        if peak_z >= cfg.burst_peak_z_threshold and peak_lag_s <= -cfg.burst_direction_min_lag_s:
            tag = "efferent-like"
        elif peak_z >= cfg.burst_peak_z_threshold and peak_lag_s >= cfg.burst_direction_min_lag_s:
            tag = "afferent-like"
        else:
            tag = "none"

        out.append(
            {
                "cluster_id": int(c),
                "xcorr": xcorr.astype(np.float32),
                "lag_axis_s": lags.astype(np.float32),
                "peak_lag_s": peak_lag_s,
                "peak_z": float(peak_z),
                "direction_tag": tag,
                "n_bursts_used": int(n_used),
            }
        )
    return out
