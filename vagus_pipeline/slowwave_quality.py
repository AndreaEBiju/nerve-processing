"""Step 11a — per-channel slow-wave quality scoring and common-mode reference.

Per spec §5 (the docx companion's Step 11.1-11.3):

Three antral mucosal full-thickness electrodes are placed 0.5 cm apart along
the gastric long axis (proximal -> middle -> distal).  Any combination of
the three may be unusable at any moment due to electrode dropout, motion
artefact, contact loss, etc.  This module:

1. Scores each channel on four diagnostics
   (`in_band_snr`, `peak_prominence`, `pairwise_coherence`, `envelope_cv`)
   per spec §5 Step 11a.  Each diagnostic is normalized to [0, 1] using the
   ``sw_*_norm_lo`` / ``sw_*_norm_hi`` config ranges, then combined via
   geometric mean into a single ``quality_score``.
2. Computes both whole-recording and rolling-window (sw_quality_window_s
   with sw_quality_overlap_s overlap) scores.
3. Applies the tiered inclusion rule:
     score >= sw_quality_good_threshold      -> "good"
     score >= sw_quality_marginal_threshold  -> "marginal"
     otherwise                                -> "bad" (excluded)
4. Builds a propagation-lag-corrected common-mode reference: estimate the
   phase lag between each surviving channel and the middle channel at the
   dominant slow-wave frequency, phase-align with a fractional-sample
   shift, and average the aligned channels weighted by their per-window
   quality scores.

When no channel reaches the marginal tier *anywhere* in the recording,
raise :class:`vagus_pipeline.config.SlowWaveUnusable` so the orchestrator
can skip Steps 11-12 and continue with the other steps.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import butter, coherence, csd, filtfilt, hilbert, sosfiltfilt, welch

from .config import PipelineConfig, SlowWaveUnusable

# Slow waves are at ~0.05-0.2 Hz; processing at the neural rate (24 kHz)
# produces numerically unstable narrow Butterworth filters.  Internally
# every per-channel computation downsamples to this rate.  Storage and
# the per-spike phase extraction in Step 11b still happen at the neural
# rate via FFT-based interpolation of the (decimated) common-mode.
_INTERNAL_SW_FS_HZ = 50.0

log = logging.getLogger("vagus.slowwave_quality")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ChannelQuality:
    snr_inband: float
    peak_prominence: float
    pairwise_coherence: float  # mean coherence with the OTHER surviving channels
    envelope_cv: float
    quality_score_whole: float
    quality_score_rolling: np.ndarray  # per-window
    quality_window_times_s: np.ndarray  # window CENTRES, seconds
    tier: str  # "good" | "marginal" | "bad"
    reason_if_excluded: str = ""


@dataclass
class CommonModeResult:
    common_mode: np.ndarray              # at config.sw_common_mode_fs_hz
    common_mode_fs_hz: float
    rolling_weights: np.ndarray          # n_channels x n_windows
    n_channels_contributing: np.ndarray  # per common-mode sample
    propagation_lag_s_between_adjacent: np.ndarray  # length n_channels - 1


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------


def score_channels(
    slowwave_channels: list[np.ndarray],
    fs: float,
    cfg: PipelineConfig,
) -> list[ChannelQuality]:
    """Score every available slow-wave channel.

    Returns one :class:`ChannelQuality` per input channel (in spatial order).
    """
    if not slowwave_channels:
        return []

    n_samples = slowwave_channels[0].size
    duration_s = n_samples / fs

    # Roll the four diagnostics over fixed windows.
    win_s = max(min(cfg.sw_quality_window_s, duration_s), 4.0)
    overlap_s = max(0.0, min(cfg.sw_quality_overlap_s, win_s - 1.0))
    win_n = int(round(win_s * fs))
    step_n = max(int(round((win_s - overlap_s) * fs)), 1)
    starts = list(range(0, max(n_samples - win_n + 1, 1), step_n))
    if starts[-1] + win_n < n_samples and (n_samples - win_n) not in starts:
        starts.append(n_samples - win_n)
    window_centres_s = np.asarray([(s + win_n / 2) / fs for s in starts], dtype=np.float32)

    qualities: list[ChannelQuality] = []
    for k, ch in enumerate(slowwave_channels):
        snr = _in_band_snr(ch, fs, cfg)
        prom = _peak_prominence(ch, fs, cfg)
        coh = _mean_pairwise_coherence(ch, slowwave_channels, k, fs, cfg)
        env_cv = _envelope_cv(ch, fs, cfg)

        rolling = np.empty(len(starts), dtype=np.float32)
        for w_i, s0 in enumerate(starts):
            seg = ch[s0 : s0 + win_n]
            others = [c[s0 : s0 + win_n] for j, c in enumerate(slowwave_channels) if j != k]
            rolling[w_i] = _combine_score(
                _in_band_snr(seg, fs, cfg),
                _peak_prominence(seg, fs, cfg),
                _pairwise_coherence_seg(seg, others, fs, cfg),
                _envelope_cv(seg, fs, cfg),
                cfg,
            )

        whole = _combine_score(snr, prom, coh, env_cv, cfg)
        tier, reason = _tier(whole, cfg)
        qualities.append(
            ChannelQuality(
                snr_inband=float(snr),
                peak_prominence=float(prom),
                pairwise_coherence=float(coh),
                envelope_cv=float(env_cv),
                quality_score_whole=float(whole),
                quality_score_rolling=rolling,
                quality_window_times_s=window_centres_s,
                tier=tier,
                reason_if_excluded=reason,
            )
        )

    return qualities


# ---------------------------------------------------------------------------
# Common-mode reference with propagation-lag correction
# ---------------------------------------------------------------------------


def build_common_mode(
    slowwave_channels: list[np.ndarray],
    qualities: list[ChannelQuality],
    fs: float,
    cfg: PipelineConfig,
) -> CommonModeResult:
    """Construct the propagation-lag-corrected common-mode slow-wave reference.

    Phase 1: per window, drop channels whose rolling tier is "bad".  Channels
    in the "marginal" tier get a weight equal to their rolling quality score
    (so they downweight); "good" channels get their score directly.  Weights
    are renormalised to sum to 1 across surviving channels per window.

    Phase 2: each surviving channel is phase-aligned to the middle channel by
    a fractional-sample delay (Welch CSD between channels at the dominant
    slow-wave frequency gives the lag; ``scipy.signal.resample`` does the
    fractional shift).  Aligned channels are summed weighted by the per-window
    weights.

    Phase 3: the per-window common-mode pieces are concatenated and downsampled
    to ``cfg.sw_common_mode_fs_hz`` for storage.  Per-window propagation lags
    (between adjacent surviving channels) are averaged across windows.
    """
    if not slowwave_channels:
        raise SlowWaveUnusable("No slow-wave channels provided.")

    n_samples = slowwave_channels[0].size
    n_ch = len(slowwave_channels)
    rolling_w = np.zeros((n_ch, len(qualities[0].quality_window_times_s)), dtype=np.float32)
    for k, q in enumerate(qualities):
        for w_i, score in enumerate(q.quality_score_rolling):
            if score >= cfg.sw_quality_good_threshold:
                rolling_w[k, w_i] = score
            elif score >= cfg.sw_quality_marginal_threshold:
                rolling_w[k, w_i] = score  # downweighted but kept
            else:
                rolling_w[k, w_i] = 0.0

    # If literally every weight is zero, refuse.
    if not np.any(rolling_w > 0):
        raise SlowWaveUnusable(
            "No slow-wave channel reaches the marginal quality tier in any window. "
            "Step 11 and Step 12 will be skipped for this recording."
        )

    # Determine which channel is "middle" by spatial order (caller passes the
    # channels already in proximal->distal order).  Middle index is len/2.
    middle_k = n_ch // 2

    # Whole-recording propagation lag estimate between adjacent channels.
    prop_lags = np.zeros(max(n_ch - 1, 0), dtype=np.float32)
    for k in range(n_ch - 1):
        prop_lags[k] = _propagation_lag_s(slowwave_channels[k], slowwave_channels[k + 1], fs, cfg)

    # Apply a static fractional-sample delay to each channel.  Downsampled
    # so the FFT-based shift doesn't choke on multi-MSample arrays.
    aligned_ds: list[np.ndarray] = []
    ds_channels: list[np.ndarray] = []
    fs_ds = fs
    for ch in slowwave_channels:
        c_ds, fs_ds = _downsample(ch, fs)
        ds_channels.append(c_ds)
    for k in range(n_ch):
        if k == middle_k:
            aligned_ds.append(ds_channels[k])
            continue
        if k < middle_k:
            lag_s = float(np.sum(prop_lags[k:middle_k]))
        else:
            lag_s = float(-np.sum(prop_lags[middle_k:k]))
        aligned_ds.append(_fractional_delay(ds_channels[k], lag_s, fs_ds))
    n_samples_ds = ds_channels[0].size

    # Per-window weighted sum, at the DOWNSAMPLED rate.
    win_s = max(min(cfg.sw_quality_window_s, n_samples_ds / fs_ds), 4.0)
    overlap_s = max(0.0, min(cfg.sw_quality_overlap_s, win_s - 1.0))
    win_n = int(round(win_s * fs_ds))
    step_n = max(int(round((win_s - overlap_s) * fs_ds)), 1)
    starts = list(range(0, max(n_samples_ds - win_n + 1, 1), step_n))
    if starts[-1] + win_n < n_samples_ds and (n_samples_ds - win_n) not in starts:
        starts.append(n_samples_ds - win_n)

    common_full = np.zeros(n_samples_ds, dtype=np.float32)
    weight_full = np.zeros(n_samples_ds, dtype=np.float32)
    n_contrib_full = np.zeros(n_samples_ds, dtype=np.uint8)
    for w_i, s0 in enumerate(starts):
        s1 = min(s0 + win_n, n_samples_ds)
        ws = rolling_w[:, w_i] if w_i < rolling_w.shape[1] else np.zeros(n_ch)
        ws_norm = ws / ws.sum() if ws.sum() > 0 else ws
        if ws.sum() <= 0:
            continue
        for k in range(n_ch):
            if ws_norm[k] <= 0:
                continue
            common_full[s0:s1] += aligned_ds[k][s0:s1] * ws_norm[k]
            weight_full[s0:s1] += ws_norm[k]
            n_contrib_full[s0:s1] += 1

    # Normalise overlapping windows.
    nz = weight_full > 0
    common_full[nz] /= np.maximum(weight_full[nz], 1e-6)

    # Downsample to the storage rate.
    decim = max(int(round(fs_ds / cfg.sw_common_mode_fs_hz)), 1)
    common_ds = common_full[::decim].copy()
    n_contrib_ds = n_contrib_full[::decim].copy()

    log.info(
        "Common-mode reference: %d windows, propagation lags between adjacent "
        "channels = %s s (proximal -> distal); stored at %.2f Hz.",
        len(starts), [f"{x:.3f}" for x in prop_lags], cfg.sw_common_mode_fs_hz,
    )

    return CommonModeResult(
        common_mode=common_ds,
        common_mode_fs_hz=float(cfg.sw_common_mode_fs_hz),
        rolling_weights=rolling_w,
        n_channels_contributing=n_contrib_ds,
        propagation_lag_s_between_adjacent=prop_lags,
    )


def expand_common_mode(common_mode_ds: np.ndarray, target_fs: float, n_samples: int, cm_fs: float) -> np.ndarray:
    """Linearly interpolate the downsampled common-mode back to ``target_fs``
    so per-spike phase extraction can sample it.  Step 11b uses this.
    """
    if common_mode_ds.size == 0:
        return np.zeros(n_samples, dtype=np.float32)
    t_old = np.arange(common_mode_ds.size, dtype=np.float64) / cm_fs
    t_new = np.arange(n_samples, dtype=np.float64) / target_fs
    return np.interp(t_new, t_old, common_mode_ds).astype(np.float32)


def serialize_quality_summary(qualities: list[ChannelQuality]) -> str:
    """JSON summary saved to ``step11a.summary_text`` for human / MATLAB
    inspection."""
    return json.dumps([
        {
            "tier": q.tier,
            "score": round(q.quality_score_whole, 4),
            "snr_inband": round(q.snr_inband, 4),
            "peak_prominence": round(q.peak_prominence, 4),
            "pairwise_coherence": round(q.pairwise_coherence, 4),
            "envelope_cv": round(q.envelope_cv, 4),
            "reason_if_excluded": q.reason_if_excluded,
        }
        for q in qualities
    ])


# ---------------------------------------------------------------------------
# Diagnostic computations
# ---------------------------------------------------------------------------


def _in_band_snr(ch: np.ndarray, fs: float, cfg: PipelineConfig) -> float:
    ch_ds, fs_ds = _downsample(ch, fs)
    if ch_ds.size < int(4 * fs_ds):
        return 0.0
    nperseg = min(int(64 * fs_ds), ch_ds.size // 2 + 1)
    try:
        f, p = welch(ch_ds - ch_ds.mean(), fs=fs_ds, nperseg=nperseg)
    except Exception:
        return 0.0
    in_band = (f >= cfg.sw_inband_low_hz) & (f <= cfg.sw_inband_high_hz)
    out_band = (
        ((f >= cfg.sw_outband_low1_hz) & (f <= cfg.sw_outband_high1_hz))
        | ((f >= cfg.sw_outband_low2_hz) & (f <= cfg.sw_outband_high2_hz))
    )
    p_in = p[in_band].sum() if in_band.any() else 0.0
    p_out = p[out_band].sum() if out_band.any() else 1e-12
    return float(p_in / max(p_out, 1e-12))


def _peak_prominence(ch: np.ndarray, fs: float, cfg: PipelineConfig) -> float:
    """Fraction of total variance that sits in the slow-wave band.

    Defined as ``std(bandpass(ch)) / std(ch)``.  Bounded in [0, 1].
    For a clean band-limited signal this is close to 1; for a noise-dominated
    channel it is close to 0.  Insensitive to absolute amplitude.
    """
    try:
        ch_ds, fs_ds = _downsample(ch, fs)
        bp = _bandpass(ch_ds, cfg.sw_inband_low_hz, cfg.sw_inband_high_hz, fs_ds)
        denom = float(np.std(ch_ds))
        if denom <= 0:
            return 0.0
        return float(min(np.std(bp) / denom, 1.0))
    except Exception:
        return 0.0


def _mean_pairwise_coherence(
    ch: np.ndarray,
    all_channels: list[np.ndarray],
    own_index: int,
    fs: float,
    cfg: PipelineConfig,
) -> float:
    others = [c for i, c in enumerate(all_channels) if i != own_index]
    return _pairwise_coherence_seg(ch, others, fs, cfg)


def _pairwise_coherence_seg(
    ch: np.ndarray,
    others: list[np.ndarray],
    fs: float,
    cfg: PipelineConfig,
) -> float:
    if not others:
        return 1.0  # single channel -> no penalty
    ch_ds, fs_ds = _downsample(ch, fs)
    if ch_ds.size < int(4 * fs_ds):
        return 1.0
    nperseg = min(int(64 * fs_ds), ch_ds.size // 2 + 1)
    scores = []
    for o in others:
        if o.size != ch.size:
            continue
        o_ds, _ = _downsample(o, fs)
        if o_ds.size != ch_ds.size:
            continue
        try:
            f, Cxy = coherence(ch_ds - ch_ds.mean(), o_ds - o_ds.mean(), fs=fs_ds, nperseg=nperseg)
            band = (f >= cfg.sw_inband_low_hz) & (f <= cfg.sw_inband_high_hz)
            if band.any():
                scores.append(float(Cxy[band].mean()))
        except Exception:
            continue
    return float(np.mean(scores)) if scores else 1.0


def _envelope_cv(ch: np.ndarray, fs: float, cfg: PipelineConfig) -> float:
    try:
        ch_ds, fs_ds = _downsample(ch, fs)
        bp = _bandpass(ch_ds, cfg.sw_inband_low_hz, cfg.sw_inband_high_hz, fs_ds)
        env = np.abs(hilbert(bp))
        mu = env.mean()
        if mu <= 0:
            return float("inf")
        return float(env.std() / mu)
    except Exception:
        return float("inf")


def _propagation_lag_s(a: np.ndarray, b: np.ndarray, fs: float, cfg: PipelineConfig) -> float:
    """Phase-lag at the dominant slow-wave frequency, from Welch CSD."""
    if a.size != b.size:
        return 0.0
    a_ds, fs_ds = _downsample(a, fs)
    b_ds, _ = _downsample(b, fs)
    if a_ds.size < int(4 * fs_ds):
        return 0.0
    nperseg = min(int(64 * fs_ds), a_ds.size // 2 + 1)
    try:
        f, Pxx = welch(a_ds - a_ds.mean(), fs=fs_ds, nperseg=nperseg)
        band = (f >= cfg.sw_inband_low_hz) & (f <= cfg.sw_inband_high_hz)
        if not band.any():
            return 0.0
        dom_f = f[band][int(np.argmax(Pxx[band]))]
        f2, Cxy = csd(a_ds - a_ds.mean(), b_ds - b_ds.mean(), fs=fs_ds, nperseg=nperseg)
        if not f2.size:
            return 0.0
        i = int(np.argmin(np.abs(f2 - dom_f)))
        phase = np.angle(Cxy[i])
        return float(phase / (2 * np.pi * max(dom_f, 1e-6)))
    except Exception:
        return 0.0


def _fractional_delay(x: np.ndarray, lag_s: float, fs: float) -> np.ndarray:
    """Apply a real-valued time delay (seconds) to ``x``.  Positive lag means
    ``x`` is advanced by ``lag_s`` (i.e. shifted earlier so it aligns with
    something that arrives ``lag_s`` later).
    """
    if abs(lag_s) < 1e-9:
        return x.astype(np.float32, copy=False)
    n = x.size
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.fft.rfft(x.astype(np.float64))
    shift = np.exp(-2j * np.pi * freqs * lag_s)
    return np.fft.irfft(spec * shift, n=n).astype(np.float32)


def _downsample(x: np.ndarray, fs: float, target_fs: float = _INTERNAL_SW_FS_HZ) -> tuple[np.ndarray, float]:
    """Downsample ``x`` (no anti-alias for the slow-wave case is fine since
    we filter below) to ``target_fs``.  Returns ``(y, actual_fs)``."""
    if fs <= target_fs * 1.2:
        return x.astype(np.float32, copy=False), float(fs)
    step = max(int(round(fs / target_fs)), 1)
    return x[::step].astype(np.float32, copy=False), float(fs / step)


def _bandpass(x: np.ndarray, low: float, high: float, fs: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth bandpass via SOS (numerically stable for narrow
    bands)."""
    nyq = fs / 2.0
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.99)
    if not (0 < low_n < high_n < 1):
        return x.astype(np.float32, copy=False)
    sos = butter(order, [low_n, high_n], btype="band", output="sos")
    return sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)


def _combine_score(snr: float, prom: float, coh: float, env_cv: float, cfg: PipelineConfig) -> float:
    s_snr = _norm_clip(snr, cfg.sw_snr_norm_lo, cfg.sw_snr_norm_hi)
    # Prominence is a variance fraction in [0,1]; a clean slow-wave channel
    # typically has > 0.3 of its variance in-band, marginal channels around
    # 0.1, dropout / noise channels < 0.05.
    s_prom = _norm_clip(prom, 0.05, 0.5)
    s_coh = _norm_clip(coh, cfg.sw_coherence_norm_lo, cfg.sw_coherence_norm_hi)
    # env_cv is INVERTED: low CV -> high score
    s_env = _norm_clip(env_cv, cfg.sw_envcv_norm_hi, cfg.sw_envcv_norm_lo, inverted=True)
    eps = 1e-4
    components = np.asarray([s_snr, s_prom, s_coh, s_env], dtype=np.float64)
    return float(np.exp(np.mean(np.log(components + eps))))


def _norm_clip(x: float, lo: float, hi: float, inverted: bool = False) -> float:
    if not np.isfinite(x):
        return 0.0
    if hi == lo:
        return 0.0
    if inverted:
        # caller passes (hi=good_threshold, lo=bad_threshold) for clarity
        if x <= lo:
            return 1.0
        if x >= hi:
            return 0.0
        return float((hi - x) / (hi - lo))
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return float((x - lo) / (hi - lo))


def _tier(score: float, cfg: PipelineConfig) -> tuple[str, str]:
    if score >= cfg.sw_quality_good_threshold:
        return "good", ""
    if score >= cfg.sw_quality_marginal_threshold:
        return "marginal", ""
    return "bad", f"quality_score={score:.3f} < marginal threshold {cfg.sw_quality_marginal_threshold:.2f}"
