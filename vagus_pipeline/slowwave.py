"""Step 11b -- per-spike slow-wave phase tagging + consistency check.

Phase is extracted from the propagation-lag-corrected common-mode reference
(produced by :mod:`slowwave_quality`) and from each individual surviving
channel.  Per-spike output has 4 columns ``[common, ch1, ch2, ch3]``; the
column for any absent or dropped channel is NaN.

Per cluster, primary stats (MRL, Rayleigh p, preferred phase) are computed
on the common-mode column; the per-channel columns give the inputs for the
consistency check:

    consistency_score = circular variance of the three per-channel
                        preferred phases (after the common-mode
                        propagation-lag correction has already aligned
                        each channel to the middle)

    robust_phase_locked = True  iff
        primary Rayleigh p < cfg.sw_primary_p
      AND consistency_score below the threshold derived from
          cfg.sw_consistency_phase_spread_deg
      AND >= 2 of the per-channel Rayleigh p < cfg.sw_per_channel_p_near
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from .config import PipelineConfig

# Downsample slow-wave inputs to this rate before bandpass + Hilbert; the
# resulting per-sample phase is then upsampled back to the neural rate.
_INTERNAL_SW_FS_HZ = 50.0

log = logging.getLogger("vagus.slowwave")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def phase_tag(
    spike_samples: np.ndarray,
    labels: np.ndarray,
    common_mode_full_fs: np.ndarray | None,
    slowwave_channels: list[np.ndarray],
    fs: float,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Compute per-spike phases and per-cluster phase-locking stats.

    Parameters
    ----------
    spike_samples, labels
        As elsewhere; one entry per spike.
    common_mode_full_fs
        The common-mode reference *expanded back to the neural sample rate*
        (use :func:`slowwave_quality.expand_common_mode`).  May be ``None``
        when Step 11a was skipped or the recording has only one channel; in
        that case the first surviving channel is used as the reference.
    slowwave_channels
        List of up to 3 1-D arrays at the neural sampling rate, in spatial
        order proximal -> middle -> distal.  Pass an empty list when no
        slow wave is available; the output reflects that with empty cluster
        results.

    Returns
    -------
    dict with keys
        ``sw_phase_per_spike`` -- N x 4 array (common, ch1, ch2, ch3)
        ``cluster``            -- list of per-cluster stats dicts
    """
    out: dict[str, Any] = {
        "sw_phase_per_spike": np.full((max(spike_samples.size, 0), 4), np.nan, dtype=np.float32),
        "cluster": [],
        "skipped": False,
    }

    n_channels = len(slowwave_channels)
    if n_channels == 0:
        out["skipped"] = True
        out["skipped_reason"] = "no slow-wave channels"
        return out
    if spike_samples.size == 0:
        return out

    # Choose primary reference for the "common" column.
    if common_mode_full_fs is not None and common_mode_full_fs.size:
        primary_signal = common_mode_full_fs
        primary_label = "common-mode"
    else:
        # No common-mode (degraded mode -- 0 or 1 usable channel after
        # quality scoring).  Use the first channel as the reference.
        primary_signal = slowwave_channels[0]
        primary_label = "slow-wave channel 1 (degraded mode)"
        log.warning("phase_tag: no common-mode reference; using %s as primary.", primary_label)

    # Extract per-sample phase from primary and each per-channel signal.
    phase_primary = _instantaneous_phase(primary_signal, fs, cfg)
    per_ch_phase = [
        _instantaneous_phase(c, fs, cfg) for c in slowwave_channels
    ]

    sp = spike_samples.astype(np.int64)
    sp_clipped = np.clip(sp, 0, phase_primary.size - 1)
    sw_phase = np.full((sp.size, 4), np.nan, dtype=np.float32)
    sw_phase[:, 0] = phase_primary[sp_clipped]
    for k in range(min(n_channels, 3)):
        sw_phase[:, k + 1] = per_ch_phase[k][np.clip(sp, 0, per_ch_phase[k].size - 1)]
    out["sw_phase_per_spike"] = sw_phase

    # Per-cluster stats.
    consistency_thr = _phase_spread_to_circular_variance(cfg.sw_consistency_phase_spread_deg)
    out["consistency_circular_variance_threshold"] = float(consistency_thr)
    unique = sorted({int(l) for l in labels if l >= 0})
    cluster_stats: list[dict[str, Any]] = []
    for c in unique:
        mask = labels == c
        if not mask.any():
            continue
        cl: dict[str, Any] = {"cluster_id": int(c)}

        # Primary (common-mode) column
        p_common = sw_phase[mask, 0]
        p_common = p_common[np.isfinite(p_common)]
        mrl_c, p_c, pref_c = _rayleigh(p_common)
        cl["mrl_common"] = float(mrl_c)
        cl["rayleigh_p_common"] = float(p_c)
        cl["preferred_phase_common"] = float(pref_c)
        hist_c, edges_c = np.histogram(p_common, bins=np.linspace(-np.pi, np.pi, 25))
        cl["phase_hist_common"] = hist_c.astype(np.int64)
        cl["phase_edges_common"] = edges_c.astype(np.float32)

        # Per-channel columns
        mrl_pc = np.full(3, np.nan, dtype=np.float32)
        p_pc = np.full(3, np.nan, dtype=np.float32)
        pref_pc = np.full(3, np.nan, dtype=np.float32)
        for k in range(3):
            p_col = sw_phase[mask, k + 1]
            p_col = p_col[np.isfinite(p_col)]
            if p_col.size > 1:
                m, pv, pr = _rayleigh(p_col)
                mrl_pc[k] = m
                p_pc[k] = pv
                pref_pc[k] = pr
        cl["mrl_per_channel"] = mrl_pc
        cl["rayleigh_p_per_channel"] = p_pc
        cl["preferred_phase_per_channel"] = pref_pc

        # Consistency check across surviving channels.
        valid_pref = pref_pc[np.isfinite(pref_pc)]
        if valid_pref.size >= 2:
            R = np.abs(np.mean(np.exp(1j * valid_pref.astype(np.float64))))
            cons = float(1.0 - R)
        else:
            cons = float("nan")
        cl["consistency_score"] = cons

        n_near = int(np.sum(p_pc < cfg.sw_per_channel_p_near))  # NaN comparisons -> False
        primary_sig = bool(p_c < cfg.sw_primary_p) and np.isfinite(p_c)
        consistent = bool(np.isfinite(cons) and cons < consistency_thr)
        cl["robust_phase_locked"] = bool(primary_sig and consistent and n_near >= 2)
        cluster_stats.append(cl)

    out["cluster"] = cluster_stats
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _instantaneous_phase(x: np.ndarray, fs: float, cfg: PipelineConfig) -> np.ndarray:
    """Bandpass to the slow-wave band + Hilbert phase, evaluated at fs.

    Internally decimates ``x`` to ~50 Hz before the bandpass (Butterworth
    cutoffs near 0.05 Hz at fs=24414 Hz produce numerically unstable
    filters), computes the analytic-signal phase, then linearly interpolates
    the unwrapped phase back to ``fs`` so per-spike lookups stay sample-accurate.
    """
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    step = max(int(round(fs / _INTERNAL_SW_FS_HZ)), 1) if fs > _INTERNAL_SW_FS_HZ * 1.2 else 1
    x_ds = x[::step].astype(np.float32, copy=False) if step > 1 else x
    fs_ds = float(fs / step)
    nyq = fs_ds / 2.0
    low = max(cfg.sw_low_hz / nyq, 1e-6)
    high = min(cfg.sw_high_hz / nyq, 0.99)
    if not (0 < low < high < 1):
        return np.zeros(n, dtype=np.float32)
    sos = butter(2, [low, high], btype="band", output="sos")
    bp = sosfiltfilt(sos, x_ds.astype(np.float64))
    phase_unwrap_ds = np.unwrap(np.angle(hilbert(bp))).astype(np.float64)
    if step == 1:
        return _wrap(phase_unwrap_ds.astype(np.float32))
    t_ds = np.arange(phase_unwrap_ds.size, dtype=np.float64) / fs_ds
    t_full = np.arange(n, dtype=np.float64) / fs
    phase_full = np.interp(t_full, t_ds, phase_unwrap_ds).astype(np.float32)
    return _wrap(phase_full)


def _wrap(p: np.ndarray) -> np.ndarray:
    return ((p + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)


def _rayleigh(phases: np.ndarray) -> tuple[float, float, float]:
    """Return (MRL, Rayleigh p, preferred phase).  Uses pingouin if
    available, otherwise the exp(-Z) approximation."""
    phases = np.asarray(phases, dtype=np.float64)
    if phases.size < 2:
        return float("nan"), float("nan"), float("nan")
    z = np.exp(1j * phases)
    mean_z = z.mean()
    mrl = float(np.abs(mean_z))
    pref = float(np.angle(mean_z))
    n = phases.size
    Z = n * mrl * mrl
    try:
        from pingouin import circ_rayleigh
        _, pval = circ_rayleigh(phases)
        rayleigh_p = float(min(max(pval, 0.0), 1.0))
    except Exception:
        rayleigh_p = float(min(max(np.exp(-Z), 0.0), 1.0))
    return mrl, rayleigh_p, pref


def _phase_spread_to_circular_variance(spread_deg: float) -> float:
    """Convert a phase-spread tolerance (degrees) to a circular-variance
    threshold suitable for the consistency check.

    Approximation: for three equally spaced phases spanning +/-X degrees the
    circular variance is approximately ``1 - sinc(2*X*pi/180)``.  We use a
    slightly more permissive form by sampling N=3 phases uniformly across
    the spread and computing R numerically.
    """
    half = np.deg2rad(spread_deg) / 2.0
    phases = np.array([-half, 0.0, half])
    R = np.abs(np.mean(np.exp(1j * phases)))
    return float(1.0 - R)
