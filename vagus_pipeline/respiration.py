"""Step 10: respiratory verification.

No dedicated EMG channel is available, so the respiratory surrogate is built
from cardiac data: instantaneous heart rate (RSA) and/or the R-wave amplitude
envelope, bandpassed to the rat-relevant respiratory band (0.1–3 Hz by default).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, welch

from .config import PipelineConfig

log = logging.getLogger("vagus.resp")


def resp_surrogate_from_rpeaks(
    rpeak_samples: np.ndarray, n_samples: int, fs: float, cfg: PipelineConfig
) -> np.ndarray:
    """Instantaneous-HR surrogate, bandpassed to the resp band.

    Returns a 1-D array of length ``n_samples``.
    """
    if rpeak_samples.size < 4:
        return np.zeros(n_samples, dtype=np.float32)
    rp = np.sort(rpeak_samples.astype(np.float64))
    rr = np.diff(rp) / fs  # seconds
    inst_hr = 1.0 / np.clip(rr, 1e-3, None)
    t_hr = rp[1:] / fs
    t_all = np.arange(n_samples) / fs
    hr_interp = np.interp(t_all, t_hr, inst_hr, left=inst_hr[0], right=inst_hr[-1])
    # Bandpass 0.1–3 Hz
    nyq = fs / 2.0
    low = cfg.resp_band_low_hz / nyq
    high = min(cfg.resp_band_high_hz, 0.99 * nyq) / nyq
    b, a = butter(2, [low, high], btype="band")
    surrogate = filtfilt(b, a, hr_interp).astype(np.float32)
    return surrogate


def verify(
    spike_samples: np.ndarray,
    labels: np.ndarray,
    resp_surrogate: np.ndarray,
    stim_events: list[tuple[int, str]] | None,
    fs: float,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Compute burst-rate periodicity, breathing rate, and compare across stim conditions.

    Returns
    -------
    dict with keys ``burst_rate_hz``, ``breathing_rate_hz``, ``rate_matches``,
    ``breathing_rate_by_condition``, ``stable_across_conditions``.
    """
    out: dict[str, Any] = {
        "burst_rate_hz": float("nan"),
        "breathing_rate_hz": float("nan"),
        "rate_matches": False,
        "breathing_rate_by_condition": {},
        "stable_across_conditions": True,
    }
    if spike_samples.size < 10 or resp_surrogate.size == 0:
        return out

    # Aggregate burst rate over all clusters (population-level)
    bin_s = 0.1
    n_bins = int(resp_surrogate.size / fs / bin_s)
    if n_bins < 8:
        return out
    edges = np.arange(n_bins + 1) * bin_s * fs
    counts, _ = np.histogram(spike_samples, bins=edges)
    rate_signal = counts.astype(np.float64) / bin_s

    fs_rate = 1.0 / bin_s
    try:
        f, pxx = welch(rate_signal - rate_signal.mean(), fs=fs_rate, nperseg=min(256, rate_signal.size))
        band = (f >= cfg.resp_band_low_hz) & (f <= min(cfg.resp_band_high_hz, fs_rate / 2 - 1e-6))
        if band.any():
            burst_rate = float(f[band][int(np.argmax(pxx[band]))])
        else:
            burst_rate = float("nan")
    except Exception:
        burst_rate = float("nan")

    # Breathing rate from resp surrogate (decimate to manageable sample rate)
    decim = max(int(round(fs / 200.0)), 1)
    surr = resp_surrogate[::decim]
    fs_surr = fs / decim
    try:
        f2, pxx2 = welch(surr - surr.mean(), fs=fs_surr, nperseg=min(2048, surr.size))
        band2 = (f2 >= cfg.resp_band_low_hz) & (f2 <= cfg.resp_band_high_hz)
        if band2.any():
            breathing_rate = float(f2[band2][int(np.argmax(pxx2[band2]))])
        else:
            breathing_rate = float("nan")
    except Exception:
        breathing_rate = float("nan")

    out["burst_rate_hz"] = burst_rate
    out["breathing_rate_hz"] = breathing_rate
    if np.isfinite(burst_rate) and np.isfinite(breathing_rate):
        out["rate_matches"] = bool(abs(burst_rate - breathing_rate) / max(breathing_rate, 1e-6) < 0.15)

    # Cross-condition breathing rate stability
    if stim_events:
        labels_per_event = sorted({l for _, l in stim_events})
        # Use ±10 s windows around each event for the by-condition surrogate
        win_s = 10.0
        rates: dict[str, list[float]] = {}
        for s, lbl in stim_events:
            t0 = max(0, s - int(win_s * fs))
            t1 = min(resp_surrogate.size, s + int(win_s * fs))
            seg = resp_surrogate[t0:t1]
            if seg.size < int(2 * fs):
                continue
            try:
                f3, pxx3 = welch(seg - seg.mean(), fs=fs, nperseg=min(8192, seg.size))
                band3 = (f3 >= cfg.resp_band_low_hz) & (f3 <= cfg.resp_band_high_hz)
                if band3.any():
                    rates.setdefault(lbl, []).append(float(f3[band3][int(np.argmax(pxx3[band3]))]))
            except Exception:
                continue
        per_cond = {k: float(np.mean(v)) for k, v in rates.items() if v}
        out["breathing_rate_by_condition"] = per_cond
        if len(per_cond) >= 2:
            vals = list(per_cond.values())
            spread = (max(vals) - min(vals)) / max(np.mean(vals), 1e-6)
            out["stable_across_conditions"] = bool(spread < 0.20)
    return out
