"""Step 1 (bandpass) and Step 2 (running noise sigma).

The signals arriving here are assumed to be notch-filtered and
motion-blanked already, per the spec.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt

from .config import PipelineConfig

log = logging.getLogger("vagus.preprocess")


def bandpass(x: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Zero-phase Butterworth bandpass.

    Designed in second-order sections through ``butter`` with ``filtfilt`` for
    zero phase distortion. Skips filtering on NaN-containing input by filling
    with zeros (the blanked-mask is carried separately).
    """
    fs = cfg.fs
    nyq = fs / 2.0
    low = cfg.bp_low_hz / nyq
    high = min(cfg.bp_high_hz, 0.99 * nyq) / nyq
    if not (0 < low < high < 1):
        raise ValueError(
            f"Bandpass corners out of range: low={cfg.bp_low_hz}, high={cfg.bp_high_hz}, fs={fs}"
        )
    b, a = butter(cfg.bp_order, [low, high], btype="band")

    y = x.astype(np.float64, copy=True)
    bad = ~np.isfinite(y)
    if bad.any():
        y[bad] = 0.0
    out = filtfilt(b, a, y, axis=-1).astype(np.float32)
    out[bad] = 0.0
    return out


def noise_sigma(
    x: np.ndarray, cfg: PipelineConfig, blanked_mask: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding median-absolute-deviation noise estimate.

    Returns
    -------
    sigma_track : 1-D array of sigma values at each window centre.
    sigma_times : 1-D array of sample indices corresponding to ``sigma_track``.
    """
    fs = cfg.fs
    win = max(int(round(cfg.noise_window_s * fs)), 1)
    n = x.size
    if win >= n:
        # one window
        s = _mad_sigma(x if blanked_mask is None else x[~blanked_mask])
        return np.asarray([s], dtype=np.float32), np.asarray([n // 2], dtype=np.int64)

    starts = np.arange(0, n - win + 1, win)
    centres = starts + win // 2
    sigmas = np.empty(starts.size, dtype=np.float32)
    for i, s0 in enumerate(starts):
        seg = x[s0 : s0 + win]
        if blanked_mask is not None:
            seg = seg[~blanked_mask[s0 : s0 + win]]
        if seg.size < 16:
            sigmas[i] = np.nan
        else:
            sigmas[i] = _mad_sigma(seg)
    # forward-fill NaNs
    if np.isnan(sigmas).any():
        idx = np.where(~np.isnan(sigmas))[0]
        if idx.size == 0:
            sigmas[:] = float(_mad_sigma(x))
        else:
            sigmas = np.interp(np.arange(sigmas.size), idx, sigmas[idx])
    return sigmas, centres.astype(np.int64)


def sigma_at(samples: np.ndarray, sigma_track: np.ndarray, sigma_times: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``sigma_track`` to arbitrary sample positions."""
    if sigma_track.size == 1:
        return np.full(samples.shape, float(sigma_track[0]), dtype=np.float32)
    return np.interp(samples, sigma_times, sigma_track).astype(np.float32)


def _mad_sigma(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(mad / 0.6745) if mad > 0 else float(np.std(x) + 1e-12)
