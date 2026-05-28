"""Step 9: peri-R-wave histogram and cardiac-locked flagging.

For each cluster, builds a histogram of spike times relative to nearby
R-peaks within ``±prwh_window_ms``. A cluster is flagged as cardiac-locked
when its peak count within ``±cardiac_lock_window_ms`` exceeds ``cardiac_peak_z``
z-scores over the baseline (bins outside the narrow lock window).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import PipelineConfig

log = logging.getLogger("vagus.cardiac")


def peri_rwave(
    spike_samples: np.ndarray,
    labels: np.ndarray,
    rpeak_samples: np.ndarray,
    fs: float,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Return per-cluster PRWH + flags + a `cleaned_spike_samples` (no flagged events).

    Output structure
    ----------------
    {
        "cluster": [
            {"cluster_id": c, "prwh": np.ndarray, "edges_ms": np.ndarray, "is_cardiac_locked": bool},
            ...
        ],
        "cleaned_spike_samples": np.ndarray,
        "cleaned_labels": np.ndarray,
        "bin_ms": float,
    }
    """
    out: dict[str, Any] = {"cluster": [], "cleaned_spike_samples": spike_samples.copy(), "cleaned_labels": labels.copy(), "bin_ms": 1.0}
    if spike_samples.size == 0 or rpeak_samples.size == 0:
        return out

    win_ms = cfg.prwh_window_ms
    bin_ms = 1.0
    edges = np.arange(-win_ms, win_ms + bin_ms, bin_ms)
    win_samples = int(round(win_ms * 1e-3 * fs))

    unique = sorted({int(l) for l in labels if l >= 0})
    flagged_mask = np.zeros(spike_samples.size, dtype=bool)
    rp_sorted = np.sort(rpeak_samples)

    for c in unique:
        idx = np.where(labels == c)[0]
        sp = spike_samples[idx]
        if sp.size == 0:
            continue
        # for each spike find nearest R-peak via searchsorted
        pos = np.searchsorted(rp_sorted, sp)
        pos_left = np.clip(pos - 1, 0, rp_sorted.size - 1)
        pos_right = np.clip(pos, 0, rp_sorted.size - 1)
        d_left = sp - rp_sorted[pos_left]
        d_right = rp_sorted[pos_right] - sp
        nearest = np.where(np.abs(d_left) <= np.abs(d_right), -d_left, d_right)  # signed
        # Actually we want spike_time - rpeak_time (signed). Recompute cleanly:
        nearest_rp = np.where(np.abs(d_left) <= np.abs(d_right), rp_sorted[pos_left], rp_sorted[pos_right])
        delta = sp - nearest_rp  # samples
        delta_ms = delta / fs * 1000.0
        inside = np.abs(delta) <= win_samples
        hist, _ = np.histogram(delta_ms[inside], bins=edges)

        lock_window = cfg.cardiac_lock_window_ms
        lock_bin_mask = (edges[:-1] >= -lock_window) & (edges[1:] <= lock_window)
        if lock_bin_mask.sum() < 1 or hist.size == 0:
            is_locked = False
        else:
            inner = hist[lock_bin_mask].astype(np.float64)
            outer = hist[~lock_bin_mask].astype(np.float64)
            base_mean = outer.mean() if outer.size > 0 else 0.0
            base_std = outer.std(ddof=1) if outer.size > 1 else 1.0
            base_std = max(base_std, 1.0)
            peak_z = (inner.max() - base_mean) / base_std if inner.size else 0.0
            is_locked = bool(peak_z >= cfg.cardiac_peak_z)

        out["cluster"].append(
            {
                "cluster_id": c,
                "prwh": hist.astype(np.int64),
                "edges_ms": edges.astype(np.float32),
                "is_cardiac_locked": is_locked,
            }
        )
        if is_locked:
            lock_samples = int(round(cfg.cardiac_lock_window_ms * 1e-3 * fs))
            flagged_mask[idx] |= np.abs(delta) <= lock_samples

    out["cleaned_spike_samples"] = spike_samples[~flagged_mask]
    out["cleaned_labels"] = labels[~flagged_mask]
    log.info("Cardiac step: %d / %d spikes flagged & removed for cleaned set.",
             int(flagged_mask.sum()), spike_samples.size)
    return out
