"""Step 12: per-cluster firing-rate traces and physiological correlations."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import PipelineConfig

log = logging.getLogger("vagus.rates")


def firing_rates(
    spike_samples: np.ndarray,
    labels: np.ndarray,
    n_samples: int,
    fs: float,
    resp_signal: np.ndarray | None,
    hr_signal: np.ndarray | None,
    sw_signal: np.ndarray | None,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Return per-cluster rate traces + correlations against resp/HR/slow-wave."""
    bin_s = cfg.rate_bin_s
    bin_samples = int(round(bin_s * fs))
    n_bins = max(n_samples // bin_samples, 1)
    edges = np.arange(n_bins + 1) * bin_samples
    t_centres = (edges[:-1] + edges[1:]) / 2.0 / fs

    out: dict[str, Any] = {
        "rate_bin_s": bin_s,
        "t_centres_s": t_centres.astype(np.float32),
        "cluster": [],
    }
    if spike_samples.size == 0:
        return out

    def _binned(signal: np.ndarray | None) -> np.ndarray | None:
        if signal is None or signal.size == 0:
            return None
        b = np.zeros(n_bins, dtype=np.float64)
        for i in range(n_bins):
            b[i] = float(signal[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0
        return b

    resp_b = _binned(resp_signal)
    hr_b = _binned(hr_signal)
    sw_b = _binned(sw_signal)

    def _corr(a: np.ndarray, b: np.ndarray | None) -> float:
        if b is None or a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    unique = sorted({int(l) for l in labels if l >= 0})
    for c in unique:
        sp = spike_samples[labels == c]
        if sp.size == 0:
            continue
        counts, _ = np.histogram(sp, bins=edges)
        rate = counts.astype(np.float32) / bin_s
        out["cluster"].append(
            {
                "cluster_id": c,
                "rate_trace": rate,
                "corr_resp": _corr(rate.astype(np.float64), resp_b),
                "corr_hr": _corr(rate.astype(np.float64), hr_b),
                "corr_sw": _corr(rate.astype(np.float64), sw_b),
            }
        )
    return out
