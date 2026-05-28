"""Step 7: per-cluster quality metrics, including full ISI distributions."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import PipelineConfig

log = logging.getLogger("vagus.quality")


def cluster_metrics(
    waveforms: np.ndarray,
    labels: np.ndarray,
    spike_samples: np.ndarray,
    fs: float,
    duration_s: float,
    cfg: PipelineConfig,
) -> list[dict[str, Any]]:
    """Compute per-cluster metrics. Returns one dict per cluster id in ascending order.

    Each entry contains:
        cluster_id, n_spikes, snr, firing_rate_hz, mean_wf, std_wf, isi_s (full),
        isi_violation_rate (refractory <1ms).
    """
    out: list[dict[str, Any]] = []
    if labels.size == 0:
        return out

    unique = sorted({int(l) for l in labels if l >= 0})
    refr_s = cfg.refractory_ms * 1e-3

    for c in unique:
        mask = labels == c
        wfs = waveforms[mask]
        samples = spike_samples[mask]
        if wfs.shape[0] == 0:
            continue
        mean_wf = wfs.mean(axis=0)
        std_wf = wfs.std(axis=0)
        # SNR: peak-to-peak of mean waveform / median std around baseline
        p2p = float(mean_wf.max() - mean_wf.min())
        noise = float(np.median(std_wf)) if std_wf.size else 0.0
        snr = p2p / noise if noise > 0 else np.inf
        firing_rate = float(samples.size / duration_s) if duration_s > 0 else 0.0
        isi_s = np.diff(np.sort(samples.astype(np.float64))) / fs if samples.size > 1 else np.zeros(0)
        isi_viol = float((isi_s < refr_s).sum() / isi_s.size) if isi_s.size > 0 else 0.0
        out.append(
            {
                "cluster_id": c,
                "n_spikes": int(samples.size),
                "snr": float(snr),
                "firing_rate_hz": firing_rate,
                "mean_wf": mean_wf.astype(np.float32),
                "std_wf": std_wf.astype(np.float32),
                "isi_s": isi_s.astype(np.float32),
                "isi_violation_rate": isi_viol,
            }
        )
    return out
