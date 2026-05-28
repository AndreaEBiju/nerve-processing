"""Step 11: slow-wave phase tagging and circular statistics.

Bandpasses the supplied slow-wave channel into ``[sw_low_hz, sw_high_hz]``,
extracts the analytic-signal phase, samples it at every spike time, and
computes per-cluster mean resultant length (MRL) plus a Rayleigh p-value.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from .config import PipelineConfig

log = logging.getLogger("vagus.slowwave")


def phase_tag(
    spike_samples: np.ndarray,
    labels: np.ndarray,
    slowwave: np.ndarray | None,
    fs: float,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Return per-spike phases and per-cluster MRL + Rayleigh p."""
    out: dict[str, Any] = {"sw_phase_per_spike": np.zeros(0, dtype=np.float32), "cluster": []}
    if slowwave is None or slowwave.size == 0 or spike_samples.size == 0:
        return out

    nyq = fs / 2.0
    low = max(cfg.sw_low_hz, 1e-4) / nyq
    high = min(cfg.sw_high_hz, 0.99 * nyq) / nyq
    if not (0 < low < high < 1):
        log.warning("Slow-wave bandpass corners invalid for fs=%g; skipping phase tagging.", fs)
        return out
    b, a = butter(2, [low, high], btype="band")
    sw = filtfilt(b, a, slowwave.astype(np.float64))
    analytic = hilbert(sw)
    phase = np.angle(analytic).astype(np.float32)
    sp = spike_samples.clip(0, phase.size - 1)
    sw_phase_per_spike = phase[sp]
    out["sw_phase_per_spike"] = sw_phase_per_spike

    unique = sorted({int(l) for l in labels if l >= 0})
    for c in unique:
        mask = labels == c
        p = sw_phase_per_spike[mask]
        if p.size == 0:
            continue
        z = np.exp(1j * p.astype(np.float64))
        mrl = float(np.abs(z.mean()))
        # Rayleigh test: Z = n * mrl^2 ; p ≈ exp(-Z) for moderate-to-large n.
        n = int(p.size)
        Z = n * mrl * mrl
        try:
            from pingouin import circ_rayleigh

            _, p_val = circ_rayleigh(p.astype(np.float64))
            rayleigh_p = float(min(max(p_val, 0.0), 1.0))
        except Exception:
            rayleigh_p = float(min(max(np.exp(-Z), 0.0), 1.0))
        hist, edges = np.histogram(p, bins=np.linspace(-np.pi, np.pi, 25))
        out["cluster"].append(
            {
                "cluster_id": c,
                "phase_hist": hist.astype(np.int64),
                "phase_edges": edges.astype(np.float32),
                "mrl": mrl,
                "rayleigh_p": rayleigh_p,
            }
        )
    return out
