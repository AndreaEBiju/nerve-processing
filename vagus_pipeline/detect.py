"""Step 3 spike detection and Step 4 waveform extraction.

Threshold crossings on the negative (or positive) phase of the filtered
signal, refractory enforcement, alignment to the local extremum, and fixed
±wf_pre/wf_post-ms waveform windows.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from .config import PipelineConfig
from .preprocess import sigma_at

log = logging.getLogger("vagus.detect")


def detect_spikes(
    filtered: np.ndarray,
    sigma_track: np.ndarray,
    sigma_times: np.ndarray,
    blanked_mask: np.ndarray | None,
    cfg: PipelineConfig,
) -> np.ndarray:
    """Return spike sample indices.

    Threshold = ``threshold_sigma * sigma(t)``. Crossings on the configured
    polarity are aligned to the local extremum within ±0.5 ms and pruned by a
    refractory window.
    """
    fs = cfg.fs
    polarity = cfg.detect_polarity
    if polarity not in ("neg", "pos", "abs"):
        raise ValueError(f"detect_polarity must be 'neg'/'pos'/'abs', got {polarity}")

    n = filtered.size
    idx = np.arange(n)
    sigma = sigma_at(idx, sigma_track, sigma_times)
    thr = cfg.threshold_sigma * sigma

    if polarity == "neg":
        crossings = filtered < -thr
    elif polarity == "pos":
        crossings = filtered > thr
    else:
        crossings = np.abs(filtered) > thr

    if blanked_mask is not None:
        crossings &= ~blanked_mask

    if not crossings.any():
        return np.zeros(0, dtype=np.int64)

    # Find rising edges of the boolean
    d = np.diff(np.concatenate(([0], crossings.view(np.int8), [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]

    win = max(int(round(0.0005 * fs)), 1)  # ±0.5 ms search for local extremum
    samples = []
    for s0, e0 in zip(starts, ends):
        a = max(0, s0 - win)
        b = min(n, e0 + win)
        seg = filtered[a:b]
        if polarity == "neg":
            local = int(np.argmin(seg))
        elif polarity == "pos":
            local = int(np.argmax(seg))
        else:
            local = int(np.argmax(np.abs(seg)))
        samples.append(a + local)
    samples = np.asarray(samples, dtype=np.int64)

    # Refractory pruning (keep the larger-amplitude event)
    refr = max(int(round(cfg.refractory_ms * 1e-3 * fs)), 1)
    samples.sort()
    keep = np.ones(samples.size, dtype=bool)
    i = 0
    while i < samples.size - 1:
        j = i + 1
        while j < samples.size and samples[j] - samples[i] < refr:
            # keep the more extreme
            if polarity == "neg":
                if filtered[samples[j]] < filtered[samples[i]]:
                    keep[i] = False
                    i = j
                else:
                    keep[j] = False
            elif polarity == "pos":
                if filtered[samples[j]] > filtered[samples[i]]:
                    keep[i] = False
                    i = j
                else:
                    keep[j] = False
            else:
                if abs(filtered[samples[j]]) > abs(filtered[samples[i]]):
                    keep[i] = False
                    i = j
                else:
                    keep[j] = False
            j += 1
        i = j
    samples = samples[keep]
    log.info("Detected %d spikes (polarity=%s)", samples.size, polarity)
    return samples


def extract_waveforms(
    filtered: np.ndarray,
    spike_samples: np.ndarray,
    cfg: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(waveforms, samples)`` where each waveform spans ±wf_pre/wf_post.

    Spikes too close to either edge are dropped, and their indices are removed
    from ``samples`` so it stays aligned with ``waveforms``.
    """
    fs = cfg.fs
    pre = int(round(cfg.wf_pre_ms * 1e-3 * fs))
    post = int(round(cfg.wf_post_ms * 1e-3 * fs))
    L = pre + post
    n = filtered.size
    valid = (spike_samples - pre >= 0) & (spike_samples + post <= n)
    samples = spike_samples[valid]
    if samples.size == 0:
        return np.zeros((0, L), dtype=np.float32), samples
    rows = np.empty((samples.size, L), dtype=np.float32)
    for i, s in enumerate(samples):
        rows[i] = filtered[s - pre : s + post]
    return rows, samples


def amplitude_histogram(filtered: np.ndarray, spike_samples: np.ndarray, n_bins: int = 100) -> dict:
    """Diagnostic histogram of detected spike amplitudes.

    Used to spot a hard cutoff at the detection threshold (motivation for the
    deferred Track-B template-matching pass).
    """
    if spike_samples.size == 0:
        return {"counts": np.zeros(n_bins, dtype=np.int64), "edges": np.linspace(0, 1, n_bins + 1)}
    amps = np.abs(filtered[spike_samples])
    counts, edges = np.histogram(amps, bins=n_bins)
    return {"counts": counts.astype(np.int64), "edges": edges.astype(np.float32)}


def template_recovery(*args, **kwargs):  # pragma: no cover - intentional stub
    """Track-B template-matching second pass (NOT implemented in v1).

    Trigger condition: the detected-amplitude histogram from
    :func:`amplitude_histogram` shows a hard cutoff right at the detection
    threshold, suggesting good waveforms are clipped by the threshold. See
    companion docx Section 3, Track B for the algorithm sketch.
    """
    raise NotImplementedError(
        "template_recovery is intentionally a stub in v1; see docstring for trigger condition."
    )
