"""Step 6: spike sorting via SpikeInterface + MountainSort5.

A KMeans fallback is provided for environments without MountainSort/SpikeInterface
so the pipeline can still run end-to-end; the chosen sorter is recorded in
``provenance.sorter`` so the consumer can tell which path ran.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Tuple

import numpy as np

from .config import PipelineConfig

log = logging.getLogger("vagus.sort")


def run_mountainsort(
    filtered: np.ndarray,
    spike_samples: np.ndarray,
    waveforms: np.ndarray,
    pca_feats: np.ndarray,
    cfg: PipelineConfig,
) -> Tuple[np.ndarray, str]:
    """Run MountainSort5 if available; otherwise fall back to a feature-space cluster.

    Returns ``(labels, sorter_name)``. Labels are integers in ``[0, K)`` where
    ``-1`` marks unsorted spikes (noise cluster).
    """
    if spike_samples.size == 0:
        return np.zeros(0, dtype=np.int64), "none"

    try:
        return _run_ms5(filtered, spike_samples, cfg), "mountainsort5"
    except Exception as e:
        log.warning("MountainSort5 unavailable or failed (%s); falling back to KMeans on PCA features.", e)
        return _kmeans_fallback(pca_feats, cfg), "kmeans_fallback"


def _run_ms5(filtered: np.ndarray, spike_samples: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    import spikeinterface.core as si
    import mountainsort5 as ms5
    from mountainsort5.util import create_cached_recording  # type: ignore

    fs = cfg.fs
    rec = si.NumpyRecording(traces_list=[filtered.astype(np.float32).reshape(-1, 1)], sampling_frequency=fs)
    rec.set_channel_locations(np.array([[0.0, 0.0]]))
    with tempfile.TemporaryDirectory() as td:
        cached = create_cached_recording(rec, folder=os.path.join(td, "rec_cache"))
        scheme = ms5.Scheme2SortingParameters(
            phase1_detect_channel_radius=50.0,
            detect_channel_radius=50.0,
            detect_threshold=cfg.threshold_sigma,
            detect_sign=-1 if cfg.detect_polarity == "neg" else 1,
            snippet_T1=int(round(cfg.wf_pre_ms * 1e-3 * fs)),
            snippet_T2=int(round(cfg.wf_post_ms * 1e-3 * fs)),
            phase1_npca_per_channel=cfg.n_pca,
            phase1_npca_per_subdivision=cfg.n_pca,
        )
        sorting = ms5.sorting_scheme2(recording=cached, sorting_parameters=scheme)
        # Map sorter results back onto our spike_samples by nearest-time within ±1 ms
        labels = np.full(spike_samples.size, -1, dtype=np.int64)
        win = max(int(round(0.001 * fs)), 1)
        for unit_id in sorting.get_unit_ids():
            unit_samples = sorting.get_unit_spike_train(unit_id=unit_id)
            for s in unit_samples:
                d = np.abs(spike_samples - s)
                j = int(d.argmin())
                if d[j] <= win:
                    labels[j] = int(unit_id)
        # Re-label so labels run 0..K-1 (preserve -1 for unsorted)
        return _compact_labels(labels)


def _kmeans_fallback(pca_feats: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = pca_feats.shape[0]
    if n < 10:
        return np.zeros(n, dtype=np.int64)

    best_k, best_labels, best_score = 1, np.zeros(n, dtype=np.int64), -1.0
    for k in range(2, min(8, n // 5) + 1):
        km = KMeans(n_clusters=k, random_state=cfg.seed, n_init=10)
        labels = km.fit_predict(pca_feats)
        try:
            score = silhouette_score(pca_feats, labels)
        except Exception:
            score = -1.0
        if score > best_score:
            best_k, best_labels, best_score = k, labels, score
    log.info("KMeans fallback chose k=%d (silhouette=%.3f)", best_k, best_score)
    return best_labels.astype(np.int64)


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    """Map present positive labels to 0..K-1, preserve -1."""
    out = labels.copy()
    present = sorted({int(l) for l in labels if l >= 0})
    mapping = {l: i for i, l in enumerate(present)}
    for old, new in mapping.items():
        out[labels == old] = new
    return out
