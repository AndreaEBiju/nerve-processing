"""Step 6: spike sorting via SpikeInterface + MountainSort5.

A KMeans fallback is provided for environments without MountainSort/SpikeInterface
so the pipeline can still run end-to-end; the chosen sorter is recorded in
``provenance.sorter`` so the consumer can tell which path ran.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
from typing import Tuple

import numpy as np

from .config import PipelineConfig

log = logging.getLogger("vagus.sort")


@contextlib.contextmanager
def _silence_c_stdio():
    """Redirect OS-level stdout/stderr to /dev/null for the duration of the
    block.  MountainSort5's isosplit5 emits its noisy ``"splitting parcel."``
    / ``"new parcel has no points"`` / ``"Size did not change..."`` messages
    via C ``printf``, so Python's ``contextlib.redirect_stdout`` does not
    catch them.  Duplicating ``/dev/null`` onto the underlying file
    descriptors does.

    The originals are restored even if MS5 raises, so a real traceback still
    surfaces to the user.
    """
    # Flush any pending Python-level output first so it appears before
    # anything from MS5 (and is not lost when we redirect).
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)


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
        labels = _run_ms5(filtered, spike_samples, cfg)
        n_units = len({int(l) for l in labels if l >= 0})
        if n_units == 0:
            # MS5 ran but its internal threshold-based detector found no
            # events above 4.5*MAD in the saved filtered signal -- this can
            # happen when the filtered signal's spike amplitudes are
            # comparable to the MAD (so the threshold sits above them).
            # The prepass already detected ``spike_samples.size`` events
            # though, and pca_feats is populated for all of them, so the
            # KMeans fallback can still produce useful clusters.
            log.warning(
                "MountainSort5 returned 0 units despite %d prepass-detected spikes; "
                "falling back to KMeans on PCA features so clusters are still produced.",
                spike_samples.size,
            )
            return _kmeans_fallback(pca_feats, cfg), "kmeans_fallback"
        return labels, "mountainsort5"
    except Exception as e:
        log.warning("MountainSort5 unavailable or failed (%s); falling back to KMeans on PCA features.", e)
        return _kmeans_fallback(pca_feats, cfg), "kmeans_fallback"


def _run_ms5(filtered: np.ndarray, spike_samples: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    import spikeinterface.core as si
    import mountainsort5 as ms5
    from mountainsort5.util import create_cached_recording  # type: ignore

    fs = cfg.fs

    # MS5 / SpikeInterface compute noise_levels from the input signal and
    # then use detect_threshold * noise_levels as the actual sample-space
    # threshold.  In practice that pipeline can give inconsistent (or zero)
    # detections when the input scale is far from the conventional
    # microvolt-ish range (e.g. raw volts, like the user's filtered signal
    # at ~1e-4) or when there's enough quantization noise to throw off
    # noise_levels.  Pre-normalising the trace so its MAD-based sigma is
    # exactly 1.0 makes ``cfg.threshold_sigma * noise_levels`` collapse to
    # the intuitive "threshold = cfg.threshold_sigma * sigma" rule
    # regardless of the original units.
    f64 = filtered.astype(np.float64)
    med = float(np.median(f64))
    mad = float(np.median(np.abs(f64 - med)))
    sigma = mad / 0.6745 if mad > 0 else float(np.std(f64) + 1e-12)
    if sigma > 0 and np.isfinite(sigma):
        scale = 1.0 / sigma
        filtered_norm = ((f64 - med) * scale).astype(np.float32)
        log.info(
            "MS5: normalised filtered trace (median=%.3g, MAD-sigma=%.3g -> scale=%.3g) "
            "so detect_threshold=%.2f is interpreted as %.2f * sigma.",
            med, sigma, scale, cfg.threshold_sigma, cfg.threshold_sigma,
        )
    else:
        filtered_norm = filtered.astype(np.float32)
        log.warning("MS5: filtered trace has zero/non-finite sigma; passing unscaled.")

    rec = si.NumpyRecording(traces_list=[filtered_norm.reshape(-1, 1)], sampling_frequency=fs)
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
        # isosplit5 (called from inside sorting_scheme2) emits a flood of
        # benign "splitting parcel." / "new parcel has no points" /
        # "Size did not change after splitting parcel" messages via C
        # printf when MS5 is detecting lots of spikes.  They mean
        # "this parcel is already unimodal" and do not affect the final
        # clustering, so suppress them at the FD level rather than
        # spamming the user's terminal.
        with _silence_c_stdio():
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
