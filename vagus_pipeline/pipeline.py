"""Per-pair orchestration of Steps 1–14."""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .audit import umap_hdbscan_audit
from .cardiac import peri_rwave
from .config import PipelineConfig, VarMap
from .detect import amplitude_histogram, detect_spikes, extract_waveforms
from .features import PCABasis, project_pca, scalar_features
from .fibertype import tag as fiber_tag
from .io_load import Recording, load_recording
from .io_discovery import RecordingPair
from .preprocess import bandpass, noise_sigma
from .quality import cluster_metrics
from .rates import firing_rates
from .respiration import resp_surrogate_from_rpeaks, verify as resp_verify
from .responder import detect as responder_detect
from .slowwave import phase_tag
from .sort import run_mountainsort

log = logging.getLogger("vagus.pipeline")


def run_prepass_on_cuff(
    neural: np.ndarray,
    blanked_mask: np.ndarray | None,
    pca_basis: PCABasis,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Steps 1-5: bandpass, sigma track, spike detection, waveform extraction,
    scalar features + PCA projection.  Returns the dict consumed by
    :func:`run_postpass_on_cuff` and persisted by :mod:`checkpoint`.
    """
    filtered = bandpass(neural, cfg)
    sigma_track, sigma_times = noise_sigma(filtered, cfg, blanked_mask)
    spikes = detect_spikes(filtered, sigma_track, sigma_times, blanked_mask, cfg)
    waveforms, spikes = extract_waveforms(filtered, spikes, cfg)
    amp_hist = amplitude_histogram(filtered, spikes)
    scalars = scalar_features(waveforms, cfg)
    pca_feats = project_pca(waveforms, pca_basis)
    return {
        "filtered": filtered,
        "sigma_track": sigma_track,
        "sigma_times": sigma_times,
        "spike_samples": spikes,
        "waveforms": waveforms,
        "amp_hist": amp_hist,
        "scalar_feats": scalars,
        "pca_feats": pca_feats,
    }


def run_postpass_on_cuff(
    prepass: dict[str, Any],
    n_samples: int,
    rpeak_samples: np.ndarray,
    slowwave: np.ndarray | None,
    stim_events: list[tuple[int, str]] | None,
    fs: float,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Steps 6-14, given a prepass result dict.  Builds the same nested
    results structure :func:`run_pipeline_on_cuff` does, including step1-5
    metadata so the .mat output looks identical no matter which mode ran.
    """
    duration_s = n_samples / fs
    filtered = prepass["filtered"]
    sigma_track = prepass["sigma_track"]
    sigma_times = prepass["sigma_times"]
    spikes = prepass["spike_samples"]
    waveforms = prepass["waveforms"]
    scalars = prepass["scalar_feats"]
    pca_feats = prepass["pca_feats"]
    amp_hist = prepass["amp_hist"]

    # Step 6
    labels, sorter_name = run_mountainsort(filtered, spikes, waveforms, pca_feats, cfg)
    # Step 7
    qmetrics = cluster_metrics(waveforms, labels, spikes, fs, duration_s, cfg)
    # Step 8
    audit = umap_hdbscan_audit(pca_feats, labels, cfg)
    # Step 9
    cardiac = peri_rwave(spikes, labels, rpeak_samples, fs, cfg)

    # Optionally use cleaned spikes for downstream
    if cfg.use_cardiac_cleaned:
        ds_samples = cardiac["cleaned_spike_samples"]
        ds_labels = cardiac["cleaned_labels"]
    else:
        ds_samples = spikes
        ds_labels = labels

    # Step 10 (respiratory verification)
    resp_surr = resp_surrogate_from_rpeaks(rpeak_samples, n_samples, fs, cfg)
    resp = resp_verify(ds_samples, ds_labels, resp_surr, stim_events, fs, cfg)
    # HR surrogate for correlations: same RR-interpolated signal but pre-bandpass
    hr_signal = _instantaneous_hr_full(rpeak_samples, n_samples, fs)
    # Step 11
    sw = phase_tag(ds_samples, ds_labels, slowwave, fs, cfg)
    # Step 12
    rates = firing_rates(
        ds_samples, ds_labels, n_samples, fs, resp_surr, hr_signal, slowwave, cfg
    )
    # Step 13
    responder = responder_detect(
        rates["cluster"], rates["t_centres_s"], stim_events, fs, cfg
    )
    # Step 14
    fibertypes = fiber_tag(scalars, ds_labels, cfg)

    return {
        "step1": {
            "bp_low": cfg.bp_low_hz,
            "bp_high": cfg.bp_high_hz,
            "bp_order": cfg.bp_order,
        },
        "step2": {"sigma_track": sigma_track, "sigma_times": sigma_times},
        "step3": {
            "spike_samples": spikes.astype(np.int64),
            "spike_times_s": (spikes.astype(np.float64) / fs).astype(np.float32),
            "threshold_sigma": cfg.threshold_sigma,
            "amplitude_hist_counts": amp_hist["counts"],
            "amplitude_hist_edges": amp_hist["edges"],
        },
        "step4": {
            "waveforms": waveforms,
            "wf_pre_ms": cfg.wf_pre_ms,
            "wf_post_ms": cfg.wf_post_ms,
            "wf_len_samples": int(waveforms.shape[1] if waveforms.ndim == 2 else 0),
        },
        "step5": {
            "scalar_feats": scalars,
            "pca_feats": pca_feats,
            "n_pca": cfg.n_pca,
        },
        "step6": {
            "labels": labels.astype(np.int64),
            "n_clusters": int(len({int(l) for l in labels if l >= 0})),
            "sorter": sorter_name,
        },
        "step7": {"cluster": qmetrics},
        "step8": audit,
        "step9": {
            "cluster": cardiac["cluster"],
            "cleaned_spike_samples": cardiac["cleaned_spike_samples"].astype(np.int64),
            "bin_ms": cardiac["bin_ms"],
        },
        "step10": resp,
        "step11": {
            "sw_phase_per_spike": sw["sw_phase_per_spike"],
            "cluster": sw["cluster"],
        },
        "step12": rates,
        "step13": {"cluster": responder},
        "step14": {"cluster": fibertypes},
    }


def run_pipeline_on_cuff(
    neural: np.ndarray,
    blanked_mask: np.ndarray | None,
    rpeak_samples: np.ndarray,
    slowwave: np.ndarray | None,
    stim_events: list[tuple[int, str]] | None,
    fs: float,
    pca_basis: PCABasis,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Convenience wrapper that runs prepass + postpass back-to-back, the
    same all-in-one entry point earlier callers used."""
    prepass = run_prepass_on_cuff(neural, blanked_mask, pca_basis, cfg)
    return run_postpass_on_cuff(
        prepass=prepass,
        n_samples=neural.size,
        rpeak_samples=rpeak_samples,
        slowwave=slowwave,
        stim_events=stim_events,
        fs=fs,
        cfg=cfg,
    )


def _instantaneous_hr_full(rpeak_samples: np.ndarray, n_samples: int, fs: float) -> np.ndarray | None:
    if rpeak_samples.size < 4:
        return None
    rp = np.sort(rpeak_samples.astype(np.float64))
    rr = np.diff(rp) / fs
    inst_hr = 1.0 / np.clip(rr, 1e-3, None)
    t_hr = rp[1:] / fs
    t_all = np.arange(n_samples) / fs
    return np.interp(t_all, t_hr, inst_hr, left=inst_hr[0], right=inst_hr[-1]).astype(np.float32)


def run_pipeline_on_pair(
    pair: RecordingPair,
    var_map: VarMap,
    pca_basis: PCABasis,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Load a pair, run Steps 1–14 per cuff, and return the assembled metrics dict."""
    rec: Recording = load_recording(pair, var_map, cfg)
    cuff_results: list[dict[str, Any]] = []
    for k, (neural, mask) in enumerate(zip(rec.neural, rec.blanked_mask)):
        log.info("[%s] Cuff %d/%d", pair.blanked_path.name, k + 1, rec.cuff_count())
        cuff_results.append(
            run_pipeline_on_cuff(
                neural=neural,
                blanked_mask=mask,
                rpeak_samples=rec.rpeak_samples,
                slowwave=rec.slowwave,
                stim_events=rec.stim_events,
                fs=rec.fs,
                pca_basis=pca_basis,
                cfg=cfg,
            )
        )

    provenance = {
        "software_version": __version__,
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "blanked_path": str(pair.blanked_path),
        "rpeak_path": str(pair.rpeak_path),
        "slowwave_path": str(pair.slowwave_path) if pair.slowwave_path else "",
        "var_map": var_map.to_dict(),
        "config": cfg.to_dict(),
        "seed": cfg.seed,
        "pca_basis_path": "",  # filled in by batch
    }

    return {
        "provenance": provenance,
        "fs": rec.fs,
        "n_cuffs": rec.cuff_count(),
        "cuff": cuff_results,
    }
