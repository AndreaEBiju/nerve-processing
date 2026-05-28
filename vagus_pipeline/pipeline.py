"""Per-pair orchestration of Steps 1–15."""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .antral_bursts import detect_bursts, xcorr as antral_xcorr
from .audit import umap_hdbscan_audit
from .cardiac import peri_rwave
from .config import PipelineConfig, SlowWaveUnusable, VarMap
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
from .slowwave_quality import (
    build_common_mode,
    expand_common_mode,
    score_channels,
    serialize_quality_summary,
)
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


def run_pair_level_slowwave(
    slowwave_channels: list[np.ndarray],
    fs: float,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Steps 11a + 12.detect_bursts -- shared across cuffs.

    Returns a dict with the artefacts needed by every cuff's postpass:
      ``qualities``, ``common_mode_result``, ``common_mode_full_fs``,
      ``burst_detection`` (per-channel + consensus times), ``skipped`` flag.

    When :class:`SlowWaveUnusable` would be raised, returns a stub
    ``{"skipped": True, "reason": ...}`` so the orchestrator can continue
    Steps 1-10, 13-15 unaffected.
    """
    if not slowwave_channels:
        return {"skipped": True, "reason": "no slow-wave channels available"}

    qualities = score_channels(slowwave_channels, fs, cfg)
    try:
        cm_res = build_common_mode(slowwave_channels, qualities, fs, cfg)
    except SlowWaveUnusable as e:
        log.warning("Steps 11+12 skipped: %s", e)
        return {"skipped": True, "reason": str(e), "qualities": qualities}

    cm_full = expand_common_mode(
        cm_res.common_mode, fs, slowwave_channels[0].size, cm_res.common_mode_fs_hz,
    )
    bursts = detect_bursts(
        slowwave_channels=slowwave_channels,
        rolling_weights=cm_res.rolling_weights,
        window_centres_s=qualities[0].quality_window_times_s if qualities else None,
        fs=fs,
        cfg=cfg,
    )
    return {
        "skipped": False,
        "qualities": qualities,
        "common_mode_result": cm_res,
        "common_mode_full_fs": cm_full,
        "burst_detection": bursts,
    }


def run_postpass_on_cuff(
    prepass: dict[str, Any],
    n_samples: int,
    rpeak_samples: np.ndarray,
    slowwave_channels: list[np.ndarray] | None,
    slowwave_artifacts: dict[str, Any] | None,
    stim_events: list[tuple[int, str]] | None,
    fs: float,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Steps 6-15, given a prepass result dict and pair-level slow-wave
    artefacts.  Builds the nested results structure described in spec §8."""
    duration_s = n_samples / fs
    filtered = prepass["filtered"]
    sigma_track = prepass["sigma_track"]
    sigma_times = prepass["sigma_times"]
    spikes = prepass["spike_samples"]
    waveforms = prepass["waveforms"]
    scalars = prepass["scalar_feats"]
    pca_feats = prepass["pca_feats"]
    amp_hist = prepass["amp_hist"]
    slowwave_channels = slowwave_channels or []

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

    # Step 10
    resp_surr = resp_surrogate_from_rpeaks(rpeak_samples, n_samples, fs, cfg)
    resp = resp_verify(ds_samples, ds_labels, resp_surr, stim_events, fs, cfg)
    hr_signal = _instantaneous_hr_full(rpeak_samples, n_samples, fs)

    # ------- Step 11 (slow wave) -------
    step11a_struct, step11b_struct = _build_step11_structs(
        slowwave_artifacts, slowwave_channels, ds_samples, ds_labels, fs, cfg,
    )

    # ------- Step 12 (antral bursts) -------
    step12_struct = _build_step12_struct(
        slowwave_artifacts, ds_samples, ds_labels, fs, cfg,
    )

    # ------- Step 13 (firing-rate series & correlations) -------
    cm_full = (slowwave_artifacts or {}).get("common_mode_full_fs")
    rates = firing_rates(
        ds_samples, ds_labels, n_samples, fs, resp_surr, hr_signal,
        cm_full if cm_full is not None and not (slowwave_artifacts or {}).get("skipped") else None,
        cfg,
    )
    # Step 14
    responder = responder_detect(rates["cluster"], rates["t_centres_s"], stim_events, fs, cfg)
    # Step 15
    fibertypes = fiber_tag(scalars, ds_labels, cfg)

    return {
        "step1": {"bp_low": cfg.bp_low_hz, "bp_high": cfg.bp_high_hz, "bp_order": cfg.bp_order},
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
        "step5": {"scalar_feats": scalars, "pca_feats": pca_feats, "n_pca": cfg.n_pca},
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
        "step11a": step11a_struct,
        "step11b": step11b_struct,
        "step12": step12_struct,
        "step13": rates,
        "step14": {"cluster": responder},
        "step15": {"cluster": fibertypes},
    }


def _build_step11_structs(
    swa: dict[str, Any] | None,
    slowwave_channels: list[np.ndarray],
    ds_samples: np.ndarray,
    ds_labels: np.ndarray,
    fs: float,
    cfg: PipelineConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the spec §8 ``step11a`` + ``step11b`` structs."""
    if swa is None or swa.get("skipped"):
        return (
            {"skipped": True, "reason": (swa or {}).get("reason", "no slow-wave")},
            {"skipped": True, "reason": (swa or {}).get("reason", "no slow-wave")},
        )
    qualities = swa["qualities"]
    cm_res = swa["common_mode_result"]
    cm_full = swa["common_mode_full_fs"]
    sw = phase_tag(ds_samples, ds_labels, cm_full, slowwave_channels, fs, cfg)

    channels_struct = [
        {
            "snr_inband": q.snr_inband,
            "peak_prominence": q.peak_prominence,
            "pairwise_coherence_with_others": q.pairwise_coherence,
            "envelope_cv": q.envelope_cv,
            "quality_score_whole": q.quality_score_whole,
            "quality_score_rolling": q.quality_score_rolling.astype(np.float32),
            "quality_window_times_s": q.quality_window_times_s.astype(np.float32),
            "tier": q.tier,
            "reason_if_excluded": q.reason_if_excluded,
        }
        for q in qualities
    ]
    step11a = {
        "channels_present": int(len(qualities)),
        "channel": channels_struct,
        "rolling_weights": cm_res.rolling_weights.astype(np.float32),
        "n_channels_contributing": cm_res.n_channels_contributing.astype(np.uint8),
        "propagation_lag_s_between_adjacent": cm_res.propagation_lag_s_between_adjacent.astype(np.float32),
        "common_mode": cm_res.common_mode.astype(np.float32),
        "common_mode_fs_hz": float(cm_res.common_mode_fs_hz),
        "summary_text": serialize_quality_summary(qualities),
        "skipped": False,
    }
    step11b = {
        "sw_phase_per_spike": sw["sw_phase_per_spike"],
        "cluster": sw["cluster"],
        "consistency_threshold_circ_var": sw.get("consistency_circular_variance_threshold", float("nan")),
        "skipped": False,
    }
    return step11a, step11b


def _build_step12_struct(
    swa: dict[str, Any] | None,
    ds_samples: np.ndarray,
    ds_labels: np.ndarray,
    fs: float,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    if swa is None or swa.get("skipped"):
        return {"skipped": True, "reason": (swa or {}).get("reason", "no slow-wave")}
    bursts = swa["burst_detection"]
    xc = antral_xcorr(ds_samples, ds_labels, bursts["consensus_burst_times_s"], fs, cfg)
    return {
        "burst_times_s_per_channel": [b.astype(np.float32) for b in bursts["burst_times_s_per_channel"]],
        "consensus_burst_times_s": bursts["consensus_burst_times_s"].astype(np.float32),
        "cluster": xc,
        "skipped": False,
    }


def run_pipeline_on_cuff(
    neural: np.ndarray,
    blanked_mask: np.ndarray | None,
    rpeak_samples: np.ndarray,
    slowwave_channels: list[np.ndarray] | None,
    stim_events: list[tuple[int, str]] | None,
    fs: float,
    pca_basis: PCABasis,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Convenience wrapper that runs prepass + postpass back-to-back.
    Pair-level slow-wave artefacts are computed inline; for batches use
    :func:`run_pair_level_slowwave` once per pair and share."""
    prepass = run_prepass_on_cuff(neural, blanked_mask, pca_basis, cfg)
    swa = run_pair_level_slowwave(slowwave_channels or [], fs, cfg)
    return run_postpass_on_cuff(
        prepass=prepass,
        n_samples=neural.size,
        rpeak_samples=rpeak_samples,
        slowwave_channels=slowwave_channels,
        slowwave_artifacts=swa,
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
    """Load a pair, run Steps 1–15 per cuff (with pair-level slow-wave
    11a + 12.detect shared), and return the assembled metrics dict."""
    rec: Recording = load_recording(pair, var_map, cfg)
    # Pair-level slow-wave artefacts (11a + 12.detect_bursts) computed once.
    swa = run_pair_level_slowwave(rec.slowwave_channels, rec.fs, cfg)
    cuff_results: list[dict[str, Any]] = []
    for k, (neural, mask) in enumerate(zip(rec.neural, rec.blanked_mask)):
        log.info("[%s] Cuff %d/%d", pair.blanked_path.name, k + 1, rec.cuff_count())
        prepass = run_prepass_on_cuff(neural, mask, pca_basis, cfg)
        cuff_results.append(
            run_postpass_on_cuff(
                prepass=prepass,
                n_samples=neural.size,
                rpeak_samples=rec.rpeak_samples,
                slowwave_channels=rec.slowwave_channels,
                slowwave_artifacts=swa,
                stim_events=rec.stim_events,
                fs=rec.fs,
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
