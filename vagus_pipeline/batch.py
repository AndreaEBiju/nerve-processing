"""Batch driver with two-pass global-PCA logic and a summary CSV."""

from __future__ import annotations

import csv
import json
import logging
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .assemble import save_mat
from .config import PipelineConfig, VarMap
from .detect import detect_spikes, extract_waveforms
from .features import PCABasis, fit_pca
from .io_discovery import (
    DEFAULT_BLANKED_TOKEN,
    DEFAULT_REQUIRED_REGEX,
    DEFAULT_RPEAK_TOKEN,
    DEFAULT_SLOWWAVE_TOKEN,
    RecordingPair,
    find_pairs,
)
from .io_load import load_recording
from .pipeline import run_pipeline_on_pair
from .preprocess import bandpass, noise_sigma

log = logging.getLogger("vagus.batch")


def _pass1_collect_waveforms(
    pair: RecordingPair, var_map: VarMap, cfg: PipelineConfig, per_pair_cap: int
) -> list[np.ndarray]:
    """Run Steps 1–4 on a pair and return per-cuff waveform arrays."""
    rec = load_recording(pair, var_map, cfg)
    out: list[np.ndarray] = []
    rng = np.random.default_rng(cfg.seed)
    for neural, mask in zip(rec.neural, rec.blanked_mask):
        filtered = bandpass(neural, cfg)
        sigma_track, sigma_times = noise_sigma(filtered, cfg, mask)
        spikes = detect_spikes(filtered, sigma_track, sigma_times, mask, cfg)
        wfs, _ = extract_waveforms(filtered, spikes, cfg)
        if wfs.shape[0] > per_pair_cap:
            sel = rng.choice(wfs.shape[0], size=per_pair_cap, replace=False)
            wfs = wfs[sel]
        out.append(wfs)
    return out


def run_batch(
    root_dir: str | Path,
    var_map: VarMap,
    cfg: PipelineConfig,
    blanked_patterns: tuple[str, ...] | list[str] | None = None,
    rpeak_patterns: tuple[str, ...] | list[str] | None = None,
    slowwave_patterns: tuple[str, ...] | list[str] | None = None,
    required_regex: str | None = DEFAULT_REQUIRED_REGEX,
    blanked_token: str | None = DEFAULT_BLANKED_TOKEN,
    rpeak_token: str | None = DEFAULT_RPEAK_TOKEN,
    slowwave_token: str | None = DEFAULT_SLOWWAVE_TOKEN,
    progress_cb: Any | None = None,
) -> dict[str, Any]:
    """Run the full two-pass batch on ``root_dir``.

    Pass 1 builds a global PCA basis; Pass 2 runs Steps 5(project)–14 per pair
    and saves a ``<stem>_metrics.mat`` next to each source file. A
    ``batch_summary.csv`` and ``batch_pca_basis.npz`` land at the batch root.

    ``required_regex`` (default :data:`DEFAULT_REQUIRED_REGEX`) gates which
    files are even considered candidates — files lacking the version tag in
    their name are silently skipped.  ``blanked_token``/``rpeak_token`` drive
    deterministic in-directory pairing.
    """
    root = Path(root_dir)
    kwargs: dict[str, Any] = {
        "required_regex": required_regex,
        "blanked_token": blanked_token,
        "rpeak_token": rpeak_token,
        "slowwave_token": slowwave_token,
    }
    if blanked_patterns:
        kwargs["blanked_patterns"] = blanked_patterns
    if rpeak_patterns:
        kwargs["rpeak_patterns"] = rpeak_patterns
    if slowwave_patterns is not None:
        kwargs["slowwave_patterns"] = slowwave_patterns
    pairs = find_pairs(root, **kwargs)
    if not pairs:
        raise RuntimeError(f"No recording pairs discovered under {root}")

    # Persist the var-map for re-runs
    (root / "batch_varmap.json").write_text(json.dumps(var_map.to_dict(), indent=2))

    # ---- Pass 1: collect waveforms across all pairs/cuffs ----
    log.info("Pass 1: collecting waveforms for global PCA basis...")
    per_pair_cap = max(cfg.pca_pool_max_spikes // max(len(pairs), 1), 100)
    pooled: list[np.ndarray] = []
    for i, pair in enumerate(pairs):
        try:
            wfs_list = _pass1_collect_waveforms(pair, var_map, cfg, per_pair_cap)
            for wfs in wfs_list:
                if wfs.shape[0] > 0:
                    pooled.append(wfs)
        except Exception as e:
            log.error("Pass 1 failed on %s: %s", pair.blanked_path, e)
        if progress_cb:
            progress_cb("pass1", i + 1, len(pairs))

    if not pooled:
        raise RuntimeError("Pass 1 yielded no waveforms; cannot fit PCA basis.")
    pooled_arr = np.concatenate(pooled, axis=0)
    pca_basis = fit_pca(pooled_arr, cfg)
    basis_path = root / "batch_pca_basis.npz"
    pca_basis.save(basis_path)
    log.info("Saved global PCA basis to %s (pooled %d waveforms)", basis_path, pooled_arr.shape[0])

    # ---- Pass 2: full pipeline per pair, save .mat per pair ----
    log.info("Pass 2: running full pipeline per pair...")
    summary_rows: list[dict[str, Any]] = []
    for i, pair in enumerate(pairs):
        row: dict[str, Any] = {
            "dir": str(pair.dir),
            "blanked": pair.blanked_path.name,
            "rpeak": pair.rpeak_path.name,
            "slowwave": pair.slowwave_path.name if pair.slowwave_path else "",
            "status": "ok",
            "reason": "",
            "n_cuffs": 0,
            "n_spikes_total": 0,
            "n_clusters_total": 0,
            "mean_snr": float("nan"),
            "n_responders": 0,
            "output_path": "",
        }
        try:
            results = run_pipeline_on_pair(pair, var_map, pca_basis, cfg)
            results["provenance"]["pca_basis_path"] = str(basis_path)
            out_path = save_mat(results, pair.dir, pair.common_stem())
            row["output_path"] = str(out_path)
            row["n_cuffs"] = int(results["n_cuffs"])
            spikes_tot, clust_tot, snrs, resp_tot = 0, 0, [], 0
            for cuff in results["cuff"]:
                spikes_tot += int(np.asarray(cuff["step3"]["spike_samples"]).size)
                clust_tot += int(cuff["step6"]["n_clusters"])
                for cl in cuff["step7"]["cluster"]:
                    if np.isfinite(cl["snr"]):
                        snrs.append(float(cl["snr"]))
                for cl in cuff["step13"]["cluster"]:
                    for c in cl.get("conditions", []) or []:
                        if c.get("is_responder"):
                            resp_tot += 1
            row["n_spikes_total"] = spikes_tot
            row["n_clusters_total"] = clust_tot
            row["mean_snr"] = float(np.mean(snrs)) if snrs else float("nan")
            row["n_responders"] = resp_tot
        except Exception as e:
            row["status"] = "failed"
            row["reason"] = f"{type(e).__name__}: {e}"
            log.error("Pair %s failed:\n%s", pair.blanked_path, traceback.format_exc())
        summary_rows.append(row)
        if progress_cb:
            progress_cb("pass2", i + 1, len(pairs))

    # ---- Summary CSV ----
    summary_path = root / "batch_summary.csv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    log.info("Wrote batch summary: %s", summary_path)

    return {
        "pairs": pairs,
        "pca_basis_path": str(basis_path),
        "summary_path": str(summary_path),
        "rows": summary_rows,
    }
