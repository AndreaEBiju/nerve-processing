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
from .checkpoint import (
    CHECKPOINT_SUFFIX,
    checkpoint_path_for,
    load_checkpoint,
    save_checkpoint,
)
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
from .pipeline import (
    run_pipeline_on_pair,
    run_prepass_on_cuff,
    run_postpass_on_cuff,
)
from .preprocess import bandpass, noise_sigma

VALID_MODES = ("full", "prepass", "resume")

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
    mode: str = "full",
    progress_cb: Any | None = None,
) -> dict[str, Any]:
    """Run the batch on ``root_dir`` in one of three modes.

    ``mode`` controls how much of the 14-step pipeline is executed:

    * ``"full"`` (default) -- discovery → Pass 1 (global PCA basis) →
      Pass 2 (Steps 1-14 per cuff, .mat output per pair).  Same behaviour
      as the original implementation.
    * ``"prepass"`` -- discovery → Pass 1 → for each pair run Steps 1-5
      per cuff and save ``<stem>_checkpoint.npz`` next to the source
      files.  No sorting, no downstream steps, no .mat output.  Intended
      for machines that can't (or don't want to) run MountainSort5 --
      typically Windows boxes.
    * ``"resume"`` -- discovery is skipped; instead the root is scanned
      for existing ``*_checkpoint.npz`` files (one per pair).  For each
      checkpoint, Steps 6-14 run and ``<stem>_metrics.mat`` lands next
      to it.  Requires ``batch_pca_basis.npz`` at the root (saved by an
      earlier prepass run).  Intended to be run on a machine that does
      have MountainSort5 installed, after the user has selectively
      copied checkpoint files from cloud storage to local disk.

    ``required_regex`` (default :data:`DEFAULT_REQUIRED_REGEX`) gates which
    files are considered candidates -- files lacking the version tag in
    their name are silently skipped.  ``blanked_token``/``rpeak_token``
    drive deterministic in-directory pairing.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    if mode == "resume":
        return _run_batch_resume(Path(root_dir), cfg, progress_cb)
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

    # ---- Pass 2: full pipeline (or prepass-only) per pair ----
    log.info("Pass 2: running %s pipeline per pair...", "prepass-only" if mode == "prepass" else "full")
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
            "mode": mode,
        }
        try:
            if mode == "prepass":
                rec = load_recording(pair, var_map, cfg)
                per_cuff_prepass = [
                    run_prepass_on_cuff(neural, mask, pca_basis, cfg)
                    for neural, mask in zip(rec.neural, rec.blanked_mask)
                ]
                ck_path = checkpoint_path_for(pair)
                save_checkpoint(ck_path, pair, rec, per_cuff_prepass, basis_path, var_map, cfg)
                row["output_path"] = str(ck_path)
                row["n_cuffs"] = rec.cuff_count()
                row["n_spikes_total"] = int(sum(p["spike_samples"].size for p in per_cuff_prepass))
            else:
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
        "mode": mode,
    }


def _run_batch_resume(root: Path, cfg: PipelineConfig, progress_cb: Any | None) -> dict[str, Any]:
    """Resume from existing ``*_checkpoint.npz`` files at ``root``.

    Steps 6-14 run per cuff; the .mat output lands next to the checkpoint
    file.  The global PCA basis must already exist at
    ``root / "batch_pca_basis.npz"`` (produced by an earlier prepass run).
    """
    basis_path = root / "batch_pca_basis.npz"
    if not basis_path.exists():
        raise FileNotFoundError(
            f"resume mode requires {basis_path} -- run a prepass first or copy the basis file across."
        )
    pca_basis = PCABasis.load(basis_path)
    log.info("Loaded global PCA basis from %s", basis_path)

    checkpoints = sorted(
        p
        for p in root.rglob(f"*{CHECKPOINT_SUFFIX}")
        if not p.name.startswith(".")  # skip AppleDouble / hidden sidecars
    )
    if not checkpoints:
        raise RuntimeError(f"No checkpoints (*{CHECKPOINT_SUFFIX}) found under {root}")
    log.info("Found %d checkpoint(s) to resume.", len(checkpoints))

    summary_rows: list[dict[str, Any]] = []
    for i, ck_path in enumerate(checkpoints):
        stem = ck_path.name[: -len(CHECKPOINT_SUFFIX)]
        row: dict[str, Any] = {
            "dir": str(ck_path.parent),
            "blanked": "",
            "rpeak": "",
            "slowwave": "",
            "status": "ok",
            "reason": "",
            "n_cuffs": 0,
            "n_spikes_total": 0,
            "n_clusters_total": 0,
            "mean_snr": float("nan"),
            "n_responders": 0,
            "output_path": "",
            "mode": "resume",
            "checkpoint_path": str(ck_path),
        }
        try:
            data = load_checkpoint(ck_path)
            prov = data["provenance"]
            row["blanked"] = Path(prov.get("blanked_path", "")).name
            row["rpeak"] = Path(prov.get("rpeak_path", "")).name
            row["slowwave"] = Path(prov.get("slowwave_path", "")).name if prov.get("slowwave_path") else ""

            cuff_results: list[dict[str, Any]] = []
            for k, cuff_data in enumerate(data["cuffs"]):
                log.info("[%s] Resuming cuff %d/%d", ck_path.name, k + 1, data["n_cuffs"])
                # Re-project waveforms through the resumed basis so the feature
                # space matches what Pass 1 produced even if the saved
                # pca_feats were computed against the same basis.
                from .features import project_pca
                pca_feats = project_pca(cuff_data["waveforms"], pca_basis)
                cuff_data["pca_feats"] = pca_feats
                cuff_results.append(
                    run_postpass_on_cuff(
                        prepass=cuff_data,
                        n_samples=data["n_samples"],
                        rpeak_samples=data["rpeak_samples"],
                        slowwave=data["slowwave"],
                        stim_events=data["stim_events"],
                        fs=data["fs"],
                        cfg=cfg,
                    )
                )

            # Build the same provenance + assembled dict the full-mode path
            # produces, so downstream consumers see one schema.
            assembled = {
                "provenance": {
                    **prov,
                    "pca_basis_path": str(basis_path),
                    "resumed_from_checkpoint": str(ck_path),
                },
                "fs": data["fs"],
                "n_cuffs": data["n_cuffs"],
                "cuff": cuff_results,
            }
            out_path = save_mat(assembled, ck_path.parent, stem)
            row["output_path"] = str(out_path)
            row["n_cuffs"] = data["n_cuffs"]
            spikes_tot, clust_tot, snrs, resp_tot = 0, 0, [], 0
            for cuff in cuff_results:
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
            log.error("Resume on %s failed:\n%s", ck_path, traceback.format_exc())
        summary_rows.append(row)
        if progress_cb:
            progress_cb("resume", i + 1, len(checkpoints))

    summary_path = root / "batch_summary_resume.csv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    log.info("Wrote resume summary: %s", summary_path)

    return {
        "pairs": [],
        "pca_basis_path": str(basis_path),
        "summary_path": str(summary_path),
        "rows": summary_rows,
        "mode": "resume",
    }
