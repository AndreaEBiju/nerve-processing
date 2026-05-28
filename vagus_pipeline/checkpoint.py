"""Per-pair checkpoint persistence for split-machine workflows.

The pipeline can be split across two machines:

1. **Prepass** (machine A, typically a Windows box without MountainSort5):
     - discover pairs;
     - fit the global PCA basis from pooled waveforms;
     - for each pair, run Steps 1-5 per cuff (bandpass, noise sigma,
       spike detection, waveform extraction, feature projection);
     - write ``<stem>_checkpoint.npz`` next to each source pair.

2. **Resume** (machine B, typically a Mac with MountainSort5 installed):
     - re-discover the same root (or just point at a single checkpoint
       on local disk);
     - load each checkpoint;
     - run Steps 6-14 per cuff (sort, quality, audit, cardiac,
       respiration, slow-wave, rates, responder, fibre-type);
     - save the final ``<stem>_metrics.mat`` next to the checkpoint.

Why this split exists
---------------------
MountainSort5's C++ dependency ``isosplit6`` does not currently ship a
pre-built wheel for Windows + Python 3.12, but Mac and Linux do.  At the
same time, the original .mat recordings often live in cloud storage
where opening many files at once causes problems (Google Drive on macOS
will try to download every selected file).  The split lets you do the
bulk preprocessing on Windows (which streams from cloud storage fine)
and then selectively bring just one small ``_checkpoint.npz`` at a time
across to the Mac to finish the heavy clustering step.

Checkpoint contents
-------------------
Everything Steps 6-14 need to resume:

  fs, n_samples, n_cuffs, rpeak_samples, slowwave (optional),
  stim_samples/stim_labels (optional), provenance (paths, var_map,
  config, software version)

Plus, per cuff k:

  cuffk_neural_raw       original notched/blanked input (kept for traceability)
  cuffk_filtered         bandpass-filtered trace          <-- needed for MountainSort5
  cuffk_blanked_mask     boolean mask of blanked samples
  cuffk_sigma_track      sliding MAD-sigma estimate
  cuffk_sigma_times      sample positions of sigma centres
  cuffk_spike_samples    detected spike sample indices
  cuffk_waveforms        n x L waveform matrix
  cuffk_pca_feats        n x n_pca projection onto the global basis
  cuffk_scalar_<name>    per-spike scalar features
  cuffk_amp_hist_*       amplitude-histogram diagnostic

The filtered signal is the largest array (~one float32 per original
sample); the rest is small.  Compressed (.npz with ``np.savez_compressed``)
a 10-minute recording is typically 80-150 MB per cuff.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import PipelineConfig, VarMap
from .io_discovery import RecordingPair
from .io_load import Recording

log = logging.getLogger("vagus.checkpoint")

CHECKPOINT_VERSION = 1
CHECKPOINT_SUFFIX = "_checkpoint.npz"


def checkpoint_path_for(pair: RecordingPair) -> Path:
    return pair.dir / f"{pair.common_stem()}{CHECKPOINT_SUFFIX}"


def save_checkpoint(
    out_path: Path,
    pair: RecordingPair,
    recording: Recording,
    per_cuff_prepass: list[dict[str, Any]],
    pca_basis_path: Path | str,
    var_map: VarMap,
    cfg: PipelineConfig,
) -> Path:
    """Persist a full prepass snapshot for one pair."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    provenance = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "software_version": __version__,
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "blanked_path": str(pair.blanked_path),
        "rpeak_path": str(pair.rpeak_path),
        "slowwave_path": str(pair.slowwave_path) if pair.slowwave_path else "",
        "pca_basis_path": str(pca_basis_path),
        "var_map": var_map.to_dict(),
        "config": cfg.to_dict(),
    }

    arrays: dict[str, np.ndarray] = {
        "_checkpoint_version": np.asarray(CHECKPOINT_VERSION),
        "_provenance_json": np.asarray(json.dumps(provenance)),
        "fs": np.asarray(recording.fs),
        "n_samples": np.asarray(recording.n_samples),
        "n_cuffs": np.asarray(len(recording.neural)),
        "rpeak_samples": recording.rpeak_samples,
    }
    if recording.slowwave is not None:
        arrays["slowwave"] = recording.slowwave
    if recording.stim_events:
        arrays["stim_samples"] = np.asarray(
            [s for s, _ in recording.stim_events], dtype=np.int64
        )
        arrays["stim_labels"] = np.asarray(
            [l for _, l in recording.stim_events], dtype=object
        ).astype(str)

    for k, res in enumerate(per_cuff_prepass):
        arrays[f"cuff{k}_neural_raw"] = recording.neural[k]
        arrays[f"cuff{k}_filtered"] = res["filtered"]
        arrays[f"cuff{k}_blanked_mask"] = recording.blanked_mask[k]
        arrays[f"cuff{k}_sigma_track"] = res["sigma_track"]
        arrays[f"cuff{k}_sigma_times"] = res["sigma_times"]
        arrays[f"cuff{k}_spike_samples"] = res["spike_samples"]
        arrays[f"cuff{k}_waveforms"] = res["waveforms"]
        arrays[f"cuff{k}_pca_feats"] = res["pca_feats"]
        arrays[f"cuff{k}_amp_hist_counts"] = res["amp_hist"]["counts"]
        arrays[f"cuff{k}_amp_hist_edges"] = res["amp_hist"]["edges"]
        for name, v in res["scalar_feats"].items():
            arrays[f"cuff{k}_scalar_{name}"] = np.asarray(v)

    np.savez_compressed(out_path, **arrays)
    size_mb = out_path.stat().st_size / 1024 / 1024
    log.info("Wrote checkpoint %s (%.1f MB)", out_path.name, size_mb)
    return out_path


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint into a dict mirroring :class:`Recording` plus
    per-cuff prepass results."""
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    version = int(z["_checkpoint_version"])
    if version != CHECKPOINT_VERSION:
        log.warning(
            "Checkpoint %s is version %d but this code expects %d; attempting to load anyway.",
            path.name, version, CHECKPOINT_VERSION,
        )

    provenance = json.loads(str(z["_provenance_json"]))
    n_cuffs = int(z["n_cuffs"])
    cuffs: list[dict[str, Any]] = []
    for k in range(n_cuffs):
        scalar = {
            name[len(f"cuff{k}_scalar_"):]: z[name]
            for name in z.files
            if name.startswith(f"cuff{k}_scalar_")
        }
        cuffs.append(
            {
                "neural_raw": z[f"cuff{k}_neural_raw"],
                "filtered": z[f"cuff{k}_filtered"],
                "blanked_mask": z[f"cuff{k}_blanked_mask"],
                "sigma_track": z[f"cuff{k}_sigma_track"],
                "sigma_times": z[f"cuff{k}_sigma_times"],
                "spike_samples": z[f"cuff{k}_spike_samples"],
                "waveforms": z[f"cuff{k}_waveforms"],
                "pca_feats": z[f"cuff{k}_pca_feats"],
                "scalar_feats": scalar,
                "amp_hist": {
                    "counts": z[f"cuff{k}_amp_hist_counts"],
                    "edges": z[f"cuff{k}_amp_hist_edges"],
                },
            }
        )

    stim_events = None
    if "stim_samples" in z.files:
        labels_arr = z["stim_labels"] if "stim_labels" in z.files else None
        samples = z["stim_samples"]
        labels = [str(l) for l in labels_arr.tolist()] if labels_arr is not None else [f"cond_{i}" for i in range(samples.size)]
        stim_events = list(zip(samples.astype(int).tolist(), labels))

    return {
        "provenance": provenance,
        "fs": float(z["fs"]),
        "n_samples": int(z["n_samples"]),
        "n_cuffs": n_cuffs,
        "rpeak_samples": z["rpeak_samples"],
        "slowwave": z["slowwave"] if "slowwave" in z.files else None,
        "stim_events": stim_events,
        "cuffs": cuffs,
    }
