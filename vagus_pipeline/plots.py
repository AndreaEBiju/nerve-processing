"""Per-cluster diagnostic plots.

Currently produces, for each pair:

* ``<stem>_sw_traces_raster.png`` -- the three slow-wave channels + the
  common-mode reference, overlaid with a spike-time raster per cluster.
* ``<stem>_burst_xcorr.png`` -- spike-to-burst cross-correlogram per
  cluster.

Matplotlib is imported lazily so headless runs without a backend don't crash.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("vagus.plots")


def save_pair_diagnostics(
    results: dict[str, Any],
    out_dir: Path | str,
    stem: str,
) -> list[Path]:
    """Render diagnostic plots for one pair's results dict.  Returns the list
    of file paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log.warning("matplotlib unavailable; skipping plots (%s).", e)
        return []

    written: list[Path] = []
    for k, cuff in enumerate(results.get("cuff", [])):
        sw11a = cuff.get("step11a", {})
        sw11b = cuff.get("step11b", {})
        if sw11a.get("skipped") or sw11b.get("skipped"):
            continue
        cm = np.asarray(sw11a.get("common_mode", []))
        cm_fs = float(sw11a.get("common_mode_fs_hz", 10.0))
        if cm.size:
            fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            t_cm = np.arange(cm.size) / cm_fs
            axes[0].plot(t_cm, cm, color="k", lw=1.2, label="common-mode")
            axes[0].set_ylabel("slow-wave (norm)")
            axes[0].legend(loc="upper right")
            axes[0].set_title(f"cuff {k+1} -- common-mode reference + raster")
            cluster = sw11b.get("cluster", []) or []
            # Per-cluster raster
            for ci, cl in enumerate(cluster):
                pl = bool(cl.get("robust_phase_locked", False))
                # Build per-cluster spike times from step3 (in samples).
                labels = np.asarray(cuff["step6"]["labels"]).ravel()
                spikes = np.asarray(cuff["step3"]["spike_samples"]).ravel()
                fs = float(results.get("fs", 24414.0))
                sp_t = spikes[labels == cl["cluster_id"]] / fs
                color = "C1" if pl else "C0"
                axes[1].vlines(sp_t, ci - 0.4, ci + 0.4, color=color, lw=0.5)
            axes[1].set_ylabel("cluster id")
            axes[1].set_xlabel("time (s)")
            axes[1].set_yticks(list(range(len(cluster))))
            out = out_dir / f"{stem}_cuff{k+1}_sw_traces_raster.png"
            fig.tight_layout()
            fig.savefig(out, dpi=140)
            plt.close(fig)
            written.append(out)

        # Burst xcorr
        s12 = cuff.get("step12", {})
        if not s12.get("skipped"):
            clusters = s12.get("cluster", []) or []
            if clusters:
                n = len(clusters)
                ncol = min(n, 3)
                nrow = int(np.ceil(n / ncol))
                fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), sharex=True)
                axes = np.atleast_1d(axes).flatten()
                for i, cl in enumerate(clusters):
                    ax = axes[i]
                    lag = np.asarray(cl["lag_axis_s"])
                    xc = np.asarray(cl["xcorr"])
                    ax.bar(lag, xc, width=(lag[1] - lag[0]) if lag.size > 1 else 0.1, color="C0")
                    ax.axvline(0.0, color="k", lw=0.5, ls="--")
                    ax.set_title(f"cl {cl['cluster_id']} -- {cl['direction_tag']}  z={cl['peak_z']:.1f}")
                    ax.set_xlabel("lag (s)")
                for j in range(n, len(axes)):
                    axes[j].set_visible(False)
                out = out_dir / f"{stem}_cuff{k+1}_burst_xcorr.png"
                fig.tight_layout()
                fig.savefig(out, dpi=140)
                plt.close(fig)
                written.append(out)
    if written:
        log.info("Wrote %d diagnostic plot(s) to %s", len(written), out_dir)
    return written
