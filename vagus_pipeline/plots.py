"""Per-cuff diagnostic plots, per IMPLEMENTATION.md §9.

Public entry points
-------------------
* :func:`render_diagnostics` -- new spec-conformant renderer.  Writes
  ``<stem>_cuff<k>_summary.pdf`` + ``<stem>_cuff<k>_supplementary.pdf``
  per cuff and one ``<stem>_cuff<k>_panelNN_<name>.png`` per panel slot.
* :func:`save_pair_diagnostics` -- legacy shim used by ``batch.py``.
  Delegates to :func:`render_diagnostics` so the existing batch wiring
  keeps working unchanged.

Design notes
------------
* Matplotlib is imported with the ``Agg`` backend so headless runs do not
  need an X server.
* Many panels need trace arrays (``filtered``, ``blanked_mask``,
  ``rpeak_samples``, ``slowwave_channels`` ...) that the pipeline does
  not currently persist into ``.mat`` (those would balloon file size).
  The renderer accepts an optional ``extras_per_cuff[k]`` dict carrying
  them so panels can render real content; when the extras are absent the
  affected panels show a centred "data missing" message instead of
  raising.  See :func:`_missing` and the top of every panel function.
* Each panel function never raises -- failures are caught at the
  per-panel level and replaced with a "panel failed: <e>" message so a
  single broken panel does not destroy the whole summary page.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("vagus.plots")

# Colour-blind safe categorical palette (Wong 2011).
WONG = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]


# ---------------------------------------------------------------------------
# Tiny utilities
# ---------------------------------------------------------------------------

def _cluster_colour(cluster_id: int) -> str:
    """Map a cluster id to a stable Wong-palette colour."""
    return WONG[(cluster_id % (len(WONG) - 1)) + 1]  # skip black


def _missing(ax, reason: str, *, polar: bool = False) -> None:
    """Render a centred grey 'data missing' message on ``ax``."""
    try:
        if polar:
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0, 0, reason, ha="center", va="center", fontsize=7,
                    color="gray", style="italic")
            return
        ax.set_axis_off()
        ax.text(0.5, 0.5, reason, ha="center", va="center", fontsize=7,
                wrap=True, transform=ax.transAxes, color="gray", style="italic")
    except Exception:
        pass


def _safe_panel(panel_name: str, fn, *args, **kwargs) -> None:
    """Run a panel callable; on exception draw the error in its primary axes."""
    try:
        fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        log.warning("Panel %s failed: %s\n%s", panel_name, e, traceback.format_exc())
        first = args[0] if args else None
        targets = first if isinstance(first, (list, tuple, np.ndarray)) else [first]
        for ax in targets:
            if ax is None:
                continue
            try:
                _missing(ax, f"panel {panel_name} failed:\n{type(e).__name__}: {e}")
            except Exception:
                pass


def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        return list(x)
    return [x]


def _step(cuff: dict, name: str) -> dict:
    """Return ``cuff[name]`` or an empty dict if missing/None."""
    s = cuff.get(name)
    return s if isinstance(s, dict) else {}


def _is_skipped(step: dict) -> bool:
    return bool(step.get("skipped"))


def _short_reason(step: dict, default: str = "step skipped") -> str:
    r = step.get("reason")
    return str(r) if r else default


# ---------------------------------------------------------------------------
# Lazy matplotlib import + figure helpers
# ---------------------------------------------------------------------------

def _import_mpl():
    """Import matplotlib lazily with the Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages  # noqa: F401
    from matplotlib import gridspec  # noqa: F401
    return matplotlib, plt


# ===========================================================================
# Panel implementations
# ===========================================================================

def panel_1a_psd_overlay(ax, results, k, extras, cfg) -> None:
    """Raw vs bandpassed PSD overlay."""
    filtered = (extras or {}).get("filtered")
    raw = (extras or {}).get("neural_raw")
    fs = float(results.get("fs", 24414.0))
    if filtered is None and raw is None:
        _missing(ax, "Panel 1a: raw/filtered trace not in extras\n"
                     "(pass via extras_per_cuff[k])")
        return
    from scipy.signal import welch

    def _midwin(x):
        n = int(min(120 * fs, x.size))
        m = x.size // 2
        return x[max(0, m - n // 2): min(x.size, m + n // 2)]

    ax.set_xscale("log"); ax.set_yscale("log")
    if raw is not None:
        x = _midwin(np.asarray(raw, dtype=np.float64))
        if x.size:
            f, pxx = welch(x, fs=fs, nperseg=min(8192, x.size))
            ax.plot(f[1:], pxx[1:], color="gray", lw=0.8, label="raw")
    if filtered is not None:
        x = _midwin(np.asarray(filtered, dtype=np.float64))
        if x.size:
            f, pxx = welch(x, fs=fs, nperseg=min(8192, x.size))
            ax.plot(f[1:], pxx[1:], color=WONG[5], lw=1.0, label="bandpassed")
    ax.axvspan(10, 40, color="#ffd6d6", alpha=0.35)
    ax.axvspan(1, 3, color="#d6e2ff", alpha=0.35)
    bp_lo = float(_step(results["cuff"][k], "step1").get("bp_low", 300))
    bp_hi = float(_step(results["cuff"][k], "step1").get("bp_high", 5000))
    ax.axvspan(bp_lo, bp_hi, color="#dff5e0", alpha=0.35)
    ax.set_xlabel("Hz", fontsize=7); ax.set_ylabel("PSD", fontsize=7)
    ax.set_title("1a  PSD raw vs band", fontsize=8)
    ax.legend(loc="lower left", fontsize=6)
    ax.tick_params(labelsize=6)


def panel_1b_fibre_bands(ax, results, k, extras, cfg) -> None:
    """Fibre-class band-power bars."""
    filtered = (extras or {}).get("filtered")
    fs = float(results.get("fs", 24414.0))
    if filtered is None:
        _missing(ax, "Panel 1b: filtered trace not in extras")
        return
    from scipy.signal import welch
    x = np.asarray(filtered, dtype=np.float64)
    n = int(min(120 * fs, x.size)); m = x.size // 2
    x = x[max(0, m - n // 2): min(x.size, m + n // 2)]
    if x.size == 0:
        _missing(ax, "Panel 1b: empty window"); return
    f, pxx = welch(x, fs=fs, nperseg=min(8192, x.size))
    df = f[1] - f[0] if f.size > 1 else 1.0
    bands = [
        ("A-like\n1500-5000", 1500, min(5000, fs / 2 - 1), WONG[1]),
        ("B/Aδ-like\n800-2000", 800, 2000, WONG[2]),
        ("C-like\n200-800", 200, 800, WONG[3]),
    ]
    labels, vals, colours = [], [], []
    for name, lo, hi, c in bands:
        m_band = (f >= lo) & (f <= hi)
        power = float(pxx[m_band].sum() * df)
        labels.append(name); vals.append(max(power, 1e-30)); colours.append(c)
    ax.bar(labels, vals, color=colours, edgecolor="black", linewidth=0.4)
    ax.set_yscale("log"); ax.set_ylabel("∫PSD (µV²)", fontsize=7)
    ax.set_title("1b  fibre-band power", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.text(0.01, 0.99,
            "Spectral fibre composition is\nsuggestive only; firm\n"
            "assignment needs CV/pharm.",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.5,
            color="#444", style="italic")


def panel_2_sigma_thr(ax, results, k, extras, cfg) -> None:
    """Sigma vs time + threshold line."""
    cuff = results["cuff"][k]
    s2 = _step(cuff, "step2"); s3 = _step(cuff, "step3")
    sig = np.asarray(s2.get("sigma_track", []), dtype=np.float64)
    t_samp = np.asarray(s2.get("sigma_times", []), dtype=np.float64)
    if sig.size == 0 or t_samp.size == 0:
        _missing(ax, "Panel 2: sigma track empty"); return
    fs = float(results.get("fs", 24414.0))
    t = t_samp / fs  # sigma_times is sample indices; convert to seconds
    thr_sigma = float(s3.get("threshold_sigma", 4.5))
    med_sigma = float(np.nanmedian(sig))
    ax.plot(t, sig, color="black", lw=0.8)
    ax.axhline(thr_sigma * med_sigma, ls="--", color=WONG[6], lw=0.8,
               label=f"{thr_sigma:.1f}·median σ")
    bm = (extras or {}).get("blanked_mask")
    blanked_frac = float("nan")
    if bm is not None:
        bm = np.asarray(bm).astype(bool)
        blanked_frac = float(bm.mean())
        if bm.any():
            d = np.diff(bm.astype(np.int8))
            starts = list(np.where(d == 1)[0] + 1)
            ends   = list(np.where(d == -1)[0] + 1)
            if bm[0]:  starts = [0] + starts
            if bm[-1]: ends   = ends + [bm.size]
            for s_i, e_i in zip(starts, ends):
                ax.axvspan(s_i / fs, e_i / fs, color="lightgray", alpha=0.4)
    ax.set_xlabel("time (s)", fontsize=7); ax.set_ylabel("σ", fontsize=7)
    title = f"2  σ track  (med={med_sigma:.3g}"
    if np.isfinite(blanked_frac):
        title += f", blanked={blanked_frac*100:.1f}%"
    title += ")"
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(labelsize=6)


def panel_3_example_trace(ax, results, k, extras, cfg) -> None:
    """5-s example trace centred where σ ≈ median σ."""
    cuff = results["cuff"][k]
    s2 = _step(cuff, "step2"); s3 = _step(cuff, "step3")
    filtered = (extras or {}).get("filtered")
    if filtered is None:
        _missing(ax, "Panel 3: filtered trace not in extras"); return
    filtered = np.asarray(filtered, dtype=np.float32)
    fs = float(results.get("fs", 24414.0))
    sig = np.asarray(s2.get("sigma_track", []), dtype=np.float64)
    t_sig_samp = np.asarray(s2.get("sigma_times", []), dtype=np.float64)
    if sig.size and t_sig_samp.size == sig.size:
        med = float(np.nanmedian(sig))
        idx = int(np.argmin(np.abs(sig - med)))
        # sigma_times is sample indices, not seconds
        centre_s = float(t_sig_samp[idx]) / fs
    else:
        centre_s = filtered.size / (2 * fs)
    half = 2.5
    a = max(0.0, centre_s - half)
    b = min(filtered.size / fs, centre_s + half)
    si, ei = int(a * fs), int(b * fs)
    if ei <= si:
        _missing(ax, "Panel 3: window empty"); return
    seg = filtered[si:ei]
    ts = np.arange(seg.size) / fs + a
    ax.plot(ts, seg, color="black", lw=0.5)
    spikes = np.asarray(s3.get("spike_samples", []), dtype=np.int64)
    in_win = spikes[(spikes >= si) & (spikes < ei)]
    if seg.size:
        y_high = float(np.nanmax(seg) * 1.05)
        if not np.isfinite(y_high) or y_high == 0:
            y_high = 1.0
        ax.vlines(in_win / fs, y_high * 0.97, y_high, color="red", lw=0.5)
    ax.set_xlabel("time (s)", fontsize=7)
    ax.set_title(f"3  example  t={a:.1f}-{b:.1f}s  n={in_win.size}", fontsize=8)
    ax.tick_params(labelsize=6)


def panel_4_amp_hist(ax, results, k, extras, cfg) -> None:
    """Peak-to-peak amplitude histogram with threshold line."""
    cuff = results["cuff"][k]
    s5 = _step(cuff, "step5"); s3 = _step(cuff, "step3")
    feats = s5.get("scalar_feats", {}) or {}
    p2p = None
    if isinstance(feats, dict):
        p2p_raw = feats.get("p2p")
        if p2p_raw is not None:
            p2p = np.asarray(p2p_raw, dtype=np.float64)
    if p2p is None or p2p.size == 0:
        cnt = np.asarray(s3.get("amplitude_hist_counts", []), dtype=np.float64)
        edg = np.asarray(s3.get("amplitude_hist_edges", []), dtype=np.float64)
        if cnt.size > 0 and edg.size == cnt.size + 1:
            centres = 0.5 * (edg[1:] + edg[:-1])
            ax.bar(centres, np.maximum(cnt, 0.5), width=np.diff(edg),
                   color=WONG[5], edgecolor="none")
            ax.set_yscale("log")
            ax.set_xlabel("|amp| at spike", fontsize=7)
            ax.set_title("4  amp hist (prepass)", fontsize=8)
            ax.tick_params(labelsize=6); return
        _missing(ax, "Panel 4: no amplitude features"); return
    p2p = p2p[np.isfinite(p2p)]
    if p2p.size == 0:
        _missing(ax, "Panel 4: p2p all non-finite"); return
    ax.hist(p2p, bins=60, color=WONG[5], edgecolor="none")
    ax.set_yscale("log")
    thr_sigma = float(s3.get("threshold_sigma", 4.5))
    ax.axvline(thr_sigma, ls="--", color=WONG[6], lw=0.6, alpha=0.6)
    title = "4  amp hist (p2p)"
    try:
        cnt, edg = np.histogram(p2p, bins=60)
        below = cnt[edg[:-1] < thr_sigma]
        above = cnt[edg[:-1] >= thr_sigma]
        if below.size and above.size and above.size > 1 and below.max() > 2 * above[0]:
            title += "  HARD CUTOFF"
    except Exception:
        pass
    ax.set_title(title, fontsize=8, color="red" if "HARD" in title else "black")
    ax.set_xlabel("p2p", fontsize=7); ax.tick_params(labelsize=6)


def panel_5_mean_waveforms(axes, results, k, extras, cfg) -> None:
    """Per-cluster mean waveform overlay (small multiples)."""
    cuff = results["cuff"][k]
    clusters = _as_list(_step(cuff, "step7").get("cluster", []))
    s4 = _step(cuff, "step4")
    pre_ms  = float(s4.get("wf_pre_ms", 1.0))
    post_ms = float(s4.get("wf_post_ms", 2.0))
    for i, ax in enumerate(axes):
        if i >= len(clusters):
            ax.set_axis_off(); continue
        cl = clusters[i]
        mw = np.asarray(cl.get("mean_wf", []), dtype=np.float64)
        sw = np.asarray(cl.get("std_wf", []), dtype=np.float64)
        if mw.size == 0:
            _missing(ax, "no waveform"); continue
        t_ms = np.linspace(-pre_ms, post_ms, mw.size)
        col = _cluster_colour(int(cl.get("cluster_id", i)))
        ax.plot(t_ms, mw, color=col, lw=1.0)
        if sw.size == mw.size:
            ax.fill_between(t_ms, mw - sw, mw + sw, color=col, alpha=0.2, linewidth=0)
        ax.axvline(0, color="gray", lw=0.4, ls=":")
        snr = float(cl.get("snr", float("nan")))
        ax.set_title(f"c{int(cl.get('cluster_id', i))}  n={int(cl.get('n_spikes', 0))} "
                     f"SNR={snr:.1f}", fontsize=6.5)
        ax.tick_params(labelsize=5)
        if i == 0:
            ax.set_ylabel("µV", fontsize=6)


def panel_6_umap_ari(ax, results, k, extras, cfg) -> None:
    """2-D UMAP scatter coloured by MS labels + ARI annotation."""
    cuff = results["cuff"][k]
    s8 = _step(cuff, "step8")
    xy = np.asarray(s8.get("umap_xy", []), dtype=np.float64)
    if xy.ndim != 2 or xy.shape[0] == 0:
        _missing(ax, "Panel 6: UMAP not run\n(too few spikes)"); return
    labels = np.asarray(_step(cuff, "step6").get("labels", []), dtype=np.int64)
    if labels.size != xy.shape[0]:
        n = min(labels.size, xy.shape[0])
        labels = labels[:n]; xy = xy[:n]
    for cid in sorted(set(int(l) for l in labels)):
        m = labels == cid
        if not m.any():
            continue
        col = "lightgray" if cid < 0 else _cluster_colour(cid)
        ax.scatter(xy[m, 0], xy[m, 1], s=2, c=col, alpha=0.6, linewidths=0,
                   label=f"c{cid}" if cid >= 0 else "noise")
    ari = float(s8.get("adjusted_rand", float("nan")))
    ari_col = "red" if (not np.isfinite(ari) or ari < 0.4) else "green"
    ax.text(0.97, 0.97, f"ARI={ari:.2f}", ha="right", va="top",
            transform=ax.transAxes, fontsize=7, weight="bold", color=ari_col)
    ax.set_title("6  UMAP (MS labels)", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    if labels.size and (labels >= 0).any():
        ax.legend(loc="lower left", fontsize=5, markerscale=2, framealpha=0.7)


def panel_7_isi_hists(axes, results, k, extras, cfg) -> None:
    """Per-cluster ISI histograms, log-x."""
    cuff = results["cuff"][k]
    clusters = _as_list(_step(cuff, "step7").get("cluster", []))
    refr_s = float(getattr(cfg, "refractory_ms", 1.0)) * 1e-3 if cfg else 1e-3
    bins = np.logspace(np.log10(0.0005), np.log10(10.0), 31)
    for i, ax in enumerate(axes):
        if i >= len(clusters):
            ax.set_axis_off(); continue
        cl = clusters[i]
        isi = np.asarray(cl.get("isi_s", []), dtype=np.float64)
        isi = isi[isi > 0]
        if isi.size == 0:
            _missing(ax, "no ISI"); continue
        col = _cluster_colour(int(cl.get("cluster_id", i)))
        ax.hist(isi, bins=bins, color=col, edgecolor="none")
        ax.set_xscale("log")
        ax.axvline(refr_s, ls="--", color="black", lw=0.5)
        viol = float(cl.get("isi_violation_rate", 0.0)) * 100
        ax.set_title(f"c{int(cl.get('cluster_id', i))} viol={viol:.1f}%",
                     fontsize=6.5)
        ax.tick_params(labelsize=5)


def _fano_curve_from_spikes(spikes_samples: np.ndarray, fs: float,
                            duration_s: float, T_axis: np.ndarray,
                            min_count: int = 5) -> np.ndarray:
    """Live Fano F(T) = var(count)/mean(count) computation."""
    out = np.full(T_axis.size, np.nan)
    if spikes_samples.size == 0 or duration_s <= 0:
        return out
    sp_s = spikes_samples.astype(np.float64) / fs
    for i, T in enumerate(T_axis):
        if T <= 0:
            continue
        nbins = int(max(1, duration_s // T))
        if nbins < 2:
            continue
        edges = np.linspace(0.0, nbins * T, nbins + 1)
        counts, _ = np.histogram(sp_s, bins=edges)
        m = float(counts.mean())
        if m < min_count:
            continue
        out[i] = float(counts.var(ddof=1) / m) if m > 0 else np.nan
    return out


def panel_7b_fano_vs_t(axes, results, k, extras, cfg) -> None:
    """Per-cluster Fano F(T) vs window size T."""
    cuff = results["cuff"][k]
    clusters = _as_list(_step(cuff, "step7").get("cluster", []))
    fs = float(results.get("fs", 24414.0))
    spikes_all = np.asarray(_step(cuff, "step3").get("spike_samples", []), dtype=np.int64)
    labels = np.asarray(_step(cuff, "step6").get("labels", []), dtype=np.int64)
    duration_s = float(spikes_all.max() / fs) if spikes_all.size else 0.0
    s2_times_samp = np.asarray(_step(cuff, "step2").get("sigma_times", []),
                               dtype=np.float64)
    if s2_times_samp.size:
        # sigma_times is sample indices; convert to seconds
        duration_s = max(duration_s, float(s2_times_samp[-1]) / fs)
    T_axis = np.logspace(np.log10(0.02), np.log10(10.0), 12)
    refr_shade = float(getattr(cfg, "fano_refractory_shading_below_s", 0.05) if cfg else 0.05)
    canonical = float(getattr(cfg, "fano_canonical_T_s", 1.0) if cfg else 1.0)
    for i, ax in enumerate(axes):
        if i >= len(clusters):
            ax.set_axis_off(); continue
        cl = clusters[i]; cid = int(cl.get("cluster_id", i))
        F = np.asarray(cl.get("fano_curve", []), dtype=np.float64)
        T = np.asarray(cl.get("fano_T_axis_s", []), dtype=np.float64)
        if F.size == 0 or T.size != F.size:
            cluster_spikes = (spikes_all[labels == cid]
                              if labels.size == spikes_all.size else spikes_all)
            T = T_axis
            F = _fano_curve_from_spikes(cluster_spikes, fs, duration_s, T)
        mask = np.isfinite(F)
        col = _cluster_colour(cid)
        ax.axhline(1.0, ls="--", color="gray", lw=0.5)
        if T.size:
            ax.axvspan(float(T.min()), refr_shade, color="lightgray", alpha=0.3)
        ax.plot(T[mask], F[mask], color=col, marker="o", lw=0.8, ms=2.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        if mask.any():
            j = int(np.argmin(np.abs(T - canonical)))
            f1 = F[j] if mask[j] else float("nan")
        else:
            f1 = float("nan")
        m_slope = mask & (T > refr_shade)
        if int(m_slope.sum()) >= 3:
            try:
                slope = float(np.polyfit(np.log10(T[m_slope]),
                                         np.log10(F[m_slope]), 1)[0])
            except Exception:
                slope = float("nan")
        else:
            slope = float("nan")
        ax.set_title(f"c{cid}  F(1s)={f1:.2f}  s={slope:+.2f}", fontsize=6.5)
        ax.tick_params(labelsize=5)


def panel_8a_prwh_stack(ax_img, ax_bars, results, k, extras, cfg) -> None:
    """Stacked peri-R-wave histogram image + per-cluster peak-z bar."""
    cuff = results["cuff"][k]
    s9 = _step(cuff, "step9")
    clusters = _as_list(s9.get("cluster", []))
    if not clusters:
        _missing(ax_img, "Panel 8a: no peri-Rwave data")
        _missing(ax_bars, ""); return
    rows, peakz, locked, cids = [], [], [], []
    bin_ms = float(s9.get("bin_ms", 1.0))
    for cl in clusters:
        prwh = np.asarray(cl.get("prwh", []), dtype=np.float64)
        if prwh.size == 0:
            continue
        rows.append(prwh)
        peakz.append(float(cl.get("peak_z", 0.0)))
        locked.append(bool(cl.get("is_cardiac_locked", False)))
        cids.append(int(cl.get("cluster_id", -1)))
    if not rows:
        _missing(ax_img, "Panel 8a: empty PRWH")
        _missing(ax_bars, ""); return
    order = np.argsort(-np.asarray(peakz))
    img = np.stack([rows[i] for i in order], axis=0)
    cids_o = [cids[i] for i in order]
    peakz_o = [peakz[i] for i in order]
    locked_o = [locked[i] for i in order]
    n_lag = img.shape[1]
    lag_max_ms = (n_lag - 1) / 2 * bin_ms
    extent = [-lag_max_ms, lag_max_ms, len(rows) - 0.5, -0.5]
    ax_img.imshow(img, aspect="auto", extent=extent, cmap="viridis",
                  interpolation="nearest")
    ax_img.set_yticks(range(len(rows)))
    ax_img.set_yticklabels([f"c{c}" for c in cids_o], fontsize=5)
    ax_img.set_xlabel("lag (ms)", fontsize=7)
    ax_img.set_title("8a  PRWH", fontsize=8)
    ax_img.tick_params(labelsize=5)
    bar_colours = ["red" if lk else "gray" for lk in locked_o]
    ax_bars.barh(range(len(rows)), peakz_o, color=bar_colours, edgecolor="none")
    ax_bars.set_yticks([]); ax_bars.invert_yaxis()
    ax_bars.set_xlabel("peak z", fontsize=7)
    ax_bars.tick_params(labelsize=5)


def panel_8b_synchrony(ax, results, k, extras, cfg) -> None:
    """Single-row representative synchrony trace.  Full 3-row version
    lives in the supplementary PDF."""
    cuff = results["cuff"][k]
    fs = float(results.get("fs", 24414.0))
    filtered = (extras or {}).get("filtered")
    rpeaks = (extras or {}).get("rpeak_samples")
    s11a = _step(cuff, "step11a")
    cm = np.asarray(s11a.get("common_mode", []), dtype=np.float64)
    cm_fs = float(s11a.get("common_mode_fs_hz", 10.0))
    s12 = _step(cuff, "step12")
    consensus = np.asarray(s12.get("consensus_burst_times_s", []), dtype=np.float64)
    if filtered is None and cm.size == 0:
        _missing(ax, "Panel 8b: traces missing\n(filtered / common-mode)"); return
    clusters = _as_list(_step(cuff, "step7").get("cluster", []))
    rep = max(clusters, key=lambda c: float(c.get("snr", -np.inf)),
              default=None) if clusters else None
    rep_cid = int(rep.get("cluster_id", -1)) if rep else None
    labels = np.asarray(_step(cuff, "step6").get("labels", []), dtype=np.int64)
    spikes = np.asarray(_step(cuff, "step3").get("spike_samples", []), dtype=np.int64)
    n_total = int(filtered.size) if filtered is not None else int(cm.size / cm_fs * fs)
    if n_total <= 0:
        _missing(ax, "Panel 8b: zero-length"); return
    centre = n_total // 2
    half = int(15 * fs)
    a, b = max(0, centre - half), min(n_total, centre + half)
    if filtered is not None:
        seg = np.asarray(filtered)[a:b]
        if seg.size:
            scale = float(np.nanmax(np.abs(seg))) + 1e-12
            ts_f = np.arange(seg.size) / fs + a / fs
            ax.plot(ts_f, seg / scale + 3.0, color="black", lw=0.4)
    if rep_cid is not None and labels.size == spikes.size:
        sp = spikes[labels == rep_cid]
        sp_t = sp[(sp >= a) & (sp < b)] / fs
        ax.vlines(sp_t, 2.7, 3.3, color="red", lw=0.5)
    if rpeaks is not None:
        rp = np.asarray(rpeaks)
        rp_t = rp[(rp >= a) & (rp < b)] / fs
        ax.vlines(rp_t, 1.0, 2.0, color="red", lw=0.7)
    if cm.size:
        cm_t = np.arange(cm.size) / cm_fs
        m = (cm_t >= a / fs) & (cm_t < b / fs)
        if m.any():
            seg_cm = cm[m]
            scale = float(np.nanmax(np.abs(seg_cm))) + 1e-12
            ax.plot(cm_t[m], seg_cm / scale, color=WONG[2], lw=0.6,
                    label="common-mode")
    if consensus.size:
        cs = consensus[(consensus >= a / fs) & (consensus < b / fs)]
        ax.vlines(cs, -0.5, 0.5, color=WONG[1], lw=0.7)
    ax.set_xlim(a / fs, b / fs)
    ax.set_yticks([])
    ax.set_xlabel("time (s)", fontsize=7)
    ax.set_title(f"8b  synchrony (cluster c{rep_cid})", fontsize=8)
    ax.tick_params(labelsize=6)


def panel_9_sw_qc(ax, results, k, extras, cfg) -> None:
    """3 slow-wave channels + QC stripe + common-mode."""
    cuff = results["cuff"][k]
    s11a = _step(cuff, "step11a")
    if _is_skipped(s11a):
        _missing(ax, f"Panel 9: step 11a skipped\n({_short_reason(s11a)})"); return
    cm = np.asarray(s11a.get("common_mode", []), dtype=np.float64)
    cm_fs = float(s11a.get("common_mode_fs_hz", 10.0))
    channels = _as_list(s11a.get("channel", []))
    sw_traces = (extras or {}).get("slowwave_channels")
    fs = float(results.get("fs", 24414.0))
    if cm.size:
        cm_t = np.arange(cm.size) / cm_fs
        scale = float(np.nanmax(np.abs(cm))) + 1e-12
        ax.plot(cm_t, cm / scale, color="black", lw=0.5, label="common-mode")
    if sw_traces is not None:
        for j, tr in enumerate(list(sw_traces)[:3]):
            tr = np.asarray(tr, dtype=np.float64)
            if tr.size == 0:
                continue
            t = np.arange(tr.size) / fs
            scale = float(np.nanmax(np.abs(tr))) + 1e-12
            ax.plot(t, tr / scale + (j + 1), color=WONG[(j % 6) + 1], lw=0.4)
    else:
        ax.text(0.99, 0.99,
                "(channel traces not in extras —\nonly common-mode shown)",
                transform=ax.transAxes, ha="right", va="top", fontsize=5,
                color="gray", style="italic")
    ymax = 4
    for j, ch in enumerate(channels[:3]):
        qs = np.asarray(ch.get("quality_score_rolling", []), dtype=np.float64)
        qt = np.asarray(ch.get("quality_window_times_s", []), dtype=np.float64)
        if qs.size and qt.size == qs.size:
            good_thr = float(getattr(cfg, "sw_quality_good_threshold", 0.5)
                             if cfg else 0.5)
            marg_thr = float(getattr(cfg, "sw_quality_marginal_threshold", 0.3)
                             if cfg else 0.3)
            for tt, sc in zip(qt, qs):
                if not np.isfinite(sc):
                    c = "lightgray"
                elif sc >= good_thr:
                    c = "#cdf2c9"
                elif sc >= marg_thr:
                    c = "#fff2c4"
                else:
                    c = "#f7cccc"
                ax.axvspan(float(tt) - 5.0, float(tt) + 5.0,
                           ymin=(j + 1 - 0.4) / (ymax + 0.5),
                           ymax=(j + 1 + 0.4) / (ymax + 0.5),
                           color=c, alpha=0.4, lw=0)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["common", "ch1", "ch2", "ch3"], fontsize=6)
    ax.set_xlabel("time (s)", fontsize=7)
    ax.set_title("9  slow-wave channels + QC + common-mode", fontsize=8)
    ax.tick_params(labelsize=6)


def panel_10_polar_phase(axes, results, k, extras, cfg) -> None:
    """Per-cluster polar phase histograms vs common-mode slow-wave."""
    cuff = results["cuff"][k]
    s11b = _step(cuff, "step11b")
    if _is_skipped(s11b):
        for ax in axes:
            _missing(ax, f"Panel 10: step 11b skipped\n({_short_reason(s11b)})",
                     polar=True)
        return
    clusters = _as_list(s11b.get("cluster", []))
    for i, ax in enumerate(axes):
        if i >= len(clusters):
            ax.set_axis_off(); continue
        cl = clusters[i]
        hist = np.asarray(cl.get("phase_hist_common", []), dtype=np.float64)
        edges = np.asarray(cl.get("phase_edges_common", []), dtype=np.float64)
        if hist.size == 0:
            _missing(ax, "no phase hist", polar=True); continue
        if edges.size != hist.size + 1:
            edges = np.linspace(-np.pi, np.pi, hist.size + 1)
        centres = 0.5 * (edges[1:] + edges[:-1])
        width = edges[1:] - edges[:-1]
        col = _cluster_colour(int(cl.get("cluster_id", i)))
        ax.bar(centres, hist, width=width, bottom=0.0,
               color=col, edgecolor="none", alpha=0.7)
        mrl = float(cl.get("mrl_common", 0.0))
        pref = float(cl.get("preferred_phase_common", 0.0))
        if mrl > 0 and hist.size:
            r_max = float(hist.max())
            if r_max > 0:
                ax.annotate("", xy=(pref, mrl * r_max), xytext=(pref, 0),
                            arrowprops=dict(arrowstyle="->", color="black", lw=0.6))
        p = float(cl.get("rayleigh_p_common", 1.0))
        ax.set_title(f"c{int(cl.get('cluster_id', i))}  MRL={mrl:.2f} p={p:.2g}",
                     fontsize=6)
        if bool(cl.get("robust_phase_locked", False)):
            ax.set_facecolor("#e8f5e9")
        ax.set_xticks([0, np.pi / 2, np.pi, -np.pi / 2])
        ax.set_xticklabels(["0", "π/2", "π", "-π/2"], fontsize=5)
        ax.set_yticks([])


def panel_11_consistency_polar(ax, results, k, extras, cfg) -> None:
    """Cross-channel preferred-phase consistency, one polar plot."""
    cuff = results["cuff"][k]
    s11b = _step(cuff, "step11b")
    if _is_skipped(s11b):
        _missing(ax, f"Panel 11: step 11b skipped\n({_short_reason(s11b)})",
                 polar=True); return
    clusters = _as_list(s11b.get("cluster", []))
    p_thr = float(getattr(cfg, "sw_primary_p", 0.05) if cfg else 0.05)
    drawn = 0
    for cl in clusters:
        if float(cl.get("rayleigh_p_common", 1.0)) >= p_thr:
            continue
        pref_per_ch = np.asarray(cl.get("preferred_phase_per_channel", []),
                                 dtype=np.float64)
        mrl_per_ch  = np.asarray(cl.get("mrl_per_channel", []), dtype=np.float64)
        pref_cm = float(cl.get("preferred_phase_common", 0.0))
        mrl_cm  = float(cl.get("mrl_common", 0.0))
        col = _cluster_colour(int(cl.get("cluster_id", -1)))
        ax.annotate("", xy=(pref_cm, mrl_cm), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=0.9))
        for ph, mr in zip(pref_per_ch, mrl_per_ch):
            ax.annotate("", xy=(float(ph), float(mr)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.5,
                                        alpha=0.5))
        drawn += 1
    ax.set_title(f"11  consistency  n={drawn}", fontsize=8)
    ax.set_xticks([0, np.pi / 2, np.pi, -np.pi / 2])
    ax.set_xticklabels(["0", "π/2", "π", "-π/2"], fontsize=5)
    ax.set_yticks([])
    if drawn == 0:
        ax.text(0, 0, "no clusters reach\nprimary phase-lock threshold",
                ha="center", va="center", fontsize=6, color="gray",
                style="italic")


def panel_12_burst_xcorr(axes, results, k, extras, cfg) -> None:
    """Spike-to-burst cross-correlogram, small multiples per cluster."""
    cuff = results["cuff"][k]
    s12 = _step(cuff, "step12")
    if _is_skipped(s12):
        for ax in axes:
            _missing(ax, f"Panel 12: step 12 skipped\n({_short_reason(s12)})")
        return
    clusters = _as_list(s12.get("cluster", []))
    for i, ax in enumerate(axes):
        if i >= len(clusters):
            ax.set_axis_off(); continue
        cl = clusters[i]
        lag = np.asarray(cl.get("lag_axis_s", []), dtype=np.float64)
        xc  = np.asarray(cl.get("xcorr", []), dtype=np.float64)
        if lag.size == 0 or xc.size == 0:
            _missing(ax, "no xcorr"); continue
        col = _cluster_colour(int(cl.get("cluster_id", i)))
        tint = {"afferent-like": "#e3f2ff",
                "efferent-like": "#fff4e3"}.get(str(cl.get("direction_tag", "")),
                                                "white")
        ax.set_facecolor(tint)
        w = (lag[1] - lag[0]) if lag.size > 1 else 0.1
        ax.bar(lag, xc, width=w, color=col, edgecolor="none")
        base = xc[(lag < -2.0) | (lag > 2.0)]
        if base.size:
            mu, sd = float(base.mean()), float(base.std())
            ax.axhspan(mu - 2 * sd, mu + 2 * sd, color="gray", alpha=0.18, lw=0)
        peak_lag = float(cl.get("peak_lag_s", 0.0))
        peak_z = float(cl.get("peak_z", 0.0))
        if xc.size:
            ax.plot([peak_lag], [float(xc.max())], "o", color="black", ms=2)
        ax.set_title(f"c{int(cl.get('cluster_id', i))}  "
                     f"{cl.get('direction_tag', '')}  "
                     f"lag={peak_lag:+.2f}s z={peak_z:.1f}", fontsize=6.5)
        ax.set_xlim(-10, 10)
        ax.axvline(0, color="black", lw=0.4, ls=":")
        ax.tick_params(labelsize=5)


def panel_13_rate_traces_stim(ax, results, k, extras, cfg) -> None:
    """Stacked per-cluster rate traces with stim vertical lines."""
    cuff = results["cuff"][k]
    s13 = _step(cuff, "step13")
    clusters = _as_list(s13.get("cluster", []))
    if not clusters:
        _missing(ax, "Panel 13: no rate traces"); return
    t = np.asarray(s13.get("t_centres_s", []), dtype=np.float64)
    if t.size == 0:
        bin_s = float(s13.get("rate_bin_s", 1.0))
        n = int(np.asarray(clusters[0].get("rate_trace", [])).size)
        t = np.arange(n) * bin_s
    responder = _as_list(_step(cuff, "step14").get("cluster", []))
    resp_by_cid = {int(r.get("cluster_id", -1)): r for r in responder}
    for i, cl in enumerate(clusters):
        rate = np.asarray(cl.get("rate_trace", []), dtype=np.float64)
        if rate.size == 0:
            continue
        mx = max(float(rate.max()), 1e-12)
        y = rate / mx + i
        col = _cluster_colour(int(cl.get("cluster_id", i)))
        ax.plot(t[:rate.size], y, color=col, lw=0.5)
        is_resp = False
        for cond in (resp_by_cid.get(int(cl.get("cluster_id", -1)), {})
                                .get("conditions", []) or []):
            if cond.get("is_responder"):
                is_resp = True; break
        tag = "R+" if is_resp else "R-"
        ax.text(t[0] if t.size else 0, i + 0.5,
                f"c{int(cl.get('cluster_id', i))} {tag}",
                fontsize=5, va="center", color=col)
    stim = (extras or {}).get("stim_events") or []
    fs = float(results.get("fs", 24414.0))
    for ev in stim:
        try:
            samp_idx, _label = ev
        except Exception:
            continue
        ax.axvline(float(samp_idx) / fs, color="black", lw=0.5, alpha=0.6)
    ax.set_ylim(-0.2, len(clusters) + 0.2)
    ax.set_yticks([])
    ax.set_xlabel("time (s)", fontsize=7)
    ax.set_title("13  rate traces + stim", fontsize=8)
    ax.tick_params(labelsize=6)


def panel_13b_fano_pre_post(ax, results, k, extras, cfg) -> None:
    """Per-condition Fano pre vs post scatter."""
    cuff = results["cuff"][k]
    responder = _as_list(_step(cuff, "step14").get("cluster", []))
    if not responder:
        _missing(ax, "Panel 13b: no Step 14 data"); return
    pts = []
    for r in responder:
        for cond in r.get("conditions", []) or []:
            if "fano_pre" in cond and "fano_post" in cond:
                pts.append((float(cond["fano_pre"]),
                            float(cond["fano_post"]),
                            int(r.get("cluster_id", -1)),
                            bool(cond.get("is_responder", False)),
                            bool(cond.get("is_regularised", False))))
    if not pts:
        _missing(ax, "Panel 13b: Fano pre/post not\ncomputed (Step 14 missing)"); return
    pre  = np.asarray([p[0] for p in pts])
    post = np.asarray([p[1] for p in pts])
    cids = [p[2] for p in pts]; resp = [p[3] for p in pts]; reg = [p[4] for p in pts]
    cols = [_cluster_colour(c) for c in cids]
    edge = ["red" if rr else "gray" for rr in reg]
    for x, y, c, e, is_resp in zip(pre, post, cols, edge, resp):
        ax.scatter(x, y, s=30, facecolors=c if is_resp else "white",
                   edgecolors=e, linewidths=1.2)
    lo = max(1e-3, float(min(pre.min(), post.min()) * 0.5))
    hi = float(max(pre.max(), post.max()) * 1.5)
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("F_pre", fontsize=7); ax.set_ylabel("F_post", fontsize=7)
    ax.set_title(f"13b  Fano pre/post  n={len(pts)}", fontsize=8)
    ax.tick_params(labelsize=6)


def panel_14a_vagal_vs_cardiac(ax, results, k, extras, cfg) -> None:
    """Vagal respiratory burst rate vs cardiac-surrogate breathing rate."""
    cuff = results["cuff"][k]
    s10 = _step(cuff, "step10")
    br_trace = np.asarray(s10.get("breathing_rate_trace", []), dtype=np.float64)
    vagal_trace = np.asarray(s10.get("vagal_respiratory_rate_trace", []),
                             dtype=np.float64)
    if br_trace.size == 0 or vagal_trace.size == 0:
        br = float(s10.get("breathing_rate_hz", float("nan")))
        bu = float(s10.get("burst_rate_hz", float("nan")))
        ok = bool(s10.get("rate_matches", False))
        flag = "✓" if ok else "✗"
        ax.set_axis_off()
        ax.text(0.5, 0.5,
                f"Panel 14a (no traces)\n\n"
                f"breathing rate: {br:.3f} Hz\n"
                f"vagal burst rate: {bu:.3f} Hz\n"
                f"matches: {ok}  [{flag}]",
                ha="center", va="center", fontsize=7, transform=ax.transAxes)
        return
    t = np.arange(br_trace.size) * float(s10.get("rate_trace_bin_s", 1.0))
    ax.plot(t, br_trace, color="red", lw=0.6, label="cardiac br")
    ax.plot(t[:vagal_trace.size], vagal_trace, color=WONG[2], lw=0.6,
            label="vagal br")
    try:
        n = min(br_trace.size, vagal_trace.size)
        r = float(np.corrcoef(br_trace[:n], vagal_trace[:n])[0, 1])
    except Exception:
        r = float("nan")
    ax.set_title(f"14a  vagal vs cardiac  r={r:.2f}", fontsize=8)
    ax.legend(fontsize=5); ax.tick_params(labelsize=6)


def panel_14b_breathing_by_condition(ax, results, k, extras, cfg) -> None:
    """Bar chart of mean breathing rate per stim condition."""
    cuff = results["cuff"][k]
    s10 = _step(cuff, "step10")
    by_cond = s10.get("breathing_rate_by_condition", {}) or {}
    if not by_cond:
        _missing(ax, "Panel 14b: no by-condition data"); return
    labels = list(by_cond.keys())
    vals = [float(v) for v in by_cond.values()]
    ax.bar(labels, vals, color=WONG[2], edgecolor="black", linewidth=0.4)
    stable = bool(s10.get("stable_across_conditions", False))
    flag = "✓" if stable else "✗"
    ax.set_title(f"14b  br by cond  [{flag}]", fontsize=8,
                 color="green" if stable else "red")
    ax.set_ylabel("Hz", fontsize=7); ax.tick_params(labelsize=6)
    for tick in ax.get_xticklabels():
        tick.set_rotation(20); tick.set_horizontalalignment("right")


def panel_15_summary_text(ax, results, k, extras, cfg, pair_name: str) -> None:
    """Monospace summary block."""
    cuff = results["cuff"][k]
    fs = float(results.get("fs", 24414.0))
    s3 = _step(cuff, "step3"); s6 = _step(cuff, "step6")
    s7 = _as_list(_step(cuff, "step7").get("cluster", []))
    s8 = _step(cuff, "step8")
    s9 = _as_list(_step(cuff, "step9").get("cluster", []))
    s10 = _step(cuff, "step10")
    s11a = _step(cuff, "step11a"); s11b = _step(cuff, "step11b")
    s12 = _step(cuff, "step12")
    s14 = _as_list(_step(cuff, "step14").get("cluster", []))

    spikes = np.asarray(s3.get("spike_samples", []))
    bm = (extras or {}).get("blanked_mask")
    blanked = (float(bm.mean()) if bm is not None and hasattr(bm, "mean")
               else float("nan"))
    duration_s = float(spikes.max() / fs) if spikes.size else 0.0
    h, rem = divmod(int(duration_s), 3600)
    m, sec = divmod(rem, 60)

    snrs  = [float(c.get("snr", np.nan)) for c in s7
             if np.isfinite(c.get("snr", np.nan))]
    viols = [float(c.get("isi_violation_rate", np.nan)) for c in s7]
    cardiac_locked = sum(1 for c in s9 if c.get("is_cardiac_locked"))
    sw_clusters = (_as_list(s11b.get("cluster", []))
                   if not _is_skipped(s11b) else [])
    primary_p_thr = float(getattr(cfg, "sw_primary_p", 0.05) if cfg else 0.05)
    primary_pl = sum(1 for c in sw_clusters
                     if float(c.get("rayleigh_p_common", 1.0)) < primary_p_thr)
    robust_pl = sum(1 for c in sw_clusters if c.get("robust_phase_locked"))
    bursts = _as_list(s12.get("cluster", [])) if not _is_skipped(s12) else []
    aff = sum(1 for c in bursts if c.get("direction_tag") == "afferent-like")
    eff = sum(1 for c in bursts if c.get("direction_tag") == "efferent-like")

    ch_tier = [str(ch.get("tier", "unusable"))
               for ch in _as_list(s11a.get("channel", []))]
    sw_usable = sum(1 for t in ch_tier if t in ("good", "marginal"))

    cond_counts: dict[str, dict[str, int]] = {}
    for r in s14:
        for cond in r.get("conditions", []) or []:
            lbl = str(cond.get("label", "?"))
            cc = cond_counts.setdefault(lbl, {"rate": 0, "reg": 0})
            if cond.get("is_responder"):
                cc["rate"] += 1
            if cond.get("is_regularised"):
                cc["reg"] += 1

    sorter = str(s6.get("sorter", "?"))
    lines: list[str] = []
    lines.append(f"Trial: {Path(pair_name).name}")
    lines.append(f"Cuff:  {k+1}")
    lines.append(f"Dur:   {h:01d}:{m:02d}:{sec:02d}   fs={fs:.0f} Hz")
    lines.append("-" * 32)
    lines.append("Detection")
    lines.append(f"  spikes: {int(spikes.size)}")
    lines.append(f"  thr σ:  {float(s3.get('threshold_sigma', float('nan'))):.1f}")
    if np.isfinite(blanked):
        lines.append(f"  blanked: {blanked*100:.1f}%")
    else:
        lines.append("  blanked: n/a")
    lines.append(f"Sorting [{sorter}]")
    lines.append(f"  clusters: {int(s6.get('n_clusters', 0))}")
    lines.append(f"  meanSNR: " +
                 (f"{float(np.mean(snrs)):.2f}" if snrs else "n/a"))
    lines.append(f"  meanISIv: " +
                 (f"{float(np.mean(viols))*100:.2f}%" if viols else "n/a"))
    ari = float(s8.get("adjusted_rand", float("nan")))
    lines.append("  ARI: " +
                 (f"{ari:.2f}" if np.isfinite(ari) else "n/a"))
    lines.append("Physiology")
    lines.append(f"  cardiac-locked: {cardiac_locked}")
    lines.append(f"  primary PL: {primary_pl}")
    lines.append(f"  robust PL:  {robust_pl}")
    lines.append(f"  af/ef:      {aff}/{eff}")
    lines.append("Respiration QC")
    rmatch = bool(s10.get("rate_matches", False))
    stable = bool(s10.get("stable_across_conditions", False))
    lines.append(f"  br match: {'✓' if rmatch else '✗'}")
    lines.append(f"  br stable: {'✓' if stable else '✗'}")
    lines.append("Slow-wave QC")
    lines.append(f"  usable ch: {sw_usable}/3")
    if cond_counts:
        lines.append("Responders (R/reg)")
        for lbl, cc in cond_counts.items():
            lines.append(f"  {lbl}: {cc['rate']}/{cc['reg']}")
    ax.set_axis_off()
    ax.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=5.5,
            ha="left", va="top", transform=ax.transAxes)


# ===========================================================================
# Summary-page layout
# ===========================================================================

def _summary_layout(plt, n_clusters: int):
    from matplotlib import gridspec
    fig = plt.figure(figsize=(16.5, 11.7), dpi=140)
    gs = gridspec.GridSpec(
        7, 12, figure=fig,
        height_ratios=[1.1, 1.0, 1.0, 1.0, 1.0, 1.2, 1.3],
        hspace=0.70, wspace=0.65,
        left=0.04, right=0.985, top=0.965, bottom=0.04,
    )
    n_wf    = max(1, min(n_clusters, 8))
    n_isi   = max(1, min(n_clusters, 6))
    n_fano  = max(1, min(n_clusters, 6))
    n_polar = max(1, min(n_clusters, 6))
    n_xcorr = max(1, min(n_clusters, 6))

    slots: dict[str, Any] = {}
    slots["1a"] = fig.add_subplot(gs[0, 0:3])
    slots["1b"] = fig.add_subplot(gs[0, 3:6])
    slots["2"]  = fig.add_subplot(gs[0, 6:9])
    slots["3"]  = fig.add_subplot(gs[0, 9:12])

    slots["4"] = fig.add_subplot(gs[1, 0:3])
    wf_gs = gs[1, 3:9].subgridspec(1, n_wf, wspace=0.4)
    slots["5"] = [fig.add_subplot(wf_gs[0, i]) for i in range(n_wf)]
    slots["6"] = fig.add_subplot(gs[1, 9:12])

    isi_gs = gs[2, 0:6].subgridspec(1, n_isi, wspace=0.45)
    slots["7"] = [fig.add_subplot(isi_gs[0, i]) for i in range(n_isi)]
    fano_gs = gs[2, 6:12].subgridspec(1, n_fano, wspace=0.45)
    slots["7b"] = [fig.add_subplot(fano_gs[0, i]) for i in range(n_fano)]

    slots["8a_img"]  = fig.add_subplot(gs[3, 0:3])
    slots["8a_bars"] = fig.add_subplot(gs[3, 3:4])
    slots["8b"]      = fig.add_subplot(gs[3, 4:12])

    slots["9"] = fig.add_subplot(gs[4, 0:12])

    polar_gs = gs[5, 0:6].subgridspec(1, n_polar, wspace=0.4)
    slots["10"] = [fig.add_subplot(polar_gs[0, i], projection="polar")
                   for i in range(n_polar)]
    slots["11"] = fig.add_subplot(gs[5, 6:8], projection="polar")
    xcorr_gs = gs[5, 8:12].subgridspec(1, n_xcorr, wspace=0.4)
    slots["12"] = [fig.add_subplot(xcorr_gs[0, i]) for i in range(n_xcorr)]

    slots["13"]  = fig.add_subplot(gs[6, 0:5])
    slots["13b"] = fig.add_subplot(gs[6, 5:7])
    slots["14a"] = fig.add_subplot(gs[6, 7:9])
    slots["14b"] = fig.add_subplot(gs[6, 9:11])
    slots["15"]  = fig.add_subplot(gs[6, 11:12])
    return fig, slots


def _render_summary_page(plt, results, k, extras, cfg, pair_name: str):
    cuff = results["cuff"][k]
    n_clusters = int(_step(cuff, "step6").get("n_clusters", 0))
    fig, slots = _summary_layout(plt, n_clusters)
    pname = Path(pair_name).name if pair_name else "trial"
    fig.suptitle(f"{pname}  --  cuff {k+1}/{int(results.get('n_cuffs', 1))}",
                 fontsize=11, y=0.992)

    _safe_panel("1a",  panel_1a_psd_overlay,        slots["1a"],  results, k, extras, cfg)
    _safe_panel("1b",  panel_1b_fibre_bands,        slots["1b"],  results, k, extras, cfg)
    _safe_panel("2",   panel_2_sigma_thr,           slots["2"],   results, k, extras, cfg)
    _safe_panel("3",   panel_3_example_trace,       slots["3"],   results, k, extras, cfg)
    _safe_panel("4",   panel_4_amp_hist,            slots["4"],   results, k, extras, cfg)
    _safe_panel("5",   panel_5_mean_waveforms,      slots["5"],   results, k, extras, cfg)
    _safe_panel("6",   panel_6_umap_ari,            slots["6"],   results, k, extras, cfg)
    _safe_panel("7",   panel_7_isi_hists,           slots["7"],   results, k, extras, cfg)
    _safe_panel("7b",  panel_7b_fano_vs_t,          slots["7b"],  results, k, extras, cfg)
    _safe_panel("8a",  panel_8a_prwh_stack,         slots["8a_img"], slots["8a_bars"],
                results, k, extras, cfg)
    _safe_panel("8b",  panel_8b_synchrony,          slots["8b"],  results, k, extras, cfg)
    _safe_panel("9",   panel_9_sw_qc,               slots["9"],   results, k, extras, cfg)
    _safe_panel("10",  panel_10_polar_phase,        slots["10"],  results, k, extras, cfg)
    _safe_panel("11",  panel_11_consistency_polar,  slots["11"],  results, k, extras, cfg)
    _safe_panel("12",  panel_12_burst_xcorr,        slots["12"],  results, k, extras, cfg)
    _safe_panel("13",  panel_13_rate_traces_stim,   slots["13"],  results, k, extras, cfg)
    _safe_panel("13b", panel_13b_fano_pre_post,     slots["13b"], results, k, extras, cfg)
    _safe_panel("14a", panel_14a_vagal_vs_cardiac,  slots["14a"], results, k, extras, cfg)
    _safe_panel("14b", panel_14b_breathing_by_condition, slots["14b"], results, k, extras, cfg)
    _safe_panel("15",  panel_15_summary_text,       slots["15"],  results, k, extras, cfg, pair_name)
    return fig, slots


# ===========================================================================
# Supplementary PDF
# ===========================================================================

def _render_supplementary(plt, pdf, results, k, extras, cfg, pair_name: str) -> None:
    """Multi-page supplementary detail per §9.5."""
    cuff = results["cuff"][k]

    clusters = _as_list(_step(cuff, "step7").get("cluster", []))
    for cl in clusters:
        fig = plt.figure(figsize=(16.5, 11.7))
        ax = fig.add_subplot(111)
        panel_8b_synchrony(ax, results, k, extras, cfg)
        ax.set_title(f"Synchrony  cluster c{int(cl.get('cluster_id', -1))}",
                     fontsize=10)
        fig.suptitle(f"{Path(pair_name).name}  cuff {k+1}", fontsize=11)
        pdf.savefig(fig); plt.close(fig)

    s11b = _step(cuff, "step11b")
    if not _is_skipped(s11b):
        for cl in _as_list(s11b.get("cluster", [])):
            fig = plt.figure(figsize=(8.27, 11.7))
            ax = fig.add_subplot(111, projection="polar")
            # build a one-cluster mini-results to reuse panel_10
            mini = {"cuff": [{"step11b": {"cluster": [cl], "skipped": False}}],
                    "fs": results.get("fs", 24414.0),
                    "n_cuffs": 1}
            panel_10_polar_phase([ax], mini, 0, extras, cfg)
            fig.suptitle(f"{Path(pair_name).name}  cuff {k+1}  "
                         f"cluster {int(cl.get('cluster_id', -1))}",
                         fontsize=11)
            pdf.savefig(fig); plt.close(fig)

    s11a = _step(cuff, "step11a")
    if not _is_skipped(s11a):
        ch = _as_list(s11a.get("channel", []))
        if ch:
            fig, axes = plt.subplots(len(ch), 4,
                                     figsize=(16.5, 2.4 * len(ch)), sharex=False)
            axes = np.atleast_2d(axes)
            for j, c in enumerate(ch):
                row_labels = ["SNR(in-band)", "peak prominence",
                              "pairwise coherence", "envelope CV"]
                metrics = [c.get("snr_inband"), c.get("peak_prominence"),
                           c.get("pairwise_coherence_with_others"),
                           c.get("envelope_cv")]
                for col, (lbl, val) in enumerate(zip(row_labels, metrics)):
                    ax = axes[j, col]
                    if val is None:
                        _missing(ax, "n/a"); continue
                    arr = np.atleast_1d(np.asarray(val))
                    if arr.size == 1:
                        ax.bar([0], [float(arr[0])], color=WONG[2])
                        ax.set_xticks([])
                        ax.set_title(f"ch{j+1} {lbl}: {float(arr[0]):.3g}",
                                     fontsize=7)
                    else:
                        ax.plot(np.arange(arr.size), arr, color=WONG[2], lw=0.7)
                        ax.set_title(f"ch{j+1} {lbl}", fontsize=7)
                    ax.tick_params(labelsize=5)
            fig.suptitle(f"{Path(pair_name).name}  cuff {k+1}  "
                         "-- slow-wave QC detail", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig); plt.close(fig)


# ===========================================================================
# Public entry points
# ===========================================================================

PANEL_SLOT_NAMES = {
    "1a": "psd_overlay", "1b": "fibre_bands", "2": "sigma_thr",
    "3": "example_trace", "4": "amp_hist", "5": "mean_waveforms",
    "6": "umap_ari", "7": "isi_hists", "7b": "fano_vs_T",
    "8a_img": "prwh_stack", "8a_bars": "prwh_peakz_bars",
    "8b": "synchrony", "9": "sw_qc",
    "10": "polar_phase", "11": "consistency_polar", "12": "burst_xcorr",
    "13": "rate_traces", "13b": "fano_pre_post",
    "14a": "vagal_vs_cardiac", "14b": "breathing_by_cond",
    "15": "summary_text",
}


def render_diagnostics(
    results: dict[str, Any],
    out_dir: Path | str,
    stem: str,
    *,
    plots_root: Path | str | None = None,
    cfg: Any = None,
    extras_per_cuff: list[dict[str, Any]] | None = None,
    pair_name: str | None = None,
) -> dict[str, list[Path]]:
    """Render the full §9 diagnostic suite per cuff.

    Outputs are written to ``plots_root / out_dir.name`` if ``plots_root``
    is set, else ``out_dir``.  Returns
    ``{'summary': [paths], 'supplementary': [paths], 'panels': [paths]}``.
    Never raises.
    """
    out_dir = Path(out_dir)
    dest_dir = Path(plots_root) / out_dir.name if plots_root is not None else out_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        _, plt = _import_mpl()
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as e:
        log.warning("matplotlib unavailable; skipping plots (%s).", e)
        return {"summary": [], "supplementary": [], "panels": []}

    pair_name = pair_name or stem
    cuffs = results.get("cuff", []) or []
    if extras_per_cuff is None:
        extras_per_cuff = [{} for _ in cuffs]
    if len(extras_per_cuff) < len(cuffs):
        extras_per_cuff = list(extras_per_cuff) + [
            {} for _ in range(len(cuffs) - len(extras_per_cuff))
        ]

    summary_paths: list[Path] = []
    supp_paths: list[Path] = []
    panel_paths: list[Path] = []

    for k, _cuff in enumerate(cuffs):
        extras = extras_per_cuff[k] if k < len(extras_per_cuff) else {}

        # --- Summary page ---
        try:
            fig, slots = _render_summary_page(plt, results, k, extras, cfg, pair_name)
            sum_pdf = dest_dir / f"{stem}_cuff{k+1}_summary.pdf"
            fig.savefig(sum_pdf, format="pdf")
            summary_paths.append(sum_pdf)
            # PNG sidecars per slot.
            try:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                for slot_id, ax_or_axes in slots.items():
                    axes = ax_or_axes if isinstance(ax_or_axes, list) else [ax_or_axes]
                    bboxes = [a.get_tightbbox(renderer) for a in axes if a is not None]
                    bboxes = [b for b in bboxes if b is not None]
                    if not bboxes:
                        continue
                    from matplotlib.transforms import Bbox
                    bbox = Bbox.union(bboxes).transformed(fig.dpi_scale_trans.inverted())
                    name = PANEL_SLOT_NAMES.get(slot_id, slot_id)
                    png_path = dest_dir / f"{stem}_cuff{k+1}_panel{slot_id}_{name}.png"
                    try:
                        fig.savefig(png_path, bbox_inches=bbox, dpi=140)
                        panel_paths.append(png_path)
                    except Exception as e:
                        log.debug("PNG sidecar %s failed: %s", png_path.name, e)
            except Exception as e:
                log.debug("PNG sidecar pass failed: %s", e)
            plt.close(fig)
        except Exception as e:
            log.warning("Summary page failed for %s cuff %d: %s\n%s",
                        stem, k + 1, e, traceback.format_exc())
            try:
                fig = plt.figure(figsize=(16.5, 11.7))
                ax = fig.add_subplot(111); ax.set_axis_off()
                ax.text(0.5, 0.5, f"plotting failed for cuff {k+1}\n{e}",
                        ha="center", va="center", fontsize=10)
                sum_pdf = dest_dir / f"{stem}_cuff{k+1}_summary.pdf"
                fig.savefig(sum_pdf); plt.close(fig); summary_paths.append(sum_pdf)
            except Exception:
                pass

        # --- Supplementary PDF ---
        try:
            supp_pdf = dest_dir / f"{stem}_cuff{k+1}_supplementary.pdf"
            with PdfPages(supp_pdf) as pdf:
                _render_supplementary(plt, pdf, results, k, extras, cfg, pair_name)
                if pdf.get_pagecount() == 0:
                    fig = plt.figure(figsize=(11.7, 8.27))
                    ax = fig.add_subplot(111); ax.set_axis_off()
                    ax.text(0.5, 0.5, "supplementary detail unavailable",
                            ha="center", va="center", fontsize=10, color="gray")
                    pdf.savefig(fig); plt.close(fig)
            supp_paths.append(supp_pdf)
        except Exception as e:
            log.warning("Supplementary page failed for %s cuff %d: %s",
                        stem, k + 1, e)

    if summary_paths or supp_paths or panel_paths:
        log.info("Wrote %d summary + %d supp + %d panel PNGs to %s",
                 len(summary_paths), len(supp_paths), len(panel_paths), dest_dir)
    return {"summary": summary_paths,
            "supplementary": supp_paths,
            "panels": panel_paths}


# ---------------------------------------------------------------------------
# Legacy shim
# ---------------------------------------------------------------------------

def save_pair_diagnostics(
    results: dict[str, Any],
    out_dir: Path | str,
    stem: str,
    plots_root: Path | str | None = None,
    *,
    cfg: Any = None,
    extras_per_cuff: list[dict[str, Any]] | None = None,
    pair_name: str | None = None,
) -> list[Path]:
    """Legacy entry point preserved for backwards compatibility.

    Delegates to :func:`render_diagnostics` and flattens all returned
    paths into a single list, matching the old return type.
    """
    bundle = render_diagnostics(
        results, out_dir, stem,
        plots_root=plots_root, cfg=cfg,
        extras_per_cuff=extras_per_cuff,
        pair_name=pair_name,
    )
    return [p for paths in bundle.values() for p in paths]
