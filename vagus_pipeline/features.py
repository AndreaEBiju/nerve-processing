"""Step 5: scalar waveform features + global PCA basis (fit/project).

The PCA basis is **fit once across the whole batch** so feature spaces are
comparable across stimulation conditions; every recording then projects onto
the same basis. This module exposes both the fitting routine (``fit_pca``)
and the projection routine (``project_pca``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig

log = logging.getLogger("vagus.features")


@dataclass
class PCABasis:
    components: np.ndarray  # (n_pca, L)
    mean: np.ndarray  # (L,)
    scale: np.ndarray  # (L,) – per-sample std used by StandardScaler
    n_pca: int

    def project(self, waveforms: np.ndarray) -> np.ndarray:
        x = (waveforms - self.mean) / self.scale
        return x @ self.components.T

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            components=self.components,
            mean=self.mean,
            scale=self.scale,
            n_pca=np.asarray(self.n_pca),
        )

    @staticmethod
    def load(path: str | Path) -> "PCABasis":
        z = np.load(path)
        return PCABasis(
            components=z["components"].astype(np.float64),
            mean=z["mean"].astype(np.float64),
            scale=z["scale"].astype(np.float64),
            n_pca=int(z["n_pca"]),
        )


def scalar_features(waveforms: np.ndarray, cfg: PipelineConfig) -> dict[str, np.ndarray]:
    """Per-spike scalar features.

    Returns
    -------
    dict with keys p2p, trough_peak_ms, halfwidth_ms, zc_slope, energy.
    """
    if waveforms.size == 0:
        empty = np.zeros(0, dtype=np.float32)
        return {"p2p": empty, "trough_peak_ms": empty, "halfwidth_ms": empty, "zc_slope": empty, "energy": empty}

    fs = cfg.fs
    n, L = waveforms.shape
    feats = {
        "p2p": np.zeros(n, dtype=np.float32),
        "trough_peak_ms": np.zeros(n, dtype=np.float32),
        "halfwidth_ms": np.zeros(n, dtype=np.float32),
        "zc_slope": np.zeros(n, dtype=np.float32),
        "energy": np.zeros(n, dtype=np.float32),
    }
    for i in range(n):
        wf = waveforms[i]
        trough = int(np.argmin(wf))
        # peak after the trough (or before if at the right edge)
        post = wf[trough:]
        peak_rel = int(np.argmax(post)) if post.size > 1 else 0
        peak = trough + peak_rel
        feats["p2p"][i] = float(wf[peak] - wf[trough])
        feats["trough_peak_ms"][i] = (peak - trough) / fs * 1000.0
        # half-width at half-trough amplitude
        thr = wf[trough] / 2.0
        below = wf <= thr
        if below.any():
            idx = np.where(below)[0]
            feats["halfwidth_ms"][i] = (idx[-1] - idx[0]) / fs * 1000.0
        # slope at the largest zero-crossing
        signs = np.sign(wf)
        zc = np.where(np.diff(signs) != 0)[0]
        if zc.size > 0:
            slopes = wf[zc + 1] - wf[zc]
            j = int(zc[np.argmax(np.abs(slopes))])
            feats["zc_slope"][i] = float(wf[j + 1] - wf[j]) * fs
        feats["energy"][i] = float(np.sum(wf * wf))
    return feats


def fit_pca(
    pooled_waveforms: np.ndarray, cfg: PipelineConfig, rng: np.random.Generator | None = None
) -> PCABasis:
    """Fit a single global PCA basis with mean+scale normalization."""
    if pooled_waveforms.shape[0] < cfg.n_pca:
        raise ValueError(
            f"Need at least {cfg.n_pca} pooled waveforms to fit a {cfg.n_pca}-component PCA basis, got {pooled_waveforms.shape[0]}."
        )
    rng = rng if rng is not None else np.random.default_rng(cfg.seed)
    cap = cfg.pca_pool_max_spikes
    if pooled_waveforms.shape[0] > cap:
        sel = rng.choice(pooled_waveforms.shape[0], size=cap, replace=False)
        pooled_waveforms = pooled_waveforms[sel]
        log.info("Subsampled pooled waveforms to PCA cap %d", cap)
    scaler = StandardScaler(with_mean=True, with_std=True)
    x = scaler.fit_transform(pooled_waveforms.astype(np.float64))
    pca = PCA(n_components=cfg.n_pca, random_state=cfg.seed)
    pca.fit(x)
    log.info(
        "Fitted PCA basis: %d components, explained variance=%.3f",
        cfg.n_pca, float(pca.explained_variance_ratio_.sum()),
    )
    return PCABasis(
        components=pca.components_.astype(np.float64),
        mean=scaler.mean_.astype(np.float64),
        scale=np.where(scaler.scale_ > 0, scaler.scale_, 1.0).astype(np.float64),
        n_pca=cfg.n_pca,
    )


def project_pca(waveforms: np.ndarray, basis: PCABasis) -> np.ndarray:
    """Project waveforms onto a previously fitted basis."""
    if waveforms.shape[0] == 0:
        return np.zeros((0, basis.n_pca), dtype=np.float32)
    if waveforms.shape[1] != basis.mean.size:
        raise ValueError(
            f"Waveform length {waveforms.shape[1]} does not match basis length {basis.mean.size}."
        )
    return basis.project(waveforms.astype(np.float64)).astype(np.float32)
