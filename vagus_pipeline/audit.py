"""Step 8: independent audit via UMAP + HDBSCAN, with agreement vs the sorter."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import PipelineConfig

log = logging.getLogger("vagus.audit")


def umap_hdbscan_audit(
    pca_feats: np.ndarray, labels: np.ndarray, cfg: PipelineConfig
) -> dict[str, Any]:
    """UMAP 2-D embedding + HDBSCAN clustering, compared to MountainSort labels.

    Returns a dict with ``umap_xy`` (N×2), ``hdbscan_labels`` (N,), ``adjusted_rand``,
    and a ``agreement`` matrix (n_ms × n_hdb confusion counts).
    """
    out: dict[str, Any] = {
        "umap_xy": np.zeros((0, 2), dtype=np.float32),
        "hdbscan_labels": np.zeros(0, dtype=np.int64),
        "adjusted_rand": float("nan"),
        "agreement": np.zeros((0, 0), dtype=np.int64),
    }
    n = pca_feats.shape[0]
    if n < max(20, cfg.hdbscan_min_cluster_size):
        log.info("Audit skipped: too few points (%d).", n)
        return out

    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=min(cfg.umap_n_neighbors, max(2, n - 1)),
            min_dist=cfg.umap_min_dist,
            n_components=cfg.umap_n_components,
            random_state=cfg.seed,
        )
        emb = reducer.fit_transform(pca_feats.astype(np.float64)).astype(np.float32)
    except Exception as e:
        log.warning("UMAP unavailable (%s); using PCA[:2] as the audit embedding.", e)
        emb = pca_feats[:, : cfg.umap_n_components].astype(np.float32)
    out["umap_xy"] = emb

    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min(cfg.hdbscan_min_cluster_size, max(5, n // 4)),
        )
        hdb_labels = clusterer.fit_predict(emb).astype(np.int64)
    except Exception as e:
        log.warning("HDBSCAN unavailable (%s); using KMeans audit fallback.", e)
        from sklearn.cluster import KMeans

        k = max(2, min(6, n // max(cfg.hdbscan_min_cluster_size, 1)))
        hdb_labels = KMeans(n_clusters=k, random_state=cfg.seed, n_init=10).fit_predict(emb).astype(np.int64)
    out["hdbscan_labels"] = hdb_labels

    if labels.size == n:
        try:
            from sklearn.metrics import adjusted_rand_score

            valid = labels >= 0
            if valid.sum() >= 2 and len(np.unique(labels[valid])) > 1 and len(np.unique(hdb_labels[valid])) > 1:
                out["adjusted_rand"] = float(
                    adjusted_rand_score(labels[valid], hdb_labels[valid])
                )
        except Exception:
            pass

        ms_ids = sorted({int(l) for l in labels if l >= 0})
        hdb_ids = sorted({int(l) for l in hdb_labels})
        agree = np.zeros((len(ms_ids), len(hdb_ids)), dtype=np.int64)
        for i, m in enumerate(ms_ids):
            for j, h in enumerate(hdb_ids):
                agree[i, j] = int(((labels == m) & (hdb_labels == h)).sum())
        out["agreement"] = agree

    return out
