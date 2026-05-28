"""Step 14: descriptive fibre-type tagging from waveform width.

Trough-to-peak duration drives a coarse three-way tag:
    < 0.7 ms : A-like
    > 1.2 ms : C-like
    otherwise: ambiguous
The breakpoints come from the companion docx §3.14; they live here so the UI
can show them and the user can override if needed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import PipelineConfig

log = logging.getLogger("vagus.fibertype")

A_LIKE_MAX_MS = 0.7
C_LIKE_MIN_MS = 1.2


def tag(
    scalar_feats: dict[str, np.ndarray],
    labels: np.ndarray,
    cfg: PipelineConfig,
) -> list[dict[str, Any]]:
    """Return one tag dict per cluster: {cluster_id, mean_trough_peak_ms, type_tag}."""
    out: list[dict[str, Any]] = []
    if "trough_peak_ms" not in scalar_feats:
        return out
    tp = scalar_feats["trough_peak_ms"]
    unique = sorted({int(l) for l in labels if l >= 0})
    for c in unique:
        mask = labels == c
        if not mask.any():
            continue
        mean_tp = float(np.median(tp[mask]))
        if mean_tp < A_LIKE_MAX_MS:
            tag_ = "A-like"
        elif mean_tp > C_LIKE_MIN_MS:
            tag_ = "C-like"
        else:
            tag_ = "ambiguous"
        out.append({"cluster_id": c, "mean_trough_peak_ms": mean_tp, "type_tag": tag_})
    return out
