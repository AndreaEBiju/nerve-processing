"""Step 13: responder detection (Masi/Zanos style).

A cluster is flagged as a responder for a stim condition when at least
``responder_frac_epoch`` of the post-stim bins exceed the per-cluster
``responder_pctile``-th percentile of the pre-stim bins.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import PipelineConfig

log = logging.getLogger("vagus.responder")


def detect(
    rate_traces: list[dict[str, Any]],
    t_centres_s: np.ndarray,
    stim_events: list[tuple[int, str]] | None,
    fs: float,
    cfg: PipelineConfig,
    pre_post_s: float = 30.0,
) -> list[dict[str, Any]]:
    """Return one entry per cluster:

    ``{"cluster_id": c, "conditions": [{"label": str, "is_responder": bool, "p_post_over_thr": float}, ...]}``
    """
    if not stim_events:
        return [{"cluster_id": rt["cluster_id"], "conditions": []} for rt in rate_traces]

    out = []
    for rt in rate_traces:
        rate = rt["rate_trace"].astype(np.float64)
        conds = []
        for sample_idx, label in stim_events:
            t_event = sample_idx / fs
            pre_mask = (t_centres_s < t_event) & (t_centres_s >= t_event - pre_post_s)
            post_mask = (t_centres_s >= t_event) & (t_centres_s <= t_event + pre_post_s)
            if pre_mask.sum() < 3 or post_mask.sum() < 3:
                conds.append({"label": label, "is_responder": False, "p_post_over_thr": 0.0})
                continue
            thr = np.percentile(rate[pre_mask], cfg.responder_pctile)
            frac_over = float((rate[post_mask] > thr).mean())
            is_resp = bool(frac_over >= cfg.responder_frac_epoch)
            conds.append({"label": label, "is_responder": is_resp, "p_post_over_thr": frac_over})
        out.append({"cluster_id": rt["cluster_id"], "conditions": conds})
    return out
