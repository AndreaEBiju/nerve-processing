"""Configuration dataclasses for the vagus nerve cuff pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PipelineConfig:
    fs: float = 24414.0625

    # Step 1 bandpass
    bp_low_hz: float = 100.0
    bp_high_hz: float = 5000.0
    bp_order: int = 4

    # Step 2 noise estimate
    noise_window_s: float = 5.0

    # Step 3 detection
    threshold_sigma: float = 4.5
    refractory_ms: float = 1.0
    detect_polarity: str = "neg"

    # Step 4 waveform window
    wf_pre_ms: float = 1.0
    wf_post_ms: float = 2.0

    # Step 5 features / PCA
    n_pca: int = 8
    pca_pool_max_spikes: int = 50_000

    # Step 6 sorter
    sorter: str = "mountainsort5"

    # Step 8 audit
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 2
    hdbscan_min_cluster_size: int = 50

    # Step 9 cardiac
    prwh_window_ms: float = 50.0
    cardiac_lock_window_ms: float = 7.5
    cardiac_peak_z: float = 3.0

    # Step 10 respiration
    resp_band_low_hz: float = 0.1
    resp_band_high_hz: float = 3.0

    # Step 11 slow wave
    sw_low_hz: float = 0.01
    sw_high_hz: float = 0.2

    # Step 12 rates
    rate_bin_s: float = 2.0

    # Step 13 responder
    responder_pctile: float = 95.0
    responder_frac_epoch: float = 1.0 / 3.0

    # Behavior toggles
    use_cardiac_cleaned: bool = False

    # Reproducibility
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VarMap:
    """User-assigned variable names for each logical role in the input files."""

    neural: str = ""
    rpeak_times: str = ""
    rpeak_units: str = "sample"  # "sample" | "sec" | "ms"
    slowwave: str | None = None
    fs: str | None = None
    stim_events: str | None = None
    stim_labels: str | None = None
    n_channels: int = 1  # cuffs in the neural variable

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
