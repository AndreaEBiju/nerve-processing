"""Configuration dataclasses for the vagus nerve cuff pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


class SlowWaveUnusable(Exception):
    """Raised when no slow-wave channel reaches the marginal quality tier
    anywhere in the recording.  Step 11 + Step 12 are skipped for that
    recording; downstream steps still run."""
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

    # ---------- Step 11 slow-wave (three antral channels, 0.5 cm spacing) ----------
    sw_low_hz: float = 0.01           # phase-extraction bandpass low
    sw_high_hz: float = 0.2           # phase-extraction bandpass high
    sw_inband_low_hz: float = 0.04    # in-band SNR numerator band
    sw_inband_high_hz: float = 0.15
    sw_outband_low1_hz: float = 0.005
    sw_outband_high1_hz: float = 0.04
    sw_outband_low2_hz: float = 0.15
    sw_outband_high2_hz: float = 0.5
    sw_quality_window_s: float = 60.0
    sw_quality_overlap_s: float = 30.0
    sw_quality_good_threshold: float = 0.7
    sw_quality_marginal_threshold: float = 0.3
    sw_coherence_min: float = 0.5
    sw_snr_norm_lo: float = 1.0
    sw_snr_norm_hi: float = 10.0
    sw_coherence_norm_lo: float = 0.3
    sw_coherence_norm_hi: float = 0.9
    sw_envcv_norm_lo: float = 1.0     # high CV -> bad
    sw_envcv_norm_hi: float = 0.2     # low CV -> good (inverted)
    sw_consistency_phase_spread_deg: float = 60.0
    sw_per_channel_p_near: float = 0.10
    sw_primary_p: float = 0.05
    sw_common_mode_fs_hz: float = 10.0   # downsample target for common-mode storage

    # ---------- Step 12 antral burst detection ----------
    burst_band_low_hz: float = 1.0
    burst_band_high_hz: float = 10.0
    burst_threshold_sigma: float = 4.0
    burst_min_duration_ms: float = 200.0
    burst_min_channels: int = 2
    burst_consensus_window_ms: float = 100.0  # tolerance for consensus across channels
    burst_xcorr_window_s: float = 10.0
    burst_xcorr_bin_s: float = 0.05
    burst_peak_z_threshold: float = 3.0
    burst_direction_min_lag_s: float = 0.5

    # Step 13 rates: 1 s bin gives twice the temporal resolution of the
    # original 2 s default for only a sqrt(2) Poisson-noise penalty per bin.
    # Critical for Step 14 (responder), where the pre/post-stim percentile
    # test gets noticeably tighter estimates with 30 vs 15 bins in a 30 s
    # window.  Override with --rate-bin-s on the CLI or the GUI spinbox.
    rate_bin_s: float = 1.0

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
    # Three antral mucosal slow-wave channels (proximal -> middle -> distal).
    # Each is either a separate variable name, the same variable with
    # different ``_channel`` index, or empty if that electrode failed.
    slowwave_ch1: str | None = None  # PROXIMAL
    slowwave_ch2: str | None = None  # MIDDLE  (reference for propagation lag)
    slowwave_ch3: str | None = None  # DISTAL
    # If all three slow-wave names point at the SAME variable, this picks
    # which column of the multi-channel matrix is ch1 / ch2 / ch3.
    # Defaults to [0, 1, 2] for natural row-order.
    slowwave_ch_indices: list[int] = field(default_factory=lambda: [0, 1, 2])
    # User-confirmed spatial order along gastric long axis.  Default is
    # the assignment as written (ch1=proximal, ch2=middle, ch3=distal).
    # Changing this swaps which channel is treated as the propagation-lag
    # reference (always the middle entry).
    slowwave_spatial_order: list[int] = field(default_factory=lambda: [1, 2, 3])
    # DEPRECATED: legacy single-channel slow-wave field, kept so old
    # batch_varmap.json files still load.  If set and the ch1/ch2/ch3
    # fields are empty, treated as slowwave_ch1 only (one-channel mode).
    slowwave: str | None = None
    fs: str | None = None
    stim_events: str | None = None
    stim_labels: str | None = None
    n_channels: int = 1  # informational; if channel_indices is set it takes precedence

    # 0-based indices of the channels in the neural array that are actual
    # nerve-cuff recordings.  Example: a 5-channel acquisition where only
    # rows 0 and 3 are cuff signals -> channel_indices=[0, 3].  When set,
    # the pipeline runs once per listed channel and ignores the rest.
    # Leave as ``None`` (or empty list) to use every channel in the array.
    channel_indices: list[int] | None = None

    # 0-based index of the slow-wave channel to use when the slow-wave
    # variable is a multi-channel matrix (e.g. a 3xN array stacking three
    # slow-wave time series).  Ignored when the variable is already 1-D
    # or a cell array of peak indices.  Default 0 = first channel.
    slowwave_channel: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
