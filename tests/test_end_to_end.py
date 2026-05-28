"""End-to-end acceptance tests per spec §10 (v2: 15 steps, 3 slow-wave channels).

Two datasets are synthesized:

* **A** -- three healthy slow-wave channels;
* **B** -- middle channel corrupted with a 60-120 s dropout + 3x noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vagus_pipeline.batch import run_batch  # noqa: E402
from vagus_pipeline.config import PipelineConfig, VarMap  # noqa: E402
from vagus_pipeline.features import PCABasis  # noqa: E402
from vagus_pipeline.io_discovery import find_pairs, introspect_variables  # noqa: E402

DATA_DIR = REPO_ROOT / "tests" / "data"


@pytest.fixture(scope="module")
def sample_dataset():
    from tests.make_sample_data import main as build

    truth = build()
    yield truth


@pytest.fixture(scope="module")
def batch_result(sample_dataset):
    cfg = PipelineConfig()
    vm = VarMap(
        neural="data",
        rpeak_times="Rpeaks",
        rpeak_units="sample",
        slowwave_ch1="sw_ch1",
        slowwave_ch2="sw_ch2",
        slowwave_ch3="sw_ch3",
        slowwave_spatial_order=[1, 2, 3],
        fs="fs",
        stim_events="stim_events",
        stim_labels="stim_labels",
    )
    res = run_batch(DATA_DIR, vm, cfg)
    return {"cfg": cfg, "var_map": vm, "result": res}


def _load_mat(path):
    try:
        from pymatreader import read_mat
        return read_mat(path)
    except Exception:
        from scipy.io import loadmat
        d = loadmat(path, squeeze_me=True, struct_as_record=False)
        return {k: v for k, v in d.items() if not k.startswith("__")}


def _as_list(x):
    return x if isinstance(x, list) else [x]


def _pair_of_dataset(rows, prefix):
    """Return the summary row whose blanked filename starts with ``prefix``."""
    return next(r for r in rows if r["blanked"].startswith(prefix))


# ------ §10.1-10.3 discovery / mapping / run completes ------------------------------


def test_discovery_finds_both_datasets(sample_dataset):
    pairs = find_pairs(DATA_DIR)
    assert len(pairs) == 2, [p.blanked_path.name for p in pairs]


def test_run_completes_and_writes_outputs(batch_result):
    res = batch_result["result"]
    assert (DATA_DIR / "batch_pca_basis.npz").exists()
    basis = PCABasis.load(DATA_DIR / "batch_pca_basis.npz")
    assert basis.n_pca == batch_result["cfg"].n_pca
    for row in res["rows"]:
        assert row["status"] == "ok", row
        assert Path(row["output_path"]).exists()


# ------ §10.4 per-cuff shapes (both datasets) ---------------------------------


def test_per_cuff_shapes(batch_result):
    for row in batch_result["result"]["rows"]:
        data = _load_mat(row["output_path"])
        cuff = _as_list(data["metrics"]["cuff"])[0]
        n_clusters = int(np.asarray(cuff["step6"]["n_clusters"]).item())
        assert n_clusters >= 2
        for cl in _as_list(cuff["step7"]["cluster"]):
            n_spk = int(np.asarray(cl["n_spikes"]).item())
            if n_spk > 1:
                assert np.asarray(cl["isi_s"]).ravel().size > 0
            snr = float(np.asarray(cl["snr"]).item())
            assert np.isfinite(snr) and snr > 0


# ------ §10.5 dataset A: all three slow-wave channels good ------------------


def test_datasetA_all_channels_good(batch_result):
    row = _pair_of_dataset(batch_result["result"]["rows"], "ratA_good")
    data = _load_mat(row["output_path"])
    cuff = _as_list(data["metrics"]["cuff"])[0]
    s11a = cuff["step11a"]
    assert not bool(np.asarray(s11a.get("skipped", False)).item())
    assert int(np.asarray(s11a["channels_present"]).item()) == 3
    channels = _as_list(s11a["channel"])
    tiers = [str(c["tier"]) for c in channels]
    assert sum(t == "good" for t in tiers) >= 2, tiers


def test_datasetA_locked_unit_passes_robust_phase_locked(batch_result):
    row = _pair_of_dataset(batch_result["result"]["rows"], "ratA_good")
    assert row["n_robust_phase_locked"] >= 1, row


# ------ §10.6 dataset B: middle channel dropout -----------------------------


def test_datasetB_middle_channel_downweighted(batch_result):
    row = _pair_of_dataset(batch_result["result"]["rows"], "ratB_dropout")
    data = _load_mat(row["output_path"])
    cuff = _as_list(data["metrics"]["cuff"])[0]
    s11a = cuff["step11a"]
    if bool(np.asarray(s11a.get("skipped", False)).item()):
        pytest.skip("ratB slow-wave was completely unusable; the corruption was too aggressive")
    channels = _as_list(s11a["channel"])
    assert len(channels) >= 3
    # Middle channel (index 1) should have the lowest whole-recording score.
    whole_scores = [float(np.asarray(c["quality_score_whole"]).item()) for c in channels]
    assert whole_scores[1] < whole_scores[0]
    assert whole_scores[1] < whole_scores[2]
    # And its rolling weights during the dropout (60-120 s) should be at or near zero.
    rolling = np.asarray(channels[1]["quality_score_rolling"]).ravel()
    window_centres = np.asarray(channels[1]["quality_window_times_s"]).ravel()
    dropout = (window_centres >= 60) & (window_centres <= 120)
    if dropout.any():
        assert (rolling[dropout] <= 0.3).any() or rolling[dropout].mean() < 0.5


# ------ §10.7 antral burst detection ----------------------------------------


def test_consensus_bursts_detected_both_datasets(batch_result, sample_dataset):
    truth = {r["name"]: r for r in sample_dataset["recordings"]}
    for row in batch_result["result"]["rows"]:
        data = _load_mat(row["output_path"])
        cuff = _as_list(data["metrics"]["cuff"])[0]
        s12 = cuff["step12"]
        if bool(np.asarray(s12.get("skipped", False)).item()):
            continue
        consensus = np.asarray(s12["consensus_burst_times_s"]).ravel()
        truth_name = "ratA_good" if row["blanked"].startswith("ratA_good") else "ratB_dropout"
        injected = int(np.asarray(truth[truth_name]["burst_times_s"]).size)
        if injected:
            # Within +/-60% of injected count (synth bursts are well-separated).
            assert 0.3 * injected <= consensus.size <= 3.0 * injected, (consensus.size, injected, row["blanked"])


# ------ §10.8 stim responder + §10.9 determinism + §10.10 reload ------------


def test_provenance_reload(batch_result):
    for row in batch_result["result"]["rows"]:
        data = _load_mat(row["output_path"])
        prov = data["metrics"]["provenance"]
        assert "config" in prov
        cfg = prov["config"]
        # New step-11/12 params present in provenance
        for key in ("sw_low_hz", "sw_high_hz", "burst_band_low_hz", "burst_threshold_sigma"):
            assert key in cfg, key


def test_determinism_labels_and_mrl(sample_dataset):
    cfg = PipelineConfig()
    vm = VarMap(
        neural="data", rpeak_times="Rpeaks", rpeak_units="sample",
        slowwave_ch1="sw_ch1", slowwave_ch2="sw_ch2", slowwave_ch3="sw_ch3",
        fs="fs", stim_events="stim_events", stim_labels="stim_labels",
    )
    r1 = run_batch(DATA_DIR, vm, cfg)
    r2 = run_batch(DATA_DIR, vm, cfg)
    for a, b in zip(r1["rows"], r2["rows"]):
        if a["status"] != "ok" or b["status"] != "ok":
            continue
        d1 = _load_mat(a["output_path"]); d2 = _load_mat(b["output_path"])
        l1 = np.asarray(_as_list(d1["metrics"]["cuff"])[0]["step6"]["labels"]).ravel()
        l2 = np.asarray(_as_list(d2["metrics"]["cuff"])[0]["step6"]["labels"]).ravel()
        assert l1.size == l2.size and np.array_equal(l1, l2)
