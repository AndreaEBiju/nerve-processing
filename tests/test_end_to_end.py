"""End-to-end acceptance tests for the vagus pipeline.

These exercise the same path as ``python run.py --no-ui`` against the
synthetic dataset built by :mod:`tests.make_sample_data`.
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
    """Build the sample dataset once per pytest session."""
    from tests.make_sample_data import main as build

    truth = build()
    yield truth


@pytest.fixture(scope="module")
def batch_result(sample_dataset):
    """Run the batch on the synthetic dataset and return its result."""
    cfg = PipelineConfig()
    vm = VarMap(
        neural="data",
        rpeak_times="rpeak_samples",
        rpeak_units="sample",
        slowwave="slow_wave",
        fs="fs",
        stim_events="stim_events",
        stim_labels="stim_labels",
        n_channels=1,
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


# 1. Discovery
def test_discovery_finds_two_pairs(sample_dataset):
    pairs = find_pairs(DATA_DIR)
    assert len(pairs) == 2, [p.blanked_path.name for p in pairs]
    # Every discovered file must carry the version tag (no decoys leaked in).
    import re
    for p in pairs:
        assert re.search(r"_v\d+\.\d+\.\d+_", p.blanked_path.name), p.blanked_path.name
        assert re.search(r"_v\d+\.\d+\.\d+_", p.rpeak_path.name), p.rpeak_path.name


def test_decoys_without_version_tag_are_skipped(sample_dataset):
    """Files like ``rat01_blankmotion.mat`` (no ``_v0.x.x_``) must be ignored."""
    pairs = find_pairs(DATA_DIR)
    names = {p.blanked_path.name for p in pairs} | {p.rpeak_path.name for p in pairs}
    for rec in sample_dataset["recordings"]:
        decoy_b = Path(rec["decoy_blanked"]).name
        decoy_r = Path(rec["decoy_rpeak"]).name
        assert decoy_b not in names, f"decoy leaked into pairs: {decoy_b}"
        assert decoy_r not in names, f"decoy leaked into pairs: {decoy_r}"


def test_recovery_variant_pairs_correctly(sample_dataset):
    """A ``..._recovery_blankmotion`` file must pair with its ``..._recovery_HRBR``."""
    pairs = find_pairs(DATA_DIR)
    recovery_pairs = [p for p in pairs if "recovery" in p.blanked_path.stem]
    if not recovery_pairs:
        return  # this dataset variant didn't include recovery -- nothing to check
    for p in recovery_pairs:
        assert "recovery" in p.rpeak_path.stem, (p.blanked_path.name, p.rpeak_path.name)


# 2. Variable introspection
def test_introspection_lists_known_names(sample_dataset):
    pairs = find_pairs(DATA_DIR)
    assert pairs, "no pairs"
    # Use the first version-tagged pair (decoys are excluded by find_pairs).
    b = introspect_variables(pairs[0].blanked_path)
    r = introspect_variables(pairs[0].rpeak_path)
    assert "data" in b and "slow_wave" in b
    assert "rpeak_samples" in r


# 3. Run completes
def test_run_completes_and_writes_outputs(batch_result):
    res = batch_result["result"]
    assert (DATA_DIR / "batch_pca_basis.npz").exists()
    basis = PCABasis.load(DATA_DIR / "batch_pca_basis.npz")
    assert basis.n_pca == batch_result["cfg"].n_pca
    for row in res["rows"]:
        assert row["status"] == "ok", row
        assert Path(row["output_path"]).exists()


# 4. Shapes/ranges per cuff
def test_per_cuff_shapes_and_ranges(batch_result, sample_dataset):
    truth = sample_dataset
    for row, truth_rec in zip(batch_result["result"]["rows"], truth["recordings"]):
        data = _load_mat(row["output_path"])
        m = data["metrics"]
        cuff = _as_list(m["cuff"])[0]

        # step1 filtered length implied via spike samples bounds + step3 amplitude_hist
        spike_samples = np.asarray(cuff["step3"]["spike_samples"]).ravel()
        # Ground-truth total spike count
        gt = (
            len(truth_rec["unit_A_samples"])
            + len(truth_rec["unit_B_samples"])
            + len(truth_rec["unit_C_samples"])
        )
        # Within a generous tolerance: synth contains a far-field ECG plus
        # stim-boosted extras on top of the labeled units, so detection count
        # can be 2× truth. Floor at 0.5× to catch silent failures.
        assert spike_samples.size >= 0.5 * gt, (spike_samples.size, gt)
        assert spike_samples.size <= 3.0 * gt, (spike_samples.size, gt)

        # n_clusters at least 2 (we injected 3 in-band units; allow over/under-split)
        n_clusters = int(np.asarray(cuff["step6"]["n_clusters"]).item())
        assert n_clusters >= 2, n_clusters

        # ISI present per cluster with >1 spike
        for cl in _as_list(cuff["step7"]["cluster"]):
            n_spk = int(np.asarray(cl["n_spikes"]).item())
            isi = np.asarray(cl["isi_s"]).ravel()
            if n_spk > 1:
                assert isi.size > 0
            snr = float(np.asarray(cl["snr"]).item())
            assert np.isfinite(snr) and snr > 0

        # Audit: ARI should be sane on clean synth data
        ari = float(np.asarray(cuff["step8"]["adjusted_rand"]).item())
        if np.isfinite(ari):
            assert ari > 0.3, ari


def test_cardiac_locked_flag_fires(batch_result, sample_dataset):
    """At least one cluster should be flagged as cardiac-locked (Unit C)."""
    for row in batch_result["result"]["rows"]:
        data = _load_mat(row["output_path"])
        cuff = _as_list(data["metrics"]["cuff"])[0]
        any_locked = False
        any_unlocked = False
        for cl in _as_list(cuff["step9"]["cluster"]):
            locked = bool(np.asarray(cl["is_cardiac_locked"]).item())
            any_locked = any_locked or locked
            any_unlocked = any_unlocked or not locked
        assert any_locked, "no cluster flagged as cardiac-locked"
        assert any_unlocked, "every cluster flagged as cardiac-locked"


def test_breathing_rate_in_range(batch_result, sample_dataset):
    for row in batch_result["result"]["rows"]:
        data = _load_mat(row["output_path"])
        cuff = _as_list(data["metrics"]["cuff"])[0]
        br = float(np.asarray(cuff["step10"]["breathing_rate_hz"]).item())
        # Injected at ~1.33 Hz; allow generous ±50% because surrogate is RSA-based
        assert 0.3 < br < 3.0, br


def test_slowwave_phase_locking(batch_result, sample_dataset):
    """At least one cluster should reach Rayleigh p<0.05 and an MRL above the median."""
    for row in batch_result["result"]["rows"]:
        data = _load_mat(row["output_path"])
        cuff = _as_list(data["metrics"]["cuff"])[0]
        mrls, ps = [], []
        for cl in _as_list(cuff["step11"]["cluster"]):
            mrls.append(float(np.asarray(cl["mrl"]).item()))
            ps.append(float(np.asarray(cl["rayleigh_p"]).item()))
        assert any(p < 0.05 for p in ps), ps
        # the most phase-locked cluster's MRL is above the median
        med = float(np.median(mrls))
        assert max(mrls) > med, (mrls, med)


def test_responder_detection(batch_result, sample_dataset):
    """At least one cluster is a responder for at least one stim condition."""
    for row in batch_result["result"]["rows"]:
        data = _load_mat(row["output_path"])
        cuff = _as_list(data["metrics"]["cuff"])[0]
        any_resp = False
        for cl in _as_list(cuff["step13"]["cluster"]):
            for c in _as_list(cl.get("conditions", []) or []):
                if c is None:
                    continue
                if bool(np.asarray(c["is_responder"]).item()):
                    any_resp = True
        assert any_resp, "no responder detected on a stim-modulated dataset"


def test_determinism(sample_dataset):
    """Two consecutive runs produce identical labels and identical MRL values."""
    cfg = PipelineConfig()
    vm = VarMap(
        neural="data", rpeak_times="rpeak_samples", rpeak_units="sample",
        slowwave="slow_wave", fs="fs", stim_events="stim_events", stim_labels="stim_labels",
    )
    res1 = run_batch(DATA_DIR, vm, cfg)
    res2 = run_batch(DATA_DIR, vm, cfg)
    for r1, r2 in zip(res1["rows"], res2["rows"]):
        d1 = _load_mat(r1["output_path"])
        d2 = _load_mat(r2["output_path"])
        c1 = _as_list(d1["metrics"]["cuff"])[0]
        c2 = _as_list(d2["metrics"]["cuff"])[0]
        l1 = np.asarray(c1["step6"]["labels"]).ravel()
        l2 = np.asarray(c2["step6"]["labels"]).ravel()
        assert l1.size == l2.size
        assert np.array_equal(l1, l2)
        m1 = [float(np.asarray(cl["mrl"]).item()) for cl in _as_list(c1["step11"]["cluster"])]
        m2 = [float(np.asarray(cl["mrl"]).item()) for cl in _as_list(c2["step11"]["cluster"])]
        assert m1 == m2


def test_prepass_then_resume_matches_full(sample_dataset, batch_result):
    """Prepass on this machine + resume on this machine should produce the
    same per-pair cluster count and mean SNR as a full-mode run.

    Mirrors the cross-machine workflow: a Windows box (no MountainSort5)
    runs prepass, ships checkpoints to a Mac, which resumes.
    """
    import shutil
    from vagus_pipeline.batch import run_batch
    from vagus_pipeline.checkpoint import CHECKPOINT_SUFFIX

    cfg = batch_result["cfg"]
    vm = batch_result["var_map"]
    full_rows = batch_result["result"]["rows"]

    # Strip every output from the full run so resume mode discovers
    # checkpoints only.  Skip macOS AppleDouble (._*) sidecars.
    for ext in ("_metrics.mat", CHECKPOINT_SUFFIX):
        for p in DATA_DIR.rglob(f"*{ext}"):
            if p.name.startswith("."):
                continue
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    pre_res = run_batch(DATA_DIR, vm, cfg, mode="prepass")
    assert all(r["status"] == "ok" for r in pre_res["rows"]), pre_res["rows"]
    # Checkpoint file exists for every pair
    cks = sorted(p for p in DATA_DIR.rglob(f"*{CHECKPOINT_SUFFIX}") if not p.name.startswith("."))
    assert len(cks) == len(pre_res["rows"])

    resume_res = run_batch(DATA_DIR, vm, cfg, mode="resume")
    assert all(r["status"] == "ok" for r in resume_res["rows"]), resume_res["rows"]
    assert len(resume_res["rows"]) == len(full_rows)

    # Match clusters + SNR between full and resume (same MS5 input, same
    # PCA basis -> identical sorter result; SNR is computed deterministically).
    for full_row, resume_row in zip(
        sorted(full_rows, key=lambda r: r["dir"]),
        sorted(resume_res["rows"], key=lambda r: r["dir"]),
    ):
        assert full_row["n_clusters_total"] == resume_row["n_clusters_total"], (full_row, resume_row)
        if np.isfinite(full_row["mean_snr"]):
            assert abs(full_row["mean_snr"] - resume_row["mean_snr"]) < 0.5, (full_row, resume_row)


def test_provenance_roundtrip(batch_result):
    for row in batch_result["result"]["rows"]:
        data = _load_mat(row["output_path"])
        prov = data["metrics"]["provenance"]
        # provenance should contain the full config and source paths
        assert "config" in prov
        cfg = prov["config"]
        assert int(np.asarray(cfg["n_pca"]).item()) == PipelineConfig().n_pca
        assert "blanked_path" in prov and "rpeak_path" in prov
        assert "pca_basis_path" in prov
