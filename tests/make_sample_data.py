"""Synthesize two labeled datasets under ``tests/data/`` per spec §9.

* **Dataset A** -- all three antral slow-wave channels are healthy.
* **Dataset B** -- the middle channel is corrupted with a mid-recording
  dropout (60-120 s) and elevated noise throughout.

Each dataset has one recording in a subfolder.  Filenames follow the
``..._notched_v0.X.Y_blankmotion[_HRBR|_slowWaves].mat`` convention so the
existing discovery + version-tag rule picks them up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scipy.io import savemat  # noqa: E402

from vagus_pipeline.config import PipelineConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _triphasic(width_ms: float, amp: float, fs: float) -> np.ndarray:
    L = max(int(round(width_ms * 1e-3 * fs * 4)), 8)
    t = np.linspace(-1.0, 1.0, L)
    sigma = 0.18
    w = -amp * np.exp(-(t ** 2) / (2 * sigma ** 2))
    w += amp * 0.3 * np.exp(-((t - 0.5) ** 2) / (2 * (sigma * 1.6) ** 2))
    return w.astype(np.float32)


def _add_template(trace: np.ndarray, samples: np.ndarray, tmpl: np.ndarray) -> None:
    for s in samples:
        end = s + tmpl.size
        if 0 <= s < trace.size and end <= trace.size:
            trace[s:end] += tmpl


def synth_recording(
    out_dir: Path,
    name: str,
    fs: float,
    duration_s: float,
    rng: np.random.Generator,
    *,
    corrupt_ch2: bool = False,
    version_tag: str = "v0.2.0",
) -> dict:
    """Build one synthetic recording.  Writes 3 ``.mat`` files into ``out_dir``.

    Returns a ground-truth dict describing what was injected.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    N = int(round(duration_s * fs))
    t = np.arange(N) / fs

    # ----- Neural noise + cardiac far-field --------------------------------
    noise_amp = 4.0
    neural = rng.standard_normal(N).astype(np.float32) * noise_amp
    ecg_period_s = 60.0 / 400.0  # ~400 bpm
    ecg_template_len = int(0.05 * fs)
    tt = np.linspace(-1, 1, ecg_template_len)
    ecg_tmpl = ((np.exp(-(tt ** 2) / 0.05) - 0.3 * np.exp(-((tt - 0.2) ** 2) / 0.10)) * 20.0).astype(np.float32)
    rpeaks_s = np.arange(ecg_period_s, duration_s - 0.1, ecg_period_s)
    rpeak_samples = np.round(rpeaks_s * fs).astype(np.int64)
    for s in rpeak_samples:
        if s + ecg_template_len <= N:
            neural[s : s + ecg_template_len] += ecg_tmpl

    # ----- Three slow-wave channels (~0.08 Hz, 0.7 s prop lag) --------------
    sw_freq = 0.08
    base_amp = 50.0  # mV-ish
    inter_ch_lag_s = 0.7
    sw_ch1 = (base_amp * np.sin(2 * np.pi * sw_freq * t)).astype(np.float32)
    sw_ch2 = (base_amp * np.sin(2 * np.pi * sw_freq * (t - inter_ch_lag_s))).astype(np.float32)
    sw_ch3 = (base_amp * np.sin(2 * np.pi * sw_freq * (t - 2 * inter_ch_lag_s))).astype(np.float32)
    sw_ch1 += 4.0 * rng.standard_normal(N).astype(np.float32)
    sw_ch2 += 4.0 * rng.standard_normal(N).astype(np.float32)
    sw_ch3 += 4.0 * rng.standard_normal(N).astype(np.float32)

    if corrupt_ch2:
        # 60-120 s = pure noise; whole-recording extra noise (3x more)
        sw_ch2 += 8.0 * rng.standard_normal(N).astype(np.float32)
        s0 = int(60 * fs); s1 = int(120 * fs)
        sw_ch2[s0:s1] = (12.0 * rng.standard_normal(s1 - s0)).astype(np.float32)

    # ----- Antral spike bursts on top of the slow waves --------------------
    sw_period_s = 1.0 / sw_freq
    # 2 bursts per slow-wave cycle, jittered by 0.4 s
    burst_times_s = []
    b = 1.0
    while b < duration_s - 1:
        burst_times_s.append(b + rng.normal(0.0, 0.2))
        burst_times_s.append(b + sw_period_s / 2 + rng.normal(0.0, 0.2))
        b += sw_period_s
    burst_times_s = np.asarray([x for x in burst_times_s if 0 < x < duration_s - 1], dtype=np.float64)

    burst_amp = 30.0
    for bt in burst_times_s:
        s0 = int(bt * fs); s1 = s0 + int(0.5 * fs)
        if s1 <= N:
            tau = np.arange(s1 - s0) / fs
            burst_wave = (burst_amp * np.sin(2 * np.pi * 3.0 * tau)).astype(np.float32)
            sw_ch1[s0:s1] += burst_wave
            if not corrupt_ch2 or (s0 < int(60 * fs) or s0 > int(120 * fs)):
                sw_ch2[s0:s1] += burst_wave
            sw_ch3[s0:s1] += burst_wave

    # ----- Neural units ----------------------------------------------------
    # Unit A: A-like, respiration-modulated rate (~80 bpm)
    a_tmpl = _triphasic(width_ms=0.5, amp=40.0, fs=fs)
    resp_freq = 80.0 / 60.0
    mod = 0.6 + 0.5 * np.sin(2 * np.pi * resp_freq * t)
    a_times = []
    next_t = 0.0
    base_rate = 12.0
    while next_t < duration_s - 0.01:
        r = base_rate * float(mod[int(next_t * fs)])
        next_t += rng.exponential(scale=1.0 / max(r, 1e-3))
        if next_t < duration_s - 0.01:
            a_times.append(next_t)
    a_samples = np.round(np.asarray(a_times) * fs).astype(np.int64)
    _add_template(neural, a_samples, a_tmpl)

    # Unit B: C-like, phase-locked to slow wave (~90 deg target)
    b_tmpl = _triphasic(width_ms=1.5, amp=30.0, fs=fs)
    # Spikes at every ~5 deg around 90 deg of the slow wave, jittered
    target_phase_deg = 90.0
    cycle_n = int(duration_s / sw_period_s)
    b_times = []
    for k in range(cycle_n):
        t_lock = k * sw_period_s + sw_period_s * ((target_phase_deg / 360.0) + 0.25)
        for _ in range(15):
            b_times.append(t_lock + rng.normal(0.0, 0.4))
    b_times = np.asarray([x for x in b_times if 0 < x < duration_s - 0.05], dtype=np.float64)
    b_samples = np.round(b_times * fs).astype(np.int64)
    _add_template(neural, b_samples, b_tmpl)

    # Unit C: cardiac-locked (small in-band unit, 3 ms after each R-peak)
    c_tmpl = _triphasic(width_ms=0.8, amp=25.0, fs=fs)
    c_times = rpeaks_s + 0.003
    c_samples = np.round(c_times * fs).astype(np.int64)
    _add_template(neural, c_samples, c_tmpl)

    # Unit D: efferent-like w.r.t. bursts (~2 s BEFORE each burst)
    d_tmpl = _triphasic(width_ms=0.6, amp=28.0, fs=fs)
    d_times = burst_times_s - 2.0
    d_times = d_times[d_times > 0]
    d_jitter = []
    for tb in d_times:
        for _ in range(6):
            d_jitter.append(tb + rng.normal(0.0, 0.1))
    d_samples = np.round(np.asarray(d_jitter) * fs).astype(np.int64)
    d_samples = d_samples[(d_samples > 0) & (d_samples < N)]
    _add_template(neural, d_samples, d_tmpl)

    # Unit E: afferent-like (~1 s AFTER each burst)
    e_tmpl = _triphasic(width_ms=0.8, amp=22.0, fs=fs)
    e_times = burst_times_s + 1.0
    e_jitter = []
    for tb in e_times:
        for _ in range(6):
            e_jitter.append(tb + rng.normal(0.0, 0.1))
    e_samples = np.round(np.asarray(e_jitter) * fs).astype(np.int64)
    e_samples = e_samples[(e_samples > 0) & (e_samples < N)]
    _add_template(neural, e_samples, e_tmpl)

    # Stim events for Step 14
    stim_events = [
        (int(duration_s * 0.4 * fs), "stim_on"),
        (int(duration_s * 0.7 * fs), "stim_off"),
    ]
    s0, s1 = stim_events[0][0], stim_events[1][0]
    extra_a_times = rng.uniform(s0 / fs, s1 / fs, size=200)
    _add_template(neural, np.round(extra_a_times * fs).astype(np.int64), a_tmpl)

    # ----- Write files ------------------------------------------------------
    blanked = out_dir / f"{name}_notched_{version_tag}_blankmotion.mat"
    rp = out_dir / f"{name}_notched_{version_tag}_blankmotion_HRBR.mat"
    sw = out_dir / f"{name}_notched_{version_tag}_blankmotion_slowWaves.mat"
    savemat(blanked, {
        "data": neural,
        "fs": float(fs),
        "stim_events": np.asarray([s for s, _ in stim_events], dtype=np.int64),
        "stim_labels": np.asarray(["stim_on", "stim_off"]),
    })
    savemat(rp, {"Rpeaks": rpeak_samples.astype(np.int64)})
    savemat(sw, {
        "sw_ch1": sw_ch1,
        "sw_ch2": sw_ch2,
        "sw_ch3": sw_ch3,
    })

    return {
        "name": name,
        "out_dir": str(out_dir),
        "blanked": str(blanked),
        "rpeak": str(rp),
        "slowwave": str(sw),
        "fs": fs,
        "duration_s": duration_s,
        "version_tag": version_tag,
        "rpeak_samples": rpeak_samples,
        "burst_times_s": burst_times_s,
        "target_sw_phase_deg": target_phase_deg,
        "stim_events": stim_events,
        "resp_freq_hz": resp_freq,
        "sw_freq_hz": sw_freq,
        "inter_ch_lag_s": inter_ch_lag_s,
        "corrupt_ch2": corrupt_ch2,
        "n_unit_A": int(a_samples.size),
        "n_unit_B": int(b_samples.size),
        "n_unit_C": int(c_samples.size),
        "n_unit_D": int(d_samples.size),
        "n_unit_E": int(e_samples.size),
    }


def main() -> dict:
    cfg = PipelineConfig()
    fs = cfg.fs
    duration_s = 180.0
    data_dir = REPO_ROOT / "tests" / "data"
    if data_dir.exists():
        for p in data_dir.rglob("*"):
            if p.is_file():
                p.unlink()
    rng = np.random.default_rng(0)
    rec_a = synth_recording(data_dir / "datasetA" / "ratA", "ratA_good", fs, duration_s, rng,
                            corrupt_ch2=False)
    rec_b = synth_recording(data_dir / "datasetB" / "ratB", "ratB_dropout", fs, duration_s, rng,
                            corrupt_ch2=True)
    print(f"Wrote dataset A: {rec_a['blanked']}")
    print(f"  units A/B/C/D/E = {rec_a['n_unit_A']}/{rec_a['n_unit_B']}/{rec_a['n_unit_C']}/{rec_a['n_unit_D']}/{rec_a['n_unit_E']}")
    print(f"  bursts: {rec_b['burst_times_s'].size}, target SW phase: {rec_a['target_sw_phase_deg']:.0f} deg")
    print(f"Wrote dataset B (ch2 dropout): {rec_b['blanked']}")

    np.savez(data_dir / "ground_truth.npz",
             recordings=np.asarray([
                 {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
                 for r in (rec_a, rec_b)
             ], dtype=object),
             fs=fs, duration_s=duration_s)
    return {"recordings": [rec_a, rec_b], "fs": fs, "duration_s": duration_s}


if __name__ == "__main__":
    main()
