"""Synthesize a labeled vagus-cuff dataset under ``tests/data/``.

Two subfolders so recursive discovery and pairing are exercised. Each
recording contains one cuff with multiple synthetic units, a far-field ECG
that the bandpass should attenuate, an in-band cardiac-locked unit (for the
Step 9 check), a respiration-modulated unit, a slow-wave phase-locked unit,
and a couple of stim events with a post-stim rate change (for Step 13).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scipy.io import savemat  # noqa: E402

from vagus_pipeline.config import PipelineConfig  # noqa: E402


def _triphasic(width_ms: float, amp: float, fs: float, asymm: float = 0.3) -> np.ndarray:
    L = max(int(round(width_ms * 1e-3 * fs * 4)), 8)
    t = np.linspace(-1.0, 1.0, L)
    sigma = 0.18  # narrower means sharper trough
    w = -amp * np.exp(-(t ** 2) / (2 * sigma ** 2))
    w += amp * asymm * np.exp(-((t - 0.5) ** 2) / (2 * (sigma * 1.6) ** 2))
    return w.astype(np.float32)


def _add_template(trace: np.ndarray, times_s: np.ndarray, fs: float, tmpl: np.ndarray) -> np.ndarray:
    samples = np.round(times_s * fs).astype(int)
    for s in samples:
        end = s + tmpl.size
        if 0 <= s < trace.size and end <= trace.size:
            trace[s:end] += tmpl
    return samples


def synth_recording(seed: int, out_dir: Path, name: str, fs: float, duration_s: float, rng: np.random.Generator, extras: dict | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    N = int(round(duration_s * fs))
    t = np.arange(N) / fs

    noise = rng.standard_normal(N).astype(np.float32) * 4.0  # ~ sigma=4

    # Far-field ECG (sub-band, ~400 bpm => 6.7 Hz) -- should be removed by bp.
    # Use a smooth ~30 ms QRS so the spectrum sits below the 100 Hz corner.
    ecg_template_len = int(0.05 * fs)
    ecg_tmpl = np.zeros(ecg_template_len, dtype=np.float32)
    tt = np.linspace(-1, 1, ecg_template_len)
    ecg_tmpl[:] = (np.exp(-(tt ** 2) / 0.05) - 0.3 * np.exp(-((tt - 0.2) ** 2) / 0.10)) * 20.0
    ecg_rate_bpm = 400.0
    rr_s = 60.0 / ecg_rate_bpm
    rpeaks_s = np.arange(rr_s, duration_s - 0.1, rr_s)
    rpeak_samples = np.round(rpeaks_s * fs).astype(np.int64)
    for s in rpeak_samples:
        end = s + ecg_template_len
        if end <= N:
            noise[s:end] += ecg_tmpl

    # Ground-truth units
    rng_u = np.random.default_rng(seed + 1)
    # Unit A — A-like (narrow), respiration-modulated rate (~80 bpm = 1.33 Hz)
    a_tmpl = _triphasic(width_ms=0.5, amp=40.0, fs=fs)
    base_rate_a = 12.0
    resp_freq = 80.0 / 60.0  # Hz
    mod = 0.6 + 0.5 * np.sin(2 * np.pi * resp_freq * t)
    a_times = []
    next_t = 0.0
    while next_t < duration_s - 0.01:
        r = base_rate_a * float(mod[int(next_t * fs)])
        next_t += rng_u.exponential(scale=1.0 / max(r, 1e-3))
        if next_t < duration_s - 0.01:
            a_times.append(next_t)
    a_times = np.asarray(a_times)
    a_samples = _add_template(noise, a_times, fs, a_tmpl)

    # Unit B — C-like (broad), slow-wave phase-locked
    b_tmpl = _triphasic(width_ms=1.5, amp=30.0, fs=fs)
    sw = 100.0 * np.sin(2 * np.pi * 0.05 * t).astype(np.float32)
    sw_period = 1.0 / 0.05
    # Locked spikes at every trough-quarter-phase (-pi/2) of slow wave: phase = -pi/2 ~ trough
    phase = (2 * np.pi * 0.05 * t + np.pi)  # shifted so trough at t=0
    target_phase = -np.pi / 2.0
    candidate_phase = np.angle(np.exp(1j * (phase - target_phase)))
    crossings = np.where((candidate_phase[:-1] >= 0) & (candidate_phase[1:] < 0))[0]
    b_times = (crossings / fs)
    # add small jitter
    b_times = b_times + rng_u.normal(scale=0.05, size=b_times.size)
    b_times = b_times[(b_times > 0.05) & (b_times < duration_s - 0.05)]
    # Sparse extra spontaneous spikes
    extra = rng_u.uniform(0, duration_s, size=80)
    b_times = np.sort(np.concatenate([b_times, extra]))
    b_samples = _add_template(noise, b_times, fs, b_tmpl)

    # Unit C — small cardiac-locked unit (in-band)
    c_tmpl = _triphasic(width_ms=0.8, amp=25.0, fs=fs)
    c_times = rpeaks_s + 0.003  # 3 ms post-R
    c_samples = _add_template(noise, c_times, fs, c_tmpl)

    # Stim events: at t = duration_s*0.4 and duration_s*0.7; post-event rate jump for unit A
    stim_events = [(int(duration_s * 0.4 * fs), "stim_on"), (int(duration_s * 0.7 * fs), "stim_off")]
    # Boost unit A rate inside [stim_on, stim_off]
    s0, s1 = stim_events[0][0], stim_events[1][0]
    t_window_s = (s0 / fs, s1 / fs)
    extra_a_times = rng_u.uniform(*t_window_s, size=200)
    _ = _add_template(noise, extra_a_times, fs, a_tmpl)

    # Real-world filenames stamp the pipeline version into the name:
    #   <prefix>_v0.1.0_blankmotion.mat / <prefix>_v0.1.0_HRBR.mat
    # The optional ``variant`` is e.g. "recovery" -> "<prefix>_v0.1.0_recovery_blankmotion.mat".
    version_tag = "v0.1.0"
    variant = (extras or {}).get("variant", "")
    prefix = f"{name}_{version_tag}" if not variant else f"{name}_{version_tag}_{variant}"
    blanked_path = out_dir / f"{prefix}_blankmotion.mat"
    rp_path = out_dir / f"{prefix}_HRBR.mat"
    savemat(blanked_path, {
        "data": noise.astype(np.float32),
        "slow_wave": sw.astype(np.float32),
        "fs": float(fs),
        "stim_events": np.asarray([s for s, _ in stim_events], dtype=np.int64),
        "stim_labels": np.asarray(["stim_on", "stim_off"]),
    })
    savemat(rp_path, {"rpeak_samples": rpeak_samples.astype(np.int64)})

    # Decoys without the version tag — discovery's required_regex must skip these.
    decoy_blanked = out_dir / f"{name}_blankmotion.mat"
    decoy_rpeak = out_dir / f"{name}_HRBR.mat"
    savemat(decoy_blanked, {"data": np.zeros(16, dtype=np.float32)})
    savemat(decoy_rpeak, {"rpeak_samples": np.zeros(4, dtype=np.int64)})

    return {
        "name": name,
        "blanked": str(blanked_path),
        "rpeak": str(rp_path),
        "decoy_blanked": str(decoy_blanked),
        "decoy_rpeak": str(decoy_rpeak),
        "fs": fs,
        "duration_s": duration_s,
        "rpeak_samples": rpeak_samples,
        "unit_A_samples": np.asarray(a_samples, dtype=np.int64),
        "unit_B_samples": np.asarray(b_samples, dtype=np.int64),
        "unit_C_samples": np.asarray(c_samples, dtype=np.int64),
        "stim_events": stim_events,
        "resp_freq_hz": resp_freq,
        "sw_freq_hz": 0.05,
        "variant": variant,
    }


def main() -> dict:
    cfg = PipelineConfig()
    fs = cfg.fs
    duration_s = 120.0  # spec calls for 120 s

    data_dir = REPO_ROOT / "tests" / "data"
    if data_dir.exists():
        for p in data_dir.rglob("*"):
            if p.is_file():
                p.unlink()
    rng = np.random.default_rng(0)
    # rat01: plain pair. rat02: also includes a "recovery" variant pair in the
    # same directory, plus untagged decoys both rats should *not* be picked up.
    rec1 = synth_recording(0, data_dir / "rat01" / "preStim", "rat01_pre", fs, duration_s, rng)
    rec2 = synth_recording(1, data_dir / "rat02" / "postStim", "rat02_post", fs, duration_s, rng,
                           extras={"variant": "recovery"})
    truth = {"recordings": [rec1, rec2], "fs": fs, "duration_s": duration_s}

    np.savez(
        data_dir / "ground_truth.npz",
        recordings=np.asarray([
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
            for r in truth["recordings"]
        ], dtype=object),
        fs=fs,
        duration_s=duration_s,
    )
    print(f"Wrote sample dataset under {data_dir}")
    print(f"  rat01: A={rec1['unit_A_samples'].size}, B={rec1['unit_B_samples'].size}, C={rec1['unit_C_samples'].size}")
    print(f"  rat02: A={rec2['unit_A_samples'].size}, B={rec2['unit_B_samples'].size}, C={rec2['unit_C_samples'].size}")
    return truth


if __name__ == "__main__":
    main()
