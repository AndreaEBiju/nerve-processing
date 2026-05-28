"""Load blanked neural data, R-peak times, and (optional) slow-wave channel.

Normalizes everything into a single :class:`Recording` dataclass so the rest
of the pipeline does not care about input format details.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig, VarMap
from .io_discovery import RecordingPair

log = logging.getLogger("vagus.io")


@dataclass
class Recording:
    pair: RecordingPair
    neural: list[np.ndarray]  # one 1-D float array per cuff
    blanked_mask: list[np.ndarray]  # one bool array per cuff (True = blanked)
    rpeak_samples: np.ndarray
    slowwave: np.ndarray | None
    stim_events: list[tuple[int, str]] | None
    fs: float
    n_samples: int

    def cuff_count(self) -> int:
        return len(self.neural)


def _read_any(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return {path.stem: np.load(path, allow_pickle=False)}
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    if suffix == ".mat":
        try:
            from pymatreader import read_mat

            return read_mat(path)
        except Exception:
            from scipy.io import loadmat

            data = loadmat(path, squeeze_me=True)
            return {k: v for k, v in data.items() if not k.startswith("__")}
    raise ValueError(f"Unsupported file type: {suffix}")


def _to_neural_list(
    arr: np.ndarray,
    n_channels_hint: int,
    channel_indices: list[int] | None = None,
) -> list[np.ndarray]:
    """Slice ``arr`` into a list of 1-D float32 cuff traces.

    Handles 1-D arrays (single cuff), 2-D arrays oriented either way
    (time-major or channel-major), and selective channel pickup via
    ``channel_indices``.  Example: a 5-channel acquisition with only
    rows 0 and 3 being nerve cuffs -> pass ``channel_indices=[0, 3]``
    and the returned list has those two traces in that order.
    """
    a = np.asarray(arr)
    if a.ndim == 1:
        if channel_indices and list(channel_indices) != [0]:
            raise ValueError(
                f"channel_indices={channel_indices} requested but neural array is 1-D (single channel)."
            )
        return [a.astype(np.float32, copy=False)]
    if a.ndim != 2:
        raise ValueError(f"Neural array must be 1-D or 2-D, got {a.shape}")

    # Orient so axis 0 is time, axis 1 is channels.
    if a.shape[0] >= a.shape[1]:
        time_view = a
    else:
        time_view = a.T
    total_chans = time_view.shape[1]

    if channel_indices:
        bad = [i for i in channel_indices if i < 0 or i >= total_chans]
        if bad:
            raise ValueError(
                f"channel_indices={channel_indices} out of range for {total_chans}-channel array (shape={a.shape})."
            )
        chans = list(channel_indices)
        log.info(
            "Using channels %s of %d in the neural array (shape=%s).",
            chans, total_chans, a.shape,
        )
    else:
        if total_chans > 2:
            log.warning(
                "Neural array has %d channels (shape=%s) but no channel_indices were given; "
                "using ALL of them as cuffs.  If only some channels are nerve recordings, "
                "set var_map.channel_indices to the cuff channel numbers.",
                total_chans, a.shape,
            )
        if n_channels_hint > 0 and total_chans != n_channels_hint:
            log.warning(
                "Neural shape %s suggests %d channels but var_map.n_channels=%d; using %d.",
                a.shape, total_chans, n_channels_hint, total_chans,
            )
        chans = list(range(total_chans))

    return [time_view[:, c].astype(np.float32, copy=False) for c in chans]


def _detect_blanked(neural: np.ndarray) -> np.ndarray:
    """Mark NaN or runs of exact zeros >= 5 samples as blanked."""
    mask = np.zeros(neural.shape, dtype=bool)
    nan = ~np.isfinite(neural)
    if nan.any():
        mask |= nan
    # find runs of zeros (length >=5)
    zeros = (neural == 0)
    if zeros.any():
        # run-length encode
        d = np.diff(np.concatenate(([0], zeros.view(np.int8), [0])))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            if e - s >= 5:
                mask[s:e] = True
    return mask


def _resolve(data: dict, name: str) -> Any:
    """Look up ``name`` in ``data`` allowing one level of dotted access.

    Examples
    --------
    >>> _resolve({"slow": {"trace": arr}}, "slow.trace")  # returns arr
    >>> _resolve({"data": arr}, "data")                    # returns arr
    """
    if name in data:
        return data[name]
    if "." in name:
        head, _, tail = name.partition(".")
        if head in data and isinstance(data[head], dict) and tail in data[head]:
            return data[head][tail]
    raise KeyError(name)


def _contains(data: dict, name: str) -> bool:
    try:
        _resolve(data, name)
        return True
    except KeyError:
        return False


def _rpeak_to_samples(values: np.ndarray, units: str, fs: float) -> np.ndarray:
    v = np.asarray(values).ravel().astype(np.float64)
    if units == "sample":
        return v.astype(np.int64)
    if units == "sec":
        return np.round(v * fs).astype(np.int64)
    if units == "ms":
        return np.round(v * fs / 1000.0).astype(np.int64)
    raise ValueError(f"Unknown rpeak_units: {units}")


def load_recording(pair: RecordingPair, var_map: VarMap, config: PipelineConfig) -> Recording:
    """Load a recording pair and normalize to a :class:`Recording` instance."""
    if not var_map.neural:
        raise ValueError("var_map.neural is empty; assign the neural variable name.")
    if not var_map.rpeak_times:
        raise ValueError("var_map.rpeak_times is empty; assign the R-peak variable name.")

    bdata = _read_any(pair.blanked_path)
    rdata = _read_any(pair.rpeak_path)

    if not _contains(bdata, var_map.neural):
        raise KeyError(f"Neural variable '{var_map.neural}' not in {pair.blanked_path.name}")
    if not _contains(rdata, var_map.rpeak_times):
        raise KeyError(f"R-peak variable '{var_map.rpeak_times}' not in {pair.rpeak_path.name}")

    fs = config.fs
    if var_map.fs and _contains(bdata, var_map.fs):
        fs_val = _resolve(bdata, var_map.fs)
        if np.isscalar(fs_val) or (hasattr(fs_val, "size") and fs_val.size == 1):
            fs = float(np.asarray(fs_val).item())
            log.info("Using fs=%g Hz from file variable %s", fs, var_map.fs)

    neural_list = _to_neural_list(
        _resolve(bdata, var_map.neural),
        var_map.n_channels,
        channel_indices=var_map.channel_indices,
    )
    n_samples = neural_list[0].size
    blanked_masks = [_detect_blanked(n) for n in neural_list]

    rpeak_samples = _rpeak_to_samples(_resolve(rdata, var_map.rpeak_times), var_map.rpeak_units, fs)
    # filter to valid range
    rpeak_samples = rpeak_samples[(rpeak_samples >= 0) & (rpeak_samples < n_samples)]

    slowwave = None
    if var_map.slowwave:
        if _contains(bdata, var_map.slowwave):
            sw = np.asarray(_resolve(bdata, var_map.slowwave)).astype(np.float32, copy=False).ravel()
            slowwave = sw
        elif pair.slowwave_path is not None:
            swdata = _read_any(pair.slowwave_path)
            if _contains(swdata, var_map.slowwave):
                slowwave = np.asarray(_resolve(swdata, var_map.slowwave)).astype(np.float32, copy=False).ravel()
        if slowwave is not None and slowwave.size != n_samples:
            log.warning(
                "Slow-wave length %d != neural length %d; resampling linearly.",
                slowwave.size, n_samples,
            )
            x_old = np.linspace(0.0, 1.0, slowwave.size, endpoint=False)
            x_new = np.linspace(0.0, 1.0, n_samples, endpoint=False)
            slowwave = np.interp(x_new, x_old, slowwave).astype(np.float32)

    stim_events: list[tuple[int, str]] | None = None
    if var_map.stim_events and _contains(bdata, var_map.stim_events):
        ev = np.asarray(_resolve(bdata, var_map.stim_events)).ravel()
        labels = None
        if var_map.stim_labels and _contains(bdata, var_map.stim_labels):
            raw = _resolve(bdata, var_map.stim_labels)
            labels = [str(x) for x in np.asarray(raw).ravel().tolist()]
        ev_samples = ev.astype(np.int64) if ev.dtype.kind in "iu" else np.round(ev * fs).astype(np.int64)
        if labels is None or len(labels) != ev_samples.size:
            labels = [f"cond_{i}" for i in range(ev_samples.size)]
        stim_events = [(int(s), str(l)) for s, l in zip(ev_samples, labels)]

    rec = Recording(
        pair=pair,
        neural=neural_list,
        blanked_mask=blanked_masks,
        rpeak_samples=rpeak_samples.astype(np.int64),
        slowwave=slowwave,
        stim_events=stim_events,
        fs=float(fs),
        n_samples=n_samples,
    )
    log.info(
        "Loaded %s: %d cuff(s), %d samples (%.1fs), %d R-peaks, slow-wave=%s",
        pair.blanked_path.name,
        rec.cuff_count(),
        n_samples,
        n_samples / fs,
        rpeak_samples.size,
        "yes" if slowwave is not None else "no",
    )
    return rec
