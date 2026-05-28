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


def _rpeak_to_samples(values, units: str, fs: float, var_name: str = "rpeak_times") -> np.ndarray:
    """Convert an R-peak variable into integer sample indices.

    The variable must be a 1-D numeric array (or 2-D with one of the axes
    equal to 1).  When the user has accidentally picked a struct, a cell
    array of variable-length segments, or a multi-dimensional matrix
    instead, the bare ``np.asarray`` call produces the cryptic
    "setting an array element with a sequence ... inhomogeneous shape"
    message.  This wrapper catches that and re-raises with the variable
    name, type and shape so the user knows which dropdown to fix.
    """
    try:
        arr = np.asarray(values)
    except Exception as e:
        raise ValueError(
            f"R-peak variable '{var_name}' could not be interpreted as a numeric array: "
            f"got type={type(values).__name__}.  Pick a variable that is a 1-D array "
            f"of sample indices (e.g. 'Rpeaks', 'peak_samples')."
        ) from e

    if arr.dtype == object:
        # Common cause: cell array of per-segment peak lists, or struct.
        sample = np.asarray(values).flat[0] if hasattr(values, "__len__") else values
        sample_descr = f"first element type={type(sample).__name__}"
        if hasattr(sample, "shape"):
            sample_descr += f", shape={sample.shape}"
        raise ValueError(
            f"R-peak variable '{var_name}' has dtype=object "
            f"(probably a struct or cell array; {sample_descr}).  "
            f"Pick a variable that is a 1-D array of sample indices directly."
        )

    # Accept 1-D, or 2-D with one axis == 1.
    if arr.ndim > 2 or (arr.ndim == 2 and 1 not in arr.shape):
        raise ValueError(
            f"R-peak variable '{var_name}' has shape {arr.shape}; expected a 1-D "
            f"array (or 2-D with one axis = 1) of sample indices."
        )
    v = arr.ravel().astype(np.float64)
    if units == "sample":
        return v.astype(np.int64)
    if units == "sec":
        return np.round(v * fs).astype(np.int64)
    if units == "ms":
        return np.round(v * fs / 1000.0).astype(np.int64)
    raise ValueError(f"Unknown rpeak_units: {units}")


def _looks_like_peak_cells(values) -> bool:
    """True if ``values`` looks like a MATLAB-style cell array of peak-sample
    arrays (a sequence whose entries are numeric 1-D arrays).
    """
    if isinstance(values, (list, tuple)):
        seq = list(values)
    elif isinstance(values, np.ndarray) and values.dtype == object and values.ndim <= 2:
        seq = list(values.flat)
    else:
        return False
    if not seq or len(seq) > 64:
        return False
    for cell in seq:
        a = np.asarray(cell) if not isinstance(cell, np.ndarray) else cell
        if a.dtype.kind not in "iuf":
            return False
        if a.ndim > 2:
            return False
        if a.size == 0:
            continue
        # Heuristic: peak indices are typically integer-valued and within
        # plausible sample range (small positive numbers up to a few billion).
        if a.size > 1 and np.any(a < 0):
            return False
    return True


def _slowwave_from_peak_cells(values, n_samples: int, var_name: str) -> np.ndarray:
    """Build a synthetic phase-coded slow-wave trace from peak sample indices.

    All peak arrays (across cells) are concatenated and sorted.  Between
    consecutive peaks, phase advances linearly from 0 to 2*pi and the
    output is ``sin(phase)`` so that the existing Step 11 Hilbert pipeline
    sees a clean sinusoid whose true phase = 0 at every detected peak.
    Before the first peak and after the last peak the trace is held at 0.
    """
    if isinstance(values, np.ndarray) and values.dtype == object:
        cells = list(values.flat)
    else:
        cells = list(values)

    pieces = []
    for c in cells:
        if c is None:
            continue
        a = np.asarray(c).ravel()
        if a.size == 0:
            continue
        pieces.append(a.astype(np.int64))
    if not pieces:
        raise ValueError(
            f"Slow-wave variable '{var_name}' is a cell array but every cell is empty; "
            f"no peaks to reconstruct phase from."
        )
    peaks = np.unique(np.concatenate(pieces))
    peaks = peaks[(peaks >= 0) & (peaks < n_samples)]
    if peaks.size < 2:
        raise ValueError(
            f"Slow-wave variable '{var_name}' has only {peaks.size} valid peak(s) within "
            f"the recording length ({n_samples}); need at least 2 to define a cycle."
        )

    log.info(
        "Slow-wave variable '%s' looks like a cell of %d cell(s) holding peak sample "
        "indices; reconstructed synthetic phase trace from %d unique peaks "
        "(median cycle length = %.3f s).",
        var_name, len(cells), peaks.size, float(np.median(np.diff(peaks))) / max(n_samples, 1),
    )

    sw = np.zeros(n_samples, dtype=np.float32)
    # Vectorised assignment: walk pairs of consecutive peaks, fill the
    # segment with sin(linspace(0, 2*pi)).
    for s0, s1 in zip(peaks[:-1], peaks[1:]):
        if s1 <= s0:
            continue
        seg_len = int(s1 - s0)
        phases = np.linspace(0.0, 2.0 * np.pi, seg_len, endpoint=False, dtype=np.float32)
        sw[s0:s1] = np.sin(phases)
    return sw


def _coerce_slowwave(values, n_samples: int, var_name: str, channel: int = 0) -> np.ndarray | None:
    """Turn the resolved slow-wave variable into a 1-D float32 trace.

    Two input shapes are supported:

    * **Continuous trace** -- a 1-D numeric array (the default).  Resampled
      linearly to ``n_samples`` if needed.
    * **Cell array / list of peak sample indices** -- common upstream of
      this pipeline (e.g. ``slowWavePeakLocs`` from a MATLAB analysis
      script, packed as a 1xK cell where each cell holds the sample
      numbers of one detector pass).  A synthetic phase-coded trace
      ``sin(2*pi * (t - prev_peak) / (next_peak - prev_peak))`` is
      reconstructed at the neural sampling rate so Step 11's Hilbert
      pipeline keeps working unchanged -- MRL, Rayleigh, and phase
      histograms are mathematically identical to what you'd get from a
      clean sinusoid running through those peaks.
    """
    # ---- Path A: cell-array of peak times -----------------------------------
    if _looks_like_peak_cells(values):
        return _slowwave_from_peak_cells(values, n_samples, var_name)

    try:
        arr = np.asarray(values)
    except Exception as e:
        raise ValueError(
            f"Slow-wave variable '{var_name}' could not be interpreted as a numeric array: "
            f"got type={type(values).__name__}.  Pick a 1-D array of samples "
            f"(e.g. 'slowWaves.trace') or a cell array of peak sample indices "
            f"(e.g. 'slowWavePeakLocs')."
        ) from e

    if arr.dtype == object:
        # Could still be a cell of peak arrays that _looks_like_peak_cells
        # didn't catch (e.g. wrapped one extra level by scipy).  Try the
        # peak-cells path one more time on the ndarray contents.
        if _looks_like_peak_cells(list(arr.flat)):
            return _slowwave_from_peak_cells(list(arr.flat), n_samples, var_name)
        sample = arr.flat[0] if arr.size else None
        descr = f"first element type={type(sample).__name__}"
        if hasattr(sample, "shape"):
            descr += f", shape={sample.shape}"
        raise ValueError(
            f"Slow-wave variable '{var_name}' has dtype=object "
            f"(probably a struct or cell array; {descr}).  "
            f"Pick a variable that is a 1-D numeric array directly "
            f"(e.g. 'slowWaves.trace' rather than the wrapping struct), "
            f"or a cell of peak sample indices (e.g. 'slowWavePeakLocs')."
        )

    if arr.ndim > 2:
        raise ValueError(
            f"Slow-wave variable '{var_name}' has shape {arr.shape}; expected a 1-D "
            f"array, a 2-D matrix of stacked channels, or a cell of peak times."
        )

    if arr.ndim == 2 and 1 not in arr.shape:
        # Multi-channel slow-wave matrix.  Orient so axis 0 = time and slice
        # the requested channel.
        time_axis_is_0 = arr.shape[0] >= arr.shape[1]
        view = arr if time_axis_is_0 else arr.T
        total_chans = view.shape[1]
        if not (0 <= channel < total_chans):
            raise ValueError(
                f"Slow-wave variable '{var_name}' has shape {arr.shape} ({total_chans} channels); "
                f"slowwave_channel={channel} is out of range.  Set var_map.slowwave_channel in "
                f"[0..{total_chans - 1}]."
            )
        sw = view[:, channel].astype(np.float32, copy=False).ravel()
        log.info(
            "Slow-wave variable '%s' is multi-channel (shape=%s); using channel %d of %d.",
            var_name, arr.shape, channel, total_chans,
        )
    else:
        sw = arr.astype(np.float32, copy=False).ravel()

    # Sanity-check the length.  A real slow-wave trace is sampled either
    # at the neural fs (sw.size == n_samples) or at a lower fs but still
    # proportional to recording length.  A "30 sample" pick is almost
    # certainly a per-segment summary, not a trace.
    if sw.size < 100:
        raise ValueError(
            f"Slow-wave variable '{var_name}' has length {sw.size}; that's too short to be "
            f"a slow-wave trace (neural length = {n_samples}).  This is almost certainly a "
            f"summary field; pick a longer variable -- typically '<struct>.trace' or '<struct>.signal'."
        )
    if sw.size < n_samples / 10000:
        raise ValueError(
            f"Slow-wave variable '{var_name}' has length {sw.size} but neural length is {n_samples}. "
            f"That ratio ({sw.size / n_samples:.2e}) is too extreme to be a real slow-wave trace; "
            f"refusing to linearly upsample garbage.  Pick the actual signal trace."
        )

    if sw.size != n_samples:
        log.warning(
            "Slow-wave length %d != neural length %d (ratio %.3g); resampling linearly to match.",
            sw.size, n_samples, sw.size / n_samples,
        )
        x_old = np.linspace(0.0, 1.0, sw.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_samples, endpoint=False)
        sw = np.interp(x_new, x_old, sw).astype(np.float32)
    return sw


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

    rpeak_samples = _rpeak_to_samples(
        _resolve(rdata, var_map.rpeak_times),
        var_map.rpeak_units,
        fs,
        var_name=var_map.rpeak_times,
    )
    # filter to valid range
    rpeak_samples = rpeak_samples[(rpeak_samples >= 0) & (rpeak_samples < n_samples)]

    slowwave = None
    if var_map.slowwave:
        sw_raw: Any | None = None
        sw_source: str | None = None
        if _contains(bdata, var_map.slowwave):
            sw_raw = _resolve(bdata, var_map.slowwave)
            sw_source = "blanked file"
        elif pair.slowwave_path is not None:
            swdata = _read_any(pair.slowwave_path)
            if _contains(swdata, var_map.slowwave):
                sw_raw = _resolve(swdata, var_map.slowwave)
                sw_source = "slow-wave file"
        if sw_raw is None:
            log.warning(
                "Slow-wave variable '%s' not found in either the blanked or slow-wave "
                "file for %s; continuing without slow-wave (Step 11 will be skipped).",
                var_map.slowwave, pair.blanked_path.name,
            )
        else:
            try:
                slowwave = _coerce_slowwave(
                    sw_raw, n_samples, var_map.slowwave, channel=var_map.slowwave_channel,
                )
            except ValueError as e:
                # Slow-wave is optional per spec -- don't crash the whole
                # recording on a bad variable pick.  Step 11 will be
                # silently skipped instead.
                log.warning(
                    "Slow-wave from %s rejected; continuing without slow-wave for %s. "
                    "Step 11 will be skipped.  Reason: %s",
                    sw_source, pair.blanked_path.name, e,
                )
                slowwave = None

    # Stim events are optional; failure here doesn't abort the whole pair.
    stim_events: list[tuple[int, str]] | None = None
    if var_map.stim_events and _contains(bdata, var_map.stim_events):
        try:
            raw_ev = _resolve(bdata, var_map.stim_events)
            ev = np.asarray(raw_ev)
            if ev.dtype == object or ev.ndim > 2:
                raise ValueError(
                    f"Stim events variable '{var_map.stim_events}' has dtype={ev.dtype} "
                    f"and shape={ev.shape}; expected a 1-D numeric array of event times."
                )
            ev = ev.ravel()
            labels = None
            if var_map.stim_labels and _contains(bdata, var_map.stim_labels):
                raw = _resolve(bdata, var_map.stim_labels)
                labels = [str(x) for x in np.asarray(raw).ravel().tolist()]
            ev_samples = ev.astype(np.int64) if ev.dtype.kind in "iu" else np.round(ev * fs).astype(np.int64)
            if labels is None or len(labels) != ev_samples.size:
                labels = [f"cond_{i}" for i in range(ev_samples.size)]
            stim_events = [(int(s), str(l)) for s, l in zip(ev_samples, labels)]
        except Exception as e:
            log.warning(
                "Stim events from '%s' rejected for %s; continuing without stim epochs. "
                "Step 13 will be skipped.  Reason: %s",
                var_map.stim_events, pair.blanked_path.name, e,
            )
            stim_events = None

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
