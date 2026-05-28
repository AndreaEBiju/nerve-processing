"""Recursive file discovery, pairing, and variable-name introspection.

The user provides two filename patterns (glob or substring) via the UI; this
module walks a root directory recursively, pairs blanked-data files with
R-peak files, and exposes the variables inside each file for role assignment.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

from .config import VarMap

log = logging.getLogger("vagus.discovery")


# Filename signatures
# -------------------
# Real-world acquisitions stamp the pipeline version into the filename
# (e.g. ``..._v0.1.0_blankmotion.mat`` / ``..._v0.1.0_HRBR.mat`` and
# ``..._v0.1.0_recovery_blankmotion.mat`` etc.). Files that lack the version
# tag — e.g. ``foo_blankmotion.mat`` on its own — are *not* meant to be used.
# ``DEFAULT_REQUIRED_REGEX`` enforces that signature; pass an empty string or
# ``None`` to disable the filter.
DEFAULT_REQUIRED_REGEX = r"_v\d+\.\d+\.\d+_"

# Token-based exact pairing (preferred). Within a directory, candidates are
# grouped by the part of the stem that remains after the configured token is
# removed — so ``rat01_v0.1.0_blankmotion`` and ``rat01_v0.1.0_HRBR`` both
# reduce to the key ``rat01_v0.1.0``, while ``rat01_v0.1.0_recovery_blankmotion``
# reduces to ``rat01_v0.1.0_recovery`` and is paired only with its matching
# recovery R-peak file.
DEFAULT_BLANKED_TOKEN = "blankmotion"
DEFAULT_RPEAK_TOKEN = "HRBR"
DEFAULT_SLOWWAVE_TOKEN = "slowWaves"

DEFAULT_BLANKED_PATTERNS = ("*blankmotion*", "*BlankMotion*", "*blank*", "*Blank*", "*BLANK*")
DEFAULT_RPEAK_PATTERNS = ("*HRBR*", "*hrbr*", "*HR*", "*hr*", "*rpeak*", "*Rpeak*", "*RPeak*")
DEFAULT_SLOWWAVE_PATTERNS = ("*slowWaves*", "*slowwaves*", "*SlowWaves*", "*slow*", "*SW*", "*sw*")


def _strip_token(stem: str, token: str | None) -> str:
    """Remove ``token`` (case-insensitive, last occurrence) from ``stem`` and
    clean the surrounding separators so the remainder can be used as a pair
    key.  Returns ``stem`` unchanged if the token is absent or empty.
    """
    if not token:
        return stem
    low = stem.lower()
    tok = token.lower()
    i = low.rfind(tok)
    if i < 0:
        return stem
    key = (stem[:i] + stem[i + len(token) :]).strip("_-. ")
    while "__" in key:
        key = key.replace("__", "_")
    return key


def _make_pair_key(stem: str, tokens: list[str]) -> str:
    """Build a recording-level pair key by stripping every role token
    (blanked / rpeak / slow-wave) from the stem.

    The motivation is that some acquisition pipelines name companion files
    using the blanked-file stem as a prefix — e.g. baseline files share a
    ``..._blankmotion_HRBR.mat`` / ``..._blankmotion_slowWaves.mat`` layout —
    so stripping a single token isn't enough to align the keys.  Stripping
    every known token collapses all companion files of the same recording to
    the same key.
    """
    s = stem
    for tok in tokens:
        if not tok:
            continue
        pat = re.compile(re.escape(tok), re.IGNORECASE)
        s = pat.sub("", s)
    # Collapse stray separators left behind by token removal
    s = re.sub(r"[_\-.]{2,}", "_", s).strip("_-. ")
    return s


@dataclass
class RecordingPair:
    dir: Path
    blanked_path: Path
    rpeak_path: Path
    slowwave_path: Path | None = None  # legacy single file (kept for back-compat)
    slowwave_paths: list[Path] | None = None  # up to 3 separate slow-wave files
    var_map: VarMap | None = None
    status: str = "ok"
    note: str = ""

    def common_stem(self) -> str:
        a, b = self.blanked_path.stem, self.rpeak_path.stem
        match = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
        stem = a[match.a : match.a + match.size].strip("_- .")
        return stem or a


def _matches_any(name: str, patterns: tuple[str, ...] | list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) or p.strip("*") in name for p in patterns)


def find_pairs(
    root_dir: str | os.PathLike,
    blanked_patterns: tuple[str, ...] | list[str] = DEFAULT_BLANKED_PATTERNS,
    rpeak_patterns: tuple[str, ...] | list[str] = DEFAULT_RPEAK_PATTERNS,
    slowwave_patterns: tuple[str, ...] | list[str] | None = DEFAULT_SLOWWAVE_PATTERNS,
    required_regex: str | None = DEFAULT_REQUIRED_REGEX,
    blanked_token: str | None = DEFAULT_BLANKED_TOKEN,
    rpeak_token: str | None = DEFAULT_RPEAK_TOKEN,
    slowwave_token: str | None = DEFAULT_SLOWWAVE_TOKEN,
    extensions: tuple[str, ...] = (".mat", ".npy", ".npz"),
) -> list[RecordingPair]:
    """Walk ``root_dir`` recursively and produce a list of recording pairs.

    Filtering order:

    1. Drop hidden / AppleDouble sidecars and any ``*_metrics.*`` outputs.
    2. Drop files whose basename does **not** match ``required_regex``
       (default: a version tag like ``_v0.1.0_``).  Pass ``required_regex=None``
       or ``""`` to disable this filter.
    3. Match candidate filenames against the configured glob/substring patterns.

    Pairing inside each directory:

    * If both ``blanked_token`` and ``rpeak_token`` are provided, build pair
      keys by stripping the token from each filename stem (e.g.
      ``rat01_v0.1.0_blankmotion`` → ``rat01_v0.1.0``) and pair files with the
      same key.  This is the preferred path because it deterministically
      separates ``..._blankmotion`` ↔ ``..._HRBR`` from
      ``..._recovery_blankmotion`` ↔ ``..._recovery_HRBR``.
    * Any remaining unmatched files fall back to the longest-common-stem
      scorer used previously.  Ambiguity is surfaced on the returned pair's
      ``status`` / ``note`` rather than silently guessed.
    """
    root = Path(root_dir)
    pairs: list[RecordingPair] = []
    req = re.compile(required_regex) if required_regex else None
    all_tokens = [t for t in (blanked_token, rpeak_token, slowwave_token) if t]

    for dirpath, _dirnames, filenames in os.walk(root):
        names = [
            f
            for f in filenames
            if Path(f).suffix.lower() in extensions
            and not f.startswith(".")  # skip hidden / AppleDouble sidecars
            and "metrics" not in Path(f).stem.lower()  # skip our own outputs
            and (req is None or req.search(f) is not None)
        ]

        # Classification priority: R-peak first, then slow-wave, then blanked.
        # This matters because real-world filenames are often nested — e.g.
        # ``..._blankmotion_HRBR.mat`` matches both the blanked AND R-peak
        # patterns, and ``..._blankmotion_slowWaves.mat`` matches both the
        # blanked AND slow-wave patterns.  Classifying the most-specific role
        # first keeps the wrong classifier from stealing the file.
        rpeaks = [f for f in names if _matches_any(f, rpeak_patterns)]
        slow = (
            [f for f in names if _matches_any(f, slowwave_patterns) and f not in rpeaks]
            if slowwave_patterns
            else []
        )
        blanked = [
            f
            for f in names
            if _matches_any(f, blanked_patterns) and f not in rpeaks and f not in slow
        ]

        if not blanked or not rpeaks:
            continue

        matched_blanked: set[str] = set()
        matched_rpeaks: set[str] = set()

        # --- Pass 1: deterministic token-based pairing -------------------
        # Build the pair key by stripping *every* role token from the stem,
        # not just the file's own token.  That collapses both
        # ``..._blankmotion``, ``..._blankmotion_HRBR``,
        # ``..._blankmotion_slowWaves`` (baseline convention) and
        # ``..._recovery_blankmotion``, ``..._recovery_HRBR``,
        # ``..._recovery_slowWaves`` (recovery convention) onto a stable key.
        if all_tokens:
            b_keyed: dict[str, list[str]] = defaultdict(list)
            for f in blanked:
                b_keyed[_make_pair_key(Path(f).stem, all_tokens)].append(f)
            r_keyed: dict[str, list[str]] = defaultdict(list)
            for f in rpeaks:
                r_keyed[_make_pair_key(Path(f).stem, all_tokens)].append(f)
            sw_keyed: dict[str, list[str]] = defaultdict(list)
            for f in slow:
                sw_keyed[_make_pair_key(Path(f).stem, all_tokens)].append(f)

            for k, b_files in b_keyed.items():
                r_files = r_keyed.get(k, [])
                if not r_files:
                    continue
                sw_for_key = sw_keyed.get(k, [])
                status, note = "ok", ""
                if len(b_files) > 1 or len(r_files) > 1:
                    status = "ambiguous"
                    note = f"multiple files share key '{k}'"
                for b, r in zip(sorted(b_files), sorted(r_files)):
                    matched_blanked.add(b)
                    matched_rpeaks.add(r)
                    pairs.append(
                        _build_pair(
                            Path(dirpath), b, r, sw_for_key, status, note,
                            prefer_exact_key=True,
                        )
                    )

        # --- Pass 2: fall back to common-stem scoring for unmatched files -
        remaining_b = [f for f in blanked if f not in matched_blanked]
        remaining_r = [f for f in rpeaks if f not in matched_rpeaks]
        if remaining_b and remaining_r:
            used_rpeaks: set[str] = set()
            for b in remaining_b:
                best, best_score = None, -1.0
                for r in remaining_r:
                    if r in used_rpeaks:
                        continue
                    score = SequenceMatcher(None, Path(b).stem, Path(r).stem).ratio()
                    if score > best_score:
                        best, best_score = r, score
                if best is None:
                    continue
                status, note = "ok", ""
                if len(remaining_b) > 1 or len(remaining_r) > 1:
                    if best_score < 0.5:
                        status, note = "ambiguous", f"best stem match score={best_score:.2f}"
                used_rpeaks.add(best)
                pairs.append(_build_pair(Path(dirpath), b, best, slow, status, note))

    pairs.sort(key=lambda p: (str(p.dir), p.blanked_path.name))
    log.info("Discovered %d recording pair(s) under %s", len(pairs), root)
    return pairs


def _build_pair(
    dirpath: Path,
    blanked_name: str,
    rpeak_name: str,
    slow_candidates: list[str],
    status: str,
    note: str,
    prefer_exact_key: bool = False,
) -> RecordingPair:
    sw_path = None
    if slow_candidates:
        if prefer_exact_key and len(slow_candidates) >= 1:
            # Caller already filtered to slow-wave candidates whose pair key
            # equals the blanked/rpeak key, so pick the first one
            # deterministically (sorted).
            sw_path = dirpath / sorted(slow_candidates)[0]
        else:
            sw_best, sw_score = None, -1.0
            for s in slow_candidates:
                score = SequenceMatcher(None, Path(blanked_name).stem, Path(s).stem).ratio()
                if score > sw_score:
                    sw_best, sw_score = s, score
            if sw_best is not None:
                sw_path = dirpath / sw_best
    return RecordingPair(
        dir=dirpath,
        blanked_path=dirpath / blanked_name,
        rpeak_path=dirpath / rpeak_name,
        slowwave_path=sw_path,
        status=status,
        note=note,
    )


# ---------------------------------------------------------------------------
# Variable-name introspection
# ---------------------------------------------------------------------------


def introspect_variables(path: str | os.PathLike) -> dict[str, dict[str, Any]]:
    """Return a mapping ``var_name -> {shape, dtype, kind}`` for the file.

    Supports .mat (v5 + v7.3 via pymatreader), .npy, and .npz.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    out: dict[str, dict[str, Any]] = {}

    if suffix == ".npy":
        arr = np.load(p, allow_pickle=False, mmap_mode="r")
        out[p.stem] = {"shape": tuple(arr.shape), "dtype": str(arr.dtype), "kind": "array"}
    elif suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            for k in z.files:
                a = z[k]
                out[k] = {"shape": tuple(a.shape), "dtype": str(a.dtype), "kind": "array"}
    elif suffix == ".mat":
        # METADATA-ONLY READ: do not load array data, which on real recordings
        # can be many GB.  v5 .mat files expose name/shape/class through
        # ``scipy.io.whosmat`` without loading; v7.3 files are HDF5 and expose
        # the same through ``h5py``.  Structs (which are typically small) are
        # loaded selectively just to enumerate their leaf field names.
        try:
            return _introspect_mat_metadata_only(p)
        except Exception as e:
            log.warning("Fast metadata read of %s failed (%s); falling back to full read.", p.name, e)
            from pymatreader import read_mat

            data = read_mat(p)
            data = {k: v for k, v in data.items() if not k.startswith("__")}
            for k, v in data.items():
                _describe(k, v, out)
                if isinstance(v, dict):
                    for ck, cv in v.items():
                        _describe(f"{k}.{ck}", cv, out)
            return out
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return out


def _introspect_mat_metadata_only(path: Path) -> dict[str, dict[str, Any]]:
    """Read variable names + shapes + dtypes from a .mat file without loading
    the array data.  Handles both v5 (scipy whosmat) and v7.3 (HDF5/h5py).

    For top-level structs, the struct itself is loaded — that's a cheap
    operation because structs hold field metadata, not bulk arrays — and its
    field names are surfaced as ``parent.child`` entries.
    """
    out: dict[str, dict[str, Any]] = {}

    # Try v5 first via scipy whosmat.
    try:
        from scipy.io import whosmat
        items = whosmat(str(path))
    except NotImplementedError:
        items = None  # signals "this is v7.3"
    except Exception:
        items = None

    if items is not None:
        from scipy.io import loadmat
        for name, shape, klass in items:
            if name.startswith("__"):
                continue
            kind = _matclass_to_kind(klass)
            out[name] = {"shape": tuple(shape), "dtype": klass, "kind": kind}
            if klass == "struct":
                try:
                    d = loadmat(str(path), variable_names=[name], squeeze_me=True, struct_as_record=False)
                    struct_val = d.get(name)
                    _enumerate_struct_fields(name, struct_val, out)
                except Exception as e:
                    log.warning("Couldn't enumerate struct '%s' in %s: %s", name, path.name, e)
        return out

    # v7.3 path: HDF5 via h5py.
    import h5py

    with h5py.File(path, "r") as f:
        for key in f.keys():
            if key.startswith("#"):  # MATLAB v7.3 internal refs
                continue
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                # v7.3 stores arrays transposed relative to MATLAB; report the
                # MATLAB-side shape (reversed).
                shape = tuple(reversed(obj.shape)) if obj.shape else ()
                out[key] = {"shape": shape, "dtype": str(obj.dtype), "kind": "array"}
            elif isinstance(obj, h5py.Group):
                out[key] = {"shape": (len(obj),), "dtype": "struct", "kind": "struct"}
                for ck in obj.keys():
                    if ck.startswith("#"):
                        continue
                    cobj = obj[ck]
                    full = f"{key}.{ck}"
                    if isinstance(cobj, h5py.Dataset):
                        shape = tuple(reversed(cobj.shape)) if cobj.shape else ()
                        out[full] = {"shape": shape, "dtype": str(cobj.dtype), "kind": "array"}
                    elif isinstance(cobj, h5py.Group):
                        out[full] = {"shape": (len(cobj),), "dtype": "struct", "kind": "struct"}
    return out


def _matclass_to_kind(klass: str) -> str:
    if klass in ("struct",):
        return "struct"
    if klass in ("cell",):
        return "sequence"
    if klass in ("char",):
        return "string"
    if klass in ("function_handle", "object"):
        return "other"
    return "array"


def _enumerate_struct_fields(parent: str, struct_val: Any, out: dict[str, dict[str, Any]]) -> None:
    """Surface field names from a loaded struct as ``parent.field`` entries."""
    if struct_val is None:
        return
    # scipy.io with struct_as_record=False gives mat_struct objects whose
    # ._fieldnames attribute lists the field names.
    fieldnames = getattr(struct_val, "_fieldnames", None)
    if fieldnames:
        for fname in fieldnames:
            try:
                fval = getattr(struct_val, fname)
            except Exception:
                continue
            _describe(f"{parent}.{fname}", fval, out)
        return
    # pymatreader gives a plain dict.
    if isinstance(struct_val, dict):
        for fname, fval in struct_val.items():
            _describe(f"{parent}.{fname}", fval, out)
        return
    # ndarray dtype with field names (older scipy default).
    if hasattr(struct_val, "dtype") and getattr(struct_val.dtype, "names", None):
        for fname in struct_val.dtype.names:
            try:
                fval = struct_val[fname]
            except Exception:
                continue
            _describe(f"{parent}.{fname}", fval, out)


def _describe(name: str, v: Any, out: dict[str, dict[str, Any]]) -> None:
    if isinstance(v, np.ndarray):
        # ndarrays with dtype=object are usually cell arrays of variable-
        # length sub-arrays; treat them as sequences so downstream pickers
        # don't try to use them as numeric traces.
        kind = "sequence" if v.dtype == object else "array"
        out[name] = {"shape": tuple(v.shape), "dtype": str(v.dtype), "kind": kind}
    elif isinstance(v, (int, float, bool, np.floating, np.integer)):
        out[name] = {"shape": (), "dtype": type(v).__name__, "kind": "scalar"}
    elif isinstance(v, (list, tuple)):
        try:
            arr = np.asarray(v)
            kind = "sequence" if arr.dtype == object else "array"
            out[name] = {"shape": tuple(arr.shape), "dtype": str(arr.dtype), "kind": kind}
        except Exception:
            out[name] = {"shape": (len(v),), "dtype": "sequence", "kind": "sequence"}
    elif isinstance(v, dict):
        out[name] = {"shape": (len(v),), "dtype": "dict", "kind": "struct"}
    elif isinstance(v, str):
        out[name] = {"shape": (len(v),), "dtype": "str", "kind": "string"}
    elif v is None:
        out[name] = {"shape": (), "dtype": "None", "kind": "none"}
    else:
        out[name] = {"shape": (), "dtype": type(v).__name__, "kind": "other"}


def autopopulate_var_map(
    blanked_vars: dict[str, dict[str, Any]],
    rpeak_vars: dict[str, dict[str, Any]],
    slowwave_vars: dict[str, dict[str, Any]] | None = None,
    fs_hint: float = 24414.0625,
) -> VarMap:
    """Best-guess variable assignment from introspected file contents.

    The "neural" variable is the largest 1- or 2-D array in the blanked file
    that is not obviously something else. R-peak is the variable whose name
    matches /rpeak|HR|R_peak|peaks/ or, failing that, the longest 1-D array
    in the R-peak file.
    """
    vm = VarMap()

    def pick_neural(vars_: dict[str, dict[str, Any]]) -> tuple[str, int]:
        best, best_size, best_chans = "", 0, 1
        for k, v in vars_.items():
            shape = v.get("shape", ())
            kind = v.get("kind", "")
            if kind != "array":
                continue
            if len(shape) == 1 and shape[0] > best_size:
                best, best_size, best_chans = k, shape[0], 1
            elif len(shape) == 2:
                rows, cols = shape
                length, n_ch = max(rows, cols), min(rows, cols)
                if length > best_size and n_ch <= 8:
                    best, best_size, best_chans = k, length, n_ch
        return best, best_chans

    vm.neural, vm.n_channels = pick_neural(blanked_vars)

    # rpeak: prefer SPECIFIC name-matches and only accept 1-D numeric arrays.
    # Generic substrings like "hr" / "br" match too broadly in HRV analysis
    # files where many variables (rate, segments, mean RR, etc.) share the
    # token but aren't sample-index arrays.
    def _is_1d_numeric(v: dict) -> bool:
        shape = v.get("shape", ())
        if v.get("kind") != "array":
            return False
        # Accept (N,) or (N, 1) / (1, N).
        if len(shape) == 1:
            return shape[0] > 0
        if len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
            return max(shape) > 0
        return False

    specific_tokens = ("rpeak", "r_peak", "rpeaks", "rwave", "r_wave", "peaks", "peak_samples", "peak_times", "qrs")
    broad_tokens = ("hr", "br", "ibi", "rri")
    name_hits: list[str] = []
    for tok in specific_tokens:
        for k, v in rpeak_vars.items():
            if tok in k.lower() and _is_1d_numeric(v) and k not in name_hits:
                name_hits.append(k)
    if not name_hits:
        for tok in broad_tokens:
            for k, v in rpeak_vars.items():
                if tok in k.lower() and _is_1d_numeric(v) and k not in name_hits:
                    name_hits.append(k)
    if name_hits:
        vm.rpeak_times = name_hits[0]
    else:
        # Last resort: longest 1-D numeric array.
        longest, longest_size = "", 0
        for k, v in rpeak_vars.items():
            if not _is_1d_numeric(v):
                continue
            shape = v.get("shape", ())
            n = max(shape) if shape else 0
            if n > longest_size:
                longest, longest_size = k, n
        vm.rpeak_times = longest

    # units heuristic for rpeak
    if vm.rpeak_times:
        v = rpeak_vars.get(vm.rpeak_times, {})
        # cannot read the values themselves here (no array), so guess from name
        n = vm.rpeak_times.lower()
        if "sec" in n or n.endswith("_s"):
            vm.rpeak_units = "sec"
        elif "ms" in n:
            vm.rpeak_units = "ms"
        else:
            vm.rpeak_units = "sample"

    # slowwave: in-blanked variable first, else external file.
    # Prefer name-matches ("slow" / "sw" / "wave" / "lfp"), but fall back to
    # the longest 1-D array in the slow-wave file if nothing obvious turns up.
    def _is_1d_numeric(v: dict) -> bool:
        if v.get("kind") != "array":
            return False
        # Reject object/dict/sequence dtypes -- they are usually nested
        # structs or ragged cell arrays that can't be stacked into a 1-D
        # trace.
        dtype = str(v.get("dtype", "")).lower()
        if dtype in ("object", "dict", "sequence", "str"):
            return False
        shape = v.get("shape", ())
        if len(shape) == 1:
            return shape[0] > 0
        if len(shape) == 2 and 1 in shape:
            return max(shape) > 0
        return False

    def _length(v: dict) -> int:
        shape = v.get("shape", ())
        return max(shape) if shape else 0

    def _pick_sw(vars_: dict[str, dict[str, Any]], allow_longest_array: bool, min_length: int = 100) -> str:
        """Pick a slow-wave variable.

        Strategy:
          1. Among entries whose **name** hints at slow-wave PEAK times
             (peakloc / peaktimes / peaks_idx / sw_peaks), accept the
             FIRST one even if it's a cell array (kind=sequence).  The
             loader reconstructs a synthetic trace from those peaks.
          2. Among entries whose name hints at a slow-wave TRACE (slow /
             wave / lfp / trace / filtered), keep only 1-D numeric
             arrays at least ``min_length`` samples long; pick the LONGEST.
          3. If nothing name-matches and ``allow_longest_array`` is True
             (dedicated slow-wave file), pick the longest 1-D numeric
             array overall.
          4. Otherwise return "" (no autopopulation).
        """
        peak_tokens = ("peakloc", "peak_loc", "peaktimes", "peak_times", "peakidx", "peak_idx", "peaks_idx", "sw_peaks", "sw_peak")
        for k in vars_:
            if any(t in k.lower() for t in peak_tokens):
                # Cell arrays show up as kind="sequence"; that's fine here.
                kind = vars_[k].get("kind", "")
                if kind in ("array", "sequence"):
                    return k

        candidates = {
            k: v for k, v in vars_.items()
            if _is_1d_numeric(v)
            and any(t in k.lower() for t in ("slow", "wave", "lfp", "trace", "filtered"))
            and _length(v) >= min_length
        }
        if candidates:
            return max(candidates, key=lambda k: _length(candidates[k]))
        if not allow_longest_array:
            return ""
        best, best_size = "", 0
        for k, v in vars_.items():
            if not _is_1d_numeric(v):
                continue
            n = _length(v)
            if n > best_size and n >= min_length:
                best, best_size = k, n
        return best

    vm.slowwave = _pick_sw(blanked_vars, allow_longest_array=False) or (
        _pick_sw(slowwave_vars, allow_longest_array=True) if slowwave_vars else ""
    )
    if not vm.slowwave:
        vm.slowwave = None

    # New 3-channel slot autopopulation.
    #
    # Priority 1: a single multi-channel matrix.  Many acquisitions stack
    # all three antral electrodes into one ``slowWaveTrace`` variable with
    # shape (N, 3) or (3, N).  When that variable exists, autopop fills
    # all three slots with the SAME name and sets ``slowwave_ch_indices``
    # to ``[0, 1, 2]`` so the loader picks each column as ch1/ch2/ch3.
    pool = slowwave_vars if slowwave_vars else blanked_vars
    multi_chan_var = ""
    if pool:
        for k, v in pool.items():
            if v.get("kind") != "array":
                continue
            shape = tuple(v.get("shape", ()))
            if len(shape) != 2:
                continue
            # Accept any axis with size in {1, 2, 3, 4} as "channels".
            if 3 in shape or 2 in shape:
                # Prefer name-matched candidates.
                if any(t in k.lower() for t in ("slow", "wave", "lfp", "trace", "filtered")):
                    multi_chan_var = k
                    break
        if not multi_chan_var:
            # No name-match; pick any 2-D array with a small "channels" axis.
            for k, v in pool.items():
                shape = tuple(v.get("shape", ()))
                if v.get("kind") == "array" and len(shape) == 2 and (3 in shape or 2 in shape):
                    multi_chan_var = k
                    break

    if multi_chan_var:
        sh = pool[multi_chan_var]["shape"]
        # Determine how many channels the matrix holds (the smaller axis).
        n_ch = min(sh) if len(sh) == 2 else 1
        if n_ch >= 3:
            indices = [0, 1, 2]
        elif n_ch == 2:
            indices = [0, 1, 1]  # repeat last so consistency check still gets >= 2 inputs
        else:
            indices = [0, 0, 0]
        vm.slowwave_ch1 = multi_chan_var
        vm.slowwave_ch2 = multi_chan_var
        vm.slowwave_ch3 = multi_chan_var if n_ch >= 3 else None
        vm.slowwave_ch_indices = indices
    else:
        # Priority 2: three distinct variables.  Look for tokens
        # ch1/ch2/ch3, prox/mid/dist, _1/_2/_3 to pick deterministically;
        # fall back to slot-by-position over the name-matched candidates.
        def _pick_sw_slot(slot_idx_1b: int) -> str:
            if not pool:
                return ""
            slot_aliases = {
                1: ("ch1", "_1", "prox", "proximal"),
                2: ("ch2", "_2", "mid", "middle"),
                3: ("ch3", "_3", "dist", "distal"),
            }
            for tok in slot_aliases[slot_idx_1b]:
                for k, v in pool.items():
                    if v.get("kind") == "array" and tok in k.lower():
                        return k
            candidates = [
                k for k, v in pool.items()
                if v.get("kind") == "array"
                and any(t in k.lower() for t in ("slow", "wave", "lfp", "trace"))
            ]
            if slot_idx_1b - 1 < len(candidates):
                return candidates[slot_idx_1b - 1]
            return ""

        vm.slowwave_ch1 = _pick_sw_slot(1) or None
        vm.slowwave_ch2 = _pick_sw_slot(2) or None
        vm.slowwave_ch3 = _pick_sw_slot(3) or None

    # fs
    fs_hits = [k for k in blanked_vars if k.lower() in ("fs", "samplerate", "sample_rate", "sr")]
    vm.fs = fs_hits[0] if fs_hits else None

    # stim
    stim_hits = [k for k in blanked_vars if "stim" in k.lower() or "event" in k.lower()]
    vm.stim_events = stim_hits[0] if stim_hits else None
    label_hits = [k for k in blanked_vars if "label" in k.lower() or "condition" in k.lower()]
    vm.stim_labels = label_hits[0] if label_hits else None

    return vm
