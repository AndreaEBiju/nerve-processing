"""Vagus pipeline entry point.

Usage:
    python run.py                     # launches the GUI
    python run.py --no-ui --root DIR --neural data --rpeak rpeak_samples
    python run.py --smoke             # regenerates sample data + runs e2e test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vagus_pipeline.config import PipelineConfig, VarMap
from vagus_pipeline.logging_setup import setup_logger


def _add_config_args(p: argparse.ArgumentParser) -> None:
    cfg = PipelineConfig()
    for name in ("bp_low_hz", "bp_high_hz", "threshold_sigma", "n_pca", "rate_bin_s", "seed"):
        val = getattr(cfg, name)
        kind = type(val)
        p.add_argument(f"--{name.replace('_', '-')}", dest=name, type=kind, default=val)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vagus nerve cuff recording pipeline")
    parser.add_argument("--no-ui", action="store_true", help="Run headless instead of launching the GUI")
    parser.add_argument("--smoke", action="store_true", help="Regenerate sample data, run the end-to-end test, and exit")
    parser.add_argument(
        "--mode",
        choices=("full", "prepass", "resume"),
        default="full",
        help=(
            "full: run all 14 steps and save .mat per pair (default). "
            "prepass: run Steps 1-5 + save <stem>_checkpoint.npz per pair; "
            "no sorting, no .mat output -- use on machines without "
            "MountainSort5. "
            "resume: skip discovery, scan for existing checkpoints, run "
            "Steps 6-14 from each one, save .mat next to the checkpoint."
        ),
    )
    parser.add_argument("--prepass", action="store_true", help="Shortcut for --mode prepass.")
    parser.add_argument("--resume", action="store_true", help="Shortcut for --mode resume.")
    parser.add_argument("--root", type=Path, help="Batch root directory (headless)")
    parser.add_argument("--neural", help="Neural variable name")
    parser.add_argument("--rpeak", help="R-peak variable name")
    parser.add_argument("--units", default="sample", choices=["sample", "sec", "ms"])
    parser.add_argument("--n-channels", dest="n_channels", type=int, default=1)
    parser.add_argument("--slowwave", default=None)
    parser.add_argument("--fs-var", dest="fs_var", default=None)
    parser.add_argument("--stim-events", dest="stim_events", default=None)
    parser.add_argument("--stim-labels", dest="stim_labels", default=None)
    parser.add_argument("--blanked-pattern", action="append", help="Override blanked patterns (repeatable)")
    parser.add_argument("--rpeak-pattern", action="append", help="Override R-peak patterns (repeatable)")
    parser.add_argument("--slowwave-pattern", action="append", help="Override slow-wave patterns (repeatable)")
    parser.add_argument(
        "--required-regex",
        dest="required_regex",
        default=None,
        help="Regex that filenames must match to be considered (default: _v\\d+\\.\\d+\\.\\d+_).  Pass '' to disable.",
    )
    parser.add_argument(
        "--blanked-token",
        dest="blanked_token",
        default=None,
        help="Token stripped from blanked filenames to build the pair key (default: blankmotion).",
    )
    parser.add_argument(
        "--rpeak-token",
        dest="rpeak_token",
        default=None,
        help="Token stripped from R-peak filenames to build the pair key (default: HRBR).",
    )
    parser.add_argument(
        "--slowwave-token",
        dest="slowwave_token",
        default=None,
        help="Token stripped from slow-wave filenames to build the pair key (default: slowWaves).",
    )
    _add_config_args(parser)
    args = parser.parse_args(argv)

    setup_logger()

    if args.smoke:
        import subprocess
        subprocess.run([sys.executable, "tests/make_sample_data.py"], check=True)
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_end_to_end.py"]).returncode
        return rc

    if not args.no_ui:
        from vagus_pipeline.ui import launch
        return launch()

    is_resume = args.resume or args.mode == "resume"
    if not args.root:
        parser.error("--no-ui requires --root")
    if not is_resume and (not args.neural or not args.rpeak):
        parser.error("--no-ui (full/prepass) requires --neural and --rpeak")
    if is_resume:
        # Resume reads everything from the checkpoint; supply harmless
        # placeholders so VarMap() construction below doesn't crash.
        args.neural = args.neural or "<from-checkpoint>"
        args.rpeak = args.rpeak or "<from-checkpoint>"

    cfg = PipelineConfig()
    for name in ("bp_low_hz", "bp_high_hz", "threshold_sigma", "n_pca", "rate_bin_s", "seed"):
        setattr(cfg, name, getattr(args, name))
    var_map = VarMap(
        neural=args.neural,
        rpeak_times=args.rpeak,
        rpeak_units=args.units,
        slowwave=args.slowwave,
        fs=args.fs_var,
        stim_events=args.stim_events,
        stim_labels=args.stim_labels,
        n_channels=args.n_channels,
    )
    from vagus_pipeline.batch import run_batch
    from vagus_pipeline.io_discovery import (
        DEFAULT_BLANKED_TOKEN,
        DEFAULT_REQUIRED_REGEX,
        DEFAULT_RPEAK_TOKEN,
        DEFAULT_SLOWWAVE_TOKEN,
    )
    required_regex = args.required_regex if args.required_regex is not None else DEFAULT_REQUIRED_REGEX
    if required_regex == "":
        required_regex = None
    blanked_token = args.blanked_token if args.blanked_token is not None else DEFAULT_BLANKED_TOKEN
    if blanked_token == "":
        blanked_token = None
    rpeak_token = args.rpeak_token if args.rpeak_token is not None else DEFAULT_RPEAK_TOKEN
    if rpeak_token == "":
        rpeak_token = None
    slowwave_token = args.slowwave_token if args.slowwave_token is not None else DEFAULT_SLOWWAVE_TOKEN
    if slowwave_token == "":
        slowwave_token = None
    mode = args.mode
    if args.prepass:
        mode = "prepass"
    if args.resume:
        mode = "resume"
    res = run_batch(
        args.root, var_map, cfg,
        blanked_patterns=args.blanked_pattern,
        rpeak_patterns=args.rpeak_pattern,
        slowwave_patterns=args.slowwave_pattern,
        required_regex=required_regex,
        blanked_token=blanked_token,
        rpeak_token=rpeak_token,
        slowwave_token=slowwave_token,
        mode=mode,
    )
    print(f"Done ({mode}). Summary:", res["summary_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
