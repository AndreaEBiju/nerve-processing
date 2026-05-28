"""Minimal PySide6 GUI for the vagus pipeline.

Flow:
1. Pick a folder.
2. Enter blanked/R-peak/slow-wave filename patterns (with default guesses).
3. Discovery preview — table of pairs, with manual fix-up for ambiguous rows.
4. Variable mapping — dropdowns per logical role, autopopulated from one file.
5. Config review — exposes key params (corners, sigma, n_pca, bin size).
6. Run — progress bar + live log; on completion, shows the batch summary.

The UI is intentionally thin: all heavy lifting is in :mod:`batch`, which is
fully usable headless via :mod:`run`.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from .batch import run_batch
from .config import PipelineConfig, VarMap
from .io_discovery import (
    DEFAULT_BLANKED_PATTERNS,
    DEFAULT_BLANKED_TOKEN,
    DEFAULT_REQUIRED_REGEX,
    DEFAULT_RPEAK_PATTERNS,
    DEFAULT_RPEAK_TOKEN,
    DEFAULT_SLOWWAVE_PATTERNS,
    DEFAULT_SLOWWAVE_TOKEN,
    autopopulate_var_map,
    find_pairs,
    introspect_variables,
)
from .logging_setup import setup_logger

log = logging.getLogger("vagus.ui")


class QtLogHandler(logging.Handler):
    def __init__(self, signal: QtCore.SignalInstance):
        super().__init__()
        self._signal = signal

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._signal.emit(self.format(record))
        except RuntimeError:
            pass


class IntrospectWorker(QtCore.QObject):
    """Reads variable metadata from one pair's files on a background thread.

    Metadata-only reads (``whosmat`` / ``h5py``) are normally millisecond-fast,
    but on cloud-backed storage (Google Drive File Stream, OneDrive, etc.) the
    first ``open()`` can stall while the file is pulled down.  Running this
    off the UI thread keeps the GUI responsive and lets the user cancel.
    """

    progress = QtCore.Signal(str)
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, pair, cfg: PipelineConfig):
        super().__init__()
        self.pair = pair
        self.cfg = cfg
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.progress.emit(f"Reading metadata: {self.pair.blanked_path.name}")
            if self._cancel:
                self.failed.emit("cancelled")
                return
            bvars = introspect_variables(self.pair.blanked_path)

            self.progress.emit(f"Reading metadata: {self.pair.rpeak_path.name}")
            if self._cancel:
                self.failed.emit("cancelled")
                return
            rvars = introspect_variables(self.pair.rpeak_path)

            svars = None
            if self.pair.slowwave_path is not None:
                self.progress.emit(f"Reading metadata: {self.pair.slowwave_path.name}")
                if self._cancel:
                    self.failed.emit("cancelled")
                    return
                svars = introspect_variables(self.pair.slowwave_path)

            self.progress.emit("Autopopulating variable mapping")
            vm = autopopulate_var_map(bvars, rvars, svars, fs_hint=self.cfg.fs)
            self.finished.emit({"bvars": bvars, "rvars": rvars, "svars": svars, "vm": vm})
        except Exception as e:
            import traceback
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class BatchWorker(QtCore.QObject):
    progress = QtCore.Signal(str, int, int)
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, root: Path, var_map: VarMap, cfg: PipelineConfig,
                 patterns: dict[str, list[str]], signature: dict[str, str | None],
                 mode: str = "full",
                 pairs: list | None = None):
        super().__init__()
        self.root = root
        self.var_map = var_map
        self.cfg = cfg
        self.patterns = patterns
        self.signature = signature
        self.mode = mode
        self.pairs = pairs

    @QtCore.Slot()
    def run(self) -> None:
        try:
            res = run_batch(
                self.root,
                self.var_map,
                self.cfg,
                blanked_patterns=self.patterns.get("blanked"),
                rpeak_patterns=self.patterns.get("rpeak"),
                slowwave_patterns=self.patterns.get("slowwave"),
                required_regex=self.signature.get("required_regex"),
                blanked_token=self.signature.get("blanked_token"),
                rpeak_token=self.signature.get("rpeak_token"),
                slowwave_token=self.signature.get("slowwave_token"),
                mode=self.mode,
                pairs=self.pairs,
                progress_cb=lambda phase, i, n: self.progress.emit(phase, i, n),
            )
            self.finished.emit(res)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class MainWindow(QtWidgets.QMainWindow):
    log_message = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vagus Nerve Cuff Pipeline")
        self.resize(1100, 720)
        self.cfg = PipelineConfig()
        self.var_map = VarMap()
        self.pairs = []

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # --- Folder picker
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Batch root:"))
        self.root_edit = QtWidgets.QLineEdit()
        row.addWidget(self.root_edit, 1)
        btn = QtWidgets.QPushButton("Browse…")
        btn.clicked.connect(self._pick_folder)
        row.addWidget(btn)
        layout.addLayout(row)

        # --- Patterns
        grid = QtWidgets.QGridLayout()
        self.pat_blanked = QtWidgets.QLineEdit(", ".join(DEFAULT_BLANKED_PATTERNS))
        self.pat_rpeak = QtWidgets.QLineEdit(", ".join(DEFAULT_RPEAK_PATTERNS))
        self.pat_slowwave = QtWidgets.QLineEdit(", ".join(DEFAULT_SLOWWAVE_PATTERNS))
        grid.addWidget(QtWidgets.QLabel("Blanked patterns:"), 0, 0); grid.addWidget(self.pat_blanked, 0, 1)
        grid.addWidget(QtWidgets.QLabel("R-peak patterns:"), 1, 0); grid.addWidget(self.pat_rpeak, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Slow-wave patterns:"), 2, 0); grid.addWidget(self.pat_slowwave, 2, 1)

        # Filename-signature rule: only files matching this regex are considered.
        # Defaults to a version tag (e.g. _v0.1.0_) so untagged decoys are excluded.
        self.required_regex = QtWidgets.QLineEdit(DEFAULT_REQUIRED_REGEX)
        self.blanked_token = QtWidgets.QLineEdit(DEFAULT_BLANKED_TOKEN)
        self.rpeak_token = QtWidgets.QLineEdit(DEFAULT_RPEAK_TOKEN)
        self.slowwave_token = QtWidgets.QLineEdit(DEFAULT_SLOWWAVE_TOKEN)
        grid.addWidget(QtWidgets.QLabel("Required filename regex:"), 3, 0); grid.addWidget(self.required_regex, 3, 1)
        grid.addWidget(QtWidgets.QLabel("Blanked pair token:"), 4, 0); grid.addWidget(self.blanked_token, 4, 1)
        grid.addWidget(QtWidgets.QLabel("R-peak pair token:"), 5, 0); grid.addWidget(self.rpeak_token, 5, 1)
        grid.addWidget(QtWidgets.QLabel("Slow-wave pair token:"), 6, 0); grid.addWidget(self.slowwave_token, 6, 1)
        layout.addLayout(grid)

        # --- Discovery + mapping
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_discover = QtWidgets.QPushButton("Discover pairs")
        self.btn_discover.clicked.connect(self._discover)
        btn_row.addWidget(self.btn_discover)
        self.btn_introspect = QtWidgets.QPushButton("Introspect variables")
        self.btn_introspect.clicked.connect(self._introspect)
        btn_row.addWidget(self.btn_introspect)
        self.btn_reuse_vm = QtWidgets.QPushButton("Reuse previous mapping")
        self.btn_reuse_vm.clicked.connect(self._reuse_varmap)
        btn_row.addWidget(self.btn_reuse_vm)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.pair_table = QtWidgets.QTableWidget(0, 6)
        self.pair_table.setHorizontalHeaderLabels(
            ["Include", "Dir", "Blanked", "R-peak", "Slow-wave", "Status"]
        )
        self.pair_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.pair_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.pair_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.pair_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.pair_table.customContextMenuRequested.connect(self._pair_table_menu)
        layout.addWidget(self.pair_table, 1)

        # Bulk include / exclude buttons (faster than right-clicking each row).
        excl_row = QtWidgets.QHBoxLayout()
        btn_excl_sel = QtWidgets.QPushButton("Exclude selected")
        btn_excl_sel.clicked.connect(lambda: self._set_inclusion(selected_only=True, include=False))
        btn_incl_sel = QtWidgets.QPushButton("Include selected")
        btn_incl_sel.clicked.connect(lambda: self._set_inclusion(selected_only=True, include=True))
        btn_excl_all = QtWidgets.QPushButton("Exclude all")
        btn_excl_all.clicked.connect(lambda: self._set_inclusion(selected_only=False, include=False))
        btn_incl_all = QtWidgets.QPushButton("Include all")
        btn_incl_all.clicked.connect(lambda: self._set_inclusion(selected_only=False, include=True))
        for w in (btn_excl_sel, btn_incl_sel, btn_excl_all, btn_incl_all):
            excl_row.addWidget(w)
        excl_row.addStretch()
        layout.addLayout(excl_row)

        # Explicit button to re-introspect from a specific row, plus a
        # cancel button (live only while introspection is running) and a
        # toggle that lets cloud-storage users skip the automatic
        # introspect-on-discovery step entirely.
        row_layout = QtWidgets.QHBoxLayout()
        row_btn = QtWidgets.QPushButton("Introspect from selected row")
        row_btn.clicked.connect(self._introspect_from_selection)
        row_layout.addWidget(row_btn)
        self.btn_cancel_introspect = QtWidgets.QPushButton("Cancel introspect")
        self.btn_cancel_introspect.setEnabled(False)
        self.btn_cancel_introspect.clicked.connect(self._cancel_introspect)
        row_layout.addWidget(self.btn_cancel_introspect)
        self.chk_auto_introspect = QtWidgets.QCheckBox("Auto-introspect on discovery")
        self.chk_auto_introspect.setChecked(True)
        self.chk_auto_introspect.setToolTip(
            "Turn off if the batch root is on slow / cloud storage and "
            "the first variable read takes too long.  You can still trigger "
            "introspection manually with the button above."
        )
        row_layout.addWidget(self.chk_auto_introspect)
        row_layout.addStretch()
        layout.addLayout(row_layout)

        # --- Variable mapping panel
        vm_box = QtWidgets.QGroupBox("Variable mapping")
        vm_layout = QtWidgets.QGridLayout(vm_box)
        self.vm_neural = QtWidgets.QComboBox(editable=True)
        self.vm_rpeak = QtWidgets.QComboBox(editable=True)
        self.vm_units = QtWidgets.QComboBox()
        self.vm_units.addItems(["sample", "sec", "ms"])
        self.vm_slowwave = QtWidgets.QComboBox(editable=True)  # legacy single -- kept for back-compat
        self.vm_slowwave.addItem("")
        # Three antral slow-wave channel dropdowns (proximal, middle, distal)
        self.vm_slowwave_ch1 = QtWidgets.QComboBox(editable=True); self.vm_slowwave_ch1.addItem("")
        self.vm_slowwave_ch2 = QtWidgets.QComboBox(editable=True); self.vm_slowwave_ch2.addItem("")
        self.vm_slowwave_ch3 = QtWidgets.QComboBox(editable=True); self.vm_slowwave_ch3.addItem("")
        # When all three ch dropdowns point to ONE variable, pick column indices.
        self.vm_slowwave_indices = QtWidgets.QLineEdit("0,1,2")
        self.vm_slowwave_indices.setToolTip(
            "When all three slow-wave dropdowns reference the SAME multi-channel\n"
            "variable, this picks which columns become ch1/ch2/ch3."
        )
        # Spatial order along gastric long axis (1-based; default 1,2,3 = ch1 prox, ch2 mid, ch3 dist)
        self.vm_slowwave_spatial = QtWidgets.QLineEdit("1,2,3")
        self.vm_slowwave_spatial.setToolTip(
            "User-confirmed spatial order along the gastric long axis.\n"
            "Format: comma-separated 1-based channel numbers, proximal -> distal.\n"
            "Example: '1,2,3' keeps the as-assigned order; '3,2,1' reverses it;\n"
            "'2,1,3' moves channel 2 to the proximal end."
        )
        self.vm_fs = QtWidgets.QComboBox(editable=True); self.vm_fs.addItem("")
        self.vm_stim = QtWidgets.QComboBox(editable=True); self.vm_stim.addItem("")
        self.vm_stim_labels = QtWidgets.QComboBox(editable=True); self.vm_stim_labels.addItem("")
        self.vm_n_channels = QtWidgets.QSpinBox(); self.vm_n_channels.setRange(1, 64); self.vm_n_channels.setValue(1)
        self.vm_slowwave_channel = QtWidgets.QSpinBox()
        self.vm_slowwave_channel.setRange(0, 31)
        self.vm_slowwave_channel.setValue(0)
        self.vm_slowwave_channel.setToolTip(
            "0-based index of which slow-wave channel to use when the\n"
            "slowwave variable is a multi-channel matrix (e.g. a 3xN\n"
            "array stacking three slow-wave time series).  Ignored when\n"
            "the variable is already 1-D or a cell array of peak indices."
        )
        self.vm_channel_indices = QtWidgets.QLineEdit()
        self.vm_channel_indices.setPlaceholderText("e.g. 0,3 -- leave blank to use all channels")
        self.vm_channel_indices.setToolTip(
            "Comma-separated 0-based indices of the channels in the neural\n"
            "array that are actual nerve-cuff recordings.\n"
            "Example: a 5-channel acquisition where rows 0 and 3 are the\n"
            "cuff signals -> type '0,3'.\n"
            "Leave blank to use every channel."
        )
        # Each editable combo gets a generous min content length so the
        # visible text doesn't truncate ``name (shape, dtype, kind)`` to a
        # few characters in the closed state.
        for combo in (self.vm_neural, self.vm_rpeak, self.vm_slowwave,
                      self.vm_slowwave_ch1, self.vm_slowwave_ch2, self.vm_slowwave_ch3,
                      self.vm_fs, self.vm_stim, self.vm_stim_labels):
            combo.setMinimumContentsLength(40)
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        # Two-column layout (label, widget) -- gives each combo the full
        # remaining row width.
        for i, (lbl, w) in enumerate([
            ("neural", self.vm_neural),
            ("rpeak_times", self.vm_rpeak),
            ("rpeak units", self.vm_units),
            ("slow-wave ch1 (proximal)", self.vm_slowwave_ch1),
            ("slow-wave ch2 (middle)", self.vm_slowwave_ch2),
            ("slow-wave ch3 (distal)", self.vm_slowwave_ch3),
            ("sw col indices (if same var)", self.vm_slowwave_indices),
            ("spatial order (1=prox..)", self.vm_slowwave_spatial),
            ("slowwave legacy (single)", self.vm_slowwave),
            ("slowwave channel idx (legacy)", self.vm_slowwave_channel),
            ("fs (optional)", self.vm_fs),
            ("stim_events (optional)", self.vm_stim),
            ("stim_labels (optional)", self.vm_stim_labels),
            ("n_channels (info)", self.vm_n_channels),
            ("cuff channel indices", self.vm_channel_indices),
        ]):
            vm_layout.addWidget(QtWidgets.QLabel(lbl), i, 0)
            vm_layout.addWidget(w, i, 1)
        vm_layout.setColumnStretch(0, 0)
        vm_layout.setColumnStretch(1, 1)
        layout.addWidget(vm_box)

        # --- Config review
        cfg_box = QtWidgets.QGroupBox("Config (key params)")
        cfg_layout = QtWidgets.QGridLayout(cfg_box)
        self.cfg_widgets: dict[str, QtWidgets.QWidget] = {}
        key_params = ["bp_low_hz", "bp_high_hz", "threshold_sigma", "n_pca", "rate_bin_s", "seed"]
        for i, name in enumerate(key_params):
            cfg_layout.addWidget(QtWidgets.QLabel(name), i // 3, (i % 3) * 2)
            val = getattr(self.cfg, name)
            w: QtWidgets.QWidget
            if isinstance(val, int):
                w = QtWidgets.QSpinBox(); w.setRange(0, 10_000); w.setValue(val)
            else:
                w = QtWidgets.QDoubleSpinBox(); w.setRange(0.0, 100_000.0); w.setDecimals(4); w.setValue(float(val))
            cfg_layout.addWidget(w, i // 3, (i % 3) * 2 + 1)
            self.cfg_widgets[name] = w
        layout.addWidget(cfg_box)

        # --- Run mode + log
        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(QtWidgets.QLabel("Mode:"))
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(["full", "prepass", "resume"])
        self.cmb_mode.setToolTip(
            "full    -- run all 14 steps and save .mat per pair (default).\n"
            "prepass -- run Steps 1-5 + write <stem>_checkpoint.npz per pair.\n"
            "           Use on machines without MountainSort5 (Windows).\n"
            "resume  -- skip discovery, scan for existing checkpoints,\n"
            "           run Steps 6-14 from each. Use on a Mac/Linux box\n"
            "           after copying checkpoints across."
        )
        run_row.addWidget(self.cmb_mode)
        self.btn_run = QtWidgets.QPushButton("Run batch")
        self.btn_run.clicked.connect(self._run_batch)
        run_row.addWidget(self.btn_run)
        self.btn_headless = QtWidgets.QPushButton("Show headless command")
        self.btn_headless.clicked.connect(self._show_headless)
        run_row.addWidget(self.btn_headless)
        self.progress = QtWidgets.QProgressBar()
        run_row.addWidget(self.progress, 1)
        layout.addLayout(run_row)

        self.log_view = QtWidgets.QPlainTextEdit(readOnly=True)
        layout.addWidget(self.log_view, 1)

        self.log_message.connect(self._append_log)
        setup_logger()
        root_logger = logging.getLogger("vagus")
        handler = QtLogHandler(self.log_message)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root_logger.addHandler(handler)

    # ------------------------------------------------------------------
    @QtCore.Slot(str)
    def _append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    def _pick_folder(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select batch root")
        if d:
            self.root_edit.setText(d)

    def _patterns(self) -> dict[str, list[str]]:
        def parse(s: str) -> list[str]:
            return [p.strip() for p in s.split(",") if p.strip()]
        return {
            "blanked": parse(self.pat_blanked.text()),
            "rpeak": parse(self.pat_rpeak.text()),
            "slowwave": parse(self.pat_slowwave.text()),
        }

    def _signature(self) -> dict[str, str | None]:
        rr = self.required_regex.text().strip()
        bt = self.blanked_token.text().strip()
        rt = self.rpeak_token.text().strip()
        st = self.slowwave_token.text().strip()
        return {
            "required_regex": rr or None,
            "blanked_token": bt or None,
            "rpeak_token": rt or None,
            "slowwave_token": st or None,
        }

    def _discover(self) -> None:
        root = self.root_edit.text().strip()
        if not root:
            QtWidgets.QMessageBox.warning(self, "Pick a folder", "Choose a batch root first.")
            return
        try:
            p = self._patterns()
            sig = self._signature()
            self.pairs = find_pairs(
                root,
                p["blanked"],
                p["rpeak"],
                p["slowwave"] or None,
                required_regex=sig["required_regex"],
                blanked_token=sig["blanked_token"],
                rpeak_token=sig["rpeak_token"],
                slowwave_token=sig["slowwave_token"],
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Discovery error", str(e))
            return
        self.pair_table.setRowCount(len(self.pairs))
        for i, pair in enumerate(self.pairs):
            # Column 0: include checkbox.
            include_item = QtWidgets.QTableWidgetItem()
            include_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            include_item.setCheckState(QtCore.Qt.Checked)
            include_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.pair_table.setItem(i, 0, include_item)
            # Columns 1-5: file info.
            self.pair_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(pair.dir.relative_to(root) if str(pair.dir).startswith(root) else pair.dir)))
            self.pair_table.setItem(i, 2, QtWidgets.QTableWidgetItem(pair.blanked_path.name))
            self.pair_table.setItem(i, 3, QtWidgets.QTableWidgetItem(pair.rpeak_path.name))
            self.pair_table.setItem(i, 4, QtWidgets.QTableWidgetItem(pair.slowwave_path.name if pair.slowwave_path else ""))
            status_item = QtWidgets.QTableWidgetItem(pair.status + (f": {pair.note}" if pair.note else ""))
            if pair.status != "ok":
                status_item.setBackground(QtGui.QColor(255, 240, 200))
            self.pair_table.setItem(i, 5, status_item)
        log.info("Discovered %d pair(s).  Uncheck the Include column on any row to skip it.", len(self.pairs))
        # Auto-introspect the first pair so the variable dropdowns populate
        # without a second click.  Runs on a worker thread (cloud-storage
        # files can stall on open) and is opt-out via the checkbox.
        if self.pairs and self.chk_auto_introspect.isChecked():
            self._introspect(pair_index=0)

    def _pair_table_menu(self, pos) -> None:
        """Right-click context menu on the discovery table."""
        if not self.pairs:
            return
        menu = QtWidgets.QMenu(self)
        act_excl = menu.addAction("Exclude selected rows")
        act_incl = menu.addAction("Include selected rows")
        menu.addSeparator()
        act_excl_all = menu.addAction("Exclude all")
        act_incl_all = menu.addAction("Include all")
        menu.addSeparator()
        act_introspect = menu.addAction("Introspect from this row")
        chosen = menu.exec(self.pair_table.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == act_excl:
            self._set_inclusion(selected_only=True, include=False)
        elif chosen == act_incl:
            self._set_inclusion(selected_only=True, include=True)
        elif chosen == act_excl_all:
            self._set_inclusion(selected_only=False, include=False)
        elif chosen == act_incl_all:
            self._set_inclusion(selected_only=False, include=True)
        elif chosen == act_introspect:
            self._introspect_from_selection()

    def _set_inclusion(self, selected_only: bool, include: bool) -> None:
        if not self.pairs:
            return
        rows = (
            {idx.row() for idx in self.pair_table.selectionModel().selectedRows()}
            if selected_only
            else set(range(self.pair_table.rowCount()))
        )
        if not rows:
            return
        state = QtCore.Qt.Checked if include else QtCore.Qt.Unchecked
        for r in rows:
            item = self.pair_table.item(r, 0)
            if item is not None:
                item.setCheckState(state)
        verb = "Included" if include else "Excluded"
        log.info("%s %d row(s)%s.", verb, len(rows), " (selected)" if selected_only else "")

    def _included_pairs(self) -> list:
        """Return only the discovered pairs whose Include checkbox is checked."""
        if not self.pairs:
            return []
        kept = []
        for i, pair in enumerate(self.pairs):
            item = self.pair_table.item(i, 0)
            if item is None or item.checkState() == QtCore.Qt.Checked:
                kept.append(pair)
        return kept

    def _introspect_from_selection(self) -> None:
        """Re-introspect from the row the user has currently selected."""
        if not self.pairs:
            QtWidgets.QMessageBox.warning(self, "Discover first", "Run discovery first.")
            return
        rows = self.pair_table.selectionModel().selectedRows()
        idx = rows[0].row() if rows else 0
        if 0 <= idx < len(self.pairs):
            self._introspect(pair_index=idx)

    def _introspect(self, pair_index: int = 0) -> None:
        if not self.pairs:
            QtWidgets.QMessageBox.warning(self, "Discover first", "Run discovery first.")
            return
        if getattr(self, "_introspect_thread", None) and self._introspect_thread.isRunning():
            log.warning("Introspection already in progress; ignoring new request.")
            return
        if pair_index >= len(self.pairs):
            pair_index = 0
        rep = self.pairs[pair_index]
        log.info("Introspecting variables from pair #%d:", pair_index + 1)
        log.info("  blanked  : %s", rep.blanked_path.name)
        log.info("  rpeak    : %s", rep.rpeak_path.name)
        if rep.slowwave_path:
            log.info("  slowwave : %s", rep.slowwave_path.name)

        self.btn_cancel_introspect.setEnabled(True)
        self._introspect_worker = IntrospectWorker(rep, self.cfg)
        self._introspect_thread = QtCore.QThread(self)
        self._introspect_worker.moveToThread(self._introspect_thread)
        self._introspect_thread.started.connect(self._introspect_worker.run)
        self._introspect_worker.progress.connect(lambda msg: log.info(msg))
        self._introspect_worker.finished.connect(self._on_introspect_finished)
        self._introspect_worker.failed.connect(self._on_introspect_failed)
        self._introspect_worker.finished.connect(self._introspect_thread.quit)
        self._introspect_worker.failed.connect(self._introspect_thread.quit)
        self._introspect_thread.start()

    def _cancel_introspect(self) -> None:
        worker = getattr(self, "_introspect_worker", None)
        if worker is not None:
            worker.cancel()
            log.info("Introspect cancel requested.")

    @QtCore.Slot(dict)
    def _on_introspect_finished(self, result: dict) -> None:
        self.btn_cancel_introspect.setEnabled(False)
        bvars = result["bvars"]
        rvars = result["rvars"]
        svars = result.get("svars")
        vm = result["vm"]
        log.info(
            "  found %d blanked / %d rpeak / %d slow-wave variable(s).",
            len(bvars), len(rvars), len(svars) if svars else 0,
        )
        self._apply_introspection(bvars, rvars, svars, vm)

    @QtCore.Slot(str)
    def _on_introspect_failed(self, msg: str) -> None:
        self.btn_cancel_introspect.setEnabled(False)
        if msg == "cancelled":
            log.info("Introspect cancelled.")
            return
        log.error("Introspect failed: %s", msg)
        QtWidgets.QMessageBox.critical(self, "Introspection error", msg[:2000])

    def _apply_introspection(self, bvars, rvars, svars, vm) -> None:

        # Populate dropdowns.  Each item shows ``name  (shape, dtype, kind)``
        # in the visible text so the user can tell at a glance which variable
        # is a 1-D array of sample indices vs. a struct full of HRV metrics;
        # only the bare name is stored as the item's data so when the user
        # picks a row, the existing _collect_var_map() reads back the same
        # string the rest of the pipeline expects.
        def _label(name: str, info: dict | None) -> str:
            if info is None:
                return name
            shape = info.get("shape", ())
            dtype = info.get("dtype", "")
            kind = info.get("kind", "")
            return f"{name}   ({shape}, {dtype}, {kind})"

        def fill(combo: QtWidgets.QComboBox, names: list[str], info_map: dict, current: str | None) -> None:
            combo.clear()
            combo.addItem("", userData="")
            # Per-item tooltip = the full labelled string, so hovering any
            # entry in the popup shows the whole name + shape even when the
            # visible text is truncated.
            for name in names:
                label = _label(name, info_map.get(name))
                combo.addItem(label, userData=name)
                idx = combo.count() - 1
                combo.setItemData(idx, label, QtCore.Qt.ToolTipRole)
            # Resize the popup view so the longest item fits.  The combobox
            # itself stays narrow (it lives inside a 4-column grid), but
            # ``QComboBox.view()`` is the dropdown widget and Qt is happy
            # for it to be wider than its parent.
            view = combo.view()
            fm = combo.fontMetrics()
            longest = max(
                (fm.horizontalAdvance(_label(n, info_map.get(n))) for n in names),
                default=200,
            )
            # Cap at 900 px so it doesn't fly off-screen on small displays.
            view.setMinimumWidth(min(longest + 40, 900))
            # Show as many items as possible without truncating vertically.
            view.setMaximumHeight(min(36 * len(names) + 8, 480))
            # Tooltip on the combo itself reflects the current selection so
            # the user can verify their pick without opening the dropdown.
            combo.currentIndexChanged.connect(
                lambda _idx, c=combo: c.setToolTip(c.currentText())
            )
            if current:
                for i in range(combo.count()):
                    if combo.itemData(i) == current:
                        combo.setCurrentIndex(i)
                        break
                else:
                    # Editable combos preserve free-text entries even if not in
                    # the introspected list.
                    combo.setEditText(current)
            combo.setToolTip(combo.currentText())

        combined_slow = {**bvars, **(svars or {})}
        fill(self.vm_neural, list(bvars.keys()), bvars, vm.neural)
        fill(self.vm_rpeak, list(rvars.keys()), rvars, vm.rpeak_times)
        sw_names = list(bvars.keys()) + (list(svars.keys()) if svars else [])
        fill(self.vm_slowwave, sw_names, combined_slow, vm.slowwave)
        fill(self.vm_slowwave_ch1, sw_names, combined_slow, vm.slowwave_ch1)
        fill(self.vm_slowwave_ch2, sw_names, combined_slow, vm.slowwave_ch2)
        fill(self.vm_slowwave_ch3, sw_names, combined_slow, vm.slowwave_ch3)
        self.vm_slowwave_indices.setText(",".join(str(i) for i in (vm.slowwave_ch_indices or [0, 1, 2])))
        self.vm_slowwave_spatial.setText(",".join(str(i) for i in (vm.slowwave_spatial_order or [1, 2, 3])))
        fill(self.vm_fs, list(bvars.keys()), bvars, vm.fs)
        fill(self.vm_stim, list(bvars.keys()), bvars, vm.stim_events)
        fill(self.vm_stim_labels, list(bvars.keys()), bvars, vm.stim_labels)
        self.vm_units.setCurrentText(vm.rpeak_units)
        self.vm_n_channels.setValue(max(vm.n_channels, 1))
        ci = vm.channel_indices or []
        self.vm_channel_indices.setText(",".join(str(i) for i in ci))
        log.info("Autopopulated variable mapping (%d / %d / %d vars).",
                 len(bvars), len(rvars), len(svars) if svars else 0)
        # Show a hint when the neural array has more channels than the user
        # is likely to want as cuffs.
        if vm.neural and vm.neural in bvars:
            shape = bvars[vm.neural].get("shape", ())
            if len(shape) == 2:
                n_ch = min(shape)
                if n_ch >= 3 and not ci:
                    log.warning(
                        "Neural variable '%s' has shape %s (~%d channels) -- "
                        "if only some of these are nerve cuffs, fill in "
                        "'cuff channel indices' (e.g. 0,3).",
                        vm.neural, shape, n_ch,
                    )

    def _reuse_varmap(self) -> None:
        root = self.root_edit.text().strip()
        if not root:
            return
        p = Path(root) / "batch_varmap.json"
        if not p.exists():
            QtWidgets.QMessageBox.information(self, "No previous mapping", f"{p} not found.")
            return
        d = json.loads(p.read_text())
        self.vm_neural.setEditText(d.get("neural", ""))
        self.vm_rpeak.setEditText(d.get("rpeak_times", ""))
        self.vm_units.setCurrentText(d.get("rpeak_units", "sample"))
        self.vm_slowwave.setEditText(d.get("slowwave") or "")
        self.vm_slowwave_ch1.setEditText(d.get("slowwave_ch1") or "")
        self.vm_slowwave_ch2.setEditText(d.get("slowwave_ch2") or "")
        self.vm_slowwave_ch3.setEditText(d.get("slowwave_ch3") or "")
        self.vm_slowwave_indices.setText(",".join(str(i) for i in (d.get("slowwave_ch_indices") or [0, 1, 2])))
        self.vm_slowwave_spatial.setText(",".join(str(i) for i in (d.get("slowwave_spatial_order") or [1, 2, 3])))
        self.vm_fs.setEditText(d.get("fs") or "")
        self.vm_stim.setEditText(d.get("stim_events") or "")
        self.vm_stim_labels.setEditText(d.get("stim_labels") or "")
        self.vm_n_channels.setValue(int(d.get("n_channels", 1)))
        ci = d.get("channel_indices") or []
        self.vm_channel_indices.setText(",".join(str(i) for i in ci))
        log.info("Loaded previous var map from %s.", p)

    def _selected_var(self, combo: QtWidgets.QComboBox) -> str:
        """Return the bare variable name behind a labelled dropdown entry.

        Items added via :meth:`_apply_introspection` carry the variable name
        as ``userData`` while their visible text includes shape/dtype hints.
        When the user types a custom value (editable combo), ``userData``
        is empty so we fall back to the trimmed visible text.
        """
        data = combo.currentData()
        if isinstance(data, str) and data:
            return data.strip()
        return combo.currentText().strip().split()[0] if combo.currentText().strip() else ""

    def _collect_var_map(self) -> VarMap:
        idx_text = self.vm_channel_indices.text().strip()
        channel_indices: list[int] | None = None
        if idx_text:
            try:
                channel_indices = [int(s.strip()) for s in idx_text.split(",") if s.strip() != ""]
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    self, "Bad channel indices",
                    f"Couldn't parse '{idx_text}' as a comma-separated list of integers. "
                    "Example: 0,3 -- leave blank to use every channel.",
                )
        def _ints(txt: str, fallback: list[int]) -> list[int]:
            try:
                return [int(s.strip()) for s in txt.split(",") if s.strip()]
            except ValueError:
                return fallback
        return VarMap(
            neural=self._selected_var(self.vm_neural),
            rpeak_times=self._selected_var(self.vm_rpeak),
            rpeak_units=self.vm_units.currentText().strip() or "sample",
            slowwave_ch1=self._selected_var(self.vm_slowwave_ch1) or None,
            slowwave_ch2=self._selected_var(self.vm_slowwave_ch2) or None,
            slowwave_ch3=self._selected_var(self.vm_slowwave_ch3) or None,
            slowwave_ch_indices=_ints(self.vm_slowwave_indices.text(), [0, 1, 2]),
            slowwave_spatial_order=_ints(self.vm_slowwave_spatial.text(), [1, 2, 3]),
            slowwave=self._selected_var(self.vm_slowwave) or None,  # legacy
            fs=self._selected_var(self.vm_fs) or None,
            stim_events=self._selected_var(self.vm_stim) or None,
            stim_labels=self._selected_var(self.vm_stim_labels) or None,
            n_channels=int(self.vm_n_channels.value()),
            channel_indices=channel_indices,
            slowwave_channel=int(self.vm_slowwave_channel.value()),
        )

    def _collect_config(self) -> PipelineConfig:
        cfg = PipelineConfig()
        for name, w in self.cfg_widgets.items():
            if isinstance(w, QtWidgets.QSpinBox):
                setattr(cfg, name, int(w.value()))
            else:
                setattr(cfg, name, float(w.value()))
        return cfg

    def _show_headless(self) -> None:
        root = self.root_edit.text().strip() or "<root>"
        vm = self._collect_var_map()
        cfg = self._collect_config()
        argv = [
            "python", "run.py", "--no-ui",
            "--root", root,
            "--neural", vm.neural,
            "--rpeak", vm.rpeak_times,
            "--units", vm.rpeak_units,
            "--n-channels", str(vm.n_channels),
        ]
        if vm.slowwave: argv += ["--slowwave", vm.slowwave]
        if vm.fs: argv += ["--fs-var", vm.fs]
        sig = self._signature()
        if sig["required_regex"]:
            argv += ["--required-regex", sig["required_regex"]]
        if sig["blanked_token"]:
            argv += ["--blanked-token", sig["blanked_token"]]
        if sig["rpeak_token"]:
            argv += ["--rpeak-token", sig["rpeak_token"]]
        if sig["slowwave_token"]:
            argv += ["--slowwave-token", sig["slowwave_token"]]
        for name in ("bp_low_hz", "bp_high_hz", "threshold_sigma", "n_pca", "rate_bin_s", "seed"):
            argv += [f"--{name}", str(getattr(cfg, name))]
        QtWidgets.QMessageBox.information(self, "Headless command", " ".join(argv))

    def _run_batch(self) -> None:
        root = self.root_edit.text().strip()
        if not root or not Path(root).exists():
            QtWidgets.QMessageBox.warning(self, "Bad root", "Pick an existing batch root.")
            return
        vm = self._collect_var_map()
        if not vm.neural or not vm.rpeak_times:
            QtWidgets.QMessageBox.warning(self, "Var map incomplete", "neural and rpeak_times are required.")
            return
        cfg = self._collect_config()
        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setMaximum(0)  # busy

        mode = self.cmb_mode.currentText()
        # In full/prepass mode, honour the per-row Include checkboxes by
        # passing only the kept pairs to the worker.  Resume mode does its
        # own checkpoint discovery and is unaffected.
        kept_pairs = None
        if mode != "resume":
            kept_pairs = self._included_pairs()
            if self.pairs and not kept_pairs:
                QtWidgets.QMessageBox.warning(
                    self, "Nothing to run",
                    "Every pair is excluded.  Re-include at least one row and try again.",
                )
                self.btn_run.setEnabled(True)
                self.progress.setMaximum(1); self.progress.setValue(0)
                return
            if kept_pairs and self.pairs and len(kept_pairs) < len(self.pairs):
                log.info(
                    "Running on %d of %d discovered pair(s) (the rest are excluded).",
                    len(kept_pairs), len(self.pairs),
                )
        self.worker = BatchWorker(
            Path(root), vm, cfg, self._patterns(), self._signature(),
            mode=mode,
            pairs=kept_pairs,
        )
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()

    @QtCore.Slot(str, int, int)
    def _on_progress(self, phase: str, i: int, n: int) -> None:
        self.progress.setMaximum(n)
        self.progress.setValue(i)
        log.info("[%s] %d/%d", phase, i, n)

    @QtCore.Slot(dict)
    def _on_finished(self, res: dict) -> None:
        self.btn_run.setEnabled(True)
        log.info("Batch finished. Summary: %s", res.get("summary_path"))
        QtWidgets.QMessageBox.information(self, "Batch finished",
                                          f"Summary written to:\n{res.get('summary_path')}")

    @QtCore.Slot(str)
    def _on_failed(self, msg: str) -> None:
        self.btn_run.setEnabled(True)
        log.error(msg)
        # Use a resizable dialog with a scrollable text view so per-pair
        # error summaries don't get clipped to ~one line per pair.
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Batch failed")
        dlg.resize(820, 480)
        v = QtWidgets.QVBoxLayout(dlg)
        v.addWidget(QtWidgets.QLabel("The batch did not complete.  Details below:"))
        text = QtWidgets.QPlainTextEdit(readOnly=True)
        text.setPlainText(msg)
        text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        font = QtGui.QFont("Menlo")
        font.setStyleHint(QtGui.QFont.Monospace)
        text.setFont(font)
        v.addWidget(text, 1)
        btn_row = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("Copy to clipboard")
        btn_copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(msg))
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_row.addStretch()
        btn_row.addWidget(btn_copy)
        btn_row.addWidget(btn_close)
        v.addLayout(btn_row)
        dlg.exec()


def launch() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()
