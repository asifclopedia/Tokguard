from typing import Any

import pyqtgraph as pg
from PyQt6.QtCore import QPoint, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QClipboard, QColor, QMouseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from analyzer import PromptAnalyzer
from telemetry import TelemetryStore


class RewriteWorker(QThread):
    finished_rewrite = pyqtSignal(str, str)

    def __init__(
        self,
        analyzer: PromptAnalyzer,
        cleaned_prompt: str,
        problems: list[str],
        recommendation: str,
    ) -> None:
        super().__init__()
        self.analyzer = analyzer
        self.cleaned_prompt = cleaned_prompt
        self.problems = problems
        self.recommendation = recommendation

    def run(self) -> None:
        rewritten, note = self.analyzer.rewrite_with_llm(
            self.cleaned_prompt, self.problems, self.recommendation
        )
        self.finished_rewrite.emit(rewritten, note)


class DiagnosticRewriteDialog(QDialog):
    apply_prompt_requested = pyqtSignal(str)

    def __init__(
        self,
        parent: QWidget,
        status: str,
        report: dict[str, Any],
        analyzer: PromptAnalyzer,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Tokguard Diagnostic ({status})")
        self.setModal(True)
        self.resize(460, 430)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #12151c;
                color: #e2e8f0;
                border: 1px solid #334155;
            }
            QLabel {
                color: #e2e8f0;
                font-size: 12px;
            }
            QTextEdit {
                background-color: #0b0d12;
                color: #f8fafc;
                border: 1px solid #283244;
                border-radius: 6px;
                font-family: Consolas, monospace;
                font-size: 11px;
                padding: 6px;
            }
            QPushButton {
                background-color: #1e293b;
                color: #f1f5f9;
                border: 1px solid #475569;
                border-radius: 5px;
                padding: 6px 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #334155;
            }
            """
        )

        layout = QVBoxLayout(self)

        problems_list = report.get("problems", [])
        prob_text = "\n".join([f"- {p}" for p in problems_list]) if problems_list else "- None"
        rec_text = str(report.get("recommendation", ""))
        cleaned_prompt = str(report.get("cleaned_prompt", ""))

        lbl_prob_hdr = QLabel("<b>Problems Detected:</b>")
        lbl_prob_val = QLabel(prob_text)
        lbl_prob_val.setWordWrap(True)

        lbl_rec_hdr = QLabel("<b>Recommendation & Local/DIY Guidance:</b>")
        lbl_rec_val = QLabel(rec_text)
        lbl_rec_val.setWordWrap(True)

        self.lbl_rewrite_status = QLabel("<b>Rewritten Prompt (generating via separate LLM...):</b>")
        self.txt_rewritten = QTextEdit()
        self.txt_rewritten.setPlainText(cleaned_prompt)

        btn_row = QHBoxLayout()
        self.btn_copy = QPushButton("Copy Rewritten")
        self.btn_apply = QPushButton("Use in Inspector")
        self.btn_close = QPushButton("Close")
        btn_row.addWidget(self.btn_copy)
        btn_row.addWidget(self.btn_apply)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_close)

        layout.addWidget(lbl_prob_hdr)
        layout.addWidget(lbl_prob_val)
        layout.addWidget(lbl_rec_hdr)
        layout.addWidget(lbl_rec_val)
        layout.addWidget(self.lbl_rewrite_status)
        layout.addWidget(self.txt_rewritten)
        layout.addLayout(btn_row)

        self.btn_copy.clicked.connect(self._copy_rewritten)
        self.btn_apply.clicked.connect(self._apply_rewritten)
        self.btn_close.clicked.connect(self.accept)

        self.worker = RewriteWorker(analyzer, cleaned_prompt, problems_list, rec_text)
        self.worker.finished_rewrite.connect(self._on_rewrite_done)
        self.worker.start()

    def _on_rewrite_done(self, rewritten_text: str, note: str) -> None:
        self.lbl_rewrite_status.setText(f"<b>Rewritten Prompt</b> <i>({note})</i>:")
        self.txt_rewritten.setPlainText(rewritten_text)

    def _copy_rewritten(self) -> None:
        cb = QApplication.clipboard()
        if cb:
            cb.setText(self.txt_rewritten.toPlainText())

    def _apply_rewritten(self) -> None:
        self.apply_prompt_requested.emit(self.txt_rewritten.toPlainText())
        self.accept()


class TokguardWidget(QWidget):
    def __init__(self, store: TelemetryStore, analyzer: PromptAnalyzer) -> None:
        super().__init__()
        self.store = store
        self.analyzer = analyzer
        self.settings = self.store.get_settings()

        self._drag_offset: QPoint | None = None
        self._minimized_pill = False
        self._suppress_clipboard = False

        # Two-blink state machine (2 full cycles = 4 phase transitions * 320ms)
        self._blink_phases_remaining = 0
        self._blink_highlight_on = False
        self._active_status = "NEUTRAL"
        self._last_report: dict[str, Any] = {}

        self._sync_analyzer_from_settings()
        self._build_ui()
        self._load_settings_into_form()
        self._refresh_telemetry_views()

        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._on_blink_tick)

        self._clipboard = QApplication.clipboard()
        if self._clipboard:
            self._clipboard.dataChanged.connect(self._on_clipboard_changed)

    def _sync_analyzer_from_settings(self) -> None:
        budget = int(self.settings.get("session_budget", "12000") or "12000")
        self.analyzer.update_context(
            project_name=self.settings.get("project_name", "Tokguard"),
            project_description=self.settings.get("project_description", ""),
            model_description=self.settings.get("model_description", ""),
            session_budget=budget,
            encoding_name=self.settings.get("encoding", "cl100k_base"),
            rewriter_endpoint=self.settings.get("rewriter_endpoint", ""),
            rewriter_model=self.settings.get("rewriter_model", ""),
            rewriter_api_key=self.settings.get("rewriter_api_key", ""),
        )

    def _get_remaining_budget(self) -> int:
        budget = int(self.settings.get("session_budget", "12000") or "12000")
        stats = self.store.get_summary_stats()
        spent = int(stats.get("total_spend", 0))
        return max(0, budget - spent)

    def _build_ui(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedSize(370, 300)
        self._apply_surface_style("#12151c", "#263042")

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(6)

        # Header Bar (Draggable + Minimize + Close)
        header = QHBoxLayout()
        header.setSpacing(4)
        self.lbl_brand = QLabel("TOKGUARD")
        self.lbl_brand.setStyleSheet(
            "color: #f8fafc; font-weight: bold; font-family: Consolas, monospace; font-size: 12px;"
        )
        self.btn_prev = QPushButton("<")
        self.btn_prev.setFixedSize(22, 22)
        self.lbl_screen = QLabel("Inspector")
        self.lbl_screen.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.btn_next = QPushButton(">")
        self.btn_next.setFixedSize(22, 22)

        self.btn_min = QPushButton("_")
        self.btn_min.setFixedSize(22, 22)
        self.btn_close = QPushButton("X")
        self.btn_close.setFixedSize(22, 22)

        header.addWidget(self.lbl_brand)
        header.addStretch()
        header.addWidget(self.btn_prev)
        header.addWidget(self.lbl_screen)
        header.addWidget(self.btn_next)
        header.addWidget(self.btn_min)
        header.addWidget(self.btn_close)
        root.addLayout(header)

        # Stacked 4 Views
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_inspector_page())
        self.stack.addWidget(self._build_graphs_page())
        self.stack.addWidget(self._build_efficiency_page())
        self.stack.addWidget(self._build_settings_page())
        root.addWidget(self.stack)

        # Bottom Status Bar with Conditional "!" Button
        status_bar = QHBoxLayout()
        status_bar.setSpacing(6)
        self.lbl_status = QLabel("READY | 0 tok")
        self.lbl_status.setStyleSheet(
            "color: #cbd5e1; font-family: Consolas, monospace; font-size: 11px;"
        )
        self.btn_bang = QPushButton("!")
        self.btn_bang.setFixedSize(26, 22)
        self.btn_bang.setVisible(False)
        self.btn_bang.setStyleSheet(
            "background-color: #dc2626; color: #ffffff; font-weight: bold; "
            "border-radius: 4px; border: 1px solid #fca5a5;"
        )

        status_bar.addWidget(self.lbl_status)
        status_bar.addStretch()
        status_bar.addWidget(self.btn_bang)
        root.addLayout(status_bar)

        # Wire navigation & window controls
        self.btn_prev.clicked.connect(lambda: self._switch_page(-1))
        self.btn_next.clicked.connect(lambda: self._switch_page(1))
        self.btn_min.clicked.connect(self._toggle_minimize)
        self.btn_close.clicked.connect(self.close)
        self.btn_bang.clicked.connect(self._open_diagnostic_dialog)

    def _build_inspector_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.txt_prompt = QTextEdit()
        self.txt_prompt.setPlaceholderText("Paste or type prompt to inspect before sending...")
        self.txt_prompt.setStyleSheet(
            "background-color: #0b0d12; color: #f1f5f9; border: 1px solid #283244; "
            "border-radius: 5px; font-family: Consolas, monospace; font-size: 11px;"
        )

        actions = QHBoxLayout()
        self.btn_inspect = QPushButton("Inspect & Log")
        self.btn_copy_clean = QPushButton("Copy Clean")
        self.chk_clip = QCheckBox("Auto-Clip")
        self.chk_clip.setStyleSheet("color: #cbd5e1; font-size: 11px;")

        actions.addWidget(self.btn_inspect)
        actions.addWidget(self.btn_copy_clean)
        actions.addWidget(self.chk_clip)

        self.lbl_inspector_meta = QLabel("Tokens: 0 | Est Out: 0 | Saved: 0")
        self.lbl_inspector_meta.setStyleSheet(
            "color: #94a3b8; font-family: Consolas, monospace; font-size: 10px;"
        )

        layout.addWidget(self.txt_prompt)
        layout.addLayout(actions)
        layout.addWidget(self.lbl_inspector_meta)

        self.btn_inspect.clicked.connect(self.run_inspection)
        self.btn_copy_clean.clicked.connect(self._copy_cleaned_prompt)
        self.chk_clip.toggled.connect(self._on_clip_toggle)

        return page

    def _build_graphs_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        mode_row = QHBoxLayout()
        self.cmb_graph_mode = QComboBox()
        self.cmb_graph_mode.addItems(
            ["Weekly Spend", "Tokens Saved", "Waste by Category"]
        )
        self.cmb_graph_mode.setStyleSheet(
            "background-color: #1e293b; color: #f8fafc; font-size: 11px; padding: 2px;"
        )
        self.btn_refresh_graph = QPushButton("Refresh")
        self.btn_refresh_graph.setFixedWidth(64)
        mode_row.addWidget(self.cmb_graph_mode)
        mode_row.addWidget(self.btn_refresh_graph)

        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#0b0d12")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)

        layout.addLayout(mode_row)
        layout.addWidget(self.plot_widget)

        self.cmb_graph_mode.currentIndexChanged.connect(self._render_graph)
        self.btn_refresh_graph.clicked.connect(self._refresh_telemetry_views)
        return page

    def _build_efficiency_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.lbl_burn = QLabel("Burn Rate: 0.0 tok/min")
        self.lbl_burn.setStyleSheet("color: #f8fafc; font-family: Consolas, monospace; font-size: 12px;")

        self.lbl_ratio = QLabel("Clean Efficiency: 100.0% | Saved: 0 tok")
        self.lbl_ratio.setStyleSheet("color: #cbd5e1; font-family: Consolas, monospace; font-size: 11px;")

        self.lbl_quota = QLabel("Remaining Budget: 0 / 0")
        self.lbl_quota.setStyleSheet("color: #94a3b8; font-family: Consolas, monospace; font-size: 11px;")

        self.bar_quota = QProgressBar()
        self.bar_quota.setRange(0, 100)
        self.bar_quota.setValue(100)
        self.bar_quota.setStyleSheet(
            "QProgressBar { background-color: #0b0d12; border: 1px solid #334155; border-radius: 4px; height: 14px; text-align: center; color: #f8fafc; }"
            "QProgressBar::chunk { background-color: #15803d; }"
        )

        self.lbl_totals = QLabel("Prompts Logged: 0 | Total Spend: 0")
        self.lbl_totals.setStyleSheet("color: #94a3b8; font-size: 11px;")

        layout.addWidget(self.lbl_burn)
        layout.addWidget(self.lbl_ratio)
        layout.addWidget(self.lbl_quota)
        layout.addWidget(self.bar_quota)
        layout.addWidget(self.lbl_totals)
        layout.addStretch()
        return page

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        form = QFormLayout()
        form.setSpacing(3)

        input_style = (
            "background-color: #0b0d12; color: #f8fafc; border: 1px solid #283244; "
            "font-size: 10px; padding: 2px;"
        )
        lbl_style = "color: #cbd5e1; font-size: 10px;"

        self.inp_proj_name = QLineEdit()
        self.inp_proj_name.setStyleSheet(input_style)

        self.inp_proj_desc = QLineEdit()
        self.inp_proj_desc.setStyleSheet(input_style)

        self.inp_model_desc = QLineEdit()
        self.inp_model_desc.setStyleSheet(input_style)

        self.inp_budget = QSpinBox()
        self.inp_budget.setRange(1000, 2000000)
        self.inp_budget.setSingleStep(1000)
        self.inp_budget.setStyleSheet(input_style)

        self.cmb_encoding = QComboBox()
        self.cmb_encoding.addItems(["cl100k_base", "o200k_base"])
        self.cmb_encoding.setStyleSheet(input_style)

        self.inp_rewriter_url = QLineEdit()
        self.inp_rewriter_url.setStyleSheet(input_style)

        self.inp_rewriter_model = QLineEdit()
        self.inp_rewriter_model.setStyleSheet(input_style)

        self.inp_rewriter_key = QLineEdit()
        self.inp_rewriter_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.inp_rewriter_key.setStyleSheet(input_style)

        for label_txt, widget in [
            ("Project:", self.inp_proj_name),
            ("Proj Desc:", self.inp_proj_desc),
            ("Model:", self.inp_model_desc),
            ("Budget:", self.inp_budget),
            ("Encoding:", self.cmb_encoding),
            ("Rewriter URL:", self.inp_rewriter_url),
            ("Rewriter Mdl:", self.inp_rewriter_model),
            ("API Key:", self.inp_rewriter_key),
        ]:
            lbl = QLabel(label_txt)
            lbl.setStyleSheet(lbl_style)
            form.addRow(lbl, widget)

        self.btn_save_settings = QPushButton("Save Profile")
        layout.addLayout(form)
        layout.addWidget(self.btn_save_settings)
        self.btn_save_settings.clicked.connect(self._save_settings_from_form)
        return page

    def _load_settings_into_form(self) -> None:
        self.inp_proj_name.setText(self.settings.get("project_name", ""))
        self.inp_proj_desc.setText(self.settings.get("project_description", ""))
        self.inp_model_desc.setText(self.settings.get("model_description", ""))
        self.inp_budget.setValue(int(self.settings.get("session_budget", "12000") or "12000"))

        enc = self.settings.get("encoding", "cl100k_base")
        idx = self.cmb_encoding.findText(enc)
        if idx >= 0:
            self.cmb_encoding.setCurrentIndex(idx)

        self.inp_rewriter_url.setText(self.settings.get("rewriter_endpoint", ""))
        self.inp_rewriter_model.setText(self.settings.get("rewriter_model", ""))
        self.inp_rewriter_key.setText(self.settings.get("rewriter_api_key", ""))

        auto_clip = self.settings.get("auto_clipboard", "1") == "1"
        self.chk_clip.setChecked(auto_clip)

    def _save_settings_from_form(self) -> None:
        updates = {
            "project_name": self.inp_proj_name.text().strip() or "Tokguard",
            "project_description": self.inp_proj_desc.text().strip(),
            "model_description": self.inp_model_desc.text().strip(),
            "session_budget": str(self.inp_budget.value()),
            "encoding": self.cmb_encoding.currentText(),
            "rewriter_endpoint": self.inp_rewriter_url.text().strip(),
            "rewriter_model": self.inp_rewriter_model.text().strip(),
            "rewriter_api_key": self.inp_rewriter_key.text().strip(),
        }
        self.store.save_settings(updates)
        self.settings = self.store.get_settings()
        self._sync_analyzer_from_settings()
        self._refresh_telemetry_views()
        self.lbl_status.setText("SAVED | Context & rewriter updated")

    def _on_clip_toggle(self, checked: bool) -> None:
        self.store.save_settings({"auto_clipboard": "1" if checked else "0"})
        self.settings["auto_clipboard"] = "1" if checked else "0"

    def _on_clipboard_changed(self) -> None:
        if self._suppress_clipboard:
            return
        if self.settings.get("auto_clipboard", "1") != "1":
            return
        if not self._clipboard:
            return
        text = self._clipboard.text()
        if text and text.strip() and text != self.txt_prompt.toPlainText():
            self.txt_prompt.setPlainText(text)
            self.run_inspection()

    def _copy_cleaned_prompt(self) -> None:
        cleaned = str(self._last_report.get("cleaned_prompt") or self.txt_prompt.toPlainText())
        if not cleaned or not self._clipboard:
            return
        self._suppress_clipboard = True
        try:
            self._clipboard.setText(cleaned)
        finally:
            self._suppress_clipboard = False
        self.lbl_status.setText("COPIED | Clean prompt on clipboard")

    def _apply_prompt_from_dialog(self, new_text: str) -> None:
        self.txt_prompt.setPlainText(new_text)
        self.run_inspection()

    def run_inspection(self) -> None:
        raw_text = self.txt_prompt.toPlainText().strip()
        if not raw_text:
            self.lbl_status.setText("EMPTY | Paste or type a prompt")
            self.btn_bang.setVisible(False)
            self._apply_surface_style("#12151c", "#263042")
            return

        remaining = self._get_remaining_budget()
        report = self.analyzer.analyze(raw_text, remaining_budget=remaining)
        self._last_report = report

        status = str(report["status"])
        p_tok = int(report["prompt_tokens"])
        e_tok = int(report["estimated_output_tokens"])
        s_tok = int(report["saved_tokens"])

        self.lbl_inspector_meta.setText(
            f"Prompt: {p_tok} tok | Est Out: {e_tok} tok | Pruned: {s_tok} tok"
        )
        self.lbl_status.setText(f"{status} | In:{p_tok} Out~{e_tok}")

        # Show "!" only on YELLOW or RED
        self.btn_bang.setVisible(status in ("YELLOW", "RED"))

        self.store.log_event(
            project_name=self.settings.get("project_name", "Tokguard"),
            prompt_tokens=p_tok,
            estimated_output=e_tok,
            saved_tokens=s_tok,
            status=status,
            problem_category=str(report["problem_category"]),
        )
        self._refresh_telemetry_views()
        self._start_two_blink(status)

    def _start_two_blink(self, status: str) -> None:
        self._active_status = status
        # 2 full blink cycles = 4 transitions (on/off/on/off)
        self._blink_phases_remaining = 4
        self._blink_highlight_on = True
        self._apply_blink_phase(True)
        self._blink_timer.start(320)

    def _on_blink_tick(self) -> None:
        self._blink_phases_remaining -= 1
        if self._blink_phases_remaining <= 0:
            self._blink_timer.stop()
            self._settle_calm_state()
            return
        self._blink_highlight_on = not self._blink_highlight_on
        self._apply_blink_phase(self._blink_highlight_on)

    def _apply_blink_phase(self, highlight: bool) -> None:
        palette = {
            "GREEN": ("#15803d", "#4ade80"),
            "YELLOW": ("#b45309", "#fbbf24"),
            "RED": ("#b91c1c", "#f87171"),
        }
        bright_bg, border_col = palette.get(self._active_status, ("#1e293b", "#64748b"))
        bg = bright_bg if highlight else "#12151c"
        self._apply_surface_style(bg, border_col)

    def _settle_calm_state(self) -> None:
        settled_palette = {
            "GREEN": ("#112419", "#15803d"),
            "YELLOW": ("#241b0c", "#b45309"),
            "RED": ("#261114", "#b91c1c"),
        }
        bg, border = settled_palette.get(self._active_status, ("#12151c", "#263042"))
        self._apply_surface_style(bg, border)

    def _apply_surface_style(self, bg_hex: str, border_hex: str) -> None:
        self.setStyleSheet(
            f"QWidget#TokguardRoot {{ background-color: {bg_hex}; border: 1px solid {border_hex}; border-radius: 8px; }}"
            f"TokguardWidget {{ background-color: {bg_hex}; border: 1px solid {border_hex}; border-radius: 8px; }}"
        )

    def _open_diagnostic_dialog(self) -> None:
        if not self._last_report:
            return
        dlg = DiagnosticRewriteDialog(self, self._active_status, self._last_report, self.analyzer)
        dlg.apply_prompt_requested.connect(self._apply_prompt_from_dialog)
        dlg.exec()

    def _switch_page(self, delta: int) -> None:
        count = self.stack.count()
        next_idx = (self.stack.currentIndex() + delta) % count
        self.stack.setCurrentIndex(next_idx)
        names = ["Inspector", "Graphs", "Session", "Settings"]
        self.lbl_screen.setText(names[next_idx])
        if next_idx == 1:
            self._render_graph()
        elif next_idx == 2:
            self._refresh_telemetry_views()

    def _toggle_minimize(self) -> None:
        self._minimized_pill = not self._minimized_pill
        if self._minimized_pill:
            self.stack.setVisible(False)
            self.btn_prev.setVisible(False)
            self.btn_next.setVisible(False)
            self.lbl_screen.setVisible(False)
            self.setFixedSize(270, 42)
        else:
            self.setFixedSize(370, 300)
            self.btn_prev.setVisible(True)
            self.btn_next.setVisible(True)
            self.lbl_screen.setVisible(True)
            self.stack.setVisible(True)

    def _refresh_telemetry_views(self) -> None:
        stats = self.store.get_summary_stats()
        budget = int(self.settings.get("session_budget", "12000") or "12000")
        total_spend = int(stats["total_spend"])
        remaining = max(0, budget - total_spend)
        pct_left = int((remaining / budget) * 100) if budget > 0 else 0

        self.lbl_burn.setText(f"Burn Rate: {stats['burn_rate']} tok/min")
        self.lbl_ratio.setText(
            f"Clean Efficiency: {stats['efficiency_pct']}% | Saved: {stats['total_saved_tokens']} tok"
        )
        self.lbl_quota.setText(f"Remaining Budget: {remaining} / {budget}")
        self.bar_quota.setValue(max(0, min(100, pct_left)))
        self.lbl_totals.setText(
            f"Prompts Logged: {stats['total_prompts']} | Total Spend: {total_spend}"
        )

        if self.stack.currentIndex() == 1:
            self._render_graph()

    def _render_graph(self) -> None:
        self.plot_widget.clear()
        mode = self.cmb_graph_mode.currentText()

        if mode in ("Weekly Spend", "Tokens Saved"):
            series = self.store.get_weekly_series()
            x_vals = list(range(len(series)))
            y_vals = [
                int(item["spent"] if mode == "Weekly Spend" else item["saved"])
                for item in series
            ]
            ticks = [[(i, str(item["day"])) for i, item in enumerate(series)]]
            self.plot_widget.getAxis("bottom").setTicks(ticks)
            bar_color = "#38bdf8" if mode == "Weekly Spend" else "#22c55e"
            bars = pg.BarGraphItem(x=x_vals, height=y_vals, width=0.55, brush=bar_color)
            self.plot_widget.addItem(bars)
        else:
            cats = self.store.get_waste_by_category()
            labels = list(cats.keys())
            values = [int(cats[k]) for k in labels]
            x_vals = list(range(len(labels)))
            short_labels = [lbl[:6] for lbl in labels]
            ticks = [[(i, s) for i, s in enumerate(short_labels)]]
            self.plot_widget.getAxis("bottom").setTicks(ticks)
            bars = pg.BarGraphItem(x=x_vals, height=values, width=0.55, brush="#f59e0b")
            self.plot_widget.addItem(bars)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event and event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event and self._drag_offset is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            new_top_left = event.globalPosition().toPoint() - self._drag_offset
            self.move(new_top_left)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def closeEvent(self, event: Any) -> None:
        self._blink_timer.stop()
        self.store.close()
        super().closeEvent(event)
