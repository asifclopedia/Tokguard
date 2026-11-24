# Tokguard

**Context-Aware LLM Pre-Flight, Rewrite & Token Telemetry Widget for Windows**

Tokguard is a lightweight, always-on-top floating Windows HUD (`PyQt6` + `tiktoken` + `pyqtgraph` + `sqlite3`) built for developers who treat context windows, KV prompt caches, and token budgets as finite systems resources. It intercepts or stages prompts before dispatch, catches token-wasting failure modes locally in milliseconds, and integrates with a separate rewriter LLM when a prompt needs tightening.

---

## Core Features

- **Two-Blink Traffic-Light Pre-Flight Signal:**
  - **Green:** Well-scoped, cache-safe, and within remaining token budget.
  - **Yellow:** Optimization available (noisy logs/diffs, volatile prefix, missing invariants, or local script offload).
  - **Red:** Critical risk (budget exhaustion, likely mid-response `max_tokens` truncation, or underspecified retry loop).
  - Pulses **exactly twice** on evaluation, then settles into a low-distraction tinted state.
- **Conditional `!` Diagnostic & Separate-LLM Rewriter:**
  - Appears automatically on **Yellow** or **Red** alerts.
  - Displays root-cause failure modes, local/CLI offload advice, and generates a tightened prompt via your configured rewriter endpoint (local Ollama or any OpenAI-compatible chat API).
- **Pre-Flight Static Checks (`analyzer.py`):**
  - **Diff & Stack-Trace Pruner:** Strips `node_modules`, `site-packages`, `__pycache__`, `.git/`, `dist/`, and duplicate log lines while tracking exact tokens saved.
  - **Prompt-Cache Prefix Guard:** Flags volatile markers (`timestamp`, `request_id`, `trace_id`, `uuid`, ISO timestamps) in the opening 250 characters that invalidate prefix caching.
  - **Output Truncation & Scarcity Predictor:** Estimates `prompt_tokens + estimated_output_tokens` against remaining session quota to prevent mid-file cutoffs.
  - **Project & Model Context Alignment:** Cross-checks prompts against your configured **Project Name**, **Project Description**, and **Model Description**.
  - **Local-First Offload Advisor:** Flags deterministic data/regex/sorting transforms better handled by a local script or CLI one-liner.
- **SQLite Telemetry & Multi-View HUD (`telemetry.py` & `ui_widget.py`):**
  - Swipe/button navigation across **Inspector**, **Telemetry Graphs** (*Weekly Spend*, *Tokens Saved*, *Waste by Category*), **Session Burn & Efficiency**, and **Project/Model/Rewriter Profile**.
  - Auto-clipboard listener toggle and one-click minimize-to-pill mode.

---

## Project Structure

```text
Tokguard/
├── .github/
│   └── workflows/
│       └── build-exe.yml   # Automated Windows x64 PyInstaller build
├── main.py                 # Application entry point
├── analyzer.py             # Token counting, pruning, heuristics & rewriter client
├── telemetry.py            # SQLite persistence (tokguard.db) & series aggregation
├── ui_widget.py            # Frameless PyQt6 HUD, two-blink timer, graphs & modal
├── requirements.txt        # Python dependencies
└── README.md
