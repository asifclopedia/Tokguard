import json
import re
import urllib.error
import urllib.request
from typing import Any

import tiktoken


class PromptAnalyzer:
    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        session_budget: int = 12000,
        project_name: str = "Tokguard",
        project_description: str = "",
        model_description: str = "",
        rewriter_endpoint: str = "http://localhost:11434/v1/chat/completions",
        rewriter_model: str = "qwen2.5-coder:1.5b",
        rewriter_api_key: str = "",
    ) -> None:
        self.project_name = project_name
        self.project_description = project_description
        self.model_description = model_description
        self.session_budget = max(500, int(session_budget))
        self.rewriter_endpoint = rewriter_endpoint.strip()
        self.rewriter_model = rewriter_model.strip()
        self.rewriter_api_key = rewriter_api_key.strip()
        self.set_encoding(encoding_name)

    def set_encoding(self, encoding_name: str) -> None:
        self.encoding_name = encoding_name
        try:
            self.encoder = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def update_context(
        self,
        project_name: str,
        project_description: str,
        model_description: str,
        session_budget: int,
        encoding_name: str,
        rewriter_endpoint: str,
        rewriter_model: str,
        rewriter_api_key: str,
    ) -> None:
        self.project_name = project_name.strip() or "Default Project"
        self.project_description = project_description.strip()
        self.model_description = model_description.strip()
        self.session_budget = max(500, int(session_budget))
        self.rewriter_endpoint = rewriter_endpoint.strip()
        self.rewriter_model = rewriter_model.strip()
        self.rewriter_api_key = rewriter_api_key.strip()
        if encoding_name != self.encoding_name:
            self.set_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def prune_noise(self, raw_text: str) -> tuple[str, int]:
        """Strips noisy paths, duplicate lines, and excessive blank lines."""
        original_tokens = self.count_tokens(raw_text)
        lines = raw_text.splitlines()
        out_lines: list[str] = []
        seen_lines: set[str] = set()
        blank_streak = 0

        noisy_substrings = [
            "node_modules",
            "site-packages",
            "__pycache__",
            ".git/",
            "dist/",
        ]

        for line in lines:
            stripped = line.strip()

            if not stripped:
                blank_streak += 1
                if blank_streak <= 1:
                    out_lines.append("")
                continue
            blank_streak = 0

            is_noisy_path = False
            for token_str in noisy_substrings:
                if token_str in stripped:
                    is_noisy_path = True
                    break
            if is_noisy_path:
                continue

            if len(stripped) > 20 and stripped in seen_lines:
                continue
            if len(stripped) > 20:
                seen_lines.add(stripped)

            out_lines.append(line)

        cleaned = "\n".join(out_lines).strip()
        saved = max(0, original_tokens - self.count_tokens(cleaned))
        return cleaned, saved

    def check_cache_prefix(self, text: str) -> str | None:
        """Detects volatile tokens near the top that invalidate KV/prompt prefix caching."""
        head = text[:250].lower()
        volatile_patterns = [
            r"\btimestamp\b",
            r"\bepoch\b",
            r"\brequest_id\b",
            r"\btrace_id\b",
            r"\buuid\b",
            r"\b202\d-\d{2}-\d{2}t\d{2}:\d{2}",
        ]
        for pat in volatile_patterns:
            if re.search(pat, head):
                return (
                    f"Volatile prefix match ('{pat}') in first 250 chars may bust KV/prompt cache; "
                    "move static instructions/context ahead of dynamic fields."
                )
        return None

    def estimate_output_and_truncation(
        self, text: str, prompt_tokens: int, remaining_budget: int
    ) -> tuple[int, str | None]:
        """Estimates output tokens and flags mid-response cutoff or quota exhaustion."""
        lower = text.lower()
        scope_mult = 1.25

        heavy_triggers = [
            "entire file",
            "whole file",
            "full codebase",
            "all files",
            "rewrite everything",
            "full test suite",
            "complete implementation",
        ]
        medium_triggers = [
            "refactor",
            "implement",
            "generate",
            "write function",
            "write class",
            "explain in detail",
        ]

        for phrase in heavy_triggers:
            if phrase in lower:
                scope_mult = 2.6
                break
        else:
            for phrase in medium_triggers:
                if phrase in lower:
                    scope_mult = 1.65
                    break

        est_output = max(120, int(prompt_tokens * scope_mult))
        total_needed = prompt_tokens + est_output

        if total_needed > remaining_budget:
            return (
                est_output,
                f"Estimated {total_needed} tokens exceeds remaining budget ({remaining_budget}); "
                "task may break mid-output or exhaust quota.",
            )
        if est_output > 3000:
            return (
                est_output,
                f"Projected output (~{est_output} tokens) is large and risks max_tokens truncation; "
                "split into smaller module or function steps.",
            )
        return est_output, None

    def check_local_offload(self, text: str) -> str | None:
        """Suggests deterministic local CLI/script handling instead of LLM tokens."""
        lower = text.lower()
        offload_patterns = [
            r"\bsort\b.*\b(csv|json|list|lines)\b",
            r"\bparse\b.*\b(csv|json|log|xml)\b",
            r"\bextract\b.*\b(regex|emails?|ips?|urls?|dates?)\b",
            r"\b(calculate|compute|convert)\b.*\b(base64|timestamp|hex|csv to json|json to csv)\b",
        ]
        for pat in offload_patterns:
            if re.search(pat, lower):
                return (
                    "Deterministic data/text transform detected; a short local Python/awk/jq/rg command "
                    "is faster, cheaper, and reproducible without LLM inference."
                )
        return None

    def check_project_and_model_alignment(self, text: str) -> list[str]:
        """Cross-checks prompt against project context and target model description."""
        issues: list[str] = []
        lower = text.lower()
        words = text.split()

        if len(words) < 5:
            issues.append(
                "Prompt is underspecified (<5 words) and likely to trigger clarification or retry loops."
            )

        action_words = ["fix", "debug", "refactor", "optimize", "implement", "build", "change", "update"]
        has_action = False
        for act in action_words:
            if re.search(rf"\b{act}\b", lower):
                has_action = True
                break

        has_spec = any(
            k in lower
            for k in [
                "expected",
                "actual",
                "error",
                "traceback",
                "input",
                "output",
                "constraint",
                "edge case",
                "test",
            ]
        )
        if has_action and not has_spec:
            issues.append(
                f"Missing concrete spec/invariants for '{self.project_name}' (expected vs. actual behavior or edge cases)."
            )

        model_lower = self.model_description.lower()
        proj_lower = f"{self.project_name} {self.project_description}".lower()

        if ("small" in model_lower or "1b" in model_lower or "3b" in model_lower or "fast" in model_lower) and (
            len(words) > 450 or "entire" in lower or "architecture" in lower
        ):
            issues.append(
                f"Target model ('{self.model_description}') may struggle with broad/large context; "
                "narrow scope or escalate to a higher-capacity model."
            )

        if "python" in proj_lower and ("c++" in lower or "rust" in lower) and "port" not in lower and "bridge" not in lower:
            issues.append(
                f"Language/stack shift detected relative to project '{self.project_name}' ({self.project_description[:80]})."
            )

        return issues

    def build_fallback_rewrite(self, cleaned_prompt: str, problems: list[str], recommendation: str) -> str:
        """Produces a structured, high-signal prompt locally when external rewriter is unreachable."""
        problem_line = "; ".join(problems) if problems else "Tighten scope and explicit constraints."
        return (
            f"Project: {self.project_name}\n"
            f"Context: {self.project_description}\n"
            f"Target Model: {self.model_description}\n"
            f"Constraints: Keep output concise, deterministic, and strictly scoped; avoid unrequested rewrites. "
            f"Address: {problem_line}\n"
            f"Action Focus: {recommendation}\n\n"
            f"Task:\n{cleaned_prompt}"
        )

    def rewrite_with_llm(
        self,
        cleaned_prompt: str,
        problems: list[str],
        recommendation: str,
        timeout_sec: int = 15,
    ) -> tuple[str, str]:
        """Calls configured OpenAI-compatible/Ollama chat endpoint; returns (rewritten_text, source_note)."""
        fallback = self.build_fallback_rewrite(cleaned_prompt, problems, recommendation)
        if not self.rewriter_endpoint or not self.rewriter_model:
            return fallback, "Local structured rewrite (no rewriter endpoint/model configured)."

        system_msg = (
            "You are a senior prompt engineer optimizing prompts for token efficiency, "
            "prompt-cache stability, and zero mid-output truncation. "
            "Rewrite the user's prompt to be crisp, unambiguous, and scoped to the project. "
            "Output ONLY the improved prompt without meta-commentary."
        )
        user_msg = (
            f"Project Name: {self.project_name}\n"
            f"Project Description: {self.project_description}\n"
            f"Target Model: {self.model_description}\n"
            f"Detected Issues: {', '.join(problems) if problems else 'None'}\n"
            f"Guidance: {recommendation}\n\n"
            f"Draft Prompt:\n{cleaned_prompt}"
        )

        payload = {
            "model": self.rewriter_model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.2,
        }

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.rewriter_api_key:
            headers["Authorization"] = f"Bearer {self.rewriter_api_key}"

        req = urllib.request.Request(
            self.rewriter_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw_body = resp.read().decode("utf-8")
                data = json.loads(raw_body)
                choices = data.get("choices", [])
                if choices and isinstance(choices, list):
                    msg = choices[0].get("message", {})
                    content = str(msg.get("content", "")).strip()
                    if content:
                        return content, f"Rewritten via {self.rewriter_model}"
                if "response" in data and str(data["response"]).strip():
                    return str(data["response"]).strip(), f"Rewritten via {self.rewriter_model}"
                return fallback, "Empty response from rewriter; used local structured rewrite."
        except urllib.error.URLError as exc:
            return fallback, f"Rewriter unreachable ({exc.reason}); used local structured rewrite."
        except Exception as exc:
            return fallback, f"Rewriter error ({exc}); used local structured rewrite."

    def analyze(self, raw_prompt: str, remaining_budget: int | None = None) -> dict[str, Any]:
        budget = self.session_budget if remaining_budget is None else max(0, int(remaining_budget))
        cleaned_prompt, saved_tokens = self.prune_noise(raw_prompt)
        prompt_tokens = self.count_tokens(cleaned_prompt)
        est_output, truncation_msg = self.estimate_output_and_truncation(
            cleaned_prompt, prompt_tokens, budget
        )
        cache_msg = self.check_cache_prefix(cleaned_prompt)
        offload_msg = self.check_local_offload(cleaned_prompt)
        alignment_issues = self.check_project_and_model_alignment(cleaned_prompt)

        red_problems: list[str] = []
        yellow_problems: list[str] = []
        recommendations: list[str] = []
        primary_category = "clean_ready"

        if truncation_msg:
            red_problems.append(truncation_msg)
            recommendations.append(
                "Split into smaller atomic tasks (one module, diff hunk, or function per turn) "
                "and state exact output boundaries."
            )
            primary_category = "truncation_risk"

        if len(raw_prompt.strip().split()) < 4:
            red_problems.append("Prompt is too brief (<4 words) to execute reliably without retry loops.")
            recommendations.append(
                "Specify exact target file/function, concrete failure or requirement, and desired output format."
            )
            if primary_category == "clean_ready":
                primary_category = "ambiguity_spec"

        if cache_msg:
            yellow_problems.append(cache_msg)
            recommendations.append("Place static instructions and stable context at the top; append dynamic values at the bottom.")
            if primary_category == "clean_ready":
                primary_category = "cache_prefix"

        if saved_tokens >= 25:
            yellow_problems.append(f"Removed {saved_tokens} noisy/duplicate tokens during pre-flight pruning.")
            recommendations.append("Use the pruned prompt text so redundant logs/paths do not eat context.")
            if primary_category == "clean_ready":
                primary_category = "noisy_pruned"

        if offload_msg:
            yellow_problems.append(offload_msg)
            recommendations.append("Run a deterministic local script or CLI one-liner instead of spending LLM tokens.")
            if primary_category == "clean_ready":
                primary_category = "local_offload"

        for issue in alignment_issues:
            if "underspecified" not in issue.lower():
                yellow_problems.append(issue)
                recommendations.append("Align explicit constraints and runtime details with the active project and model.")
                if primary_category == "clean_ready":
                    primary_category = "ambiguity_spec"

        if red_problems:
            status = "RED"
            problems = red_problems + yellow_problems
        elif yellow_problems:
            status = "YELLOW"
            problems = yellow_problems
        else:
            status = "GREEN"
            problems = []
            recommendations = ["Prompt is clear, scoped within budget, and safe to send."]

        # Deduplicate recommendations preserving order
        unique_recs: list[str] = []
        for rec in recommendations:
            if rec not in unique_recs:
                unique_recs.append(rec)

        return {
            "status": status,
            "prompt_tokens": prompt_tokens,
            "estimated_output_tokens": est_output,
            "saved_tokens": saved_tokens,
            "cleaned_prompt": cleaned_prompt,
            "problems": problems,
            "recommendation": " ".join(unique_recs),
            "problem_category": primary_category,
        }
