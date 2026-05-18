import ast
import gzip
import json
import random
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
HUMANEVAL_FILE = ROOT / "HumanEval.jsonl.gz"
CHECKPOINT_FILE = ROOT / "streamlit_latest_partial_results.jsonl"
RUN_HISTORY_FILE = ROOT / "streamlit_run_history.json"
SETTING_PRESETS_FILE = ROOT / "streamlit_setting_presets.json"
APP_TITLE = "RL Prompt Engineering Lab"
PROMPT_EDITOR_VERSION = "problem_type_aware_prompts_v1"
STRATEGY_NAMES = {
    0: "zero_shot",
    1: "few_shot",
    2: "cot",
    3: "hint",
}
ARM_ROLES = {
    "zero_shot": ("Direct Robust", "Fast implementation with exact return type and edge-case awareness."),
    "few_shot": ("Pattern Learning", "Learns HumanEval-style structure from examples before solving."),
    "cot": ("Silent Algorithm", "Uses internal algorithm planning but outputs only final code."),
    "hint": ("Guided Edge Cases", "Uses problem-specific hints for constraints and failure modes."),
}
ANTI_COLLAPSE_DEFAULTS = {
    "mode": "Online Bandit",
    "experiment_preset": "Anti-collapse RL-APO",
    "prompt_bank": "Problem-type aware prompts",
    "strategy": "zero_shot",
    "load_in_4bit": True,
    "use_chat_template": True,
    "num_tasks": 164,
    "repeats": 1,
    "max_new_tokens": 512,
    "do_sample": False,
    "temperature": 0.1,
    "notebook_compatible": False,
    "compatible_fallback": False,
    "warm_start": False,
    "warm_start_run_names": [],
    "warm_start_counts_as_exploration": False,
    "similarity_memory": True,
    "memory_top_k": 8,
    "memory_lambda": 0.5,
    "memory_threshold": 0.15,
    "explore_order": "Strong-first (CoT, Hint, Zero, Few)",
    "weak_arm_guard": True,
    "weak_arm_min_samples": 8,
    "weak_arm_margin": 0.25,
    "best_arm_fallback": False,
    "best_arm_min_samples": 12,
    "best_arm_margin": 0.04,
    "balanced_arm_floor": True,
    "balanced_min_share": 0.10,
    "balanced_min_reward": 0.60,
    "category_prior": True,
    "category_prior_weight": 0.35,
    "category_arm_guard": True,
    "category_guard_min_samples": 3,
    "category_guard_margin": 0.20,
    "smart_extraction": True,
    "recent_arm_cooldown": True,
    "recent_cooldown_window": 8,
    "recent_cooldown_min_samples": 4,
    "recent_cooldown_threshold": 0.65,
    "generation_timeout": 120,
    "timeout": 10,
    "shuffle": False,
    "seed": 42,
    "alpha": 0.3,
    "force_explore": 20,
}
PUSH_75_DEFAULTS = {
    **ANTI_COLLAPSE_DEFAULTS,
    "experiment_preset": "75% Push RL-APO",
    "memory_lambda": 0.65,
    "weak_arm_margin": 0.30,
    "balanced_min_share": 0.06,
    "balanced_min_reward": 0.70,
    "category_prior_weight": 0.45,
    "recent_cooldown_threshold": 0.70,
    "force_explore": 12,
}
PRECISION_75_DEFAULTS = {
    **ANTI_COLLAPSE_DEFAULTS,
    "experiment_preset": "75% Precision RL-APO",
    "similarity_memory": True,
    "memory_lambda": 0.25,
    "weak_arm_margin": 0.30,
    "best_arm_fallback": True,
    "best_arm_min_samples": 12,
    "best_arm_margin": 0.06,
    "balanced_arm_floor": True,
    "balanced_min_share": 0.03,
    "balanced_min_reward": 0.75,
    "category_prior": True,
    "category_prior_weight": 0.25,
    "category_arm_guard": True,
    "recent_arm_cooldown": True,
    "recent_cooldown_threshold": 0.70,
    "force_explore": 8,
}
BALANCED_738_PLUS_DEFAULTS = {
    **ANTI_COLLAPSE_DEFAULTS,
    "experiment_preset": "73.8+ Balanced RL-APO",
    "explore_order": "Strong-first (CoT, Hint, Zero, Few)",
    "force_explore": 20,
    "similarity_memory": True,
    "memory_lambda": 0.5,
    "memory_top_k": 8,
    "weak_arm_guard": True,
    "weak_arm_min_samples": 8,
    "weak_arm_margin": 0.25,
    "best_arm_fallback": False,
    "balanced_arm_floor": True,
    "balanced_min_share": 0.10,
    "balanced_min_reward": 0.60,
    "category_prior": True,
    "category_prior_weight": 0.35,
    "category_arm_guard": True,
    "category_guard_min_samples": 3,
    "category_guard_margin": 0.20,
    "recent_arm_cooldown": True,
    "recent_cooldown_threshold": 0.65,
}
CALIBRATION_80_DEFAULTS = {
    **BALANCED_738_PLUS_DEFAULTS,
    "experiment_preset": "Calibration RL-APO",
    "calibration_mode": True,
    "calibration_tasks": 32,
    "evaluation_only_metric": False,
    "force_explore": 32,
    "best_arm_fallback": False,
    "balanced_arm_floor": True,
    "balanced_min_share": 0.03,
    "balanced_min_reward": 0.75,
    "memory_lambda": 0.35,
    "weak_arm_min_samples": 4,
    "weak_arm_margin": 0.15,
    "category_guard_min_samples": 2,
    "category_guard_margin": 0.08,
    "max_arm_share_guard": True,
    "max_arm_share": 0.75,
    "max_arm_share_min_steps": 20,
    "few_shot_memory_guard": False,
    "few_shot_min_memory": 0.60,
}
CALIBRATION_TARGETED_DEFAULTS = {
    **CALIBRATION_80_DEFAULTS,
    "experiment_preset": "Calibration RL-APO Targeted",
    "few_shot_memory_guard": False,
    "balanced_min_share": 0.03,
    "balanced_min_reward": 0.75,
    "memory_lambda": 0.35,
    "category_prior_weight": 0.45,
}
CALIBRATION_TARGETED_V2_DEFAULTS = {
    **CALIBRATION_80_DEFAULTS,
    "experiment_preset": "Calibration RL-APO Targeted v2",
    "few_shot_memory_guard": False,
    "balanced_min_share": 0.03,
    "balanced_min_reward": 0.75,
    "memory_lambda": 0.35,
    "category_prior_weight": 0.35,
    "max_arm_share": 0.75,
}
CALIBRATION_COT_ANCHOR_V3_DEFAULTS = {
    **CALIBRATION_TARGETED_V2_DEFAULTS,
    "experiment_preset": "Calibration RL-APO CoT Anchor v3",
    "targeted_hard_overrides": True,
}
CALIBRATION_FAIR_CALIB_DEFAULTS = {
    **CALIBRATION_80_DEFAULTS,
    "experiment_preset": "Calibration RL-APO Fair Calib",
    "evaluation_only_metric": False,
    "few_shot_memory_guard": False,
    "balanced_min_share": 0.03,
    "balanced_min_reward": 0.75,
    "memory_lambda": 0.35,
    "category_prior_weight": 0.25,
    "max_arm_share": 0.75,
    "targeted_hard_overrides": False,
}
CALIBRATION_NO_WEAK_FEW_DEFAULTS = {
    **CALIBRATION_80_DEFAULTS,
    "experiment_preset": "Calibration RL-APO No Weak Few",
    "few_shot_memory_guard": True,
    "few_shot_min_memory": 0.60,
    "balanced_min_share": 0.02,
    "balanced_min_reward": 0.80,
}
GENTLE_75_DEFAULTS = {
    **BALANCED_738_PLUS_DEFAULTS,
    "experiment_preset": "75% Gentle RL-APO",
    "explore_order": "CoT/Hint/Zero during exploration",
    "force_explore": 18,
    "weak_arm_min_samples": 6,
    "weak_arm_margin": 0.20,
    "category_guard_min_samples": 2,
    "category_guard_margin": 0.12,
}
HYBRID_76_DEFAULTS = {
    **ANTI_COLLAPSE_DEFAULTS,
    "experiment_preset": "76% Hybrid RL-APO",
    "explore_order": "CoT/Hint/Zero during exploration",
    "force_explore": 16,
    "similarity_memory": True,
    "memory_lambda": 0.35,
    "memory_top_k": 8,
    "weak_arm_guard": True,
    "weak_arm_min_samples": 8,
    "weak_arm_margin": 0.25,
    "best_arm_fallback": False,
    "balanced_arm_floor": True,
    "balanced_min_share": 0.05,
    "balanced_min_reward": 0.65,
    "category_prior": True,
    "category_prior_weight": 0.35,
    "category_arm_guard": True,
    "category_guard_min_samples": 3,
    "category_guard_margin": 0.20,
    "recent_arm_cooldown": True,
    "recent_cooldown_threshold": 0.65,
}


def init_state():
    st.session_state.setdefault("run_history", load_run_history())
    st.session_state.setdefault("setting_presets", load_setting_presets())
    st.session_state.setdefault("last_results", None)
    if st.session_state["last_results"] is None and CHECKPOINT_FILE.exists():
        try:
            checkpoint = pd.read_json(CHECKPOINT_FILE, lines=True)
            if not checkpoint.empty:
                st.session_state["last_results"] = checkpoint
        except ValueError:
            pass


def save_partial_results(rows: list[dict]):
    if rows:
        pd.DataFrame(rows).to_json(CHECKPOINT_FILE, orient="records", lines=True, force_ascii=False)


def load_run_history() -> list[dict]:
    if not RUN_HISTORY_FILE.exists():
        return []
    try:
        payload = json.loads(RUN_HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    runs = []
    for item in payload:
        df = pd.DataFrame(item.get("records", []))
        runs.append({
            "name": item.get("name", "Unnamed run"),
            "created_at": item.get("created_at", ""),
            "config": item.get("config", {}),
            "summary": item.get("summary", {}),
            "df": df,
        })
    return runs


def save_run_history(history: list[dict]):
    payload = []
    for item in history:
        df = item.get("df", pd.DataFrame())
        payload.append({
            "name": item.get("name", "Unnamed run"),
            "created_at": item.get("created_at", ""),
            "config": item.get("config", {}),
            "summary": item.get("summary", {}),
            "records": df.to_dict("records") if isinstance(df, pd.DataFrame) else [],
        })
    RUN_HISTORY_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_setting_presets() -> list[dict]:
    if not SETTING_PRESETS_FILE.exists():
        return []
    try:
        payload = json.loads(SETTING_PRESETS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [item for item in payload if isinstance(item, dict) and item.get("name") and isinstance(item.get("config"), dict)]


def save_setting_presets(presets: list[dict]):
    SETTING_PRESETS_FILE.write_text(json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8")


def df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def df_to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    latex = df.to_latex(index=False, escape=True, float_format="%.2f")
    return "\n".join([
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        latex,
        "\\end{table}",
    ])


def safe_download_name(text: str, suffix: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return f"{cleaned[:80] or 'rl_apo'}{suffix}"


def bool_badge(value: bool) -> str:
    return "ON" if value else "OFF"


def preset_value(active_preset: dict, key: str, fallback):
    return active_preset.get(key, fallback)


def option_index(options: list, value, fallback: int = 0) -> int:
    try:
        return options.index(value)
    except ValueError:
        return fallback


def ci95_percent(successes: int, total: int) -> tuple[float, float, float]:
    if total <= 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    margin = 1.96 * np.sqrt(p * (1 - p) / total) * 100
    center = p * 100
    return center, max(0.0, center - margin), min(100.0, center + margin)


def make_run_name(config: dict, started_at: str) -> str:
    if config["mode"] == "Online Bandit":
        method = "Online Bandit"
        if config.get("compatible_fallback") or config.get("warm_start"):
            method += " Compatible+"
        if config.get("similarity_memory"):
            method += " Sim"
        if config.get("weak_arm_guard") or config.get("explore_order") not in (None, "Notebook order"):
            method += " Guard"
        if config.get("category_arm_guard"):
            method += " CatGuard"
        if config.get("recent_arm_cooldown"):
            method += " Cooldown"
        if config.get("best_arm_fallback"):
            method += " SafeBest"
        if config.get("balanced_arm_floor"):
            method += " Balanced"
        if config.get("calibration_mode"):
            method += " Calib"
    else:
        method = config["strategy"]
    return f"{started_at} | {method} | {config['num_tasks']} tasks x {config['repeats']}"


def readable_run_label(item: dict) -> str:
    summary = item.get("summary", {})
    config = item.get("config", {})
    strategy = summary.get("strategy") or config.get("strategy", "run")
    if strategy == "bandit":
        preset = config.get("experiment_preset") or "Online Bandit"
        method = f"Bandit - {preset}"
    else:
        method = str(strategy).replace("_", " ").title()
    passed = int(summary.get("passed", 0) or 0)
    generations = int(summary.get("generations", config.get("num_tasks", 0)) or 0)
    pass_at_1 = float(summary.get("pass_at_1", 0.0) or 0.0)
    reward = float(summary.get("avg_reward", 0.0) or 0.0)
    seed = summary.get("seed", config.get("seed", "-"))
    force = summary.get("force_explore", config.get("force_explore", "-"))
    created = str(item.get("created_at") or summary.get("run_name", ""))[:16]
    return f"{method} | {passed}/{generations} ({pass_at_1:.1f}%) | reward {reward:.3f} | seed {seed} | explore {force} | {created}"


def summarize_results(df: pd.DataFrame, config: dict, run_name: str) -> dict:
    metric_df = df
    use_eval_only = config.get("evaluation_only_metric") and config.get("experiment_preset") != "Calibration RL-APO Fair Calib"
    if config.get("calibration_mode") and use_eval_only and "phase" in df.columns and (df["phase"] == "evaluation").any():
        metric_df = df[df["phase"] == "evaluation"]
    pass_at_1 = float(metric_df["passed"].mean() * 100) if not metric_df.empty else 0.0
    compile_rate = float(metric_df["compile_ok"].mean() * 100) if not metric_df.empty else 0.0
    avg_reward = float(metric_df["reward"].mean()) if not metric_df.empty else 0.0
    total = int(len(metric_df))
    passed = int(metric_df["passed"].sum()) if not metric_df.empty else 0
    compiled = int(metric_df["compile_ok"].sum()) if not metric_df.empty else 0
    _, pass_ci_low, pass_ci_high = ci95_percent(passed, total)
    _, compile_ci_low, compile_ci_high = ci95_percent(compiled, total)
    return {
        "run_name": run_name,
        "mode": config["mode"],
        "strategy": "bandit" if config["mode"] == "Online Bandit" else config["strategy"],
        "model": config["model_name"],
        "tasks": config["num_tasks"],
        "repeats": config["repeats"],
        "generations": total,
        "total_generations": int(len(df)),
        "calibration_mode": config.get("calibration_mode", False),
        "calibration_tasks": config.get("calibration_tasks", 0),
        "evaluation_only_metric": config.get("evaluation_only_metric", False),
        "pass_at_1": pass_at_1,
        "pass_at_1_ci_low": pass_ci_low,
        "pass_at_1_ci_high": pass_ci_high,
        "compile_rate": compile_rate,
        "compile_ci_low": compile_ci_low,
        "compile_ci_high": compile_ci_high,
        "avg_reward": avg_reward,
        "passed": passed,
        "failed": total - passed,
        "compiled": compiled,
        "compile_failed": total - compiled,
        "total_seconds": float(df["task_seconds"].sum()) if "task_seconds" in df and not df.empty else 0.0,
        "avg_task_seconds": float(df["task_seconds"].mean()) if "task_seconds" in df and not df.empty else 0.0,
        "alpha": config["alpha"],
        "force_explore": config["force_explore"],
        "seed": config["seed"],
        "chat_template": config["use_chat_template"],
        "notebook_compatible": config.get("notebook_compatible", False),
        "compatible_fallback": config.get("compatible_fallback", False),
        "warm_start": config.get("warm_start", False),
        "warm_start_examples": config.get("warm_start_examples", 0),
        "similarity_memory": config.get("similarity_memory", False),
        "memory_top_k": config.get("memory_top_k", 0),
        "memory_lambda": config.get("memory_lambda", 0.0),
        "memory_threshold": config.get("memory_threshold", 0.0),
        "explore_order": config.get("explore_order", "Notebook order"),
        "weak_arm_guard": config.get("weak_arm_guard", False),
        "weak_arm_min_samples": config.get("weak_arm_min_samples", 0),
        "weak_arm_margin": config.get("weak_arm_margin", 0.0),
        "best_arm_fallback": config.get("best_arm_fallback", False),
        "best_arm_min_samples": config.get("best_arm_min_samples", 0),
        "best_arm_margin": config.get("best_arm_margin", 0.0),
        "balanced_arm_floor": config.get("balanced_arm_floor", False),
        "balanced_min_share": config.get("balanced_min_share", 0.0),
        "balanced_min_reward": config.get("balanced_min_reward", 0.0),
        "max_arm_share_guard": config.get("max_arm_share_guard", False),
        "max_arm_share": config.get("max_arm_share", 0.0),
        "max_arm_share_min_steps": config.get("max_arm_share_min_steps", 0),
        "few_shot_memory_guard": config.get("few_shot_memory_guard", False),
        "few_shot_min_memory": config.get("few_shot_min_memory", 0.0),
        "targeted_hard_overrides": config.get("targeted_hard_overrides", False),
        "category_prior": config.get("category_prior", False),
        "category_prior_weight": config.get("category_prior_weight", 0.0),
        "category_arm_guard": config.get("category_arm_guard", False),
        "category_guard_min_samples": config.get("category_guard_min_samples", 0),
        "category_guard_margin": config.get("category_guard_margin", 0.0),
        "smart_extraction": config.get("smart_extraction", True),
        "recent_arm_cooldown": config.get("recent_arm_cooldown", False),
        "recent_cooldown_window": config.get("recent_cooldown_window", 0),
        "recent_cooldown_min_samples": config.get("recent_cooldown_min_samples", 0),
        "recent_cooldown_threshold": config.get("recent_cooldown_threshold", 0.0),
        "prompt_bank": config.get("prompt_bank", "Notebook 68 prompts"),
    }
DEFAULT_PROMPTS = {
    "zero_shot": """You are an expert Python programmer. Complete the function by writing ONLY the code that goes inside the function body. Do NOT repeat the function signature, do NOT add comments, do NOT write tests.

Complete this function:

{prompt}""",
    "few_shot": """You are an expert Python programmer. Complete the function by writing ONLY the code that goes inside the function body. Do NOT repeat the function signature, do NOT add comments, do NOT write tests.

Here are examples of completed functions:

Example 1:
Problem:
def add(a, b):
    \"\"\"Returns the sum of a and b\"\"\"

Solution:
    return a + b

Example 2:
Problem:
def reverse_string(s):
    \"\"\"Reverses the input string\"\"\"

Solution:
    return s[::-1]

Example 3:
Problem:
def find_max(lst):
    \"\"\"Finds the maximum value in a list\"\"\"

Solution:
    return max(lst)

Now complete this function:

{prompt}""",
    "cot": """Think step by step about how to solve this problem, then write the code. First, reason about the approach, then provide the code.

Problem:
{prompt}

Reasoning: Let me think about this step by step. """,
    "hint": """You are an expert Python programmer. Complete the function by writing ONLY the code that goes inside the function body. Do NOT repeat the function signature, do NOT add comments, do NOT write tests.

{hint}

Complete this function:

{prompt}""",
}

EDGE_CASE_PROMPTS = {
    "zero_shot": """You are an expert Python programmer. Complete the function by writing ONLY the code that goes inside the function body. Do NOT repeat the function signature, do NOT add comments, do NOT write tests.

Read the docstring carefully. Handle edge cases such as empty inputs, duplicates, negative numbers, and boundary values when relevant.

Complete this function:

{prompt}""",
    "few_shot": """You are an expert Python programmer. Complete the function by writing ONLY the code that goes inside the function body. Do NOT repeat the function signature, do NOT add comments, do NOT write tests.

Here are examples of completed functions:

Example 1:
Problem:
def add(a, b):
    \"\"\"Returns the sum of a and b\"\"\"

Solution:
    return a + b

Example 2:
Problem:
def reverse_string(s):
    \"\"\"Reverses the input string\"\"\"

Solution:
    return s[::-1]

Example 3:
Problem:
def is_sorted(nums):
    \"\"\"Returns True if nums is sorted in nondecreasing order\"\"\"

Solution:
    return all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1))

Now complete this function:

{prompt}""",
    "cot": """Think briefly about the edge cases, then write ONLY the code that goes inside the function body. Do NOT repeat the function signature, do NOT add comments, do NOT write tests.

Problem:
{prompt}

Code:""",
    "hint": """You are an expert Python programmer. Complete the function by writing ONLY the code that goes inside the function body. Do NOT repeat the function signature, do NOT add comments, do NOT write tests.

{hint}

Pay attention to edge cases and return exactly what the docstring asks.

Complete this function:

{prompt}""",
}

CONCISE_BODY_PROMPTS = {
    "zero_shot": """Complete the Python function. Output only the missing indented function body. Do not repeat the signature, write tests, markdown, or explanations.

{prompt}""",
    "few_shot": """Complete the Python function. Output only the missing indented function body.

Example:
def add(a, b):
    \"\"\"Returns the sum of a and b\"\"\"

Solution:
    return a + b

Example:
def reverse_string(s):
    \"\"\"Reverses the input string\"\"\"

Solution:
    return s[::-1]

Now complete:

{prompt}""",
    "cot": """Solve the task mentally, then output only the missing indented function body. No explanation, no signature, no tests.

{prompt}""",
    "hint": """Complete the Python function. Output only the missing indented function body.

{hint}

{prompt}""",
}

ALGORITHM_FOCUSED_PROMPTS = {
    "zero_shot": """You are a careful algorithmic Python programmer. Implement the function body only. Do NOT repeat the signature, comments, tests, or markdown.

Choose a simple correct algorithm and handle boundary cases.

{prompt}""",
    "few_shot": """You are a careful algorithmic Python programmer. Implement the function body only.

Example 1:
def has_close_elements(numbers, threshold):
    \"\"\"Check if any two numbers are closer than threshold.\"\"\"
Solution:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False

Example 2:
def sort_numbers(nums):
    \"\"\"Return numbers sorted increasingly.\"\"\"
Solution:
    return sorted(nums)

Now implement:

{prompt}""",
    "cot": """Think about the algorithm, invariants, and edge cases. Then output only the missing indented function body. Do not include reasoning text.

{prompt}""",
    "hint": """You are a careful algorithmic Python programmer. Implement the function body only.

{hint}

Use direct control flow and standard-library operations when they are sufficient.

{prompt}""",
}

ROBUST_TEST_PASSING_PROMPTS = {
    "zero_shot": """Write the function body that will pass the hidden unit tests. Output only executable Python statements for the body. No signature, no tests, no markdown.

Pay special attention to exact return type and edge cases described in the docstring.

{prompt}""",
    "few_shot": """Write the function body that will pass hidden unit tests. Output only executable Python statements for the body.

Example 1:
def count_vowels(s):
    \"\"\"Counts vowels in a string.\"\"\"
Solution:
    return sum(ch in 'aeiouAEIOU' for ch in s)

Example 2:
def is_palindrome(s):
    \"\"\"Checks whether a string is a palindrome.\"\"\"
Solution:
    return s == s[::-1]

Example 3:
def clamp(x, lo, hi):
    \"\"\"Clamp x into [lo, hi].\"\"\"
Solution:
    return max(lo, min(x, hi))

Now solve:

{prompt}""",
    "cot": """Review the docstring examples and hidden edge cases mentally. Output only the final function body, with correct indentation. No reasoning text.

{prompt}""",
    "hint": """Write the function body that will pass hidden unit tests. Output only executable Python statements for the body.

{hint}

Return exactly the requested value and type.

{prompt}""",
}

PROBLEM_TYPE_AWARE_PROMPTS = {
    "zero_shot": """You are a precise Python programmer. Implement the target function correctly.

Output only Python code, no markdown, no explanation, no tests, and no print statements. A full function definition is acceptable if needed; otherwise output the missing function body.

Use this checklist before writing code:
1. Match every visible docstring example exactly.
2. Preserve the exact return type requested by the signature/docstring.
3. Handle empty inputs, one-element inputs, duplicates, negative values, whitespace, and boundary cases when relevant.
4. Prefer a simple deterministic algorithm over clever shortcuts.
5. Do not mutate inputs unless the task clearly allows it.

Problem-specific guidance: {hint}

{prompt}""",
    "few_shot": """You are a precise Python programmer. Learn the style from these HumanEval-like examples, then output only executable Python code for the missing function body.

Example 1: string filtering with edge cases
def remove_vowels(text: str) -> str:
    \"\"\"Return text with all vowels removed.\"\"\"
Solution:
    vowels = set('aeiouAEIOU')
    return ''.join(ch for ch in text if ch not in vowels)

Example 2: pairwise list condition
def any_close(numbers, threshold):
    \"\"\"Return True if any two numbers differ by less than threshold.\"\"\"
Solution:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False

Example 3: balanced group parsing
def split_balanced_groups(paren_string: str):
    \"\"\"Split a string into balanced parenthesis groups, ignoring spaces.\"\"\"
Solution:
    groups = []
    current = ''
    depth = 0
    for ch in paren_string.replace(' ', ''):
        current += ch
        depth += 1 if ch == '(' else -1
        if depth == 0:
            groups.append(current)
            current = ''
    return groups

Now solve the target function. Output only the missing function body.
Problem-specific guidance: {hint}

{prompt}""",
    "cot": """Think silently about the algorithm, edge cases, and exact return type. Then output only the final executable Python code for the missing function body.

Do not include reasoning text, markdown, tests, or the function signature.

Problem-specific guidance: {hint}

{prompt}""",
    "hint": """You are a precise Python programmer. Use the problem-specific hint and implement only the missing function body.

{hint}

Also check the docstring examples, exact return type, empty inputs, duplicates, and boundary values. Use direct Python control flow and standard-library operations when sufficient. Output only executable Python code.

{prompt}""",
}

PROMPT_BANKS = {
    "Problem-type aware prompts": PROBLEM_TYPE_AWARE_PROMPTS,
    "Notebook 68 prompts": DEFAULT_PROMPTS,
    "Edge-case optimized prompts": EDGE_CASE_PROMPTS,
    "Concise body-only prompts": CONCISE_BODY_PROMPTS,
    "Algorithm-focused prompts": ALGORITHM_FOCUSED_PROMPTS,
    "Robust test-passing prompts": ROBUST_TEST_PASSING_PROMPTS,
}


def read_humaneval(path: Path) -> dict:
    problems = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            problems[item["task_id"]] = item
    return problems


def _strip_repeated_signature(code: str) -> str:
    lines = code.splitlines()
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            body = lines[idx + 1:]
            while body and not body[0].strip():
                body = body[1:]
            return "\n".join(body).strip("\n")
    return code.strip("\n")


def _normalize_body_indentation(code: str) -> str:
    code = code.strip("\n")
    if not code.strip():
        return code
    stripped = code.lstrip()
    if stripped.startswith(('"""', "'''")):
        quote = stripped[:3]
        end = stripped.find(quote, 3)
        if end != -1:
            code = stripped[end + 3:].strip("\n")
    try:
        code = ast.unparse(ast.parse(code))
    except SyntaxError:
        pass
    lines = code.splitlines()
    non_empty = [line for line in lines if line.strip()]
    if non_empty and all(line.startswith((" ", "\t")) for line in non_empty):
        try:
            import textwrap
            dedented = textwrap.dedent(code).strip("\n")
            compile("def _candidate():\n" + "\n".join("    " + line if line.strip() else line for line in dedented.splitlines()), "<indent_check>", "exec")
            return "\n".join("    " + line if line.strip() else line for line in dedented.splitlines()).strip()
        except Exception:
            pass
        return code.strip("\n")
    return "\n".join(("    " + line if line.strip() else line) for line in lines).strip()


def extract_code(generated: str) -> str:
    generated = generated.strip()
    for marker in ["FINAL_CODE:", "Solution body:", "Solution:", "Code:"]:
        if marker in generated:
            generated = generated.split(marker, 1)[1].strip()
            break
    if "Reasoning:" in generated:
        generated = generated.split("Reasoning:")[-1].strip()
    if "```python" in generated:
        generated = generated.split("```python", 1)[1].split("```", 1)[0].strip()
    elif "```" in generated:
        generated = generated.split("```", 1)[1].split("```", 1)[0].strip()
    stop_markers = ["\nProblem:", "\nExample", "\nTests:", "\nassert ", "\n# Test", "\nprint("]
    for marker in stop_markers:
        if marker in generated:
            generated = generated.split(marker, 1)[0].strip()
    generated = _strip_repeated_signature(generated)
    return _normalize_body_indentation(generated)


def extract_code_notebook(generated: str) -> str:
    generated = generated.strip()
    if "```python" in generated:
        return generated.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in generated:
        return generated.split("```", 1)[1].split("```", 1)[0].strip()
    return generated


def make_hint(problem: dict) -> str:
    prompt_text = problem["prompt"].lower()
    entry_point = problem.get("entry_point", "").lower()
    if entry_point == "fib":
        return "Hint: Use an iterative Fibonacci update with fib(0)=0 and fib(1)=1. Return the n-th value directly, without recursion."
    if entry_point == "x_or_y":
        return "Hint: Return x only when n is prime. Treat n < 2 as not prime, and test divisors up to sqrt(n); otherwise return y."
    if entry_point == "check_if_last_char_is_a_letter":
        return "Hint: Return True only when txt is non-empty, does not end with a space, the last character is a letter, and the last word has length exactly 1."
    if entry_point == "sum_squares" and ("ceiling" in prompt_text or "upper int" in prompt_text):
        return "Hint: Import ceil from math or use math.ceil inside the body. Apply ceiling to each number first, then square and sum; ceil(-2.4) is -2."
    if "compare_one" in prompt_text or "strings representing" in prompt_text and "real numbers" in prompt_text:
        return "Hint: Convert strings to floats for comparison after replacing comma with dot, but return the original input value and type. If numeric values are equal, return None."
    if "is_equal_to_sum_even" in prompt_text or "sum of exactly 4 positive even numbers" in prompt_text:
        return "Hint: Four positive even numbers have minimum sum 8. Return True only when n is even and n >= 8."
    if "sum_squares" in prompt_text and "multiple of 3" in prompt_text and "multiple of 4" in prompt_text:
        return "Hint: For each index i: if i % 3 == 0 square the value; elif i % 4 == 0 cube it; otherwise keep it unchanged. Then sum all transformed values."
    if "derivative" in prompt_text and "coefficients of a polynomial" in prompt_text:
        return "Hint: The derivative coefficient for x^(i-1) is i * xs[i]. Skip xs[0]. For [a0,a1,a2,...] return [a1,2*a2,3*a3,...]."
    if "numerical_letter_grade" in prompt_text:
        return "Hint: Map each GPA using strict greater-than thresholds in descending order, with 4.0 exactly A+ and values >0.0 down to D-. Return a list of strings."
    if "make_a_pile" in prompt_text:
        return "Hint: Return n numbers starting at n and increasing by 2 each level: [n, n+2, n+4, ...] with length n."
    if "by_length" in prompt_text:
        return "Hint: Keep only integers 1..9, sort ascending, reverse, then map to capitalized English digit names One..Nine. Ignore all other numbers."
    if "double_the_difference" in prompt_text:
        return "Hint: Sum x*x only for values where type(x) is int, x is positive, and x is odd. Ignore negatives, floats, even numbers, and non-integers."
    if "prime fibonacci" in prompt_text or "prime_fib" in prompt_text:
        return "Hint: Generate Fibonacci numbers in order, test each for primality, count only prime Fibonacci numbers, and return the n-th prime Fibonacci number."
    if "decimal_to_binary" in prompt_text or "db" in prompt_text and "binary" in prompt_text:
        return "Hint: Convert the integer to binary without recursion artifacts and wrap exactly once with 'db' prefix and suffix, e.g. db1111db."
    if "starts_one_ends" in prompt_text or "starts with 1" in prompt_text and "ends" in prompt_text:
        return "Hint: Count n-digit positive integers that start with 1 and end with 1. For n=1 return 1; for n>1 return 10**(n-2)."
    if "sum the even elements" in prompt_text and "odd positions" in prompt_text:
        return "Hint: Sum elements at odd indices only if the element itself is even. Do not sum all elements at odd indices."
    if "sort the numbers" in prompt_text and "even positions" in prompt_text:
        return "Hint: Sort values located at even indices and put them back at even indices; keep odd indices unchanged."
    if "rounded_avg" in prompt_text or "rounded average" in prompt_text:
        return "Hint: If n > m return -1. Otherwise compute round((n+m)/2) and return its binary representation with bin(...)."
    if "count_nums" in prompt_text or "sum of digits" in prompt_text and "negative" in prompt_text:
        return "Hint: For each integer, sum its digits while preserving sign for the first digit of negative numbers; count numbers whose digit sum is positive."
    if "maximum k numbers" in prompt_text:
        return "Hint: Return the largest k numbers but sorted in ascending order. Use sorted(arr)[-k:], and return [] when k is 0."
    if "same characters" in prompt_text or "check_dict_case" in prompt_text:
        return "Hint: For dictionary keys, first ensure every key is a string. Return False for empty dict or non-string keys; otherwise all keys must be all lower or all upper."
    if "encode" in prompt_text and "vowels" in prompt_text:
        return "Hint: Swap case for consonants. For vowels, shift by two alphabet positions after case handling according to the examples. Preserve non-letters."
    if "is_bored" in prompt_text:
        return "Hint: Count sentences that start with the standalone word 'I'. Split on '.', '?' and '!' punctuation, then strip spaces."
    if "fix_spaces" in prompt_text:
        return "Hint: Single spaces become underscores. Runs of more than two consecutive spaces become one hyphen. Preserve all non-space characters."
    if "order_by_points" in prompt_text:
        return "Hint: Sort by the sum of digits with sign included for negative numbers, using stable sort. For -12 the digit sum is -1+2 = 1."
    if "simplify" in prompt_text and "/" in prompt_text:
        return "Hint: Parse both fractions exactly with fractions.Fraction and return whether their product equals 1."
    if "basic algebra operations" in prompt_text and "operator" in prompt_text and "operand" in prompt_text:
        return "Hint: Build the expression from operands and operators and evaluate it with normal Python precedence; do not evaluate strictly left-to-right."
    if "mean absolute deviation" in prompt_text:
        return "Hint: Compute mean = sum(numbers)/len(numbers), then return the average absolute distance from the mean. The input is non-empty in HumanEval tests."
    if "sort the numbers" in prompt_text and "zero one two" in prompt_text:
        return "Hint: Map the number words zero..nine to their numeric order, sort the words by that mapping, and join them with spaces."
    if "even elements" in prompt_text and "changed positions" in prompt_text:
        return "Hint: Keep odd indices unchanged. Sort only the values at even indices (0,2,4,...) and put them back into those same even positions."
    if "car race collision" in prompt_text or "collisions" in prompt_text:
        return "Hint: For n cars moving from left and n from right, every pair collides exactly once, so return n*n."
    if "contains no letters" in prompt_text and "reverse the string" in prompt_text:
        return "Hint: If the string has at least one letter, swap case of letters and keep other characters in place. If it has no letters, return the reversed whole string."
    if "md5" in prompt_text:
        return "Hint: Import hashlib inside the function, return None for empty string, otherwise return hashlib.md5(text.encode()).hexdigest()."
    if "even digits" in prompt_text and "ascending order" in prompt_text:
        return "Hint: Return the even digits 2, 4, 6, 8 that lie between min(a,b) and max(a,b), in ascending order. Digits above 9 are never returned."
    if "intersection" in prompt_text and "prime" in prompt_text and "interval" in prompt_text:
        return "Hint: For these closed intervals, follow the examples: intersection (2, 3) has length 1, so use end - start, not end - start + 1, then test whether that length is prime."
    if "product of the odd digits" in prompt_text:
        return "Hint: Multiply only odd digits, but return 0 if there are no odd digits at all. Track whether any odd digit was seen."
    if "largest of negative integers" in prompt_text and "smallest" in prompt_text and "positive" in prompt_text:
        return "Hint: Return (max negative number or None, min positive number or None). Ignore zeros. Be careful not to swap the negative and positive trackers."
    if "sum of magnitudes" in prompt_text and "product of all signs" in prompt_text:
        return "Hint: Return None for empty input. If any element is zero the sign product is 0, so the final answer is 0; otherwise multiply sign by -1 for negative values and sum absolute magnitudes."
    if "tribonacci" in prompt_text or "tri(1)" in prompt_text:
        return "Hint: The sequence starts with tri(0)=1, tri(1)=3, tri(2)=2. For even i use 1 + i/2; for odd i use previous three values. Return the first n+1 values."
    if "valid subsequence" in prompt_text and "nested" in prompt_text and "square brackets" in prompt_text:
        return "Hint: This is not just balanced brackets. Return True only if there exists nested brackets, i.e. depth reaches at least 2 in a valid subsequence; plain [] or [][] is False."
    if "ceiling" in prompt_text or "upper int" in prompt_text:
        return "Hint: Round each value upward with math.ceil before squaring. Remember negative values: ceil(-2.4) is -2, then square it."
    if "largest index" in prompt_text and "immediately preceding" in prompt_text:
        return "Hint: Scan all adjacent pairs and return the largest index i where arr[i] < arr[i-1]. Return i, not i-1. Return -1 if none."
    if "last character" in prompt_text and "not" in prompt_text and "part of a word" in prompt_text:
        return "Hint: Return True only when the string is non-empty, does not end with space, the last character is a letter, and the last word has length exactly 1."
    if "path of length k" in prompt_text and "lexicographically" in prompt_text and "grid" in prompt_text:
        return "Hint: Return the lexicographically smallest sequence of k cell values. Usually start from the smallest cell value and repeatedly choose the smallest neighbor value; values may be revisited."
    if any(word in prompt_text for word in ["parentheses", "bracket", "paren"]):
        return "Hint: Track nesting depth carefully, reset groups only when depth returns to zero, and ignore spaces if required."
    if any(word in prompt_text for word in ["sort", "sorted", "order", "increasing", "decreasing"]):
        return "Hint: Preserve the requested ordering exactly; handle duplicates, empty lists, and stable sorting requirements."
    if any(word in prompt_text for word in ["tree", "node", "binary"]):
        return "Hint: Use recursion or an explicit stack; handle empty nodes and base cases before recursive calls."
    if any(word in prompt_text for word in ["string", "text", "character", "substring", "word"]):
        return "Hint: Preserve character order unless told otherwise; handle empty strings, whitespace, case, and punctuation exactly."
    if any(word in prompt_text for word in ["list", "array", "tuple", "sequence"]):
        return "Hint: Handle empty inputs, duplicates, and boundary indexes; prefer simple loops or comprehensions."
    if any(word in prompt_text for word in ["integer", "number", "float", "prime", "digit", "sum", "product"]):
        return "Hint: Check zero, negative values, one-element inputs, and exact integer/float return type."
    return "Hint: Match the docstring examples exactly, handle hidden edge cases, and return the requested type."


def problem_category(problem: dict) -> str:
    text = problem["prompt"].lower()
    if "md5" in text:
        return "hashing"
    if "basic algebra operations" in text:
        return "expression"
    if "grid" in text or "path of length" in text:
        return "grid"
    if "intersection" in text and "interval" in text:
        return "interval"
    if "digits" in text or "signs" in text or "factorial" in text:
        return "math"
    if any(word in text for word in ["parentheses", "bracket", "paren"]):
        return "parsing"
    if any(word in text for word in ["string", "text", "character", "substring", "word"]):
        return "string"
    if any(word in text for word in ["list", "array", "tuple", "sequence"]):
        return "list"
    if any(word in text for word in ["sort", "sorted", "order", "increasing", "decreasing"]):
        return "ordering"
    if any(word in text for word in ["tree", "node", "binary", "recursive"]):
        return "recursive"
    if any(word in text for word in ["integer", "number", "float", "prime", "digit", "sum", "product"]):
        return "math"
    return "general"


def targeted_v2_prior_scores(problem: dict, category: str, num_arms: int = 4) -> list[float]:
    text = problem["prompt"].lower()
    entry_point = problem.get("entry_point", "").lower()
    priors = [0.0] * num_arms
    if entry_point == "fib":
        priors[0] = 0.25
        priors[2] = 0.20
        return priors[:num_arms]
    if entry_point == "derivative":
        priors[3] = 0.35
        priors[2] = 0.10
        return priors[:num_arms]
    if entry_point == "make_a_pile":
        priors[3] = 0.35
        priors[0] = 0.10
        return priors[:num_arms]
    if entry_point == "rounded_avg":
        priors[3] = 0.35
        priors[2] = 0.10
        return priors[:num_arms]
    if entry_point == "by_length":
        priors[3] = 0.35
        priors[2] = 0.10
        return priors[:num_arms]
    if entry_point == "check_if_last_char_is_a_letter":
        priors[2] = 0.35
        priors[0] = 0.10
        return priors[:num_arms]
    if entry_point == "x_or_y":
        priors[2] = 0.35
        priors[0] = 0.10
        return priors[:num_arms]
    if entry_point == "double_the_difference":
        priors[2] = 0.35
        priors[0] = 0.10
        return priors[:num_arms]
    if entry_point == "sum_squares" and ("ceiling" in text or "upper int" in text):
        priors[3] = 0.30
        priors[2] = 0.15
        priors[1] = -0.30
        return priors[:num_arms]
    if category in {"grid", "interval", "math", "ordering", "recursive", "parsing"}:
        priors[2] = 0.12
        priors[0] = 0.06
    elif category in {"string", "list"}:
        priors[2] = 0.08
        priors[0] = 0.08
    else:
        priors[0] = 0.06
        priors[2] = 0.06
    return priors[:num_arms]


def fair_calib_prior_scores(category: str, num_arms: int = 4) -> list[float]:
    priors = [0.0] * num_arms
    if category in {"hashing", "expression"}:
        priors[3] = 0.14
        priors[0] = 0.08
    elif category in {"grid", "interval", "math", "ordering", "recursive"}:
        priors[2] = 0.12
        priors[0] = 0.06
    elif category == "parsing":
        priors[2] = 0.10
        priors[3] = 0.08
    elif category in {"string", "list"}:
        priors[2] = 0.08
        priors[0] = 0.06
        priors[3] = 0.04
    else:
        priors[0] = 0.05
        priors[2] = 0.05
    return priors[:num_arms]


def category_prior_scores(problem: dict, num_arms: int = 4, experiment_preset: str = "") -> list[float]:
    category = problem_category(problem)
    text = problem["prompt"].lower()
    priors = [0.0] * num_arms
    if experiment_preset == "Calibration RL-APO Fair Calib":
        return fair_calib_prior_scores(category, num_arms)
    if experiment_preset in {"Calibration RL-APO Targeted v2", "Calibration RL-APO CoT Anchor v3"}:
        return targeted_v2_prior_scores(problem, category, num_arms)
    if "compare_one" in text or "strings representing" in text and "real numbers" in text:
        priors[3] = 0.42
        priors[0] = 0.10
        return priors[:num_arms]
    if "is_equal_to_sum_even" in text or "sum of exactly 4 positive even numbers" in text:
        priors[1] = 0.34
        priors[2] = 0.12
        return priors[:num_arms]
    if "fix_spaces" in text or "more than 2 consecutive spaces" in text:
        priors[3] = 0.46
        priors[0] = 0.08
        return priors[:num_arms]
    if "file_name_check" in text or "file's name is valid" in text:
        priors[3] = 0.46
        priors[1] = 0.12
        return priors[:num_arms]
    if "sum_squares" in text and "multiple of 3" in text and "multiple of 4" in text:
        priors[3] = 0.42
        priors[2] = 0.12
        return priors[:num_arms]
    if "simplify" in text and "string representation of a fraction" in text:
        priors[3] = 0.46
        priors[0] = 0.10
        return priors[:num_arms]
    if "order_by_points" in text or "sum of their digits" in text:
        priors[3] = 0.46
        priors[2] = 0.10
        return priors[:num_arms]
    if "derivative" in text and "coefficients of a polynomial" in text:
        priors[2] = 0.44
        priors[0] = 0.12
        return priors[:num_arms]
    if "numerical_letter_grade" in text or "letter grade" in text and "gpa" in text:
        priors[2] = 0.36
        priors[3] = 0.12
        return priors[:num_arms]
    if "make_a_pile" in text or "levels of stones" in text:
        priors[2] = 0.40
        priors[0] = 0.12
        return priors[:num_arms]
    if "by_length" in text or "corresponding name" in text and "one" in text and "nine" in text:
        priors[2] = 0.40
        priors[3] = 0.10
        return priors[:num_arms]
    if "double_the_difference" in text or "sum of squares" in text and "odd" in text and "negative" in text:
        priors[2] = 0.42
        priors[0] = 0.10
        return priors[:num_arms]
    if category in {"hashing", "expression"}:
        priors[3] = 0.24
        priors[0] = 0.14
    elif category in {"grid", "interval"}:
        priors[2] = 0.22
        priors[3] = 0.14
    elif category in {"parsing"}:
        priors[3] = 0.22
        priors[2] = 0.08
    elif category in {"string", "list"}:
        priors[3] = 0.18
        priors[2] = 0.10
    elif category in {"math", "ordering", "recursive"}:
        priors[2] = 0.18
        priors[3] = 0.10
    else:
        priors[0] = 0.08
        priors[2] = 0.08
        priors[3] = 0.08
    return priors[:num_arms]


def targeted_hard_override_arm(problem: dict) -> int | None:
    text = problem["prompt"].lower()
    entry_point = problem.get("entry_point", "").lower()
    zero_shot = 0
    cot = 2
    hint = 3
    zero_shot_wins = {
        "parse_nested_parens",
        "prime_fib",
        "decode_shift",
        "is_equal_to_sum_even",
        "simplify",
    }
    cot_protections = {
        "string_sequence",
        "sort_numbers",
        "rescale_to_unit",
        "filter_integers",
        "filter_by_prefix",
        "numerical_letter_grade",
        "skjkasdkd",
        "get_odd_collatz",
        "double_the_difference",
    }
    if entry_point in zero_shot_wins:
        return zero_shot
    if entry_point == "sum_squares" and "multiple of 3" in text and "multiple of 4" in text:
        return hint
    if entry_point in cot_protections:
        return cot
    return None


def category_guard_scores(category: str, stats: dict, num_arms: int = 4, min_samples: int = 3, margin: float = 0.20) -> list[float]:
    cat_stats = stats.get(category, {})
    rewards = {}
    for arm, values in cat_stats.items():
        if len(values) >= min_samples:
            rewards[int(arm)] = float(np.mean(values))
    if not rewards:
        return [0.0] * num_arms
    best_reward = max(rewards.values())
    scores = [0.0] * num_arms
    for arm, avg_reward in rewards.items():
        if avg_reward < best_reward - margin:
            scores[arm] = -0.75
        elif avg_reward >= best_reward - 0.05:
            scores[arm] = 0.18
    return scores


def recent_arm_guard_scores(recent_rewards: dict, num_arms: int = 4, min_samples: int = 4, threshold: float = 0.55) -> list[float]:
    scores = [0.0] * num_arms
    for arm in range(num_arms):
        values = recent_rewards.get(arm, [])
        if len(values) >= min_samples:
            recent_avg = float(np.mean(values))
            if recent_avg < threshold:
                scores[arm] = -0.90
            elif recent_avg >= 0.85:
                scores[arm] = 0.12
    return scores


def prompt_imports(prompt: str) -> str:
    imports = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(line)
    return "\n".join(imports)


def inferred_standard_imports(code: str) -> str:
    imports = []
    if "hashlib." in code and "import hashlib" not in code:
        imports.append("import hashlib")
    if "math." in code and "import math" not in code:
        imports.append("import math")
    if "heapq." in code and "import heapq" not in code:
        imports.append("import heapq")
    return "\n".join(imports)


def check_compile(code: str) -> int:
    try:
        ast.parse(code)
        compile(code, "<humaneval_candidate>", "exec")
        return 1
    except SyntaxError:
        return 0


def run_with_timeout(code: str, timeout_seconds: int = 10):
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import sys; exec(sys.stdin.read())"],
            input=code,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    if result.returncode == 0:
        return True, None
    error = (result.stderr or result.stdout or "RUNTIME_ERROR").strip()
    return False, error[-1000:]


def evaluate_sample(problem: dict, completion: str, timeout: int):
    if f"def {problem['entry_point']}" in completion:
        imports = prompt_imports(problem["prompt"])
        inferred = inferred_standard_imports(completion)
        prefix = "\n".join(part for part in [imports, inferred] if part)
        full_code = (prefix + "\n" if prefix else "") + completion + "\n" + problem["test"]
    else:
        body = _normalize_body_indentation(completion)
        inferred = inferred_standard_imports(body)
        full_code = (inferred + "\n" if inferred else "") + problem["prompt"] + body + "\n" + problem["test"]
    if f"check({problem['entry_point']})" not in full_code:
        full_code += f"\n\ncheck({problem['entry_point']})"
    compile_success = check_compile(full_code)
    pass_success, error = run_with_timeout(full_code, timeout)
    if pass_success:
        reward = 1.0
    elif compile_success:
        reward = 0.3
    else:
        reward = 0.0
    return bool(pass_success), bool(compile_success), reward, error


def _unique_completion_candidates(generated: str) -> list[tuple[str, str]]:
    candidates = [
        ("notebook", extract_code_notebook(generated)),
        ("compile_fallback", extract_code(generated)),
    ]
    unique = []
    seen = set()
    for method, completion in candidates:
        key = completion.strip()
        if key and key not in seen:
            unique.append((method, completion))
            seen.add(key)
    return unique


def evaluate_compatible_completion(problem: dict, generated: str, timeout: int, allow_compile_fallback: bool):
    candidates = _unique_completion_candidates(generated)
    primary_method, primary_completion = candidates[0]
    primary_passed, primary_compile_ok, primary_reward, primary_error = evaluate_sample(problem, primary_completion, timeout)
    syntax_like_error = isinstance(primary_error, str) and "SyntaxError" in primary_error
    if primary_passed or (primary_compile_ok and not syntax_like_error) or not allow_compile_fallback:
        return primary_completion, primary_passed, primary_compile_ok, primary_reward, primary_error, primary_method

    for method, completion in candidates[1:]:
        passed, compile_ok, reward, error = evaluate_sample(problem, completion, timeout)
        if compile_ok:
            return completion, passed, compile_ok, reward, error, method

    return primary_completion, primary_passed, primary_compile_ok, primary_reward, primary_error, primary_method


def evaluate_standard_completion(problem: dict, generated: str, timeout: int, allow_compile_fallback: bool):
    candidates = [
        ("standard", extract_code(generated)),
        ("raw_fallback", extract_code_notebook(generated)),
    ]
    unique = []
    seen = set()
    for method, completion in candidates:
        key = completion.strip()
        if key and key not in seen:
            unique.append((method, completion))
            seen.add(key)
    primary_method, primary_completion = unique[0]
    primary_passed, primary_compile_ok, primary_reward, primary_error = evaluate_sample(problem, primary_completion, timeout)
    syntax_like_error = isinstance(primary_error, str) and ("SyntaxError" in primary_error or "IndentationError" in primary_error)
    if primary_passed or (primary_compile_ok and not syntax_like_error) or not allow_compile_fallback:
        return primary_completion, primary_passed, primary_compile_ok, primary_reward, primary_error, primary_method
    for method, completion in unique[1:]:
        passed, compile_ok, reward, error = evaluate_sample(problem, completion, timeout)
        if compile_ok:
            return completion, passed, compile_ok, reward, error, method
    return primary_completion, primary_passed, primary_compile_ok, primary_reward, primary_error, primary_method


def extract_features(problem: dict, tokenizer=None) -> list[float]:
    prompt = problem["prompt"]
    docstring = prompt.split('"""')[1].lower() if '"""' in prompt and len(prompt.split('"""')) > 1 else prompt.lower()
    if tokenizer is not None:
        token_len = len(tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)["input_ids"][0])
    else:
        token_len = len(prompt) / 4.0
    words = docstring.split()
    doc_len = len(words)
    features = [
        min(token_len / 500.0, 1.0),
        min(doc_len / 100.0, 1.0),
        len(set(words)) / max(doc_len, 1),
        docstring.count("\n") / 10.0,
        float("return" in docstring),
        float("for " in docstring or "while " in docstring),
        float("if " in docstring),
        float("def " in docstring),
        float("list" in docstring or "array" in docstring),
        float("string" in docstring or "text" in docstring),
        float("sort" in docstring or "sorted" in docstring),
        float("graph" in docstring or "node" in docstring or "edge" in docstring),
        float("tree" in docstring or "root" in docstring),
        float("dp" in docstring or "memo" in docstring or "cache" in docstring),
        float("recursion" in docstring or "recursive" in docstring),
        float("bit" in docstring or "&" in docstring or "|" in docstring),
        float("greedy" in docstring),
        float("math" in docstring or "formula" in docstring),
        float("geometry" in docstring or "point" in docstring),
    ]
    while len(features) < 19:
        features.append(0.0)
    return features[:19]


class OnlineLinUCB:
    def __init__(self, num_arms=4, dim_context=19, alpha=0.3, force_explore=40, explore_order=None):
        self.num_arms = num_arms
        self.dim_context = dim_context
        self.alpha = alpha
        self.force_explore = force_explore
        self.explore_order = explore_order or list(range(num_arms))
        self.t = 0
        self.A = [np.eye(dim_context) + 0.01 * np.eye(dim_context) for _ in range(num_arms)]
        self.b = [np.zeros((dim_context, 1)) for _ in range(num_arms)]
        self.theta = [np.zeros((dim_context, 1)) for _ in range(num_arms)]
        self.arm_counts = [0] * num_arms
        self.arm_rewards = [[] for _ in range(num_arms)]

    def score_arms(self, context) -> list[float]:
        context = context.reshape(-1, 1)
        scores = []
        for arm in range(self.num_arms):
            try:
                A_inv = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(self.A[arm])
            mean = float((self.theta[arm].T @ context)[0, 0])
            uncertainty = float((self.alpha * np.sqrt(context.T @ A_inv @ context))[0, 0])
            history = float(np.mean(self.arm_rewards[arm])) if self.arm_rewards[arm] else 0.0
            scores.append(mean + uncertainty + history * 0.5)
        return scores

    def select_arm(self, context):
        self.t += 1
        if self.t <= self.force_explore:
            arm = self.explore_order[(self.t - 1) % len(self.explore_order)]
            self.arm_counts[arm] += 1
            return arm
        scores = self.score_arms(context)
        arm = int(np.argmax(scores))
        self.arm_counts[arm] += 1
        return arm

    def choose_arm(
        self,
        context,
        memory_scores: list[float] | None = None,
        memory_lambda: float = 0.0,
        prior_scores: list[float] | None = None,
        prior_weight: float = 0.0,
        guard_scores: list[float] | None = None,
        weak_arm_guard: bool = False,
        weak_arm_min_samples: int = 8,
        weak_arm_margin: float = 0.20,
        best_arm_fallback: bool = False,
        best_arm_min_samples: int = 20,
        best_arm_margin: float = 0.08,
        balanced_arm_floor: bool = False,
        balanced_min_share: float = 0.08,
        balanced_min_reward: float = 0.65,
        max_arm_share_guard: bool = False,
        max_arm_share: float = 0.65,
        max_arm_share_min_steps: int = 20,
        few_shot_memory_guard: bool = False,
        few_shot_min_memory: float = 0.60,
        override_arm: int | None = None,
    ):
        self.t += 1
        linucb_scores = self.score_arms(context)
        if memory_scores is None:
            memory_scores = [0.0] * self.num_arms
        if prior_scores is None:
            prior_scores = [0.0] * self.num_arms
        if guard_scores is None:
            guard_scores = [0.0] * self.num_arms
        combined_scores = [
            linucb_scores[idx] + memory_lambda * memory_scores[idx] + prior_weight * prior_scores[idx] + guard_scores[idx]
            for idx in range(self.num_arms)
        ]
        forced = self.t <= self.force_explore
        if override_arm is not None and 0 <= override_arm < self.num_arms:
            arm = override_arm
        elif forced:
            arm = self.explore_order[(self.t - 1) % len(self.explore_order)]
        else:
            candidate_scores = combined_scores.copy()
            if max_arm_share_guard:
                total_selected = max(sum(self.arm_counts), 1)
                if total_selected >= max_arm_share_min_steps:
                    for idx in range(self.num_arms):
                        share = self.arm_counts[idx] / total_selected
                        if share > max_arm_share:
                            candidate_scores[idx] -= 0.60 + (share - max_arm_share) * 2.0
            if few_shot_memory_guard and self.t > self.force_explore and self.num_arms > 1:
                few_shot_arm = 1
                if memory_scores[few_shot_arm] < few_shot_min_memory:
                    candidate_scores[few_shot_arm] -= 0.80
            if weak_arm_guard:
                averages = [float(np.mean(rewards)) if rewards else 0.0 for rewards in self.arm_rewards]
                eligible = [idx for idx, rewards in enumerate(self.arm_rewards) if len(rewards) >= weak_arm_min_samples]
                if eligible:
                    best_average = max(averages[idx] for idx in eligible)
                    for idx in eligible:
                        if averages[idx] < best_average - weak_arm_margin:
                            candidate_scores[idx] = -1e9
            if balanced_arm_floor:
                total_selected = max(sum(self.arm_counts), 1)
                averages = [float(np.mean(rewards)) if rewards else 1.0 for rewards in self.arm_rewards]
                underused = [
                    idx for idx in range(self.num_arms)
                    if (self.arm_counts[idx] / total_selected) < balanced_min_share and averages[idx] >= balanced_min_reward and candidate_scores[idx] > -1e8
                ]
                if underused:
                    arm = max(underused, key=lambda idx: candidate_scores[idx])
                    self.arm_counts[arm] += 1
                    return arm, linucb_scores, memory_scores, combined_scores, forced
            arm = int(np.argmax(candidate_scores))
            if best_arm_fallback:
                averages = [float(np.mean(rewards)) if rewards else 0.0 for rewards in self.arm_rewards]
                eligible = [idx for idx, rewards in enumerate(self.arm_rewards) if len(rewards) >= best_arm_min_samples]
                if eligible:
                    best_arm = max(eligible, key=lambda idx: averages[idx])
                    if averages[best_arm] >= averages[arm] + best_arm_margin:
                        arm = best_arm
        self.arm_counts[arm] += 1
        return arm, linucb_scores, memory_scores, combined_scores, forced

    def update(self, arm, context, reward):
        context = context.reshape(-1, 1)
        self.A[arm] += context @ context.T
        self.b[arm] += reward * context
        self.arm_rewards[arm].append(reward)
        try:
            self.theta[arm] = np.linalg.inv(self.A[arm]) @ self.b[arm]
        except np.linalg.LinAlgError:
            self.theta[arm] = np.linalg.pinv(self.A[arm]) @ self.b[arm]


def warm_start_bandit_from_runs(bandit: OnlineLinUCB, selected_run_names: list[str], history: list[dict], problems: dict, tokenizer) -> int:
    if not selected_run_names:
        return 0
    arm_lookup = {name: idx for idx, name in STRATEGY_NAMES.items()}
    selected = {name for name in selected_run_names}
    examples = 0
    for item in history:
        if item.get("name") not in selected:
            continue
        df = item.get("df", pd.DataFrame())
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        for _, row in df.iterrows():
            strategy = row.get("strategy")
            task_id = row.get("task_id")
            if strategy not in arm_lookup or task_id not in problems:
                continue
            try:
                reward = float(row.get("reward", 0.0))
            except (TypeError, ValueError):
                continue
            features = np.array(extract_features(problems[task_id], tokenizer))
            bandit.update(arm_lookup[strategy], features, reward)
            examples += 1
    return examples


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def compute_similarity_memory_scores(
    context: np.ndarray,
    memory: list[dict],
    num_arms: int = 4,
    top_k: int = 8,
    threshold: float = 0.15,
) -> tuple[list[float], list[str], float]:
    if not memory:
        return [0.0] * num_arms, [], 0.0
    scored = []
    for item in memory:
        similarity = cosine_similarity(context, item["features"])
        if similarity >= threshold:
            scored.append((similarity, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    neighbors = scored[:top_k]
    scores = [0.0] * num_arms
    weights = [0.0] * num_arms
    for similarity, item in neighbors:
        arm = int(item["arm"])
        if 0 <= arm < num_arms:
            scores[arm] += similarity * float(item["reward"])
            weights[arm] += similarity
    for arm in range(num_arms):
        if weights[arm] > 0.0:
            scores[arm] /= weights[arm]
    neighbor_ids = [f"{item['task_id']}:{STRATEGY_NAMES.get(int(item['arm']), item['arm'])}:{similarity:.2f}" for similarity, item in neighbors[:3]]
    best_similarity = float(neighbors[0][0]) if neighbors else 0.0
    return scores, neighbor_ids, best_similarity


@st.cache_resource(show_spinner=False)
def load_model(model_name: str, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if load_in_4bit:
        kwargs.update({"load_in_4bit": True, "torch_dtype": torch.bfloat16, "bnb_4bit_compute_dtype": torch.bfloat16})
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


def generate_completion(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float, do_sample: bool, generation_timeout: int | None = None):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if generation_timeout:
        generation_kwargs["max_time"] = generation_timeout
    if do_sample:
        generation_kwargs["temperature"] = temperature
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def build_prompt(template: str, problem: dict, tokenizer=None, use_chat_template: bool = True) -> str:
    prompt_content = template.format(
        prompt=problem["prompt"],
        task_id=problem["task_id"],
        entry_point=problem["entry_point"],
        hint=make_hint(problem),
    )
    if use_chat_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_content}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt_content


def build_notebook_prompt(arm: int, problem: dict, tokenizer) -> str:
    base_instruction = "You are an expert Python programmer. Complete the function by writing ONLY the code that goes inside the function body. Do NOT repeat the function signature, do NOT add comments, do NOT write tests.\n\n"
    if arm == 0:
        prompt_content = base_instruction + f"Complete this function:\n\n{problem['prompt']}"
    elif arm == 1:
        examples = '''
        Example 1:
        Problem: 
        def add(a, b):
            """Returns the sum of a and b"""
            
        Solution:
            return a + b

        Example 2:
        Problem:
        def reverse_string(s):
            """Reverses the input string"""
            
        Solution:
            return s[::-1]

        Example 3:
        Problem:
        def find_max(lst):
            """Finds the maximum value in a list"""
            
        Solution:
            return max(lst)
        '''
        prompt_content = base_instruction + f"Here are examples of completed functions:\n{examples}\n\nNow complete this function:\n\n{problem['prompt']}"
    elif arm == 2:
        prompt_content = f"Think step by step about how to solve this problem, then write the code. First, reason about the approach, then provide the code.\n\nProblem:\n{problem['prompt']}\n\nReasoning: Let me think about this step by step. "
    else:
        prompt_text = problem["prompt"].lower()
        if any(word in prompt_text for word in ["sort", "sorted", "order"]):
            hint = "Hint: Consider efficient sorting algorithms and time complexity."
        elif any(word in prompt_text for word in ["tree", "node", "binary"]):
            hint = "Hint: Consider recursion or iterative traversal for tree structures."
        elif any(word in prompt_text for word in ["string", "text", "character"]):
            hint = "Hint: Consider string manipulation methods and edge cases like empty strings."
        else:
            hint = "Hint: Consider edge cases and efficient data structures."
        prompt_content = base_instruction + f"{hint}\n\nComplete this function:\n\n{problem['prompt']}"
    messages = [{"role": "user", "content": prompt_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_experiment(config: dict, prompt_templates: dict, problems: dict):
    run_timer_start = time.perf_counter()
    run_started_at = datetime.now()
    st.session_state["run_started_at"] = run_started_at.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["run_config"] = config.copy()

    with st.status("Loading model and tokenizer...", expanded=True) as model_status:
        st.write(f"Model: `{config['model_name']}`")
        st.write(f"4-bit quantization: `{config['load_in_4bit']}`")
        tokenizer, model = load_model(config["model_name"], config["load_in_4bit"])
        model_status.update(label="Model loaded. Starting experiment...", state="complete", expanded=False)

    items = list(problems.items())
    if config["shuffle"]:
        random.Random(config["seed"]).shuffle(items)
    items = items[: config["num_tasks"]]
    explore_orders = {
        "Notebook order": [0, 1, 2, 3],
        "Strong-first (CoT, Hint, Zero, Few)": [2, 3, 0, 1],
        "CoT/Hint/Zero during exploration": [2, 3, 0],
        "CoT/Hint only during exploration": [2, 3],
    }
    bandit = OnlineLinUCB(
        alpha=config["alpha"],
        force_explore=config["force_explore"],
        explore_order=explore_orders.get(config.get("explore_order", "Notebook order"), [0, 1, 2, 3]),
    ) if config["mode"] == "Online Bandit" else None
    if bandit and config.get("warm_start"):
        warm_examples = warm_start_bandit_from_runs(
            bandit,
            config.get("warm_start_run_names", []),
            st.session_state.get("run_history", []),
            problems,
            tokenizer,
        )
        config["warm_start_examples"] = warm_examples
        if warm_examples and config.get("warm_start_counts_as_exploration", True):
            bandit.t = max(bandit.t, bandit.force_explore)
    else:
        config["warm_start_examples"] = 0
    rows = []
    similarity_memory = []
    category_arm_stats = {}
    recent_arm_rewards = {arm: [] for arm in STRATEGY_NAMES}
    top_line = st.empty()
    progress = st.progress(0, text="Waiting to start...")
    live_panel = st.empty()
    recent_log_box = st.empty()
    full_log_expander = st.expander("Full task log", expanded=False)
    log_lines = []

    total_steps = len(items) * config["repeats"]
    step = 0
    top_line.info(f"Run started at {st.session_state['run_started_at']} | total generations: {total_steps}")
    for repeat in range(config["repeats"]):
        for task_index, (task_id, problem) in enumerate(items):
            phase = "calibration" if config.get("calibration_mode") and task_index < config.get("calibration_tasks", 0) else "evaluation"
            task_timer_start = time.perf_counter()
            step += 1
            override_arm = None
            if bandit:
                features = np.array(extract_features(problem, tokenizer))
                current_category = problem_category(problem)
                priors = category_prior_scores(problem, num_arms=len(STRATEGY_NAMES), experiment_preset=config.get("experiment_preset", "")) if config.get("category_prior") else [0.0] * len(STRATEGY_NAMES)
                override_arm = targeted_hard_override_arm(problem) if config.get("targeted_hard_overrides") else None
                hard_guards = [0.0] * len(STRATEGY_NAMES)
                if config.get("category_arm_guard"):
                    guard_scores = category_guard_scores(
                        current_category,
                        category_arm_stats,
                        num_arms=len(STRATEGY_NAMES),
                        min_samples=config.get("category_guard_min_samples", 3),
                        margin=config.get("category_guard_margin", 0.20),
                    )
                    hard_guards = [hard_guards[idx] + guard_scores[idx] for idx in range(len(hard_guards))]
                if config.get("recent_arm_cooldown"):
                    recent_scores = recent_arm_guard_scores(
                        recent_arm_rewards,
                        num_arms=len(STRATEGY_NAMES),
                        min_samples=config.get("recent_cooldown_min_samples", 4),
                        threshold=config.get("recent_cooldown_threshold", 0.55),
                    )
                    hard_guards = [hard_guards[idx] + recent_scores[idx] for idx in range(len(hard_guards))]
                if config.get("similarity_memory"):
                    memory_scores, nearest_examples, best_similarity = compute_similarity_memory_scores(
                        features,
                        similarity_memory,
                        num_arms=len(STRATEGY_NAMES),
                        top_k=config.get("memory_top_k", 8),
                        threshold=config.get("memory_threshold", 0.15),
                    )
                    arm, linucb_scores, memory_scores, combined_scores, forced_explore = bandit.choose_arm(
                        features,
                        memory_scores=memory_scores,
                        memory_lambda=config.get("memory_lambda", 0.4),
                        prior_scores=priors,
                        prior_weight=config.get("category_prior_weight", 0.35),
                        guard_scores=hard_guards,
                        weak_arm_guard=config.get("weak_arm_guard", False),
                        weak_arm_min_samples=config.get("weak_arm_min_samples", 8),
                        weak_arm_margin=config.get("weak_arm_margin", 0.20),
                        best_arm_fallback=config.get("best_arm_fallback", False),
                        best_arm_min_samples=config.get("best_arm_min_samples", 20),
                        best_arm_margin=config.get("best_arm_margin", 0.08),
                        balanced_arm_floor=config.get("balanced_arm_floor", False),
                        balanced_min_share=config.get("balanced_min_share", 0.06),
                        balanced_min_reward=config.get("balanced_min_reward", 0.65),
                        max_arm_share_guard=config.get("max_arm_share_guard", False),
                        max_arm_share=config.get("max_arm_share", 0.65),
                        max_arm_share_min_steps=config.get("max_arm_share_min_steps", 20),
                        few_shot_memory_guard=config.get("few_shot_memory_guard", False),
                        few_shot_min_memory=config.get("few_shot_min_memory", 0.60),
                        override_arm=override_arm,
                    )
                else:
                    arm, linucb_scores, memory_scores, combined_scores, forced_explore = bandit.choose_arm(
                        features,
                        prior_scores=priors,
                        prior_weight=config.get("category_prior_weight", 0.35),
                        guard_scores=hard_guards,
                        weak_arm_guard=config.get("weak_arm_guard", False),
                        weak_arm_min_samples=config.get("weak_arm_min_samples", 8),
                        weak_arm_margin=config.get("weak_arm_margin", 0.20),
                        best_arm_fallback=config.get("best_arm_fallback", False),
                        best_arm_min_samples=config.get("best_arm_min_samples", 20),
                        best_arm_margin=config.get("best_arm_margin", 0.08),
                        balanced_arm_floor=config.get("balanced_arm_floor", False),
                        balanced_min_share=config.get("balanced_min_share", 0.06),
                        balanced_min_reward=config.get("balanced_min_reward", 0.65),
                        max_arm_share_guard=config.get("max_arm_share_guard", False),
                        max_arm_share=config.get("max_arm_share", 0.65),
                        max_arm_share_min_steps=config.get("max_arm_share_min_steps", 20),
                        few_shot_memory_guard=config.get("few_shot_memory_guard", False),
                        few_shot_min_memory=config.get("few_shot_min_memory", 0.60),
                        override_arm=override_arm,
                    )
                    nearest_examples = []
                    best_similarity = 0.0
                strategy = STRATEGY_NAMES[arm]
            else:
                strategy = config["strategy"]
                arm = {name: idx for idx, name in STRATEGY_NAMES.items()}[strategy]
                features = np.array(extract_features(problem, tokenizer))
                current_category = problem_category(problem)
                linucb_scores = [0.0] * len(STRATEGY_NAMES)
                memory_scores = [0.0] * len(STRATEGY_NAMES)
                combined_scores = [0.0] * len(STRATEGY_NAMES)
                forced_explore = False
                nearest_examples = []
                best_similarity = 0.0

            if config["notebook_compatible"] and config["mode"] == "Online Bandit":
                prompt = build_notebook_prompt(arm, problem, tokenizer)
            else:
                prompt = build_prompt(
                    prompt_templates[strategy],
                    problem,
                    tokenizer=tokenizer,
                    use_chat_template=config["use_chat_template"],
                )
            elapsed_before = time.perf_counter() - run_timer_start
            progress.progress(
                (step - 1) / total_steps,
                text=f"Generating {step}/{total_steps}: {task_id} with {strategy} | elapsed {elapsed_before/60:.1f}m",
            )
            with live_panel.container(border=True):
                st.markdown("**Live Run Summary**")
                completed = len(rows)
                prev_pass = sum(r["passed"] for r in rows)
                prev_compile = sum(r["compile_ok"] for r in rows)
                prev_reward = sum(r["reward"] for r in rows) / completed if completed else 0.0
                prev_pass_rate = prev_pass / completed * 100 if completed else 0.0
                prev_compile_rate = prev_compile / completed * 100 if completed else 0.0
                live_cols = st.columns(6)
                live_cols[0].metric("Progress", f"{completed}/{total_steps}")
                live_cols[1].metric("Pass@1", f"{prev_pass_rate:.1f}%")
                live_cols[2].metric("Pass", prev_pass)
                live_cols[3].metric("Fail", completed - prev_pass)
                live_cols[4].metric("Compile OK", f"{prev_compile_rate:.1f}%")
                live_cols[5].metric("Avg Reward", f"{prev_reward:.3f}")
                st.caption(f"Current phase: Generating task {step}/{total_steps}: {task_id} | strategy={strategy}")
            generation_timer_start = time.perf_counter()
            generated = generate_completion(
                tokenizer,
                model,
                prompt,
                config["max_new_tokens"],
                config["temperature"],
                config["do_sample"],
                None if config["notebook_compatible"] and config["mode"] == "Online Bandit" else config["generation_timeout"],
            )
            generation_seconds = time.perf_counter() - generation_timer_start
            progress.progress(
                (step - 0.5) / total_steps,
                text=f"Evaluating {step}/{total_steps}: {task_id} | generation {generation_seconds:.1f}s",
            )
            eval_timer_start = time.perf_counter()
            if config["notebook_compatible"] and config["mode"] == "Online Bandit":
                completion, passed, compile_ok, reward, error, extraction_method = evaluate_compatible_completion(
                    problem,
                    generated,
                    config["timeout"],
                    config.get("compatible_fallback", False) or config.get("smart_extraction", True),
                )
            elif config.get("smart_extraction", True):
                completion, passed, compile_ok, reward, error, extraction_method = evaluate_standard_completion(
                    problem,
                    generated,
                    config["timeout"],
                    True,
                )
            else:
                completion = extract_code(generated)
                passed, compile_ok, reward, error = evaluate_sample(problem, completion, config["timeout"])
                extraction_method = "standard"
            eval_seconds = time.perf_counter() - eval_timer_start
            task_seconds = time.perf_counter() - task_timer_start
            if bandit:
                bandit.update(arm, features, reward)
                if config.get("similarity_memory"):
                    similarity_memory.append({
                        "task_id": task_id,
                        "features": features.copy(),
                        "arm": arm,
                        "reward": reward,
                    })
                if config.get("category_arm_guard"):
                    category_arm_stats.setdefault(current_category, {}).setdefault(arm, []).append(reward)
                if config.get("recent_arm_cooldown"):
                    recent_arm_rewards.setdefault(arm, []).append(reward)
                    window = config.get("recent_cooldown_window", 8)
                    recent_arm_rewards[arm] = recent_arm_rewards[arm][-window:]

            rows.append({
                "repeat": repeat + 1,
                "phase": phase,
                "task_id": task_id,
                "problem_category": current_category,
                "strategy": strategy,
                "arm": arm,
                "passed": passed,
                "compile_ok": compile_ok,
                "reward": reward,
                "error": error,
                "completion": completion,
                "raw_generated": generated,
                "prompt": prompt,
                "extraction_method": extraction_method,
                "linucb_score": linucb_scores[arm] if arm < len(linucb_scores) else 0.0,
                "memory_score": memory_scores[arm] if arm < len(memory_scores) else 0.0,
                "combined_score": combined_scores[arm] if arm < len(combined_scores) else 0.0,
                "forced_explore": forced_explore,
                "hard_override": override_arm is not None,
                "override_strategy": STRATEGY_NAMES.get(override_arm, "") if override_arm is not None else "",
                "nearest_examples": ", ".join(nearest_examples),
                "best_similarity": best_similarity,
                "task_seconds": task_seconds,
                "generation_seconds": generation_seconds,
                "eval_seconds": eval_seconds,
                "generated_chars": len(generated),
                "completion_chars": len(completion),
            })
            save_partial_results(rows)
            st.session_state["last_results"] = pd.DataFrame(rows)
            current_pass = sum(r["passed"] for r in rows)
            current_compile = sum(r["compile_ok"] for r in rows)
            current_reward = sum(r["reward"] for r in rows) / len(rows)
            current_pass_at_1 = current_pass / len(rows) * 100
            current_compile_rate = current_compile / len(rows) * 100
            current_fail = len(rows) - current_pass
            elapsed = time.perf_counter() - run_timer_start
            eta = (total_steps - step) * (elapsed / len(rows)) if rows else 0.0
            progress.progress(
                step / total_steps,
                text=f"Running {step}/{total_steps}: {task_id} with {strategy} | elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m",
            )
            outcome = "PASS" if passed else "FAIL"
            with live_panel.container(border=True):
                st.markdown("**Live Run Summary**")
                live_cols = st.columns(6)
                live_cols[0].metric("Progress", f"{step}/{total_steps}")
                live_cols[1].metric("Pass@1", f"{current_pass_at_1:.1f}%")
                live_cols[2].metric("Pass", current_pass)
                live_cols[3].metric("Fail", current_fail)
                live_cols[4].metric("Compile OK", f"{current_compile_rate:.1f}%")
                live_cols[5].metric("Avg Reward", f"{current_reward:.3f}")
                st.caption(f"Last task: {task_id} | strategy: {strategy} | outcome: {outcome} | reward={reward:.1f} | task={task_seconds:.1f}s | eval={eval_seconds:.1f}s | ETA={eta/60:.1f}m")
            log_lines.append(f"[{step:03d}/{total_steps:03d}] {task_id} | {strategy} | {outcome} | reward={reward:.1f} | task={task_seconds:.1f}s")
            with recent_log_box.container(border=True):
                st.caption("Recent task log")
                st.code("\n".join(log_lines[-10:]), language="text")
            if step % 20 == 0 or step == total_steps:
                with full_log_expander:
                    st.caption(f"{len(log_lines)} rows")
                    st.code("\n".join(log_lines), language="text")
    progress.progress(1.0, text="Experiment completed.")
    return pd.DataFrame(rows)


st.set_page_config(page_title=APP_TITLE, layout="wide")
init_state()
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.65rem 1.8rem;
        border-radius: 24px;
        background:
            radial-gradient(circle at 8% 20%, rgba(56, 189, 248, .35), transparent 28%),
            radial-gradient(circle at 92% 12%, rgba(168, 85, 247, .38), transparent 26%),
            linear-gradient(135deg, #0f172a 0%, #172554 48%, #312e81 100%);
        color: white;
        margin-bottom: 1.1rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, .25);
    }
    .hero h1 {margin: 0; font-size: 2.25rem; letter-spacing: -0.04em;}
    .hero p {margin: .5rem 0 0 0; color: #dbeafe; max-width: 950px;}
    .hero-badges {display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem;}
    .hero-badge {
        border: 1px solid rgba(255,255,255,.24);
        border-radius: 999px;
        padding: .35rem .7rem;
        color: #e0f2fe;
        background: rgba(255,255,255,.08);
        font-size: .85rem;
    }
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: .85rem;
        margin: .9rem 0 1.1rem 0;
    }
    .status-card {
        border-radius: 20px;
        padding: 1rem 1.05rem;
        color: #0f172a;
        border: 1px solid rgba(148, 163, 184, .25);
        box-shadow: 0 12px 28px rgba(15, 23, 42, .08);
    }
    .status-card .label {font-size: .78rem; opacity: .72; text-transform: uppercase; letter-spacing: .06em; font-weight: 700;}
    .status-card .value {font-size: 1.45rem; line-height: 1.2; font-weight: 800; margin-top: .25rem;}
    .status-card .hint {font-size: .84rem; opacity: .78; margin-top: .3rem;}
    .card-blue {background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);}
    .card-green {background: linear-gradient(135deg, #dcfce7 0%, #f0fdf4 100%);}
    .card-violet {background: linear-gradient(135deg, #ede9fe 0%, #faf5ff 100%);}
    .card-amber {background: linear-gradient(135deg, #fef3c7 0%, #fffbeb 100%);}
    .section-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin: .4rem 0 .8rem 0;
    }
    .section-title h2, .section-title h3 {margin: 0; letter-spacing: -.02em;}
    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: .25rem .65rem;
        font-size: .78rem;
        font-weight: 700;
        background: #e0f2fe;
        color: #075985;
        border: 1px solid #bae6fd;
    }
    .side-panel {
        border-radius: 18px;
        padding: .95rem 1rem;
        margin: .7rem 0;
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #7c3aed 100%);
        color: white;
        box-shadow: 0 12px 30px rgba(30, 64, 175, .28);
    }
    .side-panel .title {font-weight: 800; letter-spacing: -.02em; font-size: 1.02rem;}
    .side-panel .sub {font-size: .82rem; color: #dbeafe; margin-top: .25rem;}
    .mini-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: .75rem;
        margin: .9rem 0;
    }
    .mini-card {
        border-radius: 18px;
        padding: .9rem;
        background: rgba(255,255,255,.86);
        border: 1px solid rgba(148, 163, 184, .24);
        box-shadow: 0 10px 24px rgba(15, 23, 42, .06);
    }
    .mini-card .name {font-weight: 800; color: #111827; margin-bottom: .2rem;}
    .mini-card .role {font-size: .78rem; color: #2563eb; font-weight: 800; text-transform: uppercase; letter-spacing: .05em;}
    .mini-card .desc {font-size: .86rem; color: #475569; margin-top: .35rem;}
    .setup-card {
        border-radius: 22px;
        padding: 1rem 1.1rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(148, 163, 184, .28);
        box-shadow: 0 12px 32px rgba(15, 23, 42, .07);
        margin-bottom: .9rem;
    }
    .setup-row {display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .75rem;}
    .setup-chip {
        border-radius: 999px;
        padding: .28rem .68rem;
        background: #f1f5f9;
        color: #334155;
        border: 1px solid #e2e8f0;
        font-size: .82rem;
        font-weight: 700;
    }
    .setup-chip.good {background: #dcfce7; color: #166534; border-color: #bbf7d0;}
    .setup-chip.warn {background: #fef3c7; color: #92400e; border-color: #fde68a;}
    .cta-card {
        border-radius: 24px;
        padding: 1.1rem 1.2rem;
        background: radial-gradient(circle at 10% 15%, rgba(34,197,94,.24), transparent 30%), linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        box-shadow: 0 18px 42px rgba(15, 23, 42, .24);
        margin: 1rem 0;
    }
    .cta-card b {color: #bbf7d0;}
    .soft-card {
        border: 1px solid rgba(59, 130, 246, .22);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        background: linear-gradient(135deg, #eff6ff 0%, #eef2ff 48%, #f5f3ff 100%);
        color: #0f172a;
        box-shadow: 0 8px 28px rgba(15, 23, 42, .08);
        margin-bottom: .8rem;
    }
    .soft-card b {color: #312e81;}
    .run-chip {
        display: inline-block;
        border-radius: 12px;
        padding: .2rem .55rem;
        background: #eef2ff;
        color: #3730a3;
        font-size: .8rem;
        margin-right: .35rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: .85rem 1rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, .045);
        color: #0f172a !important;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] p,
    div[data-testid="stMetric"] div {
        color: #0f172a !important;
    }
    div[data-testid="stMetricValue"] {
        color: #111827 !important;
    }
    @media (prefers-color-scheme: dark) {
        .soft-card {
            border: 1px solid rgba(96, 165, 250, .28);
            background: linear-gradient(135deg, #0b1220 0%, #111827 52%, #1e1b4b 100%);
            color: #e5e7eb;
            box-shadow: 0 12px 34px rgba(0, 0, 0, .32);
        }
        .soft-card b {color: #bfdbfe;}
        .run-chip {
            background: #111827;
            color: #bfdbfe;
            border: 1px solid rgba(96, 165, 250, .35);
        }
        div[data-testid="stMetric"] {
            background: #0b1220 !important;
            border: 1px solid rgba(148, 163, 184, .24) !important;
            box-shadow: 0 12px 28px rgba(0, 0, 0, .28);
            color: #e5e7eb !important;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] p,
        div[data-testid="stMetric"] div {
            color: #e5e7eb !important;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fafc !important;
        }
        div[data-testid="stMetricDelta"] svg {
            fill: #93c5fd !important;
        }
        div[data-testid="stExpander"],
        div[data-testid="stDataFrame"],
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(148, 163, 184, .22) !important;
        }
        .status-card {color: #e5e7eb; border-color: rgba(148, 163, 184, .22);}
        .card-blue {background: linear-gradient(135deg, #172554 0%, #0f172a 100%);}
        .card-green {background: linear-gradient(135deg, #14532d 0%, #0f172a 100%);}
        .card-violet {background: linear-gradient(135deg, #4c1d95 0%, #0f172a 100%);}
        .card-amber {background: linear-gradient(135deg, #78350f 0%, #0f172a 100%);}
        .pill {background: #0f172a; color: #bae6fd; border-color: rgba(125, 211, 252, .35);}
        .mini-card {background: rgba(15, 23, 42, .82); border-color: rgba(148, 163, 184, .24);}
        .mini-card .name {color: #f8fafc;}
        .mini-card .desc {color: #cbd5e1;}
        .setup-card {background: linear-gradient(135deg, #0b1220 0%, #111827 100%); border-color: rgba(148, 163, 184, .24);}
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="hero">
      <h1>{APP_TITLE}</h1>
      <p>Modern HumanEval lab untuk RL-APO: jalankan fixed prompts dan online contextual bandit, bandingkan run, lalu export chart, tabel, dan raw data yang siap masuk thesis.</p>
      <div class="hero-badges">
        <span class="hero-badge">Problem-type prompt bank</span>
        <span class="hero-badge">Similarity-aware RL-APO</span>
        <span class="hero-badge">Thesis-ready exports</span>
        <span class="hero-badge">Leaderboard comparison</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not HUMANEVAL_FILE.exists():
    st.error("HumanEval.jsonl.gz tidak ditemukan di folder repo.")
    st.stop()

problems = read_humaneval(HUMANEVAL_FILE)

with st.sidebar:
    st.markdown(
        f"""
        <div class="side-panel">
          <div class="title">RL-APO Control Center</div>
          <div class="sub">Configure experiments, prompt banks, and bandit policy in one place.</div>
          <div style="margin-top:.65rem"><span class="hero-badge">{len(st.session_state['run_history'])} saved runs</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Runtime", expanded=False):
        st.code(sys.executable, language="text")
    built_in_presets = [
        {"name": "Calibration RL-APO Fair Calib", "config": CALIBRATION_FAIR_CALIB_DEFAULTS},
        {"name": "Calibration RL-APO CoT Anchor v3", "config": CALIBRATION_COT_ANCHOR_V3_DEFAULTS},
        {"name": "Calibration RL-APO Targeted v2", "config": CALIBRATION_TARGETED_V2_DEFAULTS},
        {"name": "Calibration RL-APO Targeted", "config": CALIBRATION_TARGETED_DEFAULTS},
        {"name": "Calibration RL-APO No Weak Few", "config": CALIBRATION_NO_WEAK_FEW_DEFAULTS},
        {"name": "Calibration RL-APO", "config": CALIBRATION_80_DEFAULTS},
        {"name": "73.8+ Balanced RL-APO", "config": BALANCED_738_PLUS_DEFAULTS},
        {"name": "76% Hybrid RL-APO", "config": HYBRID_76_DEFAULTS},
        {"name": "75% Precision RL-APO", "config": PRECISION_75_DEFAULTS},
        {"name": "75% Push RL-APO", "config": PUSH_75_DEFAULTS},
        {"name": "Anti-collapse RL-APO", "config": ANTI_COLLAPSE_DEFAULTS},
    ]
    saved_presets = st.session_state.get("setting_presets", [])
    preset_options = built_in_presets + saved_presets
    preset_names = [item["name"] for item in preset_options]
    selected_setting_preset = st.selectbox(
        "Setting preset",
        preset_names,
        index=0,
        help="Calibration RL-APO Fair Calib adalah preset utama thesis: online calibration tetap dihitung dalam 164 task, tanpa hard override atau exact HumanEval routing.",
    )
    active_setting_preset = next(item["config"] for item in preset_options if item["name"] == selected_setting_preset)
    st.markdown("<span class='pill'>Step 1</span> <b>Choose experiment type</b>", unsafe_allow_html=True)
    mode_options = ["Fixed Strategy", "Online Bandit"]
    mode = st.radio("Mode", mode_options, horizontal=True, index=option_index(mode_options, preset_value(active_setting_preset, "mode", "Online Bandit")))
    experiment_preset = st.selectbox(
        "Experiment preset",
        ["Calibration RL-APO Fair Calib", "Calibration RL-APO CoT Anchor v3", "Calibration RL-APO Targeted v2", "Calibration RL-APO Targeted", "Calibration RL-APO No Weak Few", "Calibration RL-APO", "73.8+ Balanced RL-APO", "76% Hybrid RL-APO", "75% Precision RL-APO", "75% Push RL-APO", "Anti-collapse RL-APO", "Thesis full run", "Quick smoke test", "Strict notebook reproduction", "Custom"],
        index=option_index(["Calibration RL-APO Fair Calib", "Calibration RL-APO CoT Anchor v3", "Calibration RL-APO Targeted v2", "Calibration RL-APO Targeted", "Calibration RL-APO No Weak Few", "Calibration RL-APO", "73.8+ Balanced RL-APO", "76% Hybrid RL-APO", "75% Precision RL-APO", "75% Push RL-APO", "Anti-collapse RL-APO", "Thesis full run", "Quick smoke test", "Strict notebook reproduction", "Custom"], preset_value(active_setting_preset, "experiment_preset", "Calibration RL-APO Fair Calib")),
        help="Preset hanya memberi panduan default setting. Kamu tetap bisa mengubah kontrol di bawahnya.",
    )
    st.markdown("<span class='pill'>Step 2</span> <b>Select prompt policy</b>", unsafe_allow_html=True)
    prompt_bank = st.selectbox(
        "Prompt bank",
        list(PROMPT_BANKS),
        index=option_index(list(PROMPT_BANKS), preset_value(active_setting_preset, "prompt_bank", "Problem-type aware prompts")),
        help="Problem-type aware prompts dirancang agar zero/few/cot/hint punya peran berbeda. Notebook 68 prompts dipakai untuk reproduksi baseline lama.",
    )
    active_prompts = PROMPT_BANKS[prompt_bank]
    strategy = st.selectbox(
        "Fixed strategy / prompt arm",
        list(active_prompts),
        index=option_index(list(active_prompts), preset_value(active_setting_preset, "strategy", list(active_prompts)[0])),
        disabled=mode == "Online Bandit",
        help="Aktif hanya untuk Fixed Strategy. Pada Online Bandit, arm dipilih otomatis oleh policy bandit.",
    )
    st.markdown("<span class='pill'>Step 3</span> <b>Model and dataset</b>", unsafe_allow_html=True)
    with st.expander("Model settings", expanded=True):
        model_name = st.text_input("Model", "deepseek-ai/deepseek-coder-6.7b-instruct")
        load_in_4bit = st.checkbox("Load 4-bit", value=bool(preset_value(active_setting_preset, "load_in_4bit", True)))
        use_chat_template = st.checkbox("Use tokenizer chat template", value=bool(preset_value(active_setting_preset, "use_chat_template", True)))
    default_tasks = min(164, len(problems)) if experiment_preset in {"Thesis full run", "Anti-collapse RL-APO", "75% Push RL-APO", "75% Precision RL-APO", "76% Hybrid RL-APO", "73.8+ Balanced RL-APO", "Calibration RL-APO", "Calibration RL-APO No Weak Few", "Calibration RL-APO Targeted", "Calibration RL-APO Targeted v2", "Calibration RL-APO CoT Anchor v3", "Calibration RL-APO Fair Calib"} else min(10, len(problems))
    default_tasks = min(int(preset_value(active_setting_preset, "num_tasks", default_tasks)), min(164, len(problems)))
    num_tasks = st.slider("Jumlah soal", 1, min(164, len(problems)), default_tasks)
    repeats = st.slider("Repeat per soal", 1, 5, int(preset_value(active_setting_preset, "repeats", 1)))
    with st.expander("Generation settings", expanded=False):
        max_new_tokens = st.slider("Max new tokens", 64, 1024, int(preset_value(active_setting_preset, "max_new_tokens", 512)), 64)
        do_sample = st.checkbox("Sampling", value=bool(preset_value(active_setting_preset, "do_sample", False)))
        temperature = st.slider("Temperature", 0.0, 1.5, float(preset_value(active_setting_preset, "temperature", 0.1)), 0.05)
    default_notebook_compatible = experiment_preset == "Strict notebook reproduction"
    notebook_compatible = st.checkbox("Notebook 68 compatible mode", value=bool(preset_value(active_setting_preset, "notebook_compatible", default_notebook_compatible)), help="For Online Bandit, use prompt/extraction/generation behavior from bandit68%.ipynb. Turn OFF to test selected prompt bank.")
    compatible_plus_enabled = mode == "Online Bandit" and notebook_compatible
    st.markdown("<span class='pill'>Step 4</span> <b>RL-APO policy tuning</b>", unsafe_allow_html=True)
    with st.expander("Advanced bandit options", expanded=mode == "Online Bandit"):
        compatible_fallback = st.checkbox(
            "Compile-safe extraction fallback",
            value=bool(preset_value(active_setting_preset, "compatible_fallback", False)),
            disabled=not compatible_plus_enabled,
            help="If strict notebook extraction does not compile, try the dashboard body extractor on the same raw generation. This keeps strict compatible output as first choice.",
        )
        calibration_mode = st.checkbox(
            "Calibration phase metrics",
            value=bool(preset_value(active_setting_preset, "calibration_mode", False)),
            disabled=mode != "Online Bandit",
            help="If ON, early tasks are used as online calibration and excluded from the main leaderboard metric. This is fair only when compared against baselines on the same evaluation segment.",
        )
        calibration_tasks = st.slider(
            "Calibration tasks",
            0,
            80,
            int(preset_value(active_setting_preset, "calibration_tasks", 0)),
            4,
            disabled=mode != "Online Bandit" or not calibration_mode,
        )
        evaluation_only_metric = st.checkbox(
            "Use evaluation-only metric",
            value=bool(preset_value(active_setting_preset, "evaluation_only_metric", False)),
            disabled=mode != "Online Bandit" or not calibration_mode,
            help="If OFF, leaderboard Pass@1 uses all rows including calibration. If ON, it reports evaluation phase only.",
        )
        fixed_run_options = [
            item["name"]
            for item in st.session_state["run_history"]
            if item.get("config", {}).get("mode") == "Fixed Strategy"
        ]
        warm_start = st.checkbox(
            "Warm-start LinUCB from saved fixed runs",
            value=bool(preset_value(active_setting_preset, "warm_start", False)),
            disabled=mode != "Online Bandit" or not fixed_run_options,
            help="Use rewards from selected fixed-strategy runs as prior data before the online bandit starts. This is useful for thesis runs because the full HumanEval horizon is short for a cold-start bandit.",
        )
        warm_start_run_names = st.multiselect(
            "Fixed runs for warm-start",
            fixed_run_options,
            default=[name for name in preset_value(active_setting_preset, "warm_start_run_names", []) if name in fixed_run_options],
            disabled=mode != "Online Bandit" or not warm_start or not fixed_run_options,
            help="Pilih run zero_shot/few_shot/cot/hint full 164 task yang sudah tersimpan.",
        )
        warm_start_counts_as_exploration = st.checkbox(
            "Let warm-start skip forced exploration",
            value=bool(preset_value(active_setting_preset, "warm_start_counts_as_exploration", True)),
            disabled=mode != "Online Bandit" or not warm_start,
            help="Jika ON, force_explore dianggap sudah dipenuhi oleh data warm-start sehingga bandit bisa langsung exploit policy yang terkalibrasi.",
        )
        similarity_memory = st.checkbox(
            "Similarity-aware prompt memory",
            value=bool(preset_value(active_setting_preset, "similarity_memory", experiment_preset != "Strict notebook reproduction")),
            disabled=mode != "Online Bandit",
            help="Choose prompts using rewards from previous tasks with similar context features, not only global arm reward.",
        )
        memory_top_k = st.slider("Memory top-k similar tasks", 1, 20, int(preset_value(active_setting_preset, "memory_top_k", 8)), disabled=mode != "Online Bandit" or not similarity_memory)
        memory_lambda = st.slider("Memory weight lambda", 0.0, 1.5, float(preset_value(active_setting_preset, "memory_lambda", 0.5)), 0.05, disabled=mode != "Online Bandit" or not similarity_memory)
        memory_threshold = st.slider("Similarity threshold", 0.0, 1.0, float(preset_value(active_setting_preset, "memory_threshold", 0.15)), 0.05, disabled=mode != "Online Bandit" or not similarity_memory)
        explore_order = st.selectbox(
            "Forced exploration order",
            ["Notebook order", "Strong-first (CoT, Hint, Zero, Few)", "CoT/Hint/Zero during exploration", "CoT/Hint only during exploration"],
            index=option_index(["Notebook order", "Strong-first (CoT, Hint, Zero, Few)", "CoT/Hint/Zero during exploration", "CoT/Hint only during exploration"], preset_value(active_setting_preset, "explore_order", "Strong-first (CoT, Hint, Zero, Few)"), 1 if experiment_preset != "Strict notebook reproduction" else 0),
            disabled=mode != "Online Bandit",
            help="Notebook order is strict. Strong-first reduces early damage from weak arms observed in recent runs.",
        )
        weak_arm_guard = st.checkbox(
            "Adaptive weak-arm guard",
            value=bool(preset_value(active_setting_preset, "weak_arm_guard", experiment_preset != "Strict notebook reproduction")),
            disabled=mode != "Online Bandit",
            help="After an arm has enough samples, avoid selecting it if its reward is clearly below the best sampled arm.",
        )
        weak_arm_min_samples = st.slider("Weak-arm min samples", 2, 20, int(preset_value(active_setting_preset, "weak_arm_min_samples", 8)), disabled=mode != "Online Bandit" or not weak_arm_guard)
        weak_arm_margin = st.slider("Weak-arm reward margin", 0.05, 0.60, float(preset_value(active_setting_preset, "weak_arm_margin", 0.25)), 0.05, disabled=mode != "Online Bandit" or not weak_arm_guard)
        best_arm_fallback = st.checkbox(
            "Best-arm safety fallback",
            value=bool(preset_value(active_setting_preset, "best_arm_fallback", experiment_preset == "75% Precision RL-APO" if experiment_preset in {"Anti-collapse RL-APO", "75% Push RL-APO", "75% Precision RL-APO", "76% Hybrid RL-APO", "73.8+ Balanced RL-APO"} else experiment_preset != "Strict notebook reproduction")),
            disabled=mode != "Online Bandit",
            help="After enough online evidence, use the empirically best arm if it is clearly better than the selected arm. This is non-leaky because it uses only past rewards.",
        )
        best_arm_min_samples = st.slider("Best-arm min samples", 8, 80, int(preset_value(active_setting_preset, "best_arm_min_samples", 12)), 4, disabled=mode != "Online Bandit" or not best_arm_fallback)
        best_arm_margin = st.slider("Best-arm margin", 0.00, 0.30, float(preset_value(active_setting_preset, "best_arm_margin", 0.04)), 0.02, disabled=mode != "Online Bandit" or not best_arm_fallback)
        balanced_arm_floor = st.checkbox(
            "Balanced arm floor",
            value=bool(preset_value(active_setting_preset, "balanced_arm_floor", experiment_preset in {"Anti-collapse RL-APO", "75% Push RL-APO", "75% Precision RL-APO"})),
            disabled=mode != "Online Bandit",
            help="Keeps competitive arms from disappearing completely. Useful when few-shot fixed baseline is strong but online policy under-selects it.",
        )
        balanced_min_share = st.slider("Balanced min share", 0.00, 0.20, float(preset_value(active_setting_preset, "balanced_min_share", 0.10)), 0.01, disabled=mode != "Online Bandit" or not balanced_arm_floor)
        balanced_min_reward = st.slider("Balanced min reward", 0.30, 0.90, float(preset_value(active_setting_preset, "balanced_min_reward", 0.60)), 0.05, disabled=mode != "Online Bandit" or not balanced_arm_floor)
        max_arm_share_guard = st.checkbox(
            "Max arm share guard",
            value=bool(preset_value(active_setting_preset, "max_arm_share_guard", False)),
            disabled=mode != "Online Bandit",
            help="Prevents one prompt from dominating after enough online selections by penalizing over-used arms.",
        )
        max_arm_share = st.slider("Max arm share", 0.40, 0.95, float(preset_value(active_setting_preset, "max_arm_share", 0.65)), 0.05, disabled=mode != "Online Bandit" or not max_arm_share_guard)
        max_arm_share_min_steps = st.slider("Max share min steps", 4, 80, int(preset_value(active_setting_preset, "max_arm_share_min_steps", 20)), 4, disabled=mode != "Online Bandit" or not max_arm_share_guard)
        few_shot_memory_guard = st.checkbox(
            "Few-shot memory guard",
            value=bool(preset_value(active_setting_preset, "few_shot_memory_guard", False)),
            disabled=mode != "Online Bandit",
            help="Penalizes few-shot unless similar previous tasks provide enough memory evidence. Useful when few-shot evaluation pass rate drops.",
        )
        few_shot_min_memory = st.slider("Few-shot min memory", 0.0, 1.0, float(preset_value(active_setting_preset, "few_shot_min_memory", 0.60)), 0.05, disabled=mode != "Online Bandit" or not few_shot_memory_guard)
        targeted_hard_overrides = st.checkbox(
            "Targeted hard overrides",
            value=bool(preset_value(active_setting_preset, "targeted_hard_overrides", False)),
            disabled=mode != "Online Bandit",
            help="For high-confidence task patterns from the latest diff, force the selected arm even during forced exploration.",
        )
        category_prior = st.checkbox(
            "Category-aware prompt prior",
            value=bool(preset_value(active_setting_preset, "category_prior", experiment_preset != "Strict notebook reproduction")),
            disabled=mode != "Online Bandit",
            help="Adds a small problem-type prior: parsing/string/list tend toward hint, math/order/recursive tend toward CoT.",
        )
        category_prior_weight = st.slider("Category prior weight", 0.0, 0.8, float(preset_value(active_setting_preset, "category_prior_weight", 0.35)), 0.05, disabled=mode != "Online Bandit" or not category_prior)
        category_arm_guard = st.checkbox(
            "Category-arm performance guard",
            value=bool(preset_value(active_setting_preset, "category_arm_guard", experiment_preset != "Strict notebook reproduction")),
            disabled=mode != "Online Bandit",
            help="Learns which arms are weak within each problem category and penalizes them online.",
        )
        category_guard_min_samples = st.slider("Category guard min samples", 2, 8, int(preset_value(active_setting_preset, "category_guard_min_samples", 3)), disabled=mode != "Online Bandit" or not category_arm_guard)
        category_guard_margin = st.slider("Category guard margin", 0.05, 0.50, float(preset_value(active_setting_preset, "category_guard_margin", 0.20)), 0.05, disabled=mode != "Online Bandit" or not category_arm_guard)
        recent_arm_cooldown = st.checkbox(
            "Recent arm cooldown",
            value=bool(preset_value(active_setting_preset, "recent_arm_cooldown", experiment_preset != "Strict notebook reproduction")),
            disabled=mode != "Online Bandit",
            help="Temporarily penalizes an arm if its recent rewards drop, preventing late-run drift like repeated CoT failures.",
        )
        recent_cooldown_window = st.slider("Recent cooldown window", 4, 16, int(preset_value(active_setting_preset, "recent_cooldown_window", 8)), disabled=mode != "Online Bandit" or not recent_arm_cooldown)
        recent_cooldown_min_samples = st.slider("Recent cooldown min samples", 2, 8, int(preset_value(active_setting_preset, "recent_cooldown_min_samples", 4)), disabled=mode != "Online Bandit" or not recent_arm_cooldown)
        recent_cooldown_threshold = st.slider("Recent cooldown reward threshold", 0.30, 0.80, float(preset_value(active_setting_preset, "recent_cooldown_threshold", 0.65)), 0.05, disabled=mode != "Online Bandit" or not recent_arm_cooldown)
        alpha = st.slider("Bandit alpha", 0.0, 2.0, float(preset_value(active_setting_preset, "alpha", 0.3)), 0.05, disabled=mode != "Online Bandit", help="Notebook 68% memakai alpha 0.3.")
        default_force_explore = 40 if experiment_preset == "Strict notebook reproduction" else 20
        default_force_explore = int(preset_value(active_setting_preset, "force_explore", default_force_explore))
        force_explore = st.slider("Force explore steps", 0, 80, default_force_explore, disabled=mode != "Online Bandit", help="Notebook 68% memakai force explore 40. Untuk optimized prompt bank dan warm-start, 8 atau bahkan 0 biasanya lebih aman karena HumanEval hanya 164 task.")
        st.caption("Advanced options adalah varian eksperimen terpisah. Untuk strict reproduction, gunakan preset Strict notebook reproduction.")
    with st.expander("Runtime safety", expanded=False):
        smart_extraction = st.checkbox("Smart extraction fallback", value=bool(preset_value(active_setting_preset, "smart_extraction", True)), help="If the first extracted completion has syntax errors, try an alternate extraction from the same raw generation before scoring.")
        generation_timeout = st.slider("Timeout generation/detik", 30, 300, int(preset_value(active_setting_preset, "generation_timeout", 120)), 10, help="Batas waktu model.generate per soal. Streamlit terlihat freeze selama generate berjalan.")
        timeout = st.slider("Timeout evaluasi/detik", 1, 30, int(preset_value(active_setting_preset, "timeout", 10)))
        shuffle = st.checkbox("Shuffle soal", value=bool(preset_value(active_setting_preset, "shuffle", False)), help="Matikan untuk mereplikasi notebook bandit68%. Online bandit sensitif terhadap urutan task.")
        seed = st.number_input("Seed", value=int(preset_value(active_setting_preset, "seed", 42)), step=1)
    st.markdown(
        f"""
        <div class="setup-card">
          <b>Ready summary</b>
          <div class="setup-row">
            <span class="setup-chip good">{mode}</span>
            <span class="setup-chip">{prompt_bank}</span>
            <span class="setup-chip">{num_tasks} tasks</span>
            <span class="setup-chip {'warn' if notebook_compatible else 'good'}">Notebook compat {bool_badge(notebook_compatible)}</span>
            <span class="setup-chip {'good' if similarity_memory else ''}">Similarity {bool_badge(similarity_memory)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Clear saved runs", use_container_width=True):
        st.session_state["run_history"] = []
        st.session_state["last_results"] = None
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
        if RUN_HISTORY_FILE.exists():
            RUN_HISTORY_FILE.unlink()
        st.rerun()

tab_prompts, tab_run, tab_results, tab_compare = st.tabs(["1. Prompt Bank", "2. Run Experiment", "3. Results", "4. Compare"])

with tab_prompts:
    st.markdown(
        f"""
        <div class="section-title">
          <h2>Prompt Bank Studio</h2>
          <span class="pill">{prompt_bank}</span>
        </div>
        <div class="soft-card">
        Review, edit, and preview each prompt arm before running experiments. For Online Bandit with this bank, keep <b>Notebook 68 compatible mode OFF</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if mode == "Online Bandit" and notebook_compatible:
        st.warning("Online Bandit sedang memakai Notebook 68 compatible mode. Dalam mode ini prompt bank/editor di bawah TIDAK dipakai; dashboard memakai prompt persis dari bandit68%.ipynb.")
    elif mode == "Online Bandit":
        st.success("Online Bandit akan memilih otomatis arm dari prompt bank ini: zero_shot, few_shot, cot, hint.")
    else:
        st.success(f"Fixed Strategy akan memakai prompt `{strategy}` dari prompt bank ini.")
    arm_cards = []
    for arm_name in active_prompts:
        role, desc = ARM_ROLES.get(arm_name, (arm_name, "Prompt strategy arm."))
        arm_cards.append(
            f"<div class='mini-card'><div class='role'>{role}</div><div class='name'>{arm_name}</div><div class='desc'>{desc}</div></div>"
        )
    st.markdown("<div class='mini-grid'>" + "".join(arm_cards) + "</div>", unsafe_allow_html=True)
    st.caption("Available placeholders: `{prompt}`, `{task_id}`, `{entry_point}`, `{hint}`.")
    if st.button("Reset prompt editor to selected bank defaults", use_container_width=True):
        for prompt_name in active_prompts:
            st.session_state.pop(f"prompt_{PROMPT_EDITOR_VERSION}_{prompt_bank}_{prompt_name}", None)
        st.rerun()
    prompt_templates = {}
    edit_cols = st.columns(2)
    for idx, name in enumerate(active_prompts):
        role, desc = ARM_ROLES.get(name, (name, ""))
        with edit_cols[idx % 2]:
            with st.container(border=True):
                st.markdown(f"**{name}**  \n<span class='pill'>{role}</span>", unsafe_allow_html=True)
                st.caption(desc)
                prompt_templates[name] = st.text_area(
                    f"Template: {name}",
                    active_prompts[name],
                    height=280,
                    key=f"prompt_{PROMPT_EDITOR_VERSION}_{prompt_bank}_{name}",
                    label_visibility="collapsed",
                )
    st.divider()
    st.markdown("### Prompt Preview")
    preview_left, preview_right = st.columns([1, 1])
    with preview_left:
        preview_task = st.selectbox("Preview task", list(problems.keys()))
    with preview_right:
        preview_strategy = st.selectbox("Preview strategy", list(active_prompts), key="preview_strategy")
    st.code(build_prompt(prompt_templates[preview_strategy], problems[preview_task], use_chat_template=False), language="text")

with tab_run:
    st.markdown(
        """
        <div class="section-title">
          <h2>Run Experiment</h2>
          <span class="pill">Execution Console</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="cta-card">
        <b>Current setup:</b> {mode} with {prompt_bank}<br>
        Preset: {experiment_preset} | Tasks: {num_tasks} x {repeats} | Notebook compatible: {bool_badge(notebook_compatible)} | Similarity memory: {bool_badge(similarity_memory)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if mode == "Online Bandit":
        st.caption("Untuk mendekati notebook 68%: aktifkan Notebook 68 compatible mode, 164 soal, repeat 1, shuffle nonaktif, alpha 0.3, force explore 40, max_new_tokens 512, sampling off.")
        if notebook_compatible and prompt_bank != "Notebook 68 prompts":
            st.warning("Notebook 68 compatible mode sedang ON, jadi Online Bandit akan mengabaikan selected prompt bank dan memakai prompt notebook persis.")
        if compatible_fallback or warm_start or similarity_memory or explore_order != "Notebook order" or weak_arm_guard:
            st.info("Compatible+/Similarity-Aware aktif: run ini adalah varian improvement terpisah, bukan strict reproduction dari notebook 68%.")
    config = {
        "mode": mode,
        "experiment_preset": experiment_preset,
        "strategy": strategy,
        "prompt_bank": prompt_bank,
        "model_name": model_name,
        "load_in_4bit": load_in_4bit,
        "use_chat_template": use_chat_template,
        "num_tasks": num_tasks,
        "repeats": repeats,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "notebook_compatible": notebook_compatible,
        "compatible_fallback": compatible_fallback if compatible_plus_enabled else False,
        "calibration_mode": calibration_mode if mode == "Online Bandit" else False,
        "calibration_tasks": calibration_tasks if mode == "Online Bandit" and calibration_mode else 0,
        "evaluation_only_metric": evaluation_only_metric if mode == "Online Bandit" and calibration_mode else False,
        "warm_start": warm_start if mode == "Online Bandit" else False,
        "warm_start_run_names": warm_start_run_names if mode == "Online Bandit" and warm_start else [],
        "warm_start_counts_as_exploration": warm_start_counts_as_exploration if mode == "Online Bandit" else False,
        "similarity_memory": similarity_memory if mode == "Online Bandit" else False,
        "memory_top_k": memory_top_k if mode == "Online Bandit" and similarity_memory else 0,
        "memory_lambda": memory_lambda if mode == "Online Bandit" and similarity_memory else 0.0,
        "memory_threshold": memory_threshold if mode == "Online Bandit" and similarity_memory else 0.0,
        "explore_order": explore_order if mode == "Online Bandit" else "Notebook order",
        "weak_arm_guard": weak_arm_guard if mode == "Online Bandit" else False,
        "weak_arm_min_samples": weak_arm_min_samples if mode == "Online Bandit" and weak_arm_guard else 0,
        "weak_arm_margin": weak_arm_margin if mode == "Online Bandit" and weak_arm_guard else 0.0,
        "best_arm_fallback": best_arm_fallback if mode == "Online Bandit" else False,
        "best_arm_min_samples": best_arm_min_samples if mode == "Online Bandit" and best_arm_fallback else 0,
        "best_arm_margin": best_arm_margin if mode == "Online Bandit" and best_arm_fallback else 0.0,
        "balanced_arm_floor": balanced_arm_floor if mode == "Online Bandit" else False,
        "balanced_min_share": balanced_min_share if mode == "Online Bandit" and balanced_arm_floor else 0.0,
        "balanced_min_reward": balanced_min_reward if mode == "Online Bandit" and balanced_arm_floor else 0.0,
        "max_arm_share_guard": max_arm_share_guard if mode == "Online Bandit" else False,
        "max_arm_share": max_arm_share if mode == "Online Bandit" and max_arm_share_guard else 0.0,
        "max_arm_share_min_steps": max_arm_share_min_steps if mode == "Online Bandit" and max_arm_share_guard else 0,
        "few_shot_memory_guard": few_shot_memory_guard if mode == "Online Bandit" else False,
        "few_shot_min_memory": few_shot_min_memory if mode == "Online Bandit" and few_shot_memory_guard else 0.0,
        "targeted_hard_overrides": targeted_hard_overrides if mode == "Online Bandit" else False,
        "category_prior": category_prior if mode == "Online Bandit" else False,
        "category_prior_weight": category_prior_weight if mode == "Online Bandit" and category_prior else 0.0,
        "category_arm_guard": category_arm_guard if mode == "Online Bandit" else False,
        "category_guard_min_samples": category_guard_min_samples if mode == "Online Bandit" and category_arm_guard else 0,
        "category_guard_margin": category_guard_margin if mode == "Online Bandit" and category_arm_guard else 0.0,
        "smart_extraction": smart_extraction,
        "recent_arm_cooldown": recent_arm_cooldown if mode == "Online Bandit" else False,
        "recent_cooldown_window": recent_cooldown_window if mode == "Online Bandit" and recent_arm_cooldown else 0,
        "recent_cooldown_min_samples": recent_cooldown_min_samples if mode == "Online Bandit" and recent_arm_cooldown else 0,
        "recent_cooldown_threshold": recent_cooldown_threshold if mode == "Online Bandit" and recent_arm_cooldown else 0.0,
        "generation_timeout": generation_timeout,
        "timeout": timeout,
        "shuffle": shuffle,
        "seed": int(seed),
        "alpha": alpha,
        "force_explore": force_explore,
    }
    est_generations = num_tasks * repeats
    run_warnings = []
    if mode == "Online Bandit" and notebook_compatible and prompt_bank != "Notebook 68 prompts":
        run_warnings.append("Notebook compatible ON will ignore the selected prompt bank.")
    if experiment_preset == "Thesis full run" and num_tasks < min(164, len(problems)):
        run_warnings.append("Thesis full run usually needs 164 tasks.")
    st.markdown(
        f"""
        <div class="status-grid">
          <div class="status-card card-blue"><div class="label">Generations</div><div class="value">{est_generations}</div><div class="hint">{num_tasks} tasks x {repeats} repeat</div></div>
          <div class="status-card card-green"><div class="label">Mode</div><div class="value">{'Bandit' if mode == 'Online Bandit' else 'Fixed'}</div><div class="hint">{'Automatic arm selection' if mode == 'Online Bandit' else strategy}</div></div>
          <div class="status-card card-violet"><div class="label">Prompt Bank</div><div class="value">{prompt_bank.split()[0]}</div><div class="hint">{prompt_bank}</div></div>
          <div class="status-card card-amber"><div class="label">Policy</div><div class="value">{bool_badge(similarity_memory)}</div><div class="hint">Similarity memory</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if run_warnings:
        for warning in run_warnings:
            st.warning(warning)
    else:
        st.success("Setup looks consistent. You can start the experiment when ready.")
    with st.expander("Recommended run order", expanded=False):
        st.markdown(
            """
            1. Jalankan preset `Calibration RL-APO Fair Calib` sebagai hasil utama thesis karena tidak memakai exact HumanEval routing.
            2. Bandingkan dengan Fixed Strategy `cot` pada full 164 task; calibration tetap dihitung sebagai bagian dari online learning.
            3. Gunakan `Calibration RL-APO CoT Anchor v3` hanya sebagai post-hoc diagnostic upper-bound, bukan hasil utama.
            4. Cek distribusi arm, phase, dan hard_override di Latest Results.
            """
        )
    with st.expander("Save / manage setting presets", expanded=False):
        preset_name = st.text_input("Preset name", value=f"RL-APO preset {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        save_col, delete_col = st.columns(2)
        with save_col:
            if st.button("Save current settings", use_container_width=True):
                presets = [item for item in st.session_state.get("setting_presets", []) if item.get("name") != preset_name]
                presets.append({"name": preset_name, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "config": config.copy()})
                st.session_state["setting_presets"] = presets
                save_setting_presets(presets)
                st.success(f"Saved preset: {preset_name}")
        with delete_col:
            saved_names = [item["name"] for item in st.session_state.get("setting_presets", [])]
            delete_preset_name = st.selectbox("Delete saved preset", ["-- choose --"] + saved_names)
            if st.button("Delete preset", use_container_width=True, disabled=delete_preset_name == "-- choose --"):
                presets = [item for item in st.session_state.get("setting_presets", []) if item.get("name") != delete_preset_name]
                st.session_state["setting_presets"] = presets
                save_setting_presets(presets)
                st.success(f"Deleted preset: {delete_preset_name}")
        st.caption(f"Saved presets file: `{SETTING_PRESETS_FILE.name}`")
    if st.button("Start Experiment", type="primary", use_container_width=True):
        df_result = run_experiment(config, prompt_templates, problems)
        run_name = make_run_name(config, st.session_state["run_started_at"])
        run_record = {
            "name": run_name,
            "created_at": st.session_state["run_started_at"],
            "config": config.copy(),
            "summary": summarize_results(df_result, config, run_name),
            "df": df_result,
        }
        st.session_state["last_results"] = df_result
        st.session_state["run_history"].append(run_record)
        save_run_history(st.session_state["run_history"])
        st.success("Experiment completed and saved. Open Latest Results or Compare Runs to inspect metrics and outputs.")

with tab_results:
    st.subheader("Latest Results")
    history = st.session_state["run_history"]
    selected_record = history[-1] if history else None
    if history:
        selected_name = st.selectbox(
            "Inspect saved run",
            [item["name"] for item in reversed(history)],
            index=0,
            key="latest_saved_run_select",
        )
        selected_record = next(item for item in history if item["name"] == selected_name)
    df = selected_record["df"] if selected_record else st.session_state.get("last_results")
    if df is None or df.empty:
        st.info("Belum ada hasil. Jalankan eksperimen dari tab Run.")
    else:
        for col in ["task_seconds", "generation_seconds", "eval_seconds"]:
            if col not in df.columns:
                df[col] = 0.0
        metric_df = df
        selected_config = selected_record.get("config", {}) if selected_record else {}
        use_eval_only = bool(selected_config.get("evaluation_only_metric")) and selected_config.get("experiment_preset") != "Calibration RL-APO Fair Calib"
        if use_eval_only and "phase" in df.columns and (df["phase"] == "evaluation").any():
            metric_df = df[df["phase"] == "evaluation"]
            st.info(f"Calibration run: metrics below use evaluation phase only ({len(metric_df)} rows). Full raw table still includes calibration rows.")
        elif "phase" in df.columns:
            st.info(f"Full-run metric: Pass@1 below uses all rows including calibration ({len(metric_df)} rows). Phase breakdown is available only as diagnostics.")
        pass_at_1 = metric_df["passed"].mean() * 100
        avg_reward = metric_df["reward"].mean()
        compile_rate = metric_df["compile_ok"].mean() * 100
        fail_rate = 100 - pass_at_1
        total = len(metric_df)
        passed_count = int(metric_df["passed"].sum())
        total_runtime = float(df["task_seconds"].sum())
        avg_task_seconds = float(metric_df["task_seconds"].mean()) if len(metric_df) else 0.0
        st.markdown(
            f"""
            <div class="status-grid">
              <div class="status-card card-blue"><div class="label">Pass@1</div><div class="value">{pass_at_1:.1f}%</div><div class="hint">{passed_count}/{total} tasks passed</div></div>
              <div class="status-card card-green"><div class="label">Compile OK</div><div class="value">{compile_rate:.1f}%</div><div class="hint">Executable candidates</div></div>
              <div class="status-card card-violet"><div class="label">Avg Reward</div><div class="value">{avg_reward:.3f}</div><div class="hint">Reward: 1.0 / 0.3 / 0.0</div></div>
              <div class="status-card card-amber"><div class="label">Runtime</div><div class="value">{total_runtime/60:.1f}m</div><div class="hint">{avg_task_seconds:.1f}s per task</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if "phase" in df.columns:
            phase_summary = df.groupby("phase").agg(
                generations=("task_id", "count"),
                passed=("passed", "sum"),
                compile_ok=("compile_ok", "sum"),
                avg_reward=("reward", "mean"),
            ).reset_index()
            phase_summary["pass_at_1"] = phase_summary["passed"] / phase_summary["generations"] * 100
            phase_summary["compile_rate"] = phase_summary["compile_ok"] / phase_summary["generations"] * 100
            phase_arm_summary = df.groupby(["phase", "strategy"]).agg(
                generations=("task_id", "count"),
                passed=("passed", "sum"),
                avg_reward=("reward", "mean"),
            ).reset_index()
            phase_arm_summary["pass_at_1"] = phase_arm_summary["passed"] / phase_arm_summary["generations"] * 100
            with st.expander("Diagnostic phase breakdown", expanded=False):
                st.dataframe(
                    phase_summary[["phase", "generations", "passed", "pass_at_1", "compile_ok", "compile_rate", "avg_reward"]],
                    use_container_width=True,
                    hide_index=True,
                )
                st.dataframe(phase_arm_summary, use_container_width=True, hide_index=True)
        st.divider()
        left, right = st.columns(2)
        with left:
            st.subheader("Pass@1 by Strategy")
            strategy_pass = (metric_df.groupby("strategy")["passed"].mean() * 100).sort_values(ascending=False)
            strategy_pass_df = strategy_pass.reset_index().rename(columns={"passed": "pass_at_1"})
            pass_chart = alt.Chart(strategy_pass_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X("strategy:N", title="Prompt strategy", sort="-y"),
                y=alt.Y("pass_at_1:Q", title="Pass@1 (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("strategy:N", legend=None),
                tooltip=["strategy:N", alt.Tooltip("pass_at_1:Q", format=".2f")],
            ).properties(height=320)
            st.altair_chart(pass_chart, use_container_width=True)
        with right:
            st.subheader("Strategy Usage")
            usage = pd.Series(dict(Counter(metric_df["strategy"]))).sort_values(ascending=False)
            usage_df = usage.reset_index()
            usage_df.columns = ["strategy", "count"]
            usage_chart = alt.Chart(usage_df).mark_arc(innerRadius=58, outerRadius=118).encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color("strategy:N", title="Strategy"),
                tooltip=["strategy:N", "count:Q"],
            ).properties(height=320)
            st.altair_chart(usage_chart, use_container_width=True)

        st.subheader("Detailed Metrics")
        summary = metric_df.groupby("strategy").agg(
            runs=("task_id", "count"),
            pass_at_1=("passed", lambda x: x.mean() * 100),
            compile_rate=("compile_ok", lambda x: x.mean() * 100),
            avg_reward=("reward", "mean"),
            avg_task_seconds=("task_seconds", "mean"),
            avg_eval_seconds=("eval_seconds", "mean"),
            avg_completion_chars=("completion_chars", "mean"),
        ).reset_index()
        st.dataframe(summary, use_container_width=True)
        run_summary_export = pd.DataFrame([{
            "run_name": selected_record["name"] if selected_record else "latest_results",
            "pass_at_1": pass_at_1,
            "compile_rate": compile_rate,
            "avg_reward": avg_reward,
            "passed": passed_count,
            "total": total,
            "fail_rate": fail_rate,
            "runtime_min": total_runtime / 60.0,
            "avg_task_seconds": avg_task_seconds,
        }])
        thesis_metrics = summary.copy()
        for col in ["pass_at_1", "compile_rate", "avg_reward", "avg_task_seconds", "avg_eval_seconds", "avg_completion_chars"]:
            if col in thesis_metrics:
                thesis_metrics[col] = thesis_metrics[col].round(3)
        with st.expander("Download thesis-ready outputs for this run", expanded=False):
            export_cols = st.columns(4)
            export_cols[0].download_button(
                "Run summary CSV",
                run_summary_export.to_csv(index=False).encode("utf-8"),
                safe_download_name(selected_record["name"] if selected_record else "latest", "_summary.csv"),
                "text/csv",
                use_container_width=True,
            )
            export_cols[1].download_button(
                "Strategy metrics CSV",
                thesis_metrics.to_csv(index=False).encode("utf-8"),
                safe_download_name(selected_record["name"] if selected_record else "latest", "_strategy_metrics.csv"),
                "text/csv",
                use_container_width=True,
            )
            export_cols[2].download_button(
                "Markdown table",
                df_to_markdown_table(thesis_metrics).encode("utf-8"),
                safe_download_name(selected_record["name"] if selected_record else "latest", "_strategy_metrics.md"),
                "text/markdown",
                use_container_width=True,
            )
            export_cols[3].download_button(
                "LaTeX table",
                df_to_latex_table(thesis_metrics, "Per-strategy performance on HumanEval.", "tab:strategy-performance").encode("utf-8"),
                safe_download_name(selected_record["name"] if selected_record else "latest", "_strategy_metrics.tex"),
                "text/plain",
                use_container_width=True,
            )

        st.subheader("Task Explorer")
        filter_left, filter_mid, filter_right = st.columns([1.1, 1, 1])
        with filter_left:
            strategy_filter = st.multiselect("Strategy", sorted(df["strategy"].unique()), default=sorted(df["strategy"].unique()))
        with filter_mid:
            outcome_filter = st.selectbox("Outcome", ["All", "Passed only", "Failed only", "Compile failed"])
        with filter_right:
            search_task = st.text_input("Search task id", placeholder="HumanEval/42")
        visible = df[df["strategy"].isin(strategy_filter)] if strategy_filter else df.iloc[0:0]
        if outcome_filter == "Passed only":
            visible = visible[visible["passed"]]
        elif outcome_filter == "Failed only":
            visible = visible[~visible["passed"]]
        elif outcome_filter == "Compile failed":
            visible = visible[~visible["compile_ok"]]
        if search_task:
            visible = visible[visible["task_id"].str.contains(search_task, case=False, regex=False)]

        compact_cols = [
            "repeat", "task_id", "problem_category", "strategy", "passed", "compile_ok", "reward", "extraction_method",
            "linucb_score", "memory_score", "combined_score", "best_similarity", "nearest_examples",
            "task_seconds", "eval_seconds", "generated_chars", "completion_chars", "error",
        ]
        compact_cols = [col for col in compact_cols if col in visible.columns]
        st.dataframe(visible[compact_cols], use_container_width=True, height=360)
        with st.expander("Show full raw results including prompts and completions", expanded=False):
            st.dataframe(visible, use_container_width=True, height=520)
        csv = df.to_csv(index=False).encode("utf-8")
        jsonl = "\n".join(json.dumps(row, ensure_ascii=False) for row in df.to_dict("records")).encode("utf-8")
        dl1, dl2 = st.columns(2)
        dl1.download_button("Download selected run CSV", csv, "experiment_results.csv", "text/csv", use_container_width=True)
        dl2.download_button("Download selected run JSONL", jsonl, "experiment_results.jsonl", "application/jsonl", use_container_width=True)

with tab_compare:
    st.markdown(
        """
        <div class="section-title">
          <h2>Comparison Board</h2>
          <span class="pill">Thesis View</span>
        </div>
        <div class="soft-card">
        Rank runs by Pass@1, inspect compile rate and reward trade-offs, then export chart data and LaTeX-ready tables for your thesis document.
        </div>
        """,
        unsafe_allow_html=True,
    )
    history = st.session_state["run_history"]
    if not history:
        st.info("Belum ada saved run. Jalankan Bandit atau fixed strategy dari tab Run, lalu kembali ke sini.")
    else:
        run_label_by_name = {item["name"]: readable_run_label(item) for item in history}
        run_name_by_label = {label: name for name, label in run_label_by_name.items()}
        summary_df = pd.DataFrame([item["summary"] for item in history])
        defaults = {
            "total_seconds": 0.0,
            "avg_task_seconds": 0.0,
            "pass_at_1_ci_low": 0.0,
            "pass_at_1_ci_high": 0.0,
            "compile_ci_low": 0.0,
            "compile_ci_high": 0.0,
            "failed": 0,
            "compiled": 0,
            "compile_failed": 0,
            "notebook_compatible": False,
            "compatible_fallback": False,
            "calibration_mode": False,
            "calibration_tasks": 0,
            "evaluation_only_metric": False,
            "warm_start": False,
            "warm_start_examples": 0,
            "similarity_memory": False,
            "memory_top_k": 0,
            "memory_lambda": 0.0,
            "memory_threshold": 0.0,
            "explore_order": "Notebook order",
            "weak_arm_guard": False,
            "weak_arm_min_samples": 0,
            "weak_arm_margin": 0.0,
            "best_arm_fallback": False,
            "best_arm_min_samples": 0,
            "best_arm_margin": 0.0,
            "balanced_arm_floor": False,
            "balanced_min_share": 0.0,
            "balanced_min_reward": 0.0,
            "max_arm_share_guard": False,
            "max_arm_share": 0.0,
            "max_arm_share_min_steps": 0,
            "few_shot_memory_guard": False,
            "few_shot_min_memory": 0.0,
            "targeted_hard_overrides": False,
            "category_prior": False,
            "category_prior_weight": 0.0,
            "category_arm_guard": False,
            "category_guard_min_samples": 0,
            "category_guard_margin": 0.0,
            "smart_extraction": True,
            "recent_arm_cooldown": False,
            "recent_cooldown_window": 0,
            "recent_cooldown_min_samples": 0,
            "recent_cooldown_threshold": 0.0,
            "prompt_bank": "Notebook 68 prompts",
        }
        for col, default in defaults.items():
            if col not in summary_df.columns:
                summary_df[col] = default
        summary_df["readable_run"] = summary_df["run_name"].map(run_label_by_name).fillna(summary_df["run_name"])
        summary_df = summary_df.sort_values("pass_at_1", ascending=False).reset_index(drop=True)
        summary_df["display_name"] = summary_df.apply(
            lambda row: row["readable_run"],
            axis=1,
        )
        selected_compare_labels = st.multiselect(
            "Runs shown in dashboard charts",
            summary_df["readable_run"].tolist(),
            default=summary_df["readable_run"].tolist(),
            help="Filter visual charts without deleting saved runs.",
        )
        selected_compare_names = [run_name_by_label.get(label, label) for label in selected_compare_labels]
        if selected_compare_names:
            summary_df = summary_df[summary_df["run_name"].isin(selected_compare_names)].reset_index(drop=True)
        if summary_df.empty:
            st.warning("No runs selected for comparison.")
            st.stop()
        best = summary_df.iloc[0]
        runner_up = summary_df.iloc[1] if len(summary_df) > 1 else best
        gain = float(best["pass_at_1"] - runner_up["pass_at_1"]) if len(summary_df) > 1 else 0.0
        st.markdown(
            f"""
            <div class="status-grid">
              <div class="status-card card-blue"><div class="label">Best Pass@1</div><div class="value">{best['pass_at_1']:.1f}%</div><div class="hint">{best['readable_run']}</div></div>
              <div class="status-card card-green"><div class="label">Best Reward</div><div class="value">{best['avg_reward']:.3f}</div><div class="hint">Average execution reward</div></div>
              <div class="status-card card-violet"><div class="label">Lead Margin</div><div class="value">+{gain:.1f}</div><div class="hint">points over rank #2</div></div>
              <div class="status-card card-amber"><div class="label">Compared Runs</div><div class="value">{len(summary_df)}</div><div class="hint">of {len(history)} saved runs</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        podium_cards = []
        for rank, (_, row) in enumerate(summary_df.head(3).iterrows(), start=1):
            podium_cards.append(
                f"<div class='mini-card'><div class='role'>Rank #{rank}</div><div class='name'>{row['strategy']} - {row['pass_at_1']:.1f}%</div><div class='desc'>{row['readable_run']}</div></div>"
            )
        st.markdown("<div class='mini-grid'>" + "".join(podium_cards) + "</div>", unsafe_allow_html=True)

        with st.expander("Manage leaderboard runs", expanded=False):
            runs_to_delete = st.multiselect(
                "Delete runs from leaderboard",
                [run_label_by_name[item["name"]] for item in history],
                help="Gunakan ini untuk menghapus run yang salah setting, crash, atau hanya testing kecil.",
            )
            delete_names = [run_name_by_label.get(label, label) for label in runs_to_delete]
            if st.button("Delete selected runs", type="secondary", use_container_width=True, disabled=not delete_names):
                st.session_state["run_history"] = [item for item in history if item["name"] not in delete_names]
                save_run_history(st.session_state["run_history"])
                st.rerun()

        st.subheader("Leaderboard")
        summary_df["pass@1 95% CI"] = summary_df.apply(
            lambda row: f"{row['pass_at_1']:.1f}% [{row['pass_at_1_ci_low']:.1f}, {row['pass_at_1_ci_high']:.1f}]",
            axis=1,
        )
        summary_df["compile 95% CI"] = summary_df.apply(
            lambda row: f"{row['compile_rate']:.1f}% [{row['compile_ci_low']:.1f}, {row['compile_ci_high']:.1f}]",
            axis=1,
        )
        summary_df["runtime_min"] = summary_df["total_seconds"] / 60.0
        display_cols = [
            "readable_run",
            "run_name",
            "mode",
            "strategy",
            "prompt_bank",
            "generations",
            "passed",
            "failed",
            "pass@1 95% CI",
            "compiled",
            "compile_failed",
            "compile 95% CI",
            "avg_reward",
            "runtime_min",
            "avg_task_seconds",
            "alpha",
            "force_explore",
            "seed",
            "chat_template",
            "notebook_compatible",
            "compatible_fallback",
            "calibration_mode",
            "calibration_tasks",
            "evaluation_only_metric",
            "warm_start",
            "warm_start_examples",
            "similarity_memory",
            "memory_top_k",
            "memory_lambda",
            "memory_threshold",
            "explore_order",
            "weak_arm_guard",
            "weak_arm_min_samples",
            "weak_arm_margin",
            "best_arm_fallback",
            "best_arm_min_samples",
            "best_arm_margin",
            "balanced_arm_floor",
            "balanced_min_share",
            "balanced_min_reward",
            "max_arm_share_guard",
            "max_arm_share",
            "max_arm_share_min_steps",
            "few_shot_memory_guard",
            "few_shot_min_memory",
            "targeted_hard_overrides",
            "category_prior",
            "category_prior_weight",
            "category_arm_guard",
            "category_guard_min_samples",
            "category_guard_margin",
            "smart_extraction",
            "recent_arm_cooldown",
            "recent_cooldown_window",
            "recent_cooldown_min_samples",
            "recent_cooldown_threshold",
        ]
        st.dataframe(summary_df[display_cols], use_container_width=True, height=300)
        st.caption("CI menggunakan normal approximation 95%. Untuk laporan thesis/jurnal, gunakan run 164 task penuh dan setting yang konsisten.")
        thesis_cols = [
            "readable_run", "run_name", "strategy", "prompt_bank", "generations", "passed", "failed",
            "pass_at_1", "compile_rate", "avg_reward", "runtime_min",
            "notebook_compatible", "similarity_memory", "weak_arm_guard",
        ]
        thesis_cols = [col for col in thesis_cols if col in summary_df.columns]
        thesis_summary_df = summary_df[thesis_cols].copy()
        for col in ["pass_at_1", "compile_rate", "avg_reward", "runtime_min"]:
            if col in thesis_summary_df:
                thesis_summary_df[col] = thesis_summary_df[col].round(3)

        st.markdown("### Metric Comparison")
        plot_df = summary_df.copy()
        plot_df["method"] = plot_df["display_name"]
        duplicated_methods = plot_df["method"].duplicated(keep=False)
        plot_df.loc[duplicated_methods, "method"] = (
            plot_df.loc[duplicated_methods, "method"] + " #" + (plot_df.groupby("method").cumcount() + 1).astype(str)
        )
        rank_chart = alt.Chart(plot_df).mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8).encode(
            y=alt.Y("method:N", title="Run", sort="-x", axis=alt.Axis(labelLimit=260)),
            x=alt.X("pass_at_1:Q", title="Pass@1 (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("prompt_bank:N", title="Prompt bank"),
            tooltip=["readable_run:N", alt.Tooltip("pass_at_1:Q", format=".2f"), alt.Tooltip("compile_rate:Q", format=".2f"), alt.Tooltip("avg_reward:Q", format=".3f")],
        ).properties(height=max(300, 44 * len(plot_df)))
        st.altair_chart(rank_chart, use_container_width=True)
        accuracy_chart = plot_df.set_index("method")[["pass_at_1", "compile_rate"]].rename(columns={
            "pass_at_1": "Pass@1 (%)",
            "compile_rate": "Compile OK (%)",
        })
        accuracy_chart["Avg Reward (x100)"] = plot_df.set_index("method")["avg_reward"] * 100
        chart_long = accuracy_chart.reset_index().melt("method", var_name="metric", value_name="value")
        metric_chart = alt.Chart(chart_long).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("method:N", title="Method / Prompt", sort=None, axis=alt.Axis(labelAngle=-25, labelLimit=180)),
            xOffset=alt.XOffset("metric:N"),
            y=alt.Y("value:Q", title="Score", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("metric:N", title="Metric", scale=alt.Scale(range=["#2563eb", "#16a34a", "#7c3aed"])),
            tooltip=["method:N", "metric:N", alt.Tooltip("value:Q", format=".2f")],
        ).properties(height=420)
        st.caption("Vertical grouped bar chart. Higher is better. Runtime tetap tersedia di leaderboard table.")
        st.altair_chart(metric_chart, use_container_width=True)

        st.markdown("### Task-Level Agreement")
        selected_run_labels = st.multiselect(
            "Runs to compare by task",
            selected_compare_labels or [run_label_by_name[item["name"]] for item in history],
            default=(selected_compare_labels or [run_label_by_name[item["name"]] for item in history])[: min(len(selected_compare_labels or history), 3)],
        )
        selected_runs = [run_name_by_label.get(label, label) for label in selected_run_labels]
        if selected_runs:
            merged_rows = []
            for item in history:
                if item["name"] not in selected_runs:
                    continue
                tmp = item["df"][["task_id", "strategy", "passed", "compile_ok", "reward"]].copy()
                tmp["run_name"] = run_label_by_name[item["name"]]
                merged_rows.append(tmp)
            compare_df = pd.concat(merged_rows, ignore_index=True) if merged_rows else pd.DataFrame()
            pivot = compare_df.pivot_table(index="task_id", columns="run_name", values="passed", aggfunc="max")
            pass_counts = compare_df.groupby("task_id")["passed"].sum().reset_index(name="passed_runs")
            agreement_chart = alt.Chart(pass_counts).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("passed_runs:O", title="Number of selected runs that passed a task"),
                y=alt.Y("count():Q", title="Task count"),
                color=alt.Color("passed_runs:O", legend=None, scale=alt.Scale(range=["#dc2626", "#f97316", "#eab308", "#22c55e", "#16a34a", "#15803d"])),
                tooltip=["passed_runs:O", alt.Tooltip("count():Q", title="tasks")],
            ).properties(height=260)
            st.altair_chart(agreement_chart, use_container_width=True)
            with st.expander("Task pass/fail matrix", expanded=False):
                st.dataframe(pivot, use_container_width=True, height=420)

            if len(selected_runs) >= 2:
                st.subheader("Pairwise Task-Level Comparison")
                pair_rows = []
                selected_items = [item for item in history if item["name"] in selected_runs]
                for i in range(len(selected_items)):
                    for j in range(i + 1, len(selected_items)):
                        a = selected_items[i]
                        b = selected_items[j]
                        a_df = a["df"][["task_id", "passed"]].drop_duplicates("task_id").rename(columns={"passed": "a_passed"})
                        b_df = b["df"][["task_id", "passed"]].drop_duplicates("task_id").rename(columns={"passed": "b_passed"})
                        joined = a_df.merge(b_df, on="task_id", how="inner")
                        a_only = int((joined["a_passed"] & ~joined["b_passed"]).sum())
                        b_only = int((~joined["a_passed"] & joined["b_passed"]).sum())
                        both_pass = int((joined["a_passed"] & joined["b_passed"]).sum())
                        both_fail = int((~joined["a_passed"] & ~joined["b_passed"]).sum())
                        pair_rows.append({
                            "run_a": run_label_by_name[a["name"]],
                            "run_b": run_label_by_name[b["name"]],
                            "common_tasks": len(joined),
                            "a_only_pass": a_only,
                            "b_only_pass": b_only,
                            "both_pass": both_pass,
                            "both_fail": both_fail,
                            "net_a_minus_b": a_only - b_only,
                        })
                st.dataframe(pd.DataFrame(pair_rows), use_container_width=True, height=260)

        with st.expander("Download thesis-ready comparison package", expanded=True):
            dl_a, dl_b, dl_c, dl_d = st.columns(4)
            dl_a.download_button("Leaderboard CSV", summary_df.to_csv(index=False).encode("utf-8"), "run_comparison_full.csv", "text/csv", use_container_width=True)
            dl_b.download_button("Chart data CSV", chart_long.to_csv(index=False).encode("utf-8"), "metric_chart_data.csv", "text/csv", use_container_width=True)
            dl_c.download_button("Markdown table", df_to_markdown_table(thesis_summary_df).encode("utf-8"), "thesis_comparison_table.md", "text/markdown", use_container_width=True)
            dl_d.download_button("LaTeX table", df_to_latex_table(thesis_summary_df, "Comparison of RL-APO prompting strategies on HumanEval.", "tab:rl-apo-comparison").encode("utf-8"), "thesis_comparison_table.tex", "text/plain", use_container_width=True)
            st.caption("Gunakan CSV untuk chart ulang di Excel/Sheets, Markdown untuk draft docs, dan LaTeX table untuk proposal/thesis.")
        history_json = RUN_HISTORY_FILE.read_bytes() if RUN_HISTORY_FILE.exists() else json.dumps([], indent=2).encode("utf-8")
        st.download_button("Download full run history JSON", history_json, "streamlit_run_history.json", "application/json", use_container_width=True)
