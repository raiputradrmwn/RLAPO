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
APP_TITLE = "RL Prompt Engineering Lab"
PROMPT_EDITOR_VERSION = "notebook_68_prompts_v4"
STRATEGY_NAMES = {
    0: "zero_shot",
    1: "few_shot",
    2: "cot",
    3: "hint",
}


def init_state():
    st.session_state.setdefault("run_history", load_run_history())
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
    else:
        method = config["strategy"]
    return f"{started_at} | {method} | {config['num_tasks']} tasks x {config['repeats']}"


def summarize_results(df: pd.DataFrame, config: dict, run_name: str) -> dict:
    pass_at_1 = float(df["passed"].mean() * 100) if not df.empty else 0.0
    compile_rate = float(df["compile_ok"].mean() * 100) if not df.empty else 0.0
    avg_reward = float(df["reward"].mean()) if not df.empty else 0.0
    total = int(len(df))
    passed = int(df["passed"].sum()) if not df.empty else 0
    compiled = int(df["compile_ok"].sum()) if not df.empty else 0
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
    lines = code.splitlines()
    non_empty = [line for line in lines if line.strip()]
    if non_empty and all(line.startswith((" ", "\t")) for line in non_empty):
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
    stop_markers = ["\nProblem:", "\nExample", "\nTests:", "\nassert ", "\n# Test"]
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
    if any(word in prompt_text for word in ["sort", "sorted", "order"]):
        return "Hint: Consider efficient sorting algorithms and time complexity."
    if any(word in prompt_text for word in ["tree", "node", "binary"]):
        return "Hint: Consider recursion or iterative traversal for tree structures."
    if any(word in prompt_text for word in ["string", "text", "character"]):
        return "Hint: Consider string manipulation methods and edge cases like empty strings."
    return "Hint: Consider edge cases and efficient data structures."


def check_compile(code: str) -> int:
    try:
        ast.parse(code)
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
    full_code = problem["prompt"] + completion + "\n" + problem["test"]
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
    def __init__(self, num_arms=4, dim_context=19, alpha=0.3, force_explore=40):
        self.num_arms = num_arms
        self.dim_context = dim_context
        self.alpha = alpha
        self.force_explore = force_explore
        self.t = 0
        self.A = [np.eye(dim_context) + 0.01 * np.eye(dim_context) for _ in range(num_arms)]
        self.b = [np.zeros((dim_context, 1)) for _ in range(num_arms)]
        self.theta = [np.zeros((dim_context, 1)) for _ in range(num_arms)]
        self.arm_counts = [0] * num_arms
        self.arm_rewards = [[] for _ in range(num_arms)]

    def select_arm(self, context):
        self.t += 1
        if self.t <= self.force_explore:
            arm = (self.t - 1) % self.num_arms
            self.arm_counts[arm] += 1
            return arm
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
        arm = int(np.argmax(scores))
        self.arm_counts[arm] += 1
        return arm

    def update(self, arm, context, reward):
        context = context.reshape(-1, 1)
        self.A[arm] += context @ context.T
        self.b[arm] += reward * context
        self.arm_rewards[arm].append(reward)
        try:
            self.theta[arm] = np.linalg.inv(self.A[arm]) @ self.b[arm]
        except np.linalg.LinAlgError:
            self.theta[arm] = np.linalg.pinv(self.A[arm]) @ self.b[arm]


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
    bandit = OnlineLinUCB(alpha=config["alpha"], force_explore=config["force_explore"]) if config["mode"] == "Online Bandit" else None
    rows = []
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
        for task_id, problem in items:
            task_timer_start = time.perf_counter()
            step += 1
            if bandit:
                features = np.array(extract_features(problem, tokenizer))
                arm = bandit.select_arm(features)
                strategy = STRATEGY_NAMES[arm]
            else:
                strategy = config["strategy"]
                arm = {name: idx for idx, name in STRATEGY_NAMES.items()}[strategy]
                features = np.array(extract_features(problem, tokenizer))

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
            completion = extract_code_notebook(generated) if config["notebook_compatible"] and config["mode"] == "Online Bandit" else extract_code(generated)
            passed, compile_ok, reward, error = evaluate_sample(problem, completion, config["timeout"])
            eval_seconds = time.perf_counter() - eval_timer_start
            task_seconds = time.perf_counter() - task_timer_start
            if bandit:
                bandit.update(arm, features, reward)

            rows.append({
                "repeat": repeat + 1,
                "task_id": task_id,
                "strategy": strategy,
                "arm": arm,
                "passed": passed,
                "compile_ok": compile_ok,
                "reward": reward,
                "error": error,
                "completion": completion,
                "raw_generated": generated,
                "prompt": prompt,
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
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="hero">
      <h1>{APP_TITLE}</h1>
      <p>Interactive HumanEval runner untuk fixed prompting dan online contextual bandit. Simpan beberapa run, bandingkan Bandit vs Zero-shot vs Few-shot vs CoT vs Hint, dan telusuri output tanpa tenggelam dalam log 164 soal.</p>
      <div class="hero-badges">
        <span class="hero-badge">One-click experiment</span>
        <span class="hero-badge">Run comparison</span>
        <span class="hero-badge">Expandable 164-task log</span>
        <span class="hero-badge">CSV / JSONL export</span>
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
    st.header("Experiment Setup")
    st.caption("Gunakan Python environment yang sama dengan notebook supaya CUDA/model cache terbaca.")
    st.markdown(f"<span class='run-chip'>{len(st.session_state['run_history'])} saved runs</span>", unsafe_allow_html=True)
    with st.expander("Runtime", expanded=False):
        st.code(sys.executable, language="text")
    mode = st.radio("Mode", ["Fixed Strategy", "Online Bandit"])
    strategy = st.selectbox("Fixed strategy", list(DEFAULT_PROMPTS), disabled=mode == "Online Bandit")
    model_name = st.text_input("Model", "deepseek-ai/deepseek-coder-6.7b-instruct")
    load_in_4bit = st.checkbox("Load 4-bit", value=True)
    use_chat_template = st.checkbox("Use tokenizer chat template", value=True)
    num_tasks = st.slider("Jumlah soal", 1, min(164, len(problems)), 5)
    repeats = st.slider("Repeat per soal", 1, 5, 1)
    max_new_tokens = st.slider("Max new tokens", 64, 1024, 512, 64)
    do_sample = st.checkbox("Sampling", value=False)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.1, 0.05)
    notebook_compatible = st.checkbox("Notebook 68 compatible mode", value=True, help="For Online Bandit, use prompt/extraction/generation behavior from bandit68%.ipynb.")
    generation_timeout = st.slider("Timeout generation/detik", 30, 300, 120, 10, help="Batas waktu model.generate per soal. Streamlit terlihat freeze selama generate berjalan.")
    timeout = st.slider("Timeout evaluasi/detik", 1, 30, 10)
    shuffle = st.checkbox("Shuffle soal", value=False, help="Matikan untuk mereplikasi notebook bandit68%. Online bandit sensitif terhadap urutan task.")
    seed = st.number_input("Seed", value=42, step=1)
    alpha = st.slider("Bandit alpha", 0.0, 2.0, 0.3, 0.05, disabled=mode != "Online Bandit", help="Notebook 68% memakai alpha 0.3.")
    force_explore = st.slider("Force explore steps", 0, 80, 40, disabled=mode != "Online Bandit", help="Notebook 68% memakai force explore 40.")
    if st.button("Clear saved runs", use_container_width=True):
        st.session_state["run_history"] = []
        st.session_state["last_results"] = None
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
        if RUN_HISTORY_FILE.exists():
            RUN_HISTORY_FILE.unlink()
        st.rerun()

tab_prompts, tab_run, tab_results, tab_compare = st.tabs(["Prompt Lab", "Run", "Latest Results", "Compare Runs"])

with tab_prompts:
    st.subheader("Prompt Templates")
    st.write("Gunakan `{prompt}`, `{task_id}`, `{entry_point}`, dan `{hint}` sebagai placeholder.")
    if st.button("Reset prompt editor to notebook 68 defaults", use_container_width=True):
        for prompt_name in DEFAULT_PROMPTS:
            st.session_state.pop(f"prompt_{PROMPT_EDITOR_VERSION}_{prompt_name}", None)
        st.rerun()
    prompt_templates = {}
    cols = st.columns(2)
    for idx, name in enumerate(DEFAULT_PROMPTS):
        with cols[idx % 2]:
            prompt_templates[name] = st.text_area(
                name,
                DEFAULT_PROMPTS[name],
                height=260,
                key=f"prompt_{PROMPT_EDITOR_VERSION}_{name}",
            )
    preview_task = st.selectbox("Preview task", list(problems.keys()))
    preview_strategy = st.selectbox("Preview strategy", list(DEFAULT_PROMPTS), key="preview_strategy")
    st.code(build_prompt(prompt_templates[preview_strategy], problems[preview_task], use_chat_template=False), language="text")

with tab_run:
    st.subheader("Run Experiment")
    st.markdown(
        """
        <div class="soft-card">
        Klik tombol di bawah untuk mulai. Hasil run akan otomatis disimpan di session sehingga kamu bisa menjalankan Bandit, lalu Zero-shot, Few-shot, CoT, dan Hint, kemudian membandingkannya di tab <b>Compare Runs</b>. Logic prompt dan bandit dikembalikan ke konfigurasi notebook 68%, sementara evaluator timeout dibuat lebih aman agar tidak stuck.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if mode == "Online Bandit":
        st.caption("Untuk mendekati notebook 68%: aktifkan Notebook 68 compatible mode, 164 soal, repeat 1, shuffle nonaktif, alpha 0.3, force explore 40, max_new_tokens 512, sampling off.")
    config = {
        "mode": mode,
        "strategy": strategy,
        "model_name": model_name,
        "load_in_4bit": load_in_4bit,
        "use_chat_template": use_chat_template,
        "num_tasks": num_tasks,
        "repeats": repeats,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "notebook_compatible": notebook_compatible,
        "generation_timeout": generation_timeout,
        "timeout": timeout,
        "shuffle": shuffle,
        "seed": int(seed),
        "alpha": alpha,
        "force_explore": force_explore,
    }
    est_generations = num_tasks * repeats
    quick1, quick2, quick3, quick4 = st.columns(4)
    quick1.metric("Total generations", est_generations)
    quick2.metric("Mode", "Bandit" if mode == "Online Bandit" else "Fixed")
    quick3.metric("Current strategy", "auto" if mode == "Online Bandit" else strategy)
    quick4.metric("Saved runs", len(st.session_state["run_history"]))
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
        pass_at_1 = df["passed"].mean() * 100
        avg_reward = df["reward"].mean()
        compile_rate = df["compile_ok"].mean() * 100
        fail_rate = 100 - pass_at_1
        total = len(df)
        passed_count = int(df["passed"].sum())
        total_runtime = float(df["task_seconds"].sum())
        avg_task_seconds = float(df["task_seconds"].mean()) if len(df) else 0.0
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Pass@1", f"{pass_at_1:.1f}%")
        col2.metric("Avg Reward", f"{avg_reward:.3f}")
        col3.metric("Compile OK", f"{compile_rate:.1f}%")
        col4.metric("Passed", f"{passed_count}/{total}")
        col5.metric("Fail Rate", f"{fail_rate:.1f}%")
        col6.metric("Runtime", f"{total_runtime/60:.1f}m", f"{avg_task_seconds:.1f}s/task")

        st.divider()
        left, right = st.columns(2)
        with left:
            st.subheader("Pass@1 by Strategy")
            strategy_pass = (df.groupby("strategy")["passed"].mean() * 100).sort_values(ascending=False)
            st.bar_chart(strategy_pass)
        with right:
            st.subheader("Strategy Usage")
            usage = pd.Series(dict(Counter(df["strategy"]))).sort_values(ascending=False)
            st.bar_chart(usage)

        st.subheader("Detailed Metrics")
        summary = df.groupby("strategy").agg(
            runs=("task_id", "count"),
            pass_at_1=("passed", lambda x: x.mean() * 100),
            compile_rate=("compile_ok", lambda x: x.mean() * 100),
            avg_reward=("reward", "mean"),
            avg_task_seconds=("task_seconds", "mean"),
            avg_eval_seconds=("eval_seconds", "mean"),
            avg_completion_chars=("completion_chars", "mean"),
        ).reset_index()
        st.dataframe(summary, use_container_width=True)

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

        compact_cols = ["repeat", "task_id", "strategy", "passed", "compile_ok", "reward", "task_seconds", "eval_seconds", "generated_chars", "completion_chars", "error"]
        st.dataframe(visible[compact_cols], use_container_width=True, height=360)
        with st.expander("Show full raw results including prompts and completions", expanded=False):
            st.dataframe(visible, use_container_width=True, height=520)
        csv = df.to_csv(index=False).encode("utf-8")
        jsonl = "\n".join(json.dumps(row, ensure_ascii=False) for row in df.to_dict("records")).encode("utf-8")
        dl1, dl2 = st.columns(2)
        dl1.download_button("Download selected run CSV", csv, "experiment_results.csv", "text/csv", use_container_width=True)
        dl2.download_button("Download selected run JSONL", jsonl, "experiment_results.jsonl", "application/jsonl", use_container_width=True)

with tab_compare:
    st.subheader("Compare Saved Runs")
    history = st.session_state["run_history"]
    if not history:
        st.info("Belum ada saved run. Jalankan Bandit atau fixed strategy dari tab Run, lalu kembali ke sini.")
    else:
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
        }
        for col, default in defaults.items():
            if col not in summary_df.columns:
                summary_df[col] = default
        summary_df = summary_df.sort_values("pass_at_1", ascending=False).reset_index(drop=True)
        best = summary_df.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Best run", best["strategy"])
        c2.metric("Best Pass@1", f"{best['pass_at_1']:.1f}%")
        c3.metric("Best Avg Reward", f"{best['avg_reward']:.3f}")
        c4.metric("Best Runtime", f"{best['total_seconds']/60:.1f}m")
        c5.metric("Saved runs", len(history))

        with st.expander("Manage leaderboard runs", expanded=False):
            runs_to_delete = st.multiselect(
                "Delete runs from leaderboard",
                [item["name"] for item in history],
                help="Gunakan ini untuk menghapus run yang salah setting, crash, atau hanya testing kecil.",
            )
            if st.button("Delete selected runs", type="secondary", use_container_width=True, disabled=not runs_to_delete):
                st.session_state["run_history"] = [item for item in history if item["name"] not in runs_to_delete]
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
            "run_name",
            "mode",
            "strategy",
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
        ]
        st.dataframe(summary_df[display_cols], use_container_width=True, height=300)
        st.caption("CI menggunakan normal approximation 95%. Untuk laporan thesis/jurnal, gunakan run 164 task penuh dan setting yang konsisten.")

        st.subheader("Metric Comparison")
        plot_df = summary_df.copy()
        plot_df["method"] = plot_df["strategy"].replace({"bandit": "Online Bandit"})
        duplicated_methods = plot_df["method"].duplicated(keep=False)
        plot_df.loc[duplicated_methods, "method"] = (
            plot_df.loc[duplicated_methods, "method"] + " #" + (plot_df.groupby("method").cumcount() + 1).astype(str)
        )
        accuracy_chart = plot_df.set_index("method")[["pass_at_1", "compile_rate"]].rename(columns={
            "pass_at_1": "Pass@1 (%)",
            "compile_rate": "Compile OK (%)",
        })
        accuracy_chart["Avg Reward (x100)"] = plot_df.set_index("method")["avg_reward"] * 100
        chart_long = accuracy_chart.reset_index().melt("method", var_name="metric", value_name="value")
        metric_chart = alt.Chart(chart_long).mark_bar().encode(
            x=alt.X("method:N", title="Method / Prompt", sort=None, axis=alt.Axis(labelAngle=-35)),
            xOffset=alt.XOffset("metric:N"),
            y=alt.Y("value:Q", title="Score", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=["method:N", "metric:N", alt.Tooltip("value:Q", format=".2f")],
        ).properties(height=420)
        st.caption("Vertical grouped bar chart. Higher is better. Runtime tetap tersedia di leaderboard table.")
        st.altair_chart(metric_chart, use_container_width=True)

        st.subheader("Strategy Comparison")
        selected_runs = st.multiselect(
            "Runs to compare by task",
            [item["name"] for item in history],
            default=[item["name"] for item in history[-min(len(history), 3):]],
        )
        if selected_runs:
            merged_rows = []
            for item in history:
                if item["name"] not in selected_runs:
                    continue
                tmp = item["df"][["task_id", "strategy", "passed", "compile_ok", "reward"]].copy()
                tmp["run_name"] = item["name"]
                merged_rows.append(tmp)
            compare_df = pd.concat(merged_rows, ignore_index=True) if merged_rows else pd.DataFrame()
            pivot = compare_df.pivot_table(index="task_id", columns="run_name", values="passed", aggfunc="max")
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
                            "run_a": a["name"],
                            "run_b": b["name"],
                            "common_tasks": len(joined),
                            "a_only_pass": a_only,
                            "b_only_pass": b_only,
                            "both_pass": both_pass,
                            "both_fail": both_fail,
                            "net_a_minus_b": a_only - b_only,
                        })
                st.dataframe(pd.DataFrame(pair_rows), use_container_width=True, height=260)

        all_csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download comparison CSV", all_csv, "run_comparison.csv", "text/csv", use_container_width=True)
        history_json = RUN_HISTORY_FILE.read_bytes() if RUN_HISTORY_FILE.exists() else json.dumps([], indent=2).encode("utf-8")
        st.download_button("Download full run history JSON", history_json, "streamlit_run_history.json", "application/json", use_container_width=True)
