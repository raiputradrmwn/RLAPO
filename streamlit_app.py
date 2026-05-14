import ast
import gzip
import json
import random
import sys
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
HUMANEVAL_FILE = ROOT / "HumanEval.jsonl.gz"
APP_TITLE = "RL Prompt Engineering Lab"
PROMPT_EDITOR_VERSION = "optimized_prompts_v3"
STRATEGY_NAMES = {
    0: "zero_shot",
    1: "few_shot",
    2: "cot",
    3: "hint",
}


def init_state():
    st.session_state.setdefault("run_history", [])
    st.session_state.setdefault("last_results", None)


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
    return {
        "run_name": run_name,
        "mode": config["mode"],
        "strategy": "bandit" if config["mode"] == "Online Bandit" else config["strategy"],
        "model": config["model_name"],
        "tasks": config["num_tasks"],
        "repeats": config["repeats"],
        "generations": int(len(df)),
        "pass_at_1": pass_at_1,
        "compile_rate": compile_rate,
        "avg_reward": avg_reward,
        "passed": int(df["passed"].sum()) if not df.empty else 0,
        "alpha": config["alpha"],
        "force_explore": config["force_explore"],
        "seed": config["seed"],
        "chat_template": config["use_chat_template"],
    }
DEFAULT_PROMPTS = {
    "zero_shot": """You are an expert Python programmer. Complete the function by writing ONLY the code that goes inside the function body. Do NOT repeat the function signature, do NOT add comments, do NOT write tests.

Focus on correctness. Read the docstring carefully, handle edge cases, and prefer simple Python standard-library code.

Complete this function:

{prompt}""",
    "few_shot": """You are an expert Python programmer. Complete the function by writing ONLY the code that goes inside the function body. Do NOT repeat the function signature, do NOT add comments, do NOT write tests.

Follow the examples exactly: output only the missing indented function body.

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

Example 4:
Problem:
def is_sorted(nums):
    \"\"\"Returns True if nums is sorted in nondecreasing order\"\"\"

Solution:
    return all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1))

Now complete this function:

{prompt}""",
    "cot": """Think step by step internally about how to solve this problem, including edge cases, then write the code. Your final answer must be ONLY the code that goes inside the function body. Do NOT repeat the function signature, do NOT add comments, do NOT write tests.

Problem:
{prompt}

Code:""",
    "hint": """You are an expert Python programmer. Complete the function by writing ONLY the code that goes inside the function body. Do NOT repeat the function signature, do NOT add comments, do NOT write tests.

{hint}

Before writing code, consider boundary cases such as empty inputs, duplicates, negative numbers, ordering, and type-specific behavior when relevant. Output only the final function body.

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
    result = {"success": False, "error": None}

    def target():
        try:
            exec_globals = {}
            exec(code, exec_globals)
            result["success"] = True
        except Exception as exc:
            result["error"] = repr(exc)

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        return False, "TIMEOUT"
    return result["success"], result.get("error")


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
    def __init__(self, num_arms=4, dim_context=19, alpha=0.15, force_explore=12):
        self.num_arms = num_arms
        self.dim_context = dim_context
        self.alpha = alpha
        self.force_explore = force_explore
        self.t = 0
        self.arm_priors = [0.0, 0.02, 0.08, 0.10]
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
            A_inv = np.linalg.pinv(self.A[arm])
            mean = float((self.theta[arm].T @ context)[0, 0])
            uncertainty = float((self.alpha * np.sqrt(context.T @ A_inv @ context))[0, 0])
            history = float(np.mean(self.arm_rewards[arm])) if self.arm_rewards[arm] else 0.0
            scores.append(mean + uncertainty + history * 0.5 + self.arm_priors[arm])
        arm = int(np.argmax(scores))
        self.arm_counts[arm] += 1
        return arm

    def update(self, arm, context, reward):
        context = context.reshape(-1, 1)
        self.A[arm] += context @ context.T
        self.b[arm] += reward * context
        self.arm_rewards[arm].append(reward)
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


def generate_completion(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float, do_sample: bool):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
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


def run_experiment(config: dict, prompt_templates: dict, problems: dict):
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
            step += 1
            if bandit:
                features = np.array(extract_features(problem, tokenizer))
                arm = bandit.select_arm(features)
                strategy = STRATEGY_NAMES[arm]
            else:
                strategy = config["strategy"]
                arm = {name: idx for idx, name in STRATEGY_NAMES.items()}[strategy]
                features = np.array(extract_features(problem, tokenizer))

            prompt = build_prompt(
                prompt_templates[strategy],
                problem,
                tokenizer=tokenizer,
                use_chat_template=config["use_chat_template"],
            )
            generated = generate_completion(
                tokenizer,
                model,
                prompt,
                config["max_new_tokens"],
                config["temperature"],
                config["do_sample"],
            )
            completion = extract_code(generated)
            passed, compile_ok, reward, error = evaluate_sample(problem, completion, config["timeout"])
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
                "generated_chars": len(generated),
                "completion_chars": len(completion),
            })
            current_pass = sum(r["passed"] for r in rows)
            current_compile = sum(r["compile_ok"] for r in rows)
            current_reward = sum(r["reward"] for r in rows) / len(rows)
            current_pass_at_1 = current_pass / len(rows) * 100
            current_compile_rate = current_compile / len(rows) * 100
            current_fail = len(rows) - current_pass
            progress.progress(
                step / total_steps,
                text=f"Running {step}/{total_steps}: {task_id} with {strategy}",
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
                st.caption(f"Last task: {task_id} | strategy: {strategy} | outcome: {outcome} | reward={reward:.1f}")
            log_lines.append(f"[{step:03d}/{total_steps:03d}] {task_id} | {strategy} | {outcome} | reward={reward:.1f}")
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
    timeout = st.slider("Timeout evaluasi/detik", 1, 30, 10)
    shuffle = st.checkbox("Shuffle soal", value=False, help="Matikan untuk mereplikasi notebook bandit68%. Online bandit sensitif terhadap urutan task.")
    seed = st.number_input("Seed", value=42, step=1)
    alpha = st.slider("Bandit alpha", 0.0, 2.0, 0.15, 0.05, disabled=mode != "Online Bandit", help="Rekomendasi dashboard: 0.10-0.20. Lebih kecil berarti lebih cepat exploit arm terbaik.")
    force_explore = st.slider("Force explore steps", 0, 80, 12, disabled=mode != "Online Bandit", help="Rekomendasi dashboard: 8-16. Force explore 40 terlalu lama jika beberapa arm lemah.")
    if st.button("Clear saved runs", use_container_width=True):
        st.session_state["run_history"] = []
        st.session_state["last_results"] = None
        st.rerun()

tab_prompts, tab_run, tab_results, tab_compare = st.tabs(["Prompt Lab", "Run", "Latest Results", "Compare Runs"])

with tab_prompts:
    st.subheader("Prompt Templates")
    st.write("Gunakan `{prompt}`, `{task_id}`, `{entry_point}`, dan `{hint}` sebagai placeholder.")
    if st.button("Reset prompt editor to optimized defaults", use_container_width=True):
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
        Klik tombol di bawah untuk mulai. Hasil run akan otomatis disimpan di session sehingga kamu bisa menjalankan Bandit, lalu Zero-shot, Few-shot, CoT, dan Hint, kemudian membandingkannya di tab <b>Compare Runs</b>. Default sekarang memakai optimized prompts v3 dan bandit yang lebih cepat exploit arm kuat.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if mode == "Online Bandit":
        st.caption("Rekomendasi saat ini untuk mengejar skor lebih tinggi: 164 soal, repeat 1, shuffle nonaktif dulu, chat template aktif, alpha 0.15, force explore 12, max_new_tokens 512, sampling off. Jika hasil terlalu bias ke satu arm, coba alpha 0.20 atau force explore 16.")
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
        pass_at_1 = df["passed"].mean() * 100
        avg_reward = df["reward"].mean()
        compile_rate = df["compile_ok"].mean() * 100
        fail_rate = 100 - pass_at_1
        total = len(df)
        passed_count = int(df["passed"].sum())
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Pass@1", f"{pass_at_1:.1f}%")
        col2.metric("Avg Reward", f"{avg_reward:.3f}")
        col3.metric("Compile OK", f"{compile_rate:.1f}%")
        col4.metric("Passed", f"{passed_count}/{total}")
        col5.metric("Fail Rate", f"{fail_rate:.1f}%")

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

        compact_cols = ["repeat", "task_id", "strategy", "passed", "compile_ok", "reward", "generated_chars", "completion_chars", "error"]
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
        summary_df = summary_df.sort_values("pass_at_1", ascending=False).reset_index(drop=True)
        best = summary_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best run", best["strategy"])
        c2.metric("Best Pass@1", f"{best['pass_at_1']:.1f}%")
        c3.metric("Best Avg Reward", f"{best['avg_reward']:.3f}")
        c4.metric("Saved runs", len(history))

        st.subheader("Leaderboard")
        display_cols = [
            "run_name",
            "mode",
            "strategy",
            "generations",
            "passed",
            "pass_at_1",
            "compile_rate",
            "avg_reward",
            "seed",
            "chat_template",
        ]
        st.dataframe(summary_df[display_cols], use_container_width=True, height=300)

        chart_df = summary_df.set_index("run_name")[["pass_at_1", "compile_rate", "avg_reward"]]
        st.subheader("Metric Comparison")
        st.bar_chart(chart_df[["pass_at_1", "compile_rate"]])
        st.caption("Avg reward has a different scale, so it is shown separately.")
        st.bar_chart(chart_df[["avg_reward"]])

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

        all_csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download comparison CSV", all_csv, "run_comparison.csv", "text/csv", use_container_width=True)
