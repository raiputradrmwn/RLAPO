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
STRATEGY_NAMES = {
    0: "zero_shot",
    1: "few_shot",
    2: "cot",
    3: "hint",
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

Reasoning: Let me think about this step by step.""",
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


def extract_code(generated: str) -> str:
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


def extract_features(problem: dict) -> list[float]:
    prompt = problem["prompt"]
    docstring = prompt.split('"""')[1].lower() if '"""' in prompt and len(prompt.split('"""')) > 1 else prompt.lower()
    words = docstring.split()
    doc_len = len(words)
    return [
        min(len(prompt) / 2000.0, 1.0),
        min(doc_len / 100.0, 1.0),
        len(set(words)) / max(doc_len, 1),
        docstring.count("\n") / 10.0,
        float("return" in docstring),
        float("for " in docstring or "while " in docstring),
        float("if " in docstring),
        float("list" in docstring or "array" in docstring),
        float("string" in docstring or "text" in docstring),
        float("sort" in docstring or "sorted" in docstring),
        float("graph" in docstring or "node" in docstring or "edge" in docstring),
        float("tree" in docstring or "root" in docstring),
        float("dp" in docstring or "memo" in docstring or "cache" in docstring),
        float("recursive" in docstring),
        float("math" in docstring or "formula" in docstring),
    ]


class OnlineLinUCB:
    def __init__(self, num_arms=4, dim_context=15, alpha=0.3, force_explore=8):
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
    live_metrics = st.columns(4)
    log_box = st.container(border=True)
    log_lines = []

    total_steps = len(items) * config["repeats"]
    step = 0
    top_line.info(f"Run started at {st.session_state['run_started_at']} | total generations: {total_steps}")
    for repeat in range(config["repeats"]):
        for task_id, problem in items:
            step += 1
            if bandit:
                features = np.array(extract_features(problem))
                arm = bandit.select_arm(features)
                strategy = STRATEGY_NAMES[arm]
            else:
                strategy = config["strategy"]
                arm = {name: idx for idx, name in STRATEGY_NAMES.items()}[strategy]
                features = np.array(extract_features(problem))

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
            progress.progress(
                step / total_steps,
                text=f"Running {step}/{total_steps}: {task_id} with {strategy}",
            )
            live_metrics[0].metric("Progress", f"{step}/{total_steps}")
            live_metrics[1].metric("Live Pass@1", f"{current_pass_at_1:.1f}%")
            live_metrics[2].metric("Compile OK", f"{current_compile_rate:.1f}%")
            live_metrics[3].metric("Avg Reward", f"{current_reward:.3f}")
            outcome = "PASS" if passed else "FAIL"
            log_lines.append(f"[{step:03d}/{total_steps:03d}] {task_id} | {strategy} | {outcome} | reward={reward:.1f}")
            with log_box:
                st.code("\n".join(log_lines[-12:]), language="text")
    progress.progress(1.0, text="Experiment completed.")
    return pd.DataFrame(rows)


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #111827 0%, #1f2937 52%, #312e81 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {margin: 0; font-size: 2rem;}
    .hero p {margin: .45rem 0 0 0; color: #d1d5db;}
    .section-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem;
        background: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="hero">
      <h1>{APP_TITLE}</h1>
      <p>Interactive HumanEval runner untuk fixed prompting dan online contextual bandit. Edit prompt, jalankan model lokal, evaluasi Pass@1, dan ekspor hasil eksperimen.</p>
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
    shuffle = st.checkbox("Shuffle soal", value=True)
    seed = st.number_input("Seed", value=42, step=1)
    alpha = st.slider("Bandit alpha", 0.0, 2.0, 0.3, 0.05, disabled=mode != "Online Bandit")
    force_explore = st.slider("Force explore steps", 0, 80, 40, disabled=mode != "Online Bandit")

tab_prompts, tab_run, tab_results = st.tabs(["Prompt Lab", "Run", "Results"])

with tab_prompts:
    st.subheader("Prompt Templates")
    st.write("Gunakan `{prompt}`, `{task_id}`, `{entry_point}`, dan `{hint}` sebagai placeholder.")
    prompt_templates = {}
    cols = st.columns(2)
    for idx, name in enumerate(DEFAULT_PROMPTS):
        with cols[idx % 2]:
            prompt_templates[name] = st.text_area(name, DEFAULT_PROMPTS[name], height=260)
    preview_task = st.selectbox("Preview task", list(problems.keys()))
    preview_strategy = st.selectbox("Preview strategy", list(DEFAULT_PROMPTS), key="preview_strategy")
    st.code(build_prompt(prompt_templates[preview_strategy], problems[preview_task], use_chat_template=False), language="text")

with tab_run:
    st.subheader("Run Experiment")
    st.info("Klik tombol di bawah untuk mulai. Status loading model, progress, dan log task akan muncul real-time di area ini.")
    if mode == "Online Bandit":
        st.caption("Untuk mendekati hasil notebook 68.3%, pakai 164 soal, repeat 1, seed 42, shuffle aktif, chat template aktif, alpha 0.3, force explore 40, max_new_tokens 512, sampling off.")
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
    st.metric("Total generations", est_generations)
    if st.button("Start Experiment", type="primary", use_container_width=True):
        st.session_state["last_results"] = run_experiment(config, prompt_templates, problems)
        st.success("Experiment completed. Open the Results tab to inspect metrics and outputs.")

with tab_results:
    st.subheader("Latest Results")
    df = st.session_state.get("last_results")
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

        st.subheader("Raw Results")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        jsonl = "\n".join(json.dumps(row, ensure_ascii=False) for row in df.to_dict("records")).encode("utf-8")
        st.download_button("Download CSV", csv, "experiment_results.csv", "text/csv")
        st.download_button("Download JSONL", jsonl, "experiment_results.jsonl", "application/jsonl")
