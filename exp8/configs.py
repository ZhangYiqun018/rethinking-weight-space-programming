"""Sampling configurations and benchmarks for the Exp8 config matrix (REPRO_PLAN.md §3, Phase 2a).

Config A is the blog's protocol. B is the model card's non-thinking benchmark setting. C, D and E keep thinking on
but remove or counter the looping in different ways. Every config is applied identically to the base and the edited
model.
"""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Same user-turn instruction as the authors' math evaluation, so that config A reproduces their protocol (H1).
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

THINK_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
NOTHINK_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}

CONFIGS = {
    "A": {"thinking": True, "sampling": {**THINK_SAMPLING, "presence_penalty": 1.5}, "max_tokens": 32768},
    "B": {"thinking": False, "sampling": {**NOTHINK_SAMPLING, "presence_penalty": 1.5}, "max_tokens": 8192},
    "C": {"thinking": True, "sampling": {**THINK_SAMPLING, "presence_penalty": 0.0}, "max_tokens": 32768},
    "D": {"thinking": True, "sampling": {**THINK_SAMPLING, "presence_penalty": 1.5, "repetition_penalty": 1.1},
          "max_tokens": 32768},
    # Budget forcing: thinking is cut at think_budget tokens, then </think> is forced and the answer gets
    # answer_budget more tokens, so the total stays within config A's 32,768.
    "E": {"thinking": True, "sampling": {**THINK_SAMPLING, "presence_penalty": 1.5}, "max_tokens": 32768,
          "think_budget": 16384, "answer_budget": 4096},
}

# Qwen's documented phrase for stopping thinking early.
FORCE_CLOSE = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"

SAMPLES = {"math500": 4, "gsm8k": 2, "ifeval": 2, "practice_diag": 4, "practice_select": 2}

PRACTICE_SPLIT_SIZE = 128  # per source (GSM8K train, MATH train), per split


def practice_split(name):
    """Disjoint practice splits for the scripted steering baseline (Phase 2b), drawn from the authors' practice pool
    (GSM8K train, MATH train; no overlap with the test sets): 'diag' finds the direction, 'select' picks the strength."""
    import random
    out = []
    for source in ("gsm8k_train", "math_train"):
        rows = [json.loads(line) for line in open(os.path.join(DATA_DIR, "practice", f"{source}.jsonl"))]
        random.Random(f"wsp-{source}").shuffle(rows)
        start = {"diag": 0, "select": PRACTICE_SPLIT_SIZE}[name]
        out += rows[start:start + PRACTICE_SPLIT_SIZE]
    return [{"id": r["id"], "prompt": r["problem"] + "\n" + MATH_INSTRUCTION, "answer": r["answer"]} for r in out]


def load_benchmark(name):
    """Rows with id, prompt (the user turn) and the fields the grader needs."""
    if name.startswith("practice_"):
        return practice_split(name[len("practice_"):])
    path = os.path.join(DATA_DIR, {"math500": "math500.jsonl", "gsm8k": "gsm8k.jsonl", "ifeval": "ifeval.jsonl"}[name])
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if name == "math500":
        return [{"id": r["unique_id"], "prompt": r["problem"] + "\n" + MATH_INSTRUCTION, "answer": r["answer"]}
                for r in rows]
    if name == "gsm8k":
        return [{"id": r["id"], "prompt": r["problem"] + "\n" + MATH_INSTRUCTION, "answer": r["answer"]} for r in rows]
    return [{"id": f"ifeval-{r['key']}", "prompt": r["prompt"], "instruction_id_list": r["instruction_id_list"],
             "kwargs": r["kwargs"]} for r in rows]
