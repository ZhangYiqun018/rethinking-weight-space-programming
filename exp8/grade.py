"""Grade generation files and compare models.

Math (MATH-500, GSM8K):
- Primary rule, same as the authors': read the text after the last </think> (the whole text if there is none),
  take its last \\boxed{}, and check it with math-verify against the gold answer.
- Strict rule: with thinking on, an answer cut off before </think> counts as no answer.

IFEval: strict prompt-level accuracy with the official instruction checkers (via lm-eval), on the text after the
last </think>; empty if thinking was on and never closed.

Confidence intervals come from a bootstrap over problems: all samples of one problem are resampled together, and
edited-minus-base differences are paired on the same problems.

    python grade.py results/raw results/summary.json
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# IFEval checkers tokenize with NLTK punkt_tab. Its data (English only) ships in nltk_data/ next to this file,
# so grading never downloads. Install punkt_tab into nltk_data/ before grading.
NLTK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
# NLTK refuses data that other accounts can write to; extraction can leave it group/world-writable.
for _root, _dirs, _files in os.walk(NLTK_DIR):
    for _p in [_root] + [os.path.join(_root, f) for f in _files]:
        os.chmod(_p, os.stat(_p).st_mode & ~0o022)
os.environ["NLTK_DATA"] = NLTK_DIR
from configs import CONFIGS, load_benchmark  # noqa: E402

# <model>__<config>__<bench>__s<seed>.jsonl; the first smoke run used single underscores
FILE_RES = [re.compile(r"(?P<model>[a-z0-9.-]+)__(?P<config>[A-Z])__(?P<bench>[a-z0-9_]+)__s(?P<seed>\d+)\.jsonl$"),
            re.compile(r"(?P<model>[a-z]+)_(?P<config>[A-Z])_(?P<bench>[a-z0-9]+)_s(?P<seed>\d+)\.jsonl$")]


def answer_part(text):
    return text.split("</think>")[-1] if "</think>" in text else text


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
    return None


def math_correct(pred, gold):
    from math_verify import parse, verify
    if pred is None:
        return False
    try:
        return bool(verify(parse("\\boxed{" + gold + "}"), parse("\\boxed{" + pred + "}")))
    except Exception:
        return False


def ifeval_correct(row, response):
    from lm_eval.tasks.ifeval import instructions_registry
    import langdetect  # the language checker is random unless seeded; seeded so regrading is reproducible
    langdetect.DetectorFactory.seed = 0
    if not response.strip():
        return False
    for k, iid in enumerate(row["instruction_id_list"]):
        ins = instructions_registry.INSTRUCTION_DICT[iid](iid)
        ins.build_description(**{a: v for a, v in row["kwargs"][k].items() if v is not None})
        args = ins.get_instruction_args()
        if args and "prompt" in args:
            ins.build_description(prompt=row["prompt"])
        if not ins.check_following(response):
            return False
    return True


def grade_file(path, bench, thinking):
    rows = {r["id"]: r for r in load_benchmark(bench)}
    per_problem = defaultdict(lambda: defaultdict(list))
    for rec in map(json.loads, open(path)):
        row, text = rows[rec["id"]], rec["text"]
        closed = "</think>" in text or not thinking
        if bench == "ifeval":
            ok = ifeval_correct(row, answer_part(text).strip() if closed else "")
            strict = ok
        else:
            ok = math_correct(last_boxed(answer_part(text)), row["answer"])
            strict = ok and closed
        p = per_problem[rec["id"]]
        p["acc"].append(ok)
        p["acc_strict"].append(strict)
        p["cut"].append(rec["finish_reason"] == "length" or rec.get("forced", False))
        p["tokens"].append(rec["completion_tokens"])
    return {pid: {k: np.mean(v) for k, v in d.items()} for pid, d in per_problem.items()}


def bootstrap(values, n=2000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(raw_dir, out_path):
    graded = {}
    for name in sorted(os.listdir(raw_dir)):
        m = next((m for m in (r.match(name) for r in FILE_RES) if m), None)
        if not m:
            continue
        key = (m["model"], m["config"], m["bench"], int(m["seed"]))
        probs = grade_file(os.path.join(raw_dir, name), m["bench"], CONFIGS[m["config"]]["thinking"])
        if not probs:
            print("empty, skipped:", name, flush=True)
            continue
        graded[key] = probs
        print("graded", name, flush=True)

    summary = {"runs": [], "paired": []}
    for (model, config, bench, seed), probs in sorted(graded.items()):
        entry = {"model": model, "config": config, "bench": bench, "seed": seed, "n_problems": len(probs)}
        for metric in ("acc", "acc_strict", "cut", "tokens"):
            mean, lo, hi = bootstrap([p[metric] for p in probs.values()])
            entry[metric] = {"mean": mean, "ci95": [lo, hi]}
        summary["runs"].append(entry)
    for (model, config, bench, seed), probs in sorted(graded.items()):
        if model == "base" or ("base", config, bench, seed) not in graded:
            continue
        base = graded[("base", config, bench, seed)]
        shared = sorted(set(probs) & set(base))
        for metric in ("acc", "acc_strict", "cut"):
            mean, lo, hi = bootstrap([probs[p][metric] - base[p][metric] for p in shared])
            summary["paired"].append({"model": model, "config": config, "bench": bench, "seed": seed, "metric": metric,
                                      "minus_base": mean, "ci95": [lo, hi], "n_problems": len(shared)})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(summary, open(out_path, "w"), indent=1)

    print(f"\n{'model':12s} {'cfg':3s} {'bench':16s} seed  acc%  [95% CI]        strict%  cut%   tokens")
    for e in summary["runs"]:
        a, s, c = e["acc"], e["acc_strict"], e["cut"]
        print(f"{e['model']:12s} {e['config']:3s} {e['bench']:16s} {e['seed']:4d}  {100*a['mean']:5.1f} "
              f"[{100*a['ci95'][0]:5.1f},{100*a['ci95'][1]:5.1f}]  {100*s['mean']:6.1f}  {100*c['mean']:5.1f}  "
              f"{e['tokens']['mean']:7.0f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
