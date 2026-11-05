"""Exp8 cross-config comparison: is the edited model under the blog's protocol (config A) better than the unedited
model under any other decoding setting? Paired bootstrap over problems (each problem's sample mean), per benchmark.

    uv run --no-project --with "math-verify[antlr4_13_2]==0.9.0" --with "lm-eval[ifeval]" --with numpy \
        python -I results/cross_config.py > results/exp8_cross_config.md
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp8"))
from configs import CONFIGS  # noqa: E402
from grade import grade_file  # noqa: E402

ART = os.path.join(ROOT, "results", "runs")
TASK_OF = {"A": "exp8_config_A", "B": "exp8_config_BE", "C": "exp8_config_C", "D": "exp8_config_D", "E": "exp8_config_BE"}
BENCHES = ["math500", "gsm8k", "ifeval"]
REF = ("edited", "A")


def per_problem(model, cfg, bench):
    path = os.path.join(ART, TASK_OF[cfg], "raw", f"{model}__{cfg}__{bench}__s0.jsonl")
    return {pid: d["acc"] for pid, d in grade_file(path, bench, CONFIGS[cfg]["thinking"]).items()}


def paired(a, b, n=4000, seed=0):
    ids = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in ids])
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return 100 * d.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5), len(ids)


def main():
    print("# Exp8: edited model under the blog's protocol (A) vs every configuration\n")
    print("Accuracy % (mean over samples, then problems). Δ = edited-A minus the row, paired over problems, 95% CI.\n")
    for bench in BENCHES:
        ref = per_problem(*REF, bench)
        print(f"## {bench}\n\n| model | config | acc % | Δ (edited-A − row) | 95% CI |\n|---|---|---|---|---|")
        for model in ("base", "edited"):
            for cfg in "ABCDE":
                if (model, cfg) == REF:
                    print(f"| {model} | {cfg} | {100 * np.mean(list(ref.values())):.1f} | (reference) | |")
                    continue
                row = per_problem(model, cfg, bench)
                m, lo, hi, n = paired(ref, row)
                print(f"| {model} | {cfg} | {100 * np.mean(list(row.values())):.1f} | {m:+.1f} | [{lo:+.1f}, {hi:+.1f}] |")
        print()


if __name__ == "__main__":
    main()
