"""Parse every saved reply -> results/networks/*.npz, evaluate with ../eval_minatar.py, write scores + summary.

    python -I score_write.py [--raw results/raw] [--episodes 1000]

Evaluations are cached in results/eval_cache/<npz stem>.json (resume-safe). Authors' reference score per game comes
from ../results/phase1b.jsonl (our eval of their hand.npz, same protocol).
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python -I drops the script dir
from edits_parser import apply_reply, to_npz_arrays, write_npz

EVAL = os.path.join(HERE, "..", "eval_minatar.py")
PHASE1B = os.path.join(HERE, "..", "results", "phase1b.jsonl")
FIELDS = ["game", "condition", "rep", "model", "statements", "skipped", "nonzero", "mean", "se", "network"]


def authors_scores():
    ref = {}
    for line in open(PHASE1B):
        d = json.loads(line)
        if d["policy"].endswith("/hand.npz") and d["policy"].startswith("net:networks/"):
            ref[d["game"]] = d["mean"]
    return ref


def evaluate(game, npz, episodes, cache_dir):
    cache = os.path.join(cache_dir, os.path.basename(npz)[:-4] + f"__{episodes}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    out = subprocess.run([sys.executable, "-I", EVAL, "--game", game, "--policy", f"net:{npz}",
                          "--episodes", str(episodes)], capture_output=True, text=True, check=True).stdout
    res = json.loads(out.strip().splitlines()[-1])
    with open(cache, "w") as f:
        json.dump(res, f)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "results", "raw"))
    ap.add_argument("--episodes", type=int, default=1000)
    args = ap.parse_args()
    res_dir = os.path.dirname(os.path.abspath(args.raw))
    net_dir, cache_dir = os.path.join(res_dir, "networks"), os.path.join(res_dir, "eval_cache")
    os.makedirs(net_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    ref = authors_scores()
    rows = []
    for path in sorted(glob.glob(os.path.join(args.raw, "*.json"))):
        rec = json.load(open(path))
        game, condition = os.path.splitext(rec["prompt_file"])[0].split("__")
        res = apply_reply(rec["output_text"])
        npz = os.path.join(net_dir, os.path.basename(path)[:-5] + ".npz")
        write_npz(res, npz)
        nonzero = sum(int(np.count_nonzero(a)) for a in to_npz_arrays(res).values())
        ev = evaluate(game, npz, args.episodes, cache_dir)
        rows.append({"game": game, "condition": condition, "rep": rec["rep"], "model": rec["model_requested"],
                     "statements": res.statements, "skipped": res.skipped, "nonzero": nonzero,
                     "mean": round(ev["mean"], 3), "se": round(ev["se"], 3), "network": os.path.relpath(npz, HERE)})
        print(rows[-1], flush=True)
    with open(os.path.join(res_dir, "scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    summary = []
    for key in sorted({(r["game"], r["condition"]) for r in rows}):
        m = np.array([r["mean"] for r in rows if (r["game"], r["condition"]) == key])
        q1, med, q3 = np.percentile(m, [25, 50, 75])
        summary.append({"game": key[0], "condition": key[1], "n": len(m), "median": round(float(med), 3),
                        "q1": round(float(q1), 3), "q3": round(float(q3), 3), "authors_score": ref.get(key[0]),
                        "share_ge_0.9x_authors": round(float(np.mean(m >= 0.9 * ref[key[0]])), 3)})
    with open(os.path.join(res_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(summary[0])) if summary else None
        if w:
            w.writeheader()
            w.writerows(summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
