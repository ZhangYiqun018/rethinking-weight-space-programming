#!/usr/bin/env python3
"""Generate results/exp5_phase1b.md from exp5/results/phase1b.jsonl and the
authors' eval JSON. Numbers only; re-run to regenerate the file exactly."""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OURS = ROOT / "exp5" / "results" / "phase1b.jsonl"
THEIRS = ROOT / "upstream" / "lm_understand-logs" / "exp5_write_from_scratch" / "eval" / "minatar_eval.json"
OUT = ROOT / "results" / "exp5_phase1b.md"

GAMES = ["breakout", "seaquest", "asterix"]
BUDGETS = [("ppo100", "100M"), ("ppo500", "500M")]
REF_POLICIES = ["random", "heuristic", "heuristic_keepdir", "heuristic_sim"]


def f1(x):
    return f"{x:.1f}"


ours = [json.loads(line) for line in OURS.read_text().splitlines() if line.strip()]
by_game_policy = {(r["game"], r["policy"]): r for r in ours}
theirs = json.loads(THEIRS.read_text())

lines = []
lines.append("# Exp5 (MinAtar) — phase 1b results tables")
lines.append("")

# Table A: hand network, authors vs ours
lines.append("## Table A — Astra network: authors' eval vs our re-eval (1,000 episodes each)")
lines.append("")
lines.append("Rows are the Astra-written network (`hand.npz`) per game; diff = ours − authors; "
             "diff/SE = diff in units of the combined SE sqrt(se1² + se2²).")
lines.append("")
lines.append("| game | authors mean ± se | ours mean ± se | diff | diff (combined SE) |")
lines.append("|---|---|---|---|---|")
for g in GAMES:
    t = theirs[g]["hand"]
    o = by_game_policy[(g, f"net:networks/{g}/hand.npz")]
    diff = o["mean"] - t["mean"]
    comb = math.sqrt(t["se"] ** 2 + o["se"] ** 2)
    lines.append(
        f"| {g} | {f1(t['mean'])} ± {f1(t['se'])} | {f1(o['mean'])} ± {f1(o['se'])} "
        f"| {f1(diff)} | {f1(diff / comb)} |"
    )
lines.append("")

# Table B: authors' PPO seeds
lines.append("## Table B — authors' PPO seeds, per game and budget")
lines.append("")
lines.append("Rows are the authors' PPO evaluations per game and training budget; "
             "best/median/worst are over per-seed mean returns; blog seed = the seed the blog showed (the best one).")
lines.append("")
lines.append("| game | budget | seeds | best | median | worst | blog seed |")
lines.append("|---|---|---|---|---|---|---|")
for g in GAMES:
    for prefix, label in BUDGETS:
        seeds = {k: v for k, v in theirs[g].items() if k.startswith(prefix + "_s")}
        means = sorted(v["mean"] for v in seeds.values())
        best_key = max(seeds, key=lambda k: seeds[k]["mean"])
        lines.append(
            f"| {g} | {label} | {len(seeds)} | {f1(means[-1])} | {f1(statistics.median(means))} "
            f"| {f1(means[0])} | {best_key} |"
        )
lines.append("")

# Table C: our reference policies
lines.append("## Table C — our reference policies (1,000 episodes each)")
lines.append("")
lines.append("Rows are our reference-policy evaluations on official MinAtar 1.0.15; "
             "cut = share of episodes truncated at 10,000 steps.")
lines.append("")
lines.append("| game | policy | mean ± se | mean episode length | cut at 10,000 steps |")
lines.append("|---|---|---|---|---|")
for g in GAMES:
    for p in REF_POLICIES:
        key = (g, p)
        if key not in by_game_policy:
            continue
        r = by_game_policy[key]
        lines.append(
            f"| {g} | {p} | {f1(r['mean'])} ± {f1(r['se'])} | {f1(r['mean_length'])} "
            f"| {r['cut_at_max_steps'] * 100:.1f}% |"
        )
lines.append("")

OUT.write_text("\n".join(lines))
print(f"wrote {OUT}")
