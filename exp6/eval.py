"""Run with .venv/bin/python -I eval.py gate (or evaluate)."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def local_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = local_module("match")
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def run(a, b, seed, prefix, policies):
    path = RESULTS / f"{prefix}_{a}_vs_{b}.json"
    if path.exists():
        return json.loads(path.read_text())
    print(f"{prefix}: {a} vs {b}", flush=True)
    result = M.play(policies[a], policies[b], seed=seed)
    result.update(player_a=a, player_b=b)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def gate():
    policies = {f"b{i}": M.network(f"budget_{i}") for i in (0, 50)}
    policies.update({f"s{i}": M.network(f"ppo_same_size_lr3e-5_s{i}") for i in range(3)})
    expected = {"b0": [0.426, 0.263, 0.363], "b50": [0.742, 0.700, 0.785]}
    lines = ["# Validation gate", "", "2,000 games per pair; seed 31337; exactly 1,000 games per color.",
             "The unreleased ppo_ref helper prevents bitwise PRNG replication; the prescribed tolerance is used.",
             "", "| Astra | PPO | Published | Measured | 95% Wilson | Difference (pp) | Pass |",
             "|---|---|---:|---:|---|---:|---|"]
    passed = True
    for a in expected:
        for i, reference in enumerate(expected[a]):
            r = run(a, f"s{i}", 31337, "gate", policies)
            ok = abs(r["score"] - reference) <= 0.035
            passed &= ok
            lo, hi = r["wilson95"]
            lines.append(f"| {a} | s{i} | {reference:.1%} | {r['score']:.2%} | "
                         f"{lo:.2%}-{hi:.2%} | {100*(r['score']-reference):+.2f} | {ok} |")
    lines += ["", f"Gate: {'PASS' if passed else 'FAIL'}.",
              "", "Engine: pgx go v1, 9x9, komi 7.5, two-pass/repetition/162-move termination. "
              "Input is interleaved own/opponent/White-to-move; row-vector affine layers, six ReLUs; "
              "illegal actions have negative-infinite logits; categorical sampling at temperature one. "
              "Finished states are frozen and rewards are indexed by color, not randomized pgx player ID."]
    (ROOT / "VALIDATION.md").write_text("\n".join(lines) + "\n")
    return passed


def evaluate():
    assert "Gate: PASS." in (ROOT / "VALIDATION.md").read_text()
    assert (ROOT / "PREREG.md").exists()
    rules = local_module("rules")
    policies = {"R0": rules.r0, "R1": rules.r1, "random": M.random_policy}
    policies.update({f"b{i}": M.network(f"budget_{i}") for i in (0, 10, 25, 50)})
    policies.update({f"s{i}": M.network(f"ppo_same_size_lr3e-5_s{i}") for i in range(3)})
    opponents = ["random", "s0", "s1", "s2", "b0", "b10", "b25", "b50"]
    for index, a in enumerate(["R0", "R1"]):
        for j, b in enumerate(opponents):
            run(a, b, 61000 + 100 * index + j, "eval", policies)
    for j, a in enumerate(["b0", "b50"]):
        run(a, "random", 61200 + j, "eval", policies)
    report_results()


def report_results():
    def load(a, b):
        prefix = "gate" if a in ["b0", "b50"] and b.startswith("s") else "eval"
        path = RESULTS / f"{prefix}_{a}_vs_{b}.json"
        return json.loads(path.read_text()) if path.exists() else None

    lines = ["# Results", "", "Win rate of the row player; brackets are 95% Wilson intervals. "
             "Every measured cell uses 2,000 games, 1,000 per color. "
             "Unmeasured cells are marked --. Astra/PPO cells reuse the validation matches.",
             "", "| Player | Random | PPO s0 | PPO s1 | PPO s2 | Astra b0 | Astra b50 |",
             "|---|---|---|---|---|---|---|"]
    for a in ["R0", "R1", "b0", "b50"]:
        row = [a]
        for b in ["random", "s0", "s1", "s2", "b0", "b50"]:
            r = load(a, b)
            row.append("--" if r is None else
                       f"{r['score']:.2%} [{r['wilson95'][0]:.2%}, {r['wilson95'][1]:.2%}]")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "Additional preregistered matchups (also in raw JSON):", ""]
    for a in ["R0", "R1"]:
        for b in ["b10", "b25"]:
            r = load(a, b)
            lines.append(f"- {a} vs Astra {b}: {r['score']:.2%} "
                         f"[{r['wilson95'][0]:.2%}, {r['wilson95'][1]:.2%}].")
    lines += ["", "## Preregistered decision", ""]
    decisions = {}
    for a in ["R0", "R1"]:
        values = [load(a, f"s{i}")["score"] for i in range(3)]
        weak = all(x >= t for x, t in zip(values, [0.426, 0.263, 0.363]))
        strong = all(x >= t for x, t in zip(values, [0.742, 0.700, 0.785]))
        decisions[a] = (weak, strong)
        lines.append(f"- {a}: weakened criterion {'met' if weak else 'not met'}; "
                     f"strongly weakened criterion {'met' if strong else 'not met'}.")
    verdict = ("strongly weakened" if any(v[1] for v in decisions.values()) else
               "weakened" if any(v[0] for v in decisions.values()) else
               "neither weakened nor strongly weakened by the preregistered rule")
    lines += ["", f"Decision: **{verdict}**.", "",
              f"R1 wins only {load('R1', 'b0')['score']:.2%} directly against Astra b0 and "
              f"{load('R1', 'b50')['score']:.2%} against Astra b50. "
              "The PPO comparison therefore does not establish general playing-strength equivalence: "
              "the opponent-specific results are nontransitive. Beating these released PPO policies "
              "is achievable with a fixed tactical script, while Astra still decisively beats that script.", "",
              "This decision concerns the specified fixed-rule baseline comparison. "
              "Failure of these two heuristics would not establish that PPO is strong, "
              "nor prove that other fixed scripts cannot beat it.", "",
              "## Caveats", "",
              "- Hidden small PPO and practice weak/strong opponents were not released and cannot be tested.",
              "- Only two fixed heuristics were tested; no result-driven tuning, training, or sequence search.",
              "- Point-estimate thresholds use the published numbers, as preregistered. "
              "Intervals are per-match, not simultaneous; the published estimates also have sampling uncertainty.",
              "- The original shared evaluation helper and environment lockfile were absent. "
              "This independently implemented engine passed all six prescribed tolerance checks, "
              "but does not claim bitwise reproduction of the original PRNG stream.",
              "- R1 uses exact one-move group/liberty calculations and softmax(10 * score). "
              "Its raw-input access is comparable, but its computation is not constrained to the network architecture.",
              "- The 162-move cap, komi 7.5, and repetition-loss rule are pgx's task-specific Go setting."]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["gate", "evaluate"])
    args = parser.parse_args()
    if args.stage == "evaluate":
        evaluate()
    elif not gate():
        sys.exit(1)
