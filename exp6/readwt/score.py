"""Score per-role predictions, paired W/noW errors, and collect forward-audit code."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("readwt_schema", ROOT / "schema.py")
SCHEMA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SCHEMA)
PAIRS = [("b0", "b50"), ("b0", "ppo"), ("b50", "ppo")]


def truth():
    roles = {r: json.loads((ROOT / "truth" / f"{r}.json").read_text()) for r in SCHEMA.ROLES}
    pairs = json.loads((ROOT / "truth/pairwise.json").read_text())
    return roles, {name: row["probability"] for name, row in pairs.items()}


def score_prediction(data, condition, key, actual, actual_pairs):
    SCHEMA.validate(data, condition)
    mapping = key if condition == "W" else {r: r for r in SCHEMA.ROLES}
    assert set(mapping.values()) == set(SCHEMA.ROLES)
    role_errors, identities = {}, {}
    for name, row in data["networks"].items():
        role = mapping[name]
        target = actual[role]
        errors = {metric: 100 * abs(row[metric] - target[metric]) for metric in SCHEMA.RATES}
        errors["mean_game_length"] = abs(row["mean_game_length"] - target["mean_game_length"])
        first = row["first_move"]
        actual_first = target["first_move"]
        point_errors = [100 * abs(a - b) for a, b in
                        zip(first["probabilities"], actual_first["probabilities"])]
        errors["first_move_probability"] = statistics.mean(point_errors)
        errors["p_tengen"] = 100 * abs(first["p_tengen"] - actual_first["p_tengen"])
        predicted_top = {p["row"] * 9 + p["col"] for p in first["top3"]}
        # Credit ties at the true third-largest board-point probability.
        threshold = sorted(actual_first["probabilities"][:81], reverse=True)[2]
        top_accuracy = sum(actual_first["probabilities"][i] >= threshold - 1e-10
                           for i in predicted_top) / 3
        errors["first_move_top3_error"] = 100 * (1 - top_accuracy)
        role_errors[role] = {"absolute_errors": errors,
                             "first_move_point_errors_pp": point_errors,
                             "top3_accuracy": top_accuracy}
        if condition == "W":
            identities[role] = {"label": name, "predicted_role": row["role"],
                                "correct": row["role"] == role,
                                "confidence": row["role_confidence"]}
    predicted_pairs = {}
    for name, probability in data["pairwise"].items():
        a, b = name.split("_vs_")
        a, b = mapping[a], mapping[b]
        if (a, b) not in PAIRS:
            a, b, probability = b, a, 1 - probability
        predicted_pairs[f"{a}_vs_{b}"] = probability
    pair_errors = {pair: 100 * abs(predicted_pairs[pair] - target)
                   for pair, target in actual_pairs.items()}
    means = {metric: statistics.mean(r["absolute_errors"][metric] for r in role_errors.values())
             for metric in next(iter(role_errors.values()))["absolute_errors"]}
    means["pairwise_winrate"] = statistics.mean(pair_errors.values())
    return {"role_errors": role_errors, "pairwise_errors_pp": pair_errors,
            "mean_absolute_errors": means, "identities": identities,
            "role_accuracy": statistics.mean(r["correct"] for r in identities.values())
            if identities else None}


def forward_audit(sessions):
    lines = ["# Forward audit", "",
             "This is a conservative review queue, not proof of compliance. All W code blocks "
             "are reproduced, including blocks with no recognized matrix operations. This covers "
             "aliased functions, dynamically loaded code, persistent files, and computations split "
             "across turns that static syntax cannot reliably classify.",
             "", "Products of weight matrices and single-layer responses are allowed; flags alone "
             "are not violations. The experimenter must determine whether an input was propagated "
             "through the full policy to logits, including across blocks.", ""]
    for directory in sorted(sessions.glob("*")):
        state_path = directory / "state.json"
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text())
        if state["metadata"]["condition"] != "W":
            continue
        log = directory / "turns.jsonl"
        if not log.exists():
            continue
        lines += [f"## {directory.name}", "", f"Fake: {state['metadata']['fake']}.", ""]
        for raw in log.read_text().splitlines():
            event = json.loads(raw)
            if event["event"] != "code":
                continue
            code = event["code"]
            try:
                tree = ast.parse(code)
                ops = [node for node in ast.walk(tree)
                       if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult))
                       or (isinstance(node, ast.Call) and
                           ((isinstance(node.func, ast.Attribute) and node.func.attr in
                             {"dot", "matmul", "einsum", "tensordot", "multi_dot", "mm", "bmm", "gemm"})
                            or (isinstance(node.func, ast.Name) and node.func.id in
                                {"dot", "matmul", "einsum", "tensordot", "multi_dot", "mm", "bmm", "gemm"})))]
                loops = any(isinstance(n, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp))
                            for n in ast.walk(tree))
                flagged = len(ops) > 1 or (bool(ops) and loops) or "multi_dot" in code
                note = f"Recognized matrix operations: {len(ops)}; loop/comprehension: {loops}; multi-layer candidate: {flagged}."
            except SyntaxError:
                note = "Unparsed code: manual review required."
            lines += [f"### Turn {event['turn']}, run {event['run']}", "", note, "",
                      "````python", code, "````", ""]
    (ROOT / "forward_audit.md").write_text("\n".join(lines) + "\n")


def main(args):
    assert args.reps > 0, "--reps must be positive"
    actual, pairs = truth()
    sessions = ROOT / "sessions"
    rows, missing = [], []
    for condition in ["W", "noW"]:
        for rep in range(args.reps):
            directory = sessions / f"{'fake_' if args.fake else ''}{condition}_r{rep}"
            predictions = directory / "predictions.json"
            if not predictions.exists():
                missing.append(str(directory))
                continue
            state = json.loads((directory / "state.json").read_text())
            assert state["phase"] == "done"
            assert state["metadata"]["fake"] == args.fake
            data = json.loads(predictions.read_text())
            key = json.loads((directory / "key.json").read_text()) if condition == "W" else None
            rows.append({"condition": condition, "rep": rep, "session": str(directory),
                         **score_prediction(data, condition, key, actual, pairs)})
    summary = {}
    for condition in ["W", "noW"]:
        selected = [r for r in rows if r["condition"] == condition]
        if not selected:
            continue
        summary[condition] = {
            "completed_reps": len(selected),
            "mean_absolute_errors": {q: statistics.mean(r["mean_absolute_errors"][q] for r in selected)
                                     for q in selected[0]["mean_absolute_errors"]},
            "per_role_mae": {role: {q: statistics.mean(r["role_errors"][role]["absolute_errors"][q]
                                                     for r in selected)
                                    for q in selected[0]["role_errors"][role]["absolute_errors"]}
                             for role in SCHEMA.ROLES},
            "role_accuracy": statistics.mean(r["role_accuracy"] for r in selected)
            if condition == "W" else None}
    paired = []
    for rep in range(args.reps):
        w = next((r for r in rows if r["condition"] == "W" and r["rep"] == rep), None)
        no = next((r for r in rows if r["condition"] == "noW" and r["rep"] == rep), None)
        if w is not None and no is not None:
            paired.append({"rep": rep,
                           "W_minus_noW": {q: w["mean_absolute_errors"][q] - no["mean_absolute_errors"][q]
                                           for q in w["mean_absolute_errors"]}})
    differences = {q: {"mean_W_minus_noW": statistics.mean(r["W_minus_noW"][q] for r in paired),
                       "paired_differences": [r["W_minus_noW"][q] for r in paired],
                       "n_pairs": len(paired)}
                   for q in paired[0]["W_minus_noW"]} if paired else {}
    result = {"fake": args.fake, "expected_reps_per_condition": args.reps,
              "complete": not missing, "missing": missing, "sessions": rows,
              "summary": summary, "paired": paired, "paired_summary": differences,
              "units": "percentage points except mean_game_length (moves); role/top3 accuracy is a fraction",
              "interpretation": "Negative W_minus_noW means weights improved accuracy. "
                                "Fake sessions are pipeline tests, never scientific evidence. "
                                "No composite average mixes move-count and probability errors."}
    destination = ROOT / ("fake_scores.json" if args.fake else "scores.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    forward_audit(sessions)
    print(f"Saved {destination}; {len(rows)} sessions; {len(paired)} pairs; missing={len(missing)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--reps", type=int, default=3)
    main(parser.parse_args())
