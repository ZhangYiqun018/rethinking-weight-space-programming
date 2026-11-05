"""Verify all saved games and record provenance without executing upstream files."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = sorted((ROOT / "results").glob("*.json"))
    assert len(paths) == 24, len(paths)
    expected = {f"gate_b{budget}_vs_s{seed}" for budget in [0, 50] for seed in range(3)}
    expected |= {f"eval_{rule}_vs_{opp}" for rule in ["R0", "R1"]
                 for opp in ["random", "s0", "s1", "s2", "b0", "b10", "b25", "b50"]}
    expected |= {"eval_b0_vs_random", "eval_b50_vs_random"}
    assert {p.stem for p in paths} == expected
    total = 0
    for path in paths:
        r = json.loads(path.read_text())
        records = r["records"]
        assert r["games"] == len(records) == 2000
        assert sum(x["a_black"] for x in records) == 1000
        assert sum(x["a_won"] for x in records) == r["wins"]
        assert r["score"] == r["wins"] / 2000
        if path.stem.startswith("gate_"):
            expected_seed = 31337
        elif r["player_a"] in ["R0", "R1"]:
            opponents = ["random", "s0", "s1", "s2", "b0", "b10", "b25", "b50"]
            expected_seed = 61000 + 100 * (r["player_a"] == "R1") + opponents.index(r["player_b"])
        else:
            expected_seed = 61200 + (r["player_a"] == "b50")
        assert r["seed"] == expected_seed and r["batch_size"] == 250
        assert all(2 <= x["length"] <= 162 for x in records)
        assert all(x["two_passes"] or x["repetition"] or x["length"] == 162 for x in records)
        assert r["wilson95"][0] <= r["score"] <= r["wilson95"][1]
        p, n, z = r["score"], 2000, 1.959963984540054
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        assert abs(r["wilson95"][0] - (center - radius)) < 1e-12
        assert abs(r["wilson95"][1] - (center + radius)) < 1e-12
        assert abs(sum(x["length"] for x in records) / 2000 - r["mean_length"]) < 1e-12
        for color, key in [(True, "score_as_black"), (False, "score_as_white")]:
            assert sum(x["a_won"] for x in records if x["a_black"] == color) / 1000 == r[key]
        total += len(records)
    networks = ROOT.parent / "upstream/lm_understand-logs/exp6_go_from_scratch/networks"
    names = [f"budget_{i}" for i in [0, 10, 25, 50]]
    names += [f"ppo_same_size_lr3e-5_s{i}" for i in range(3)]
    manifest = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ["pgx", "jax", "jaxlib", "numpy"]},
        "device": "JAX CPU",
        "total_games": total,
        "source_sha256": {p.name: digest(p) for p in sorted(ROOT.glob("*.py"))},
        "prereg_sha256": digest(ROOT / "PREREG.md"),
        "requirements_sha256": digest(ROOT / "requirements.txt"),
        "results_sha256": {p.name: digest(p) for p in paths},
        "network_sha256": {str(p.relative_to(networks)): digest(p)
                           for name in names for p in sorted((networks / name).glob("*.npy"))},
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PASS: {len(paths)} matches, {total:,} games; balanced colors, aggregates, "
          "lengths and terminal flags verified; SHA-256 provenance saved.")


if __name__ == "__main__":
    main()
