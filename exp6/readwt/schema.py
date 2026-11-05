"""Strict prediction schema shared by harness, fake subject, and scorer."""
import math

ROLES = ["b0", "b50", "ppo"]
RATES = ["capture_rate", "atari_escape_rate", "self_atari_rate", "eye_fill_rate",
         "pass_before_60_rate", "winrate_vs_random", "winrate_vs_R1"]


def probability(value):
    assert type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate(data, condition):
    names = ["X", "Y", "Z"] if condition == "W" else ROLES
    assert set(data) == {"networks", "pairwise"}
    assert set(data["networks"]) == set(names)
    for name, row in data["networks"].items():
        assert set(row) == set(RATES + ["mean_game_length", "first_move", "role",
                                       "role_confidence", "strategy"])
        for key in RATES:
            probability(row[key])
        length = row["mean_game_length"]
        assert type(length) in (int, float) and math.isfinite(length) and 2 <= length <= 162
        assert row["role"] in ROLES
        if condition == "W":
            probability(row["role_confidence"])
        else:
            assert row["role"] == name and row["role_confidence"] is None
        assert isinstance(row["strategy"], str) and len(row["strategy"].split()) <= 200
        first = row["first_move"]
        assert set(first) == {"probabilities", "top3", "p_tengen"}
        p = first["probabilities"]
        assert isinstance(p, list) and len(p) == 82
        for x in p:
            probability(x)
        assert abs(sum(p) - 1) < 1e-5
        assert abs(first["p_tengen"] - p[40]) < 1e-7
        probability(first["p_tengen"])
        assert isinstance(first["top3"], list) and len(first["top3"]) == 3
        seen = set()
        previous = 2.0
        for point in first["top3"]:
            assert set(point) == {"row", "col", "probability"}
            r, c = point["row"], point["col"]
            assert type(r) is int and type(c) is int and 0 <= r < 9 and 0 <= c < 9
            index = r * 9 + c
            assert index not in seen
            seen.add(index)
            probability(point["probability"])
            assert abs(point["probability"] - p[index]) < 1e-7
            assert point["probability"] <= previous + 1e-7
            previous = point["probability"]
        assert min(p[i] for i in seen) + 1e-7 >= max(p[i] for i in range(81) if i not in seen)
    pairs = [f"{names[i]}_vs_{names[j]}" for i in range(3) for j in range(i + 1, 3)]
    assert set(data["pairwise"]) == set(pairs)
    for p in data["pairwise"].values():
        probability(p)
    return data


def fake_prediction(condition):
    names = ["X", "Y", "Z"] if condition == "W" else ROLES
    networks = {}
    for i, name in enumerate(names):
        networks[name] = {**{key: 0.5 for key in RATES}, "mean_game_length": 100,
                          "first_move": {"probabilities": [1 / 82] * 82,
                                         "top3": [{"row": 0, "col": c, "probability": 1 / 82}
                                                  for c in range(3)],
                                         "p_tengen": 1 / 82},
                          "role": ROLES[i], "role_confidence": 1 / 3 if condition == "W" else None,
                          "strategy": "Scripted fake prediction for harness validation only."}
    return validate({"networks": networks,
                     "pairwise": {f"{names[i]}_vs_{names[j]}": 0.5
                                  for i in range(3) for j in range(i + 1, 3)}}, condition)
