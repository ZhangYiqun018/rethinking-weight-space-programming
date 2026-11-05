"""Offline tests: schema, role/permutation scoring, SSE, retry, and dry-run logs."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def local(name):
    spec = importlib.util.spec_from_file_location(f"test_{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H, S = local("run_subject"), local("score")


def prediction(actual, pairs, mapping, condition):
    data = H.SCHEMA.fake_prediction(condition)
    for label, role in mapping.items():
        row = data["networks"][label]
        for key in H.SCHEMA.RATES + ["mean_game_length", "first_move"]:
            row[key] = copy.deepcopy(actual[role][key])
        row["role"] = role
        row["role_confidence"] = 1 if condition == "W" else None
    for name in data["pairwise"]:
        a, b = [mapping[x] for x in name.split("_vs_")]
        data["pairwise"][name] = pairs[f"{a}_vs_{b}"] if (a, b) in S.PAIRS else 1 - pairs[f"{b}_vs_{a}"]
    return data


def main():
    actual, pairs = S.truth()
    for permutation in itertools.permutations(H.SCHEMA.ROLES):
        mapping = dict(zip(["X", "Y", "Z"], permutation))
        data = prediction(actual, pairs, mapping, "W")
        result = S.score_prediction(data, "W", mapping, actual, pairs)
        assert all(abs(v) < 1e-12 for v in result["mean_absolute_errors"].values())
        assert result["role_accuracy"] == 1
        bad = copy.deepcopy(data)
        bad["networks"]["X"]["capture_rate"] += 0.1
        bad["networks"]["X"]["mean_game_length"] += 3
        changed = S.score_prediction(bad, "W", mapping, actual, pairs)
        assert abs(changed["role_errors"][mapping["X"]]["absolute_errors"]["capture_rate"] - 10) < 1e-9
        assert changed["role_errors"][mapping["X"]]["absolute_errors"]["mean_game_length"] == 3
    data = prediction(actual, pairs, {r: r for r in H.SCHEMA.ROLES}, "noW")
    result = S.score_prediction(data, "noW", None, actual, pairs)
    assert all(v == 0 for v in result["mean_absolute_errors"].values())
    assert result["role_accuracy"] is None
    invalid = H.SCHEMA.fake_prediction("W")
    invalid["networks"]["X"]["first_move"]["probabilities"][0] = float("nan")
    try:
        H.SCHEMA.validate(invalid, "W")
    except AssertionError:
        pass
    else:
        raise AssertionError("NaN accepted")

    class Stream:
        def __enter__(self):
            return iter([b": heartbeat\n", b"data: invalid\n",
                         b'data: {"type":"response.output_text.delta","delta":"ignored"}\n',
                         b'data: {"type":"response.completed","response":{"output":[],"id":"offline"}}\n'])

        def __exit__(self, *args):
            return False

    def offline(request, timeout):
        body = json.loads(request.data)
        assert body["stream"] is True and body["store"] is False
        assert request.full_url == H.URL and timeout == 900
        assert len(body["input"]) == 3
        return Stream()

    with patch.object(H.urllib.request, "urlopen", offline):
        response = H.post({"store": False, "input": [{"role": "user", "content": "a"},
                                                   {"role": "assistant", "content": "b"},
                                                   {"role": "user", "content": "c"}]})
        assert response["id"] == "offline"
    attempts = []

    def flaky(body):
        attempts.append(body)
        if len(attempts) < 3:
            raise ConnectionError("offline retry fixture")
        return {"ok": True}

    with patch.object(H, "post", flaky), patch.object(H.time, "sleep") as sleep:
        assert H.call({"fixture": True}) == {"ok": True}
        assert [x.args[0] for x in sleep.call_args_list] == [2, 4]
    assert len(attempts) == 3
    # Interruptions and run caps are tested without executing any subject code.
    with tempfile.TemporaryDirectory(prefix=".harness-test-", dir=ROOT) as temporary:
        temporary = Path(temporary)
        (temporary / "prompt.md").write_text((ROOT / "prompt.md").read_text())
        args = SimpleNamespace(condition="noW", rep=9, fake=True,
                               model="test-model", effort="xhigh")
        with patch.object(H, "ROOT", temporary), patch.object(H.S, "run") as sandbox_run:
            sandbox_run.return_value = {"output": "MOCKED UNIT TEST", "returncode": 0}
            directory = temporary / "sessions/fake_noW_r9"
            directory.mkdir(parents=True)
            state = H.setup(directory, args)
            state.update(phase="reply_saved", reply=H.fake(0, "noW"))
            H.atomic(directory / "state.json", state)
            H.run(args)
            assert sandbox_run.call_count == 2
            before = (directory / "turns.jsonl").read_bytes()
            H.run(args)
            assert (directory / "turns.jsonl").read_bytes() == before
            assert sandbox_run.call_count == 2
            args.rep = 10
            directory = temporary / "sessions/fake_noW_r10"
            directory.mkdir(parents=True)
            state = H.setup(directory, args)
            state.update(phase="executing", runs=1, reply=H.fake(0, "noW"))
            H.atomic(directory / "state.json", state)
            try:
                H.run(args)
            except RuntimeError as exc:
                assert "ambiguous" in str(exc)
            else:
                raise AssertionError("Ambiguous code execution was replayed")
            assert sandbox_run.call_count == 2
            args.rep = 11
            directory = temporary / "sessions/fake_noW_r11"
            directory.mkdir(parents=True)
            state = H.setup(directory, args)
            state.update(phase="reply_saved", runs=40, reply=H.fake(0, "noW"))
            H.atomic(directory / "state.json", state)
            H.run(args)
            assert sandbox_run.call_count == 2
            assert json.loads((directory / "state.json").read_text())["runs"] == 40
    # The initial prompt may differ only in the explicitly marked sections.
    for condition in ["W", "noW"]:
        text = H.render(condition)
        assert "<!--" not in text
        assert f"CONDITION {condition}:" in text
        assert f"CONDITION {'noW' if condition == 'W' else 'W'}:" not in text
        for rep in range(3):
            session = ROOT / "sessions" / f"fake_{condition}_r{rep}"
            state = json.loads((session / "state.json").read_text())
            assert state["phase"] == "done" and state["runs"] == 2 and state["turn"] == 2
            events = [json.loads(x) for x in (session / "turns.jsonl").read_text().splitlines()]
            assert sum(e["event"] == "code" for e in events) == 2
            assert sum(e["event"] == "output" for e in events) == 2
            requests = [e for e in events if e["event"] == "request"]
            assert [len(e["conversation"]) for e in requests] == [1, 3, 5]
            assert not (session / "work/key.json").exists()
            if condition == "W":
                assert len(list((session / "work/nets").glob("*/*.npy"))) == 42
            else:
                assert not (session / "work/nets").exists()
    print("PASS: six role permutations and pairwise inversion; exact/perturbed errors; "
          "noW role alignment; NaN rejection; offline SSE/retries; condition-only prompt differences; "
          "six fake sessions, two code runs each, full conversation replay, isolated work layout; "
          "saved-reply recovery, completed-session idempotence, ambiguous-execution refusal, 40-run cap.")


if __name__ == "__main__":
    main()
