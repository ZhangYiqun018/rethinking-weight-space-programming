"""Responses/SSE subject harness. --fake never contacts an API."""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sys
import time
import urllib.error
import urllib.request

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def local(name):
    spec = importlib.util.spec_from_file_location(f"readwt_{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C, S, SCHEMA = local("common"), local("sandbox_runner"), local("schema")
URL = os.environ.get("WSP_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/") + "/responses"


def atomic(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def log(session, event, **fields):
    with (session / "turns.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": event,
                                "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                **fields}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def render(condition):
    text = (ROOT / "prompt.md").read_text()
    for name in ["W", "noW"]:
        pattern = rf"<!-- {name} -->(.*?)<!-- /{name} -->"
        text = re.sub(pattern, lambda match: match[1] if name == condition else "", text, flags=re.S)
    return text


def post(body):
    req = urllib.request.Request(URL, json.dumps({**body, "stream": True}).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer" + " " + os.environ.get("WSP_API_KEY", "")})
    with urllib.request.urlopen(req, timeout=900) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:])
            except json.JSONDecodeError:
                continue
            if event.get("type") == "response.completed":
                return event["response"]
            if event.get("type") in ("response.failed", "response.incomplete", "error"):
                raise ConnectionError(f"SSE {event.get('type')}: {json.dumps(event)[:300]}")
    raise ConnectionError("stream closed before response.completed")


def call(body):
    for attempt in range(1, 7):
        try:
            return post(body)
        except urllib.error.HTTPError as exc:
            detail = exc.read(300).decode("utf-8", "replace")
            if not (exc.code == 429 or exc.code >= 500) or attempt == 6:
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, urllib.error.URLError, ConnectionError, json.JSONDecodeError):
            if attempt == 6:
                raise
        time.sleep(min(2 ** attempt, 120))


def extract(response):
    text = []
    for item in response.get("output", []):
        if item.get("type") == "message":
            text.extend(c.get("text", "") for c in item.get("content", [])
                        if c.get("type") == "output_text")
    return "".join(text)


def fake(turn, condition):
    if turn == 0:
        return "```python\nfrom pathlib import Path\nprint(sorted(str(p) for p in Path('.').rglob('*')))\n```"
    if turn == 1:
        forbidden = str(C.EXP6 / "RESULTS.md")
        return f"```python\nfrom pathlib import Path\nprint(Path({forbidden!r}).read_text())\n```"
    return "```json\n" + json.dumps(SCHEMA.fake_prediction(condition)) + "\n```"


def setup(session, args):
    work = session / "work"
    work.mkdir(parents=True, exist_ok=True)
    metadata = {"condition": args.condition, "rep": args.rep, "fake": args.fake,
                "model": args.model, "effort": args.effort}
    state_path = session / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        assert state["metadata"] == metadata, "Session arguments differ from saved run."
        return state
    if args.condition == "W":
        key_path = session / "key.json"
        if key_path.exists():
            key = json.loads(key_path.read_text())
        else:
            roles = list(C.ROLES)
            secrets.SystemRandom().shuffle(roles)
            key = dict(zip(["X", "Y", "Z"], roles))
            atomic(key_path, key)
        assert set(key) == {"X", "Y", "Z"} and set(key.values()) == set(C.ROLES)
        for name, role in key.items():
            destination = work / "nets" / name
            destination.mkdir(parents=True, exist_ok=True)
            source = C.REPO / "upstream/lm_understand-logs/exp6_go_from_scratch/networks" / C.ROLES[role]
            for tensor in sorted(source.glob("*.npy")):
                shutil.copyfile(tensor, destination / tensor.name)
            assert len(list(destination.glob("*.npy"))) == 14
    else:
        assert not (work / "nets").exists()
    prompt = render(args.condition)
    state = {"metadata": metadata, "messages": [{"role": "user", "content": prompt}],
             "phase": "ready", "turn": 0, "runs": 0, "format_errors": 0,
             "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    atomic(state_path, state)
    log(session, "initial_prompt", prompt=prompt, metadata=metadata)
    return state


def run(args):
    assert args.rep >= 0
    name = f"{'fake_' if args.fake else ''}{args.condition}_r{args.rep}"
    session = ROOT / "sessions" / name
    session.mkdir(parents=True, exist_ok=True)
    with (session / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = setup(session, args)
        path = session / "state.json"
        if state["phase"] == "done":
            print(f"Already complete: {session}")
            return
        if state["phase"] == "executing":
            raise RuntimeError("Interrupted code execution has ambiguous side effects. "
                               "Do not rerun automatically; the experimenter must review state and work.")
        if not args.fake:
            proof = local("test_sandbox").check(session / "work")
            atomic(session / "sandbox_preflight.json", proof)
            if proof["status"] != "PASS":
                raise RuntimeError("Real session refused: sandbox isolation preflight did not PASS.")
        while state["turn"] < 80:
            if state["phase"] == "ready":
                log(session, "request", turn=state["turn"], conversation=state["messages"])
                if args.fake:
                    reply = fake(state["turn"], args.condition)
                    response = {"fake": True}
                else:
                    response = call({"model": args.model, "reasoning": {"effort": args.effort,
                                                                       "summary": "detailed"},
                                     "store": False, "input": state["messages"]})
                    reply = extract(response)
                state.update(phase="reply_saved", reply=reply, response=response)
                atomic(path, state)
            # Recover a saved response even if interrupted before its log append.
            events = [json.loads(line) for line in (session / "turns.jsonl").read_text().splitlines()]
            if not any(e["event"] == "reply" and e["turn"] == state["turn"] for e in events):
                log(session, "reply", turn=state["turn"], reply=state["reply"],
                    response=state.get("response", {"recovered_from_state": True}))
            reply = state["reply"]
            block = re.fullmatch(r"\s*```(python|json)\s*\n(.*?)\n```\s*", reply, re.S)
            if block and block[1] == "python":
                if state["runs"] >= 40:
                    feedback = "Run limit reached (40). Return your final JSON now."
                else:
                    state["runs"] += 1
                    state["phase"] = "executing"
                    atomic(path, state)
                    log(session, "code", turn=state["turn"], run=state["runs"], code=block[2])
                    output = S.run(block[2], session / "work")
                    log(session, "output", turn=state["turn"], run=state["runs"], **output)
                    feedback = json.dumps(output)
            elif block and block[1] == "json":
                try:
                    data = SCHEMA.validate(json.loads(block[2]), args.condition)
                except (ValueError, AssertionError, KeyError, TypeError, IndexError) as exc:
                    feedback = f"Invalid final JSON ({type(exc).__name__}). Follow the complete schema and retry."
                    state["format_errors"] += 1
                else:
                    atomic(session / "predictions.json", data)
                    state["phase"] = "done"
                    state["messages"].append({"role": "assistant", "content": reply})
                    atomic(path, state)
                    log(session, "complete", turn=state["turn"], fake=args.fake)
                    print(f"Saved {session / 'predictions.json'}")
                    return
            else:
                feedback = "Reply with exactly one python or json fenced block and nothing else."
                state["format_errors"] += 1
            state["messages"] += [{"role": "assistant", "content": reply},
                                  {"role": "user", "content": feedback}]
            state.update(phase="ready", turn=state["turn"] + 1)
            atomic(path, state)
            if state["format_errors"] >= 6:
                raise RuntimeError("Too many invalid replies; saved for review.")
        raise RuntimeError("80-turn limit reached without a valid final prediction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["W", "noW"], required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--model", default=os.environ.get("WSP_MODEL"), required=not os.environ.get("WSP_MODEL"))
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--fake", action="store_true")
    run(parser.parse_args())
