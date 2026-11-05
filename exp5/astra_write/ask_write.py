"""Ask a model to write the MinAtar network weights (Exp5 reliability) via the configured model endpoint (Responses API).

    python -I ask_write.py --prompts prompts/breakout__orig.txt --reps 1 --out results/raw
    python -I ask_write.py --prompts prompts --reps 5 --concurrency 4 --dry-run

instructions = the game's instructions.md from the authors' release (game = prompt file name before '__').
No `tools` field is ever sent. Existing output files are skipped (resume-safe).
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
UP_RUNS = os.path.join(HERE, "..", "..", "upstream", "lm_understand-logs", "exp5_write_from_scratch", "runs")
URL = os.environ.get("WSP_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/") + "/responses"
MAX_TRIES = 3  # each retry of a paid xhigh call costs about $1.6
TIMEOUT_S = 600


def prompt_files(path):
    if os.path.isdir(path):
        return sorted(p for p in glob.glob(os.path.join(path, "*__*.txt")) if not p.endswith("__diff.txt"))
    return [path]


def game_of(prompt_file):
    return os.path.basename(prompt_file).split("__")[0]


def instructions_for(prompt_file):
    with open(os.path.join(UP_RUNS, game_of(prompt_file), "instructions.md"), encoding="utf-8") as f:
        return f.read()


def out_path(out_dir, prompt_file, rep, model):
    stem = os.path.splitext(os.path.basename(prompt_file))[0]
    return os.path.join(out_dir, f"{stem}__r{rep}__{model.replace('/', '_')}.json")


def post(body):
    # Streamed: an xhigh reply takes 10+ minutes, and a non-streamed request sat silent that long and never returned
    # (an idle connection dropped somewhere on the way). With SSE, events keep flowing; the final response object
    # comes in the response.completed event. TIMEOUT_S bounds each read, i.e. the longest silence.
    req = urllib.request.Request(URL, json.dumps({**body, "stream": True}).encode(), {"Content-Type": "application/json", "Authorization": "Bearer" + " " + os.environ.get("WSP_API_KEY", "")})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:])
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "response.completed":
                return ev["response"]
            if ev.get("type") in ("response.failed", "response.incomplete", "error"):
                raise ConnectionError(f"stream ended with {ev.get('type')}: {json.dumps(ev)[:300]}")
    raise ConnectionError("stream closed before response.completed")


def call_with_retry(body, label):
    for attempt in range(1, MAX_TRIES + 1):
        try:
            return post(body)
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode("utf-8", "replace")
            if not (e.code == 429 or e.code >= 500) or attempt == MAX_TRIES:
                raise RuntimeError(f"HTTP {e.code}: {detail}")
            why = f"HTTP {e.code}"
        except (TimeoutError, urllib.error.URLError, ConnectionError, json.JSONDecodeError) as e:
            if attempt == MAX_TRIES:
                raise RuntimeError(f"{type(e).__name__}: {e}")
            why = type(e).__name__
        delay = min(2 ** attempt, 120)
        print(f"[retry] {label}: {why}, attempt {attempt}/{MAX_TRIES}, sleeping {delay}s", file=sys.stderr)
        time.sleep(delay)


def extract(resp):
    text, summaries = [], []
    for item in resp.get("output") or []:
        if item.get("type") == "message":
            text += [c.get("text", "") for c in item.get("content") or [] if c.get("type") == "output_text"]
        elif item.get("type") == "reasoning":
            summaries += [s.get("text", "") for s in item.get("summary") or []]
    return "".join(text), summaries


def run_one(prompt_file, rep, args):
    with open(prompt_file, encoding="utf-8") as f:
        prompt = f.read()
    instructions = instructions_for(prompt_file)
    path = out_path(args.out, prompt_file, rep, args.model)
    body = {"model": args.model, "instructions": instructions, "input": prompt,
            "reasoning": {"effort": args.effort, "summary": "detailed"}, "store": False}
    if args.max_output_tokens:
        body["max_output_tokens"] = args.max_output_tokens
    t0 = time.time()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    resp = call_with_retry(body, os.path.basename(path))
    wall = time.time() - t0
    text, summaries = extract(resp)
    rec = {"model_requested": args.model, "model_returned": resp.get("model"), "effort": args.effort,
           "prompt_file": os.path.basename(prompt_file),
           "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
           "instructions_sha256": hashlib.sha256(instructions.encode()).hexdigest(), "rep": rep,
           "output_text": text, "reasoning_summaries": summaries, "usage": resp.get("usage"),
           "wall_time_s": round(wall, 2), "started_utc": started,
           "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "response_tools": resp.get("tools"), "status": resp.get("status")}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="prompt .txt file or directory of <game>__<cond>.txt")
    ap.add_argument("--model", default=os.environ.get("WSP_MODEL"), required=not os.environ.get("WSP_MODEL"))
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "raw"))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-output-tokens", type=int, default=0, help="0 = provider default")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    calls = [(p, r) for p in prompt_files(args.prompts) for r in range(args.reps)]
    todo = [(p, r) for p, r in calls if not os.path.exists(out_path(args.out, p, r, args.model))]
    print(f"{len(calls)} calls, {len(calls) - len(todo)} already done, {len(todo)} to run")
    if args.dry_run:
        tot = 0
        for p, r in todo:
            toks = (os.path.getsize(p) + len(instructions_for(p))) // 4
            tot += toks
            print(f"  {os.path.basename(p)} rep {r}  ~{toks} input tokens -> {os.path.basename(out_path(args.out, p, r, args.model))}")
        print(f"total ~{tot} input tokens (chars/4)")
        return
    os.makedirs(args.out, exist_ok=True)
    failures = 0
    with ThreadPoolExecutor(args.concurrency) as ex:
        futs = {ex.submit(run_one, p, r, args): (p, r) for p, r in todo}
        for fut in as_completed(futs):
            p, r = futs[fut]
            try:
                print("saved", fut.result())
            except Exception as e:
                failures += 1
                print(f"[FAILED] {os.path.basename(p)} rep {r}: {e}", file=sys.stderr)
    print(f"done, {failures} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
