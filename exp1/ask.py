"""Send Exp1 prompts to the configured model endpoint (Responses API) and save one JSON per call.

    python ask.py --prompts prompts/a9f535cc__orig.txt --model MODEL_ID --effort low --out results/raw
    python ask.py --prompts prompts --model <id> --effort high --reps 5 --concurrency 4 --dry-run

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
URL = os.environ.get("WSP_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/") + "/responses"
MAX_TRIES = 6
TIMEOUT_S = 900


def prompt_files(path):
    return sorted(glob.glob(os.path.join(path, "*.txt"))) if os.path.isdir(path) else [path]


def out_path(out_dir, prompt_file, rep, model):
    stem = os.path.splitext(os.path.basename(prompt_file))[0]
    return os.path.join(out_dir, f"{stem}__r{rep}__{model.replace('/', '_')}.json")


def post(body):
    # Streamed (SSE): a long reasoning reply can sit silent for many minutes, and a non-streamed request then risks an
    # idle connection being dropped on the way. The final response object comes in the response.completed event;
    # TIMEOUT_S bounds the longest silence between events.
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


def run_one(prompt_file, rep, args, instructions):
    prompt = open(prompt_file).read()
    path = out_path(args.out, prompt_file, rep, args.model)
    body = {"model": args.model, "instructions": instructions, "input": prompt,
            "reasoning": {"effort": args.effort, "summary": "detailed"}, "store": False}
    t0 = time.time()
    resp = call_with_retry(body, os.path.basename(path))
    wall = time.time() - t0
    text, summaries = extract(resp)
    rec = {"model_requested": args.model, "model_returned": resp.get("model"), "effort": args.effort,
           "prompt_file": os.path.basename(prompt_file),
           "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "rep": rep,
           "output_text": text, "reasoning_summaries": summaries, "usage": resp.get("usage"),
           "wall_time_s": round(wall, 2), "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "response_tools": resp.get("tools"), "status": resp.get("status")}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="prompt .txt file or directory of them")
    ap.add_argument("--model", default=os.environ.get("WSP_MODEL"), required=not os.environ.get("WSP_MODEL"))
    ap.add_argument("--effort", default="high")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "raw"))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    instructions = open(os.path.join(HERE, "..", "upstream", "lm_understand-logs", "exp1_which_game", "runs", "breakout", "instructions.md")).read()
    calls = [(p, r) for p in prompt_files(args.prompts) for r in range(args.reps)]
    todo = [(p, r) for p, r in calls if not os.path.exists(out_path(args.out, p, r, args.model))]
    print(f"{len(calls)} calls, {len(calls) - len(todo)} already done, {len(todo)} to run")
    if args.dry_run:
        tot = 0
        for p, r in todo:
            toks = (os.path.getsize(p) + len(instructions)) // 4
            tot += toks
            print(f"  {os.path.basename(p)} rep {r}  ~{toks} input tokens")
        print(f"total ~{tot} input tokens (chars/4)")
        return
    os.makedirs(args.out, exist_ok=True)
    failures = 0
    with ThreadPoolExecutor(args.concurrency) as ex:
        futs = {ex.submit(run_one, p, r, args, instructions): (p, r) for p, r in todo}
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
