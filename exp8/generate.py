"""Generate answers for one (model, config, benchmark, seed) job against a running vLLM OpenAI-compatible server.

Prompts are rendered locally with the model's chat template, so the thinking switch is explicit, and sent to
/v1/completions. Output is one JSON line per sample. Rerunning skips samples already written, so preempted jobs
can resume.

    python generate.py --served MODEL_ID --tokenizer <model_dir> \
        --config A --bench math500 --seed 0 --out results/raw/base_A_math500_s0.jsonl
"""
import argparse
import asyncio
import hashlib
import json
import os

import aiohttp
from transformers import AutoTokenizer

from configs import CONFIGS, FORCE_CLOSE, SAMPLES, load_benchmark


def request_seed(seed, row_id, sample):
    return int(hashlib.sha256(f"{seed}:{row_id}:{sample}".encode()).hexdigest()[:8], 16)


async def complete(session, url, body):
    for attempt in range(8):
        try:
            async with session.post(f"{url.rstrip(chr(47))}/completions", json=body) as r:
                if r.status == 200:
                    return (await r.json())["choices"][0]
                text = await r.text()
                if r.status < 500:
                    raise RuntimeError(f"HTTP {r.status}: {text[:500]}")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        await asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"request failed after retries: {body.get('seed')}")


def server_for(args, row_id, sample):
    """Spread requests over the servers deterministically (all samples of one problem go to one server)."""
    urls = args.url.split(",")
    return urls[int(hashlib.sha256(row_id.encode()).hexdigest()[:8], 16) % len(urls)]


async def one_sample(session, args, cfg, tok, row, sample):
    url = server_for(args, row["id"], sample)
    prompt = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=cfg["thinking"])
    base = {"model": args.served, "skip_special_tokens": False, **cfg["sampling"],
            "seed": request_seed(args.seed, row["id"], sample)}
    record = {"id": row["id"], "sample": sample, "forced": False}
    if "think_budget" not in cfg:
        c = await complete(session, url, {**base, "prompt": prompt, "max_tokens": cfg["max_tokens"]})
        record.update(text=c["text"], finish_reason=c["finish_reason"])
    else:
        think = await complete(session, url, {**base, "prompt": prompt, "max_tokens": cfg["think_budget"],
                                                   "stop": ["</think>"]})
        forced = think["finish_reason"] == "length"
        closer = FORCE_CLOSE if forced else "</think>\n\n"
        ans = await complete(session, url, {**base, "prompt": prompt + think["text"] + closer,
                                                 "max_tokens": cfg["answer_budget"]})
        record.update(text=think["text"] + closer + ans["text"], finish_reason=ans["finish_reason"], forced=forced)
    record["completion_tokens"] = len(tok(record["text"], add_special_tokens=False)["input_ids"])
    return record


async def main(args):
    cfg = CONFIGS[args.config]
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = load_benchmark(args.bench)
    if args.limit:
        rows = rows[:args.limit]
    done = set()
    if os.path.exists(args.out):
        done = {(r["id"], r["sample"]) for r in map(json.loads, open(args.out))}
    todo = [(row, s) for row in rows for s in range(SAMPLES[args.bench]) if (row["id"], s) not in done]
    print(f"{args.out}: {len(done)} done, {len(todo)} to generate", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=6 * 3600)
    # aiohttp caps a session at 100 connections by default, which left each of 8 servers ~12 requests
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers={"Authorization": "Bearer" + " " + os.environ.get("WSP_API_KEY", "")}) as session:
        async def run(row, s):
            async with sem:
                return await one_sample(session, args, cfg, tok, row, s)
        with open(args.out, "a") as f:
            for i, fut in enumerate(asyncio.as_completed([run(r, s) for r, s in todo]), 1):
                f.write(json.dumps(await fut) + "\n")
                f.flush()
                if i % 200 == 0:
                    print(f"{args.out}: {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("WSP_API_BASE", "http://127.0.0.1:8000/v1"), help="API base URL including /v1, or several separated by commas")
    ap.add_argument("--served", default=os.environ.get("WSP_MODEL"), required=not os.environ.get("WSP_MODEL"))
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--bench", required=True, choices=sorted(SAMPLES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="first N problems only (smoke test)")
    asyncio.run(main(ap.parse_args()))
