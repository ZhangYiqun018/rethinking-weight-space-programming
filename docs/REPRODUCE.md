# Reproducing the experiments

[← Back to the README](../README.md)

This page collects the technical details for re-running the experiments. Run commands from the repository root
unless a script's usage says otherwise.

- [Read the blog offline](#read-the-blog-offline)
- [Entry scripts per experiment](#entry-scripts-per-experiment)
- [Environments](#environments)
- [Authors' artifacts](#authors-artifacts)
- [Model endpoint](#model-endpoint)
- [Exp6 sandbox](#exp6-sandbox)
- [Exp8 benchmarks and raw evidence](#exp8-benchmarks-and-raw-evidence)
- [Figure and layout checks](#figure-and-layout-checks)
- [Scope: what is not shipped](#scope-what-is-not-shipped)
- [Licenses](#licenses)

## Read the blog offline

Open [blog/index.html](../blog/index.html) directly in a browser. It works from `file://`, including the bundled
figure data, and has an **EN / 中文** toggle. No build system or server is required to read it.

## Entry scripts per experiment

Entry scripts are relative to each experiment's folder. Verdicts and evidence are in
[results/CLAIMS.md](../results/CLAIMS.md).

| Experiment | Folder | Entry scripts |
|---|---|---|
| Exp1: identify games from weights | [exp1/](../exp1/) | `make_prompts.py`; `ask.py`; `score.py`; `stats_baseline.py` |
| Exp2: diffusion model shapes | [exp2/](../exp2/) | `baseline.py`; `score_replies.py` |
| Exp3: language and copying heads | [exp3/](../exp3/) | `baseline.py`; `prevhead.py` |
| Exp4: car edit and open-loop control | [exp4/](../exp4/) | `run_h1.py`; `run_h8.py`; `run_h8b.py`; `run_h8c.py`; `openloop_check.py` |
| Exp5: freshly written game networks and PPO | [exp5/](../exp5/) | `eval_minatar.py`; `ppo/ppo_pgx.py`; `astra_write/make_prompts.py`; `astra_write/ask_write.py`; `astra_write/score_write.py` |
| Exp6: Go networks and reading their behaviour | [exp6/](../exp6/) | `eval.py`; `check_engine.py`; `readwt/truth.py`; `readwt/run_subject.py`; `readwt/score.py` |
| Exp7: sycophancy from weight distances | [exp7/](../exp7/) | `prepare.py`; `run_training.py`; `blind.py`; `dist_test.py`; `score.py`; `confirm.py` |
| Exp8: math edit, decoding and steering | [exp8/](../exp8/) | `generate.py`; `grade.py`; `apply_edit.py`; `steer/directions.py`; `steer/make_edit.py` |

## Environments

Use **Python 3.12** for the scientific environments (Exp6 records 3.12.13). The release builder and HTML validator
use the standard library; compilation alone does not establish scientific dependency compatibility. Use separate
environments for JAX, PyTorch and model serving.

The existing environment snapshots are `exp6/requirements.txt`, `exp7/requirements.txt`,
`exp6/readwt/subject_requirements.txt` and `exp6/readwt/truth_requirements.txt`. These record the original
environment; platform-specific wheels may require the matching package index or an appropriate local build. There
is no universal requirements file for all eight experiments.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r exp6/requirements.txt
python -I blog/validate.py
python -I blog/verify/scan_public.py
python -I blog/verify/figs/check_data.py
```

- **Exp1–3** need NumPy.
- **Exp4** needs JAX and the authors' Kinetix environment.
- **Exp5** evaluation uses official MinAtar 1.0.15 and NumPy, and PPO uses JAX/PGX/Optax. PPO training ran on a
  single GPU; see `exp5/ppo/` for the training script and hyper-parameters.
- **Exp8** needs PyTorch, Transformers, Accelerate, aiohttp, NumPy, `math-verify[antlr4_13_2]==0.9.0`, and
  `lm-eval[ifeval]`; install a compatible model server separately.

See imports and the experiment preregistrations/reports for the exact protocols.

## Authors' artifacts

Obtain the authors' MIT-licensed release separately from
[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs). Clone it into
`upstream/lm_understand-logs` beneath the repository root. It is **not shipped here**. Scripts read the original
networks, prompts, environment inputs and evaluation data there.

Exp1 reads networks and prompt headers directly from that checkout; its local answer key is included. Exp5's
author-network evaluations expect local `exp5/networks/<game>/` inputs: copy the required networks from
`upstream/lm_understand-logs/exp5_write_from_scratch/networks/` before running, or supply the upstream file path to
`--policy net:PATH`. The external build manifest lists byte-identical network exclusions and their exact upstream
locations.

## Model endpoint

Callers use an OpenAI-compatible endpoint. Set `WSP_API_BASE` (default `http://127.0.0.1:8000/v1`), `WSP_API_KEY`,
and `WSP_MODEL` in your environment. Supply your own endpoint credentials; no key is bundled. `--model` overrides the
model for Exp1, Exp5-write and Exp6-read; Exp8 uses `--served` and optionally `--url` (API base including `/v1`,
comma-separated for multiple servers). Exp2/3 have local scoring/baseline scripts, with no separate endpoint caller.

The model endpoint must implement the API used by the experiment: Responses with SSE and reasoning settings for the
reading/writing callers, or completions with the sampling extensions used by Exp8. A chat-completions-only endpoint
is insufficient. These portability changes preserve retry, scoring and experimental logic.

```sh
export WSP_API_BASE=http://127.0.0.1:8000/v1
export WSP_MODEL=your-model-id
# Set WSP_API_KEY securely in your shell environment.
python exp1/make_prompts.py --verify
python exp1/make_prompts.py --out exp1/prompts
python exp1/ask.py --prompts exp1/prompts --dry-run
mkdir -p exp5/astra_write/prompts
python exp5/astra_write/make_prompts.py
python exp5/astra_write/ask_write.py --prompts exp5/astra_write/prompts --dry-run
```

## Exp6 sandbox

Exp6's subject harness requires **macOS Seatbelt** (`sandbox-exec`); it has no unsandboxed fallback. Create
`exp6/readwt/subject_venv` using Python 3.12 and install its subject requirements, then run
`exp6/readwt/test_sandbox.py` and `test_harness.py` before real sessions. The release profile uses a default read
denial with runtime-specific allowlists; verify isolation on your installation before interpreting a run. The
separate truth environment uses its own requirements. Linux requires a separately validated equivalent sandbox.

## Exp8 benchmarks and raw evidence

Download MATH-500 (`HuggingFaceH4/MATH-500`, test), GSM8K (the official main test split) and IFEval
(`google/IFEval`, train) separately, following their licenses. Write newline-delimited JSON to
`exp8/data/math500.jsonl`, `gsm8k.jsonl`, and `ifeval.jsonl`. `exp8/configs.py:load_benchmark` defines the exact
schema:

| Benchmark | Fields |
|---|---|
| MATH-500 | `unique_id`, `problem`, `answer` |
| GSM8K | a stable `id`, `problem` (source `question`), `answer` |
| IFEval | `key`, `prompt`, `instruction_id_list`, `kwargs` |

Keep the original dataset order and identifiers for paired comparisons. The steering practice pool uses the
authors' GSM8K/MATH training inputs, converted to `id`, `problem`, `answer` in
`exp8/data/practice/gsm8k_train.jsonl` and `math_train.jsonl`; `practice_split` fixes the disjoint selection.

Install NLTK tokenizer data before IFEval grading:

```sh
python -m nltk.downloader -d exp8/nltk_data punkt_tab
```

Run `exp8/generate.py --help` and `exp8/grade.py` as documented in their modules. Regenerated per-sample responses
belong in `results/runs/<public-name>/raw/` for `results/cross_config.py`; the shipped aggregates alone cannot
reconstruct paired bootstrap intervals. This release includes named aggregate run summaries, never historical raw
replies.

## Figure and layout checks

The shipped JSON and JavaScript are sufficient to view the blog. `blog/figs/data/build.py` preserves the original
figure-building code for audit/reproduction; rebuilding all figures additionally requires regenerated raw replies,
upstream inputs and the original audit-derived evidence referenced by that script. Some historical audit inputs are
deliberately not distributed, so it is not a standalone release build step. `blog/verify/figs/check_data.py` checks
the shipped datasets without those inputs.

The `.cjs` layout/render checks need Node.js, `playwright-core`, and a browser; set `PLAYWRIGHT_MODULE` and
`CHROME_EXECUTABLE` if needed. Figure checks default to the local file URL; the layout check accepts a served blog
URL. Running render checks creates local screenshots, which are not part of this release.

The two images in the README are screenshots of Figures 4 and 10 of the blog.

## Scope: what is not shipped

Large language-model weights, checkpoints, downloaded benchmark data, virtual environments, generated prompts
derived from upstream networks, raw model replies and internal audit logs are omitted for size, provenance and
infrastructure privacy. Obtain public model weights and author-owned inputs separately, and regenerate experimental
outputs with the included entry scripts. Small aggregate results and our own eligible small networks are included;
networks byte-identical to upstream files are excluded. Experiment reports may cite historical artifacts that are
not distributed here.

## Licenses

Our code and text are under the [MIT License](../LICENSE), Copyright (c) 2026 Yiqun Zhang. The page template retains
the authors' separate MIT license in [blog/LICENSE-original.txt](../blog/LICENSE-original.txt). The upstream
artifacts and independently downloaded models/datasets retain their own licenses; our license does not relicense
them.
