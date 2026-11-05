# Experiment 8 — inventory of every author run/row mentioned in the release

Scope: `upstream/lm_understand-logs/exp8_math/`
(referred to below as `exp8/`). Sources read: `eval/RESULTS_excerpt.md`, `README.md`,
`../README.md` (the release index), `runs/*/state.json`, `code/profiles.py` (comments),
`gemma_followup/README.md`, `gemma_followup/eval/*/merged.json`, `notices/*.txt`,
plus `runs/*/turns/tNNN/prompt.txt` to locate notices.

Numbers are percentages as printed in `RESULTS_excerpt.md`. "…" reproduces a literal "…" in
the source. "—" = the source gives no value for that cell (empty cell or field absent), not "…".
"Post" = the blog post *Sparks of Weight Space Programming*; the release's own statements about
which runs are in the post are `exp8/README.md:50-57`, `exp8/README.md:145-146`,
`exp8/gemma_followup/README.md:36-44`, `exp8/gemma_followup/README.md:69-70`,
`exp8/eval/RESULTS_excerpt.md:3-6`, `../README.md:24,26`.

## 1. Table of every run / row

### 1a. Qwen3.5-0.8B, thinking ON (RESULTS_excerpt.md section "Experiment 6", lines 10–27)

All rows in this section are measured with the thinking-on protocol named in
`RESULTS_excerpt.md:10` (temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, max 32,768).
MATH-500 = 4 samples/problem, GSM8K = 2 samples/problem (`RESULTS_excerpt.md:7-8`).

| # | Model | Prompt | Budget (passes) | Used | Edit submitted | MATH-500 | GSM8K | Other numbers given | In post | Source (file:line) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen3.5-0.8B, thinking on, unedited | — | — | — | — (no run) | 7.9 | 13.8 | AIME24 0.0, AIME25 0.0, AMC23 0.5, Minerva 1.5, Olympiad 1.3; MMLU-Pro 40.3, GPQA 22.7, IFEval 21.0; MATH-500 cut_share 93.0; raw: `math500.acc` 0.079, `gsm8k.acc` 0.1384 | yes (post's table) | `eval/RESULTS_excerpt.md:14`, `:34`, `:45`; values: `eval/unedited/merged.json:27` (gsm8k), `:43` (math500), `:11` (ifeval), `:19` (gpqa), `:35` (MMLU-Pro), `:51` (aime25), `:59` (amc23), `:67` (aime24), `:75` (minerva); post mapping `README.md:52,66,70` |
| 2 | Qwen3.5-0.8B, thinking on | v2 (max effort) | 0 | 0 | no edit made: "weights identical to the unedited model" | 7.9 | 13.8 | — | no (`README.md:145-146`) | `eval/RESULTS_excerpt.md:15`; budget set "0, 5, 10, 25, 50" `code/profiles.py:31` |
| 3 | Qwen3.5-0.8B, thinking on | v3 (512 answers / paired passes, max) | 5 | 5 | no edit submitted "(= unedited model)" | 7.9 | 13.8 | — | no (`README.md:145-146`) | `eval/RESULTS_excerpt.md:16` |
| 4 | Qwen3.5-0.8B, thinking on | v3 | 10 | 10 | yes (run `p10c`, `edit/` present) | 12.3 | 25.7 | AIME24 0.0, AIME25 0.0, AMC23 1.1, Minerva 3.9, Olympiad 2.4; MMLU-Pro 40.0, GPQA 29.4, IFEval 49.0; cut_share 87.2 | yes (post's table) | `eval/RESULTS_excerpt.md:17`, `:35`, `:46`; `runs/p10c/state.json:2,7`; `eval/p10c/merged.json:11` (math500), `:27` (gsm8k), `:35` (ifeval), `:43` (olympiad), `:51` (amc23), `:59` (minerva), `:67` (MMLU-Pro), `:75` (gpqa), `:3` (aime24), `:19` (aime25); post mapping `README.md:53,67` |
| 5 | Qwen3.5-0.8B, thinking on | v3 | 25 | 25 | yes (run `p25c`, `edit/` present) | 14.1 | 25.1 | AIME24 0.0, AIME25 0.0, AMC23 1.1, Minerva 3.5, Olympiad 3.9; MMLU-Pro 44.7, GPQA 30.1, IFEval 39.9; cut_share 86.4 | yes (post's table) | `eval/RESULTS_excerpt.md:18`, `:36`, `:47`; `runs/p25c/state.json:2,7`; `eval/p25c/merged.json:19` (math500), `:27` (gsm8k), `:35` (ifeval), `:43` (olympiad), `:51` (amc23), `:59` (gpqa), `:67` (minerva), `:75` (MMLU-Pro), `:3` (aime24), `:11` (aime25); post mapping `README.md:54,68` |
| 6 | Qwen3.5-0.8B, thinking on | v3 | 50 | 50 | yes (run `p50c`, `edit/` present) | 48.1 | 67.1 | AIME24 0.6, AIME25 2.3, AMC23 23.4, Minerva 25.4, Olympiad 21.8; MMLU-Pro 45.5, GPQA 36.6, IFEval 62.3; cut_share 33.1 | yes (post's table + benchmark figure) | `eval/RESULTS_excerpt.md:19`, `:37`, `:48`; `runs/p50c/state.json:2,7`; `eval/p50c/merged.json:3` (math500), `:11` (gsm8k), `:19` (MMLU-Pro), `:27` (gpqa), `:35` (olympiad), `:43` (minerva), `:51` (amc23), `:59` (ifeval), `:67` (aime24), `:75` (aime25); post mapping `README.md:55,69` |
| 7 | Qwen3.5-0.8B, thinking on | v4 (shorter, DEV/HOLDOUT, no failure example) | 10 | 10 | no edit submitted "(= unedited model)" | 7.9 | 13.8 | — | no (`README.md:145-146`) | `eval/RESULTS_excerpt.md:20` |
| 8 | Qwen3.5-0.8B, thinking on | v4 | 25 | 25 | — (Edit cell empty) | 14.1 | 25.1 | — | no (`README.md:145-146`) | `eval/RESULTS_excerpt.md:21` |
| 9 | Qwen3.5-0.8B, thinking on | — (row gives only the run ID `p05c`; no prompt version stated) | — (not stated in the row; ID reads `p05c`) | — (not stated) | no; "its best candidate, which it did not submit (its own acceptance rule); evaluated at the user's request" | 8.6 | 17.1 | — | no (`README.md:145-146`) | `eval/RESULTS_excerpt.md:22`; ID convention "c = prompt version 3" stated for `p10c/p25c/p50c` only, `README.md:57` |
| 10 | Qwen3.5-0.8B, thinking on | — (row gives only the run ID `p10d`; no prompt version stated) | — (not stated in the row; ID reads `p10d`) | — (not stated) | no; "its best candidate, which it did not submit (its own acceptance rule); evaluated at the user's request" | 10.1 | 16.1 | — | no (`README.md:145-146`) | `eval/RESULTS_excerpt.md:23` |
| 11 | Qwen3.5-0.8B, thinking on | v5 (TF screening, fixed 1.5 SE rule) | 5 | 0 | — (Edit cell empty) | … | … | — | no (`README.md:145-146`) | `eval/RESULTS_excerpt.md:24` |
| 12 | Qwen3.5-0.8B, thinking on | v5 | 10 | — (not stated; only "used k" entries carry a number) | — (Edit cell empty) | … | … | — | no (`README.md:145-146`) | `eval/RESULTS_excerpt.md:25` |
| 13 | Qwen3.5-0.8B, thinking on | v5 | 25 | — (not stated) | — (Edit cell empty) | … | … | — | no (`README.md:145-146`) | `eval/RESULTS_excerpt.md:26` |
| 14 | Qwen3.5-0.8B, thinking on | v5 | 50 | — (not stated) | — (Edit cell empty) | … | … | — | no (`README.md:145-146`) | `eval/RESULTS_excerpt.md:27` |

The two "other math benchmarks" tables (`RESULTS_excerpt.md:32-37`, AIME24/AIME25/AMC23/Minerva/Olympiad)
and the "beyond math" table (`:43-48`, MMLU-Pro/GPQA Diamond/IFEval) contain no new runs: they repeat
rows 1, 4, 5, 6 (their "Edit" column is empty in every row of both tables). Their numbers are in the
"Other numbers given" column above.

### 1b. Gemma 3 270M IT (RESULTS_excerpt.md section "Experiment 8", lines 50–80)

Protocol for all rows: prompt v5 with "submit the best edit", sampling temperature 1.0, top_k 64,
top_p 0.95, max 8,192 tokens, MATH-500 only, 4 samples/problem (`RESULTS_excerpt.md:50-54`).
Two grading variants are tabulated: lenient (main) and strict `\boxed{}`-only (`RESULTS_excerpt.md:52-56,68`).
Run-ID letter `b` = "the runs whose prompt stated the format-agnostic grading" (`gemma_followup/README.md:44`);
the strict-prompt runs are named `g05, g10, g25, g50` (`gemma_followup/README.md:69`).

| # | Model | Prompt | Budget (passes) | Used | Edit submitted | MATH-500 (lenient) | MATH-500 (strict) | GSM8K | Other numbers given | In post | Source (file:line) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 | Gemma 3 270M IT, unedited | — | — | — | — (no run) | 6.8 | 2.8 | 4.0 (strict); merged.json: `gsm8k.acc` 0.0402, `gsm8k.acc_lenient` 0.0986 | cut_share 8.5 (`math500.cut_share` 0.085) | yes (post's table + "8.5% of its answers reach its token limit") | `eval/RESULTS_excerpt.md:50,58,72`; `gemma_followup/eval/unedited/merged.json:3,8,11,12,16`; post mapping `gemma_followup/README.md:38,48,50` |
| 16 | Gemma 3 270M IT | v5, strict grading in the prompt | 5 | 5 | — (Edit cell empty; run folder `g05` is not in the release) | 6.7 | 2.6 | — | — | no (`gemma_followup/README.md:69-70`; numbers only in RESULTS_excerpt) | `eval/RESULTS_excerpt.md:59`, `:73` |
| 17 | Gemma 3 270M IT | v5, strict grading in the prompt | 10 | 10 | — (run folder `g10` not in the release) | 6.7 | 2.9 | — | — | no (`gemma_followup/README.md:69-70`) | `eval/RESULTS_excerpt.md:60`, `:74` |
| 18 | Gemma 3 270M IT | v5, strict grading in the prompt | 25 | 25 | — (run folder `g25` not in the release) | 6.1 | 3.5 | — | — | no (`gemma_followup/README.md:69-70`) | `eval/RESULTS_excerpt.md:61`, `:75` |
| 19 | Gemma 3 270M IT | v5, strict grading in the prompt | 50 | 50 | — (run folder `g50` not in the release) | 6.2 | 3.2 | — | — | no (`gemma_followup/README.md:69-70`) | `eval/RESULTS_excerpt.md:62`, `:76` |
| 20 | Gemma 3 270M IT | v5, lenient grading stated in the prompt | 5 | 5 | yes (run `g05b`, `edit/` present) | 7.0 | 3.5 (`merged.json` `acc` 0.0355) | — | cut_share 6.95 (`math500.cut_share` 0.0695) | yes (post's table) | `eval/RESULTS_excerpt.md:63`, `:77`; `gemma_followup/runs/g05b/state.json:2,7`; `gemma_followup/eval/g05b/merged.json:3,4,8`; post mapping `gemma_followup/README.md:39,49` |
| 21 | Gemma 3 270M IT | v5, lenient grading stated in the prompt | 10 | 10 | yes (run `g10b`, `edit/` present) | 7.6 | 3.1 (`acc` 0.031) | — | cut_share 7.3 (`math500.cut_share` 0.073) | yes (post's table) | `eval/RESULTS_excerpt.md:64`, `:78`; `gemma_followup/runs/g10b/state.json:2,7`; `gemma_followup/eval/g10b/merged.json:3,4,8`; post mapping `gemma_followup/README.md:40,49` |
| 22 | Gemma 3 270M IT | v5, lenient grading stated in the prompt | 25 | 25 | yes (run `g25b`, `edit/` present) | 7.7 | 3.8 (`acc` 0.0375) | — | cut_share 10.75 (`math500.cut_share` 0.1075) | yes (post's table) | `eval/RESULTS_excerpt.md:65`, `:79`; `gemma_followup/runs/g25b/state.json:2,7`; `gemma_followup/eval/g25b/merged.json:3,4,8`; post mapping `gemma_followup/README.md:41,49` |
| 23 | Gemma 3 270M IT | v5, lenient grading stated in the prompt | 50 | 50 | yes (run `g50b`, `edit/` present) | 7.0 | 2.6 (`acc` 0.026) | — | cut_share 10.4 (`math500.cut_share` 0.104) | yes (post's table) | `eval/RESULTS_excerpt.md:66`, `:80`; `gemma_followup/runs/g50b/state.json:2,7`; `gemma_followup/eval/g50b/merged.json:3,4,8`; post mapping `gemma_followup/README.md:42,49` |

Note on the two Gemma tables: `gemma_followup/README.md:54-56` says "The strict (`\boxed{}` only) scores are
2.8 unedited and 3.5 / 3.1 / 3.8 / 2.6 edited (`math500.acc`)" — four edited values, matching rows 20–23;
`RESULTS_excerpt.md:73-76` also lists strict-table values for the four strict-prompt runs (2.6 / 2.9 / 3.5 / 3.2),
which have no `eval/` folder in the release.

### 1c. Other models mentioned with no tabulated row

| # | Model | Prompt | Budget | Used | Edit submitted | MATH-500 | GSM8K | Other numbers given | In post | Source (file:line) |
|---|---|---|---|---|---|---|---|---|---|---|
| 24 | LFM2.5-350M | — | — | "paused after 1-2 passes" | — | — | — | "chosen instead of LFM2.5" (it was replaced by Gemma 3 270M IT); unedited values in a code comment: "GSM8K 28.0 / MATH-500 11.1 with its model card's sampling", "answers in ~124 tokens without reasoning" | no (has no row in the excerpt; named only in the Experiment 8 section header at `RESULTS_excerpt.md:50`) | `eval/RESULTS_excerpt.md:50`; `code/profiles.py:48-68` (comment `:48-50`, `runs5` at `:55`); `code/evaluate.py:43` |
| 25 | Qwen3-1.7B | — | — | — | — | — | — | none | no | `code/profiles.py:8-14` (profile `qwen3`, default, `runs` = `runs2`); `code/evaluate.py:1`; `code/practice35.py:5` |
| 26 | Qwen3.5-0.8B, thinking OFF | — (base `qwen35` profile, `runs3`) | — | — | — | — (no thinking-off accuracy given) | — | sampling temperature 0.7, top_p 0.8, top_k 20, max 8,192 (`code/profiles.py:27-28`). (The comment at `code/profiles.py:23-24` gives, for *thinking ON*, "92 % of MATH-500 answers hit 32,768 tokens (MATH-500 9.3 %)"; not a thinking-off result.) | no | `code/profiles.py:15-29` (comment lines 23-25), `code/evaluate.py:38-40` |

Mentioned but with no run at all (not counted in the totals): Qwen3-0.6B, listed only as a candidate
model for "the next experiment" alongside LFM2.5-350M and Gemma 3 270M IT (`code/evaluate.py:41-44`).
Also: prompt v2 runs measured "the unedited model read 55.5 % and 50.8 % in two 128-answer passes"
(`code/profiles.py:37-38`, comment only; two such reads, no run row).

## 2. Notices 001–004: sent to which runs, at which turn

First sentence of each notice, quoted from `notices/`:

- **001** (`notices/001_pass_size.txt:1`): "Your forward-pass budget counts passes, not their size or how long they take: a generation pass with 256 answers per model ("both") or 512 answers costs the same one pass as a pass with 32."
- **002** (`notices/002_tool_facts.txt:1`): "Corrections about the tools (no action needed if they do not concern you): (1) Attention probabilities ("attn_layers") work now; give full-attention layer indices (3, 7, 11, 15, 19, 23)."
- **003** (`notices/003_submit_best.txt:1`): "Change to the submission rule (it replaces any rule about submitting no edit): at the end, submit the best edit you found, the candidate with the best measured paired gain (on HOLDOUT if you measured it there), even if that gain is not statistically significant."
- **004** (`notices/004_failure_counts.txt:1`): "Correction to the generation-pass summaries: the failure count "no_answer_finished" now counts only answers in which the grader finds no final answer at all under your evaluation's grading rule; until now it also counted answers whose stated final answer was wrong."

Delivery, from `state.json` (`notices` array) and from the turn prompt that contains the text:

| Run | `state.json` notices list | 001 | 002 | 003 | 004 |
|---|---|---|---|---|---|
| `p10c` | 001, 002 (`runs/p10c/state.json:18-21`) | turn 59 (`runs/p10c/turns/t059/prompt.txt:1`) | turn 65 (`runs/p10c/turns/t065/prompt.txt:1`) | not delivered | not delivered |
| `p25c` | 001, 002 (`runs/p25c/state.json:19-22`) | turn 25 (`runs/p25c/turns/t025/prompt.txt:1`) | turn 31 (`runs/p25c/turns/t031/prompt.txt:1`) | not delivered | not delivered |
| `p50c` | 001, 002 (`runs/p50c/state.json:20-23`) | turn 46 (`runs/p50c/turns/t046/prompt.txt:1`) | turn 51 (`runs/p50c/turns/t051/prompt.txt:1`) | not delivered | not delivered |
| `g05b` | 001, 002, 003, 004 (`gemma_followup/runs/g05b/state.json:15-20`) | rule in v5 task text, turn 1 (`turns/t001/prompt.txt:13`) | rule in v5 task text, turn 1 (`turns/t001/prompt.txt:12` "attn_layers", `:13` eval_prompt_format) | rule in v5 task text, turn 1 (`turns/t001/prompt.txt:20`) | turn 19 (`turns/t019/prompt.txt:1`) |
| `g10b` | 001, 002, 003, 004 (`gemma_followup/runs/g10b/state.json:15-20`) | rule in v5 task text, turn 1 | rule in v5 task text, turn 1 | rule in v5 task text, turn 1 (`turns/t001/prompt.txt:20`) | turn 28 (`turns/t028/prompt.txt:1`) |
| `g25b` | 001, 002, 003, 004 (`gemma_followup/runs/g25b/state.json:15-20`) | rule in v5 task text, turn 1 | rule in v5 task text, turn 1 | v5 task text at turn 1 (`turns/t001/prompt.txt:20`); task text again at conversation continuation turn 41 | turn 30 (`turns/t030/prompt.txt:1`) |
| `g50b` | 001, 002, 003, 004 (`gemma_followup/runs/g50b/state.json:15-20`) | rule in v5 task text, turn 1 | rule in v5 task text, turn 1 | v5 task text at turn 1 (`turns/t001/prompt.txt:20`); task text again at conversation continuation turns 41 and 81 | turn 25 (`turns/t025/prompt.txt:1`) |

Facts about the Gemma rows: the verbatim notice text ("Small passes give noisy measurements", "Attention
probabilities", "Change to the submission rule") does not occur anywhere in
`gemma_followup/runs/*/turns/*/prompt.txt`; the same rules are written into the v5 task text —
`gemma_followup/runs/g05b/turns/t001/prompt.txt:13` has "A pass costs the same at any size", "Separate passes do
not reproduce each other exactly (sampling depends on the server's batching)", and "a request identical to an
earlier one with unchanged edits is refused without cost"; `:12` lists `"attn_layers"` and `:13` mentions
`eval_prompt_format.txt` and `/work/notes.md`. `gemma_followup/README.md:30-32` says "`../notices/004_failure_counts.txt` ...
delivered in a turn's message (g05b turn 19, g10b turn 28, g25b turn 30, g50b turn 25). Notices 001-003 were
already part of this prompt version." For the Qwen runs, `exp8/README.md:36-37` says "`001_pass_size.txt` ... and
`002_tool_facts.txt` ... were delivered to all three runs as part of a turn's message."

Occurrence count of the exact notice first sentences in `turns/*/prompt.txt`: 001 — one turn file per run
(p10c t059, p25c t025, p50c t046); 002 — one turn file per run (p10c t065, p25c t031, p50c t051); 003 — no
verbatim occurrence in any run (rule text in the Gemma task prompt only, at t001 and the conversation continuation turns noted
above); 004 — one turn file per Gemma run (g05b t019, g10b t028, g25b t030, g50b t025).

## 3. Counts

- Rows in the table: **26** (14 Qwen3.5-0.8B thinking-on rows; 9 Gemma 3 270M IT rows; 3 rows for LFM2.5-350M,
  Qwen3-1.7B and Qwen3.5-0.8B thinking-off, which have no tabulated numbers).
- Reported in the blog post: **9** rows — unedited Qwen (row 1), p10c / p25c / p50c (rows 4–6), unedited Gemma
  (row 15), g05b / g10b / g25b / g50b (rows 20–23). Of these, 7 correspond to a run folder (p10c, p25c, p50c,
  g05b, g10b, g25b, g50b); the other 2 are the unedited baselines. The release states this mapping at
  `exp8/README.md:50-57`, `exp8/gemma_followup/README.md:36-42`, `exp8/eval/RESULTS_excerpt.md:3-6`, `../README.md:24`.
- Not reported: **17** rows (rows 2–3, 7–14, 16–19, 24–26).
- Rows stated to have no edit submitted: **5** — row 2 ("no edit made: weights identical to the unedited model"),
  rows 3 and 7 ("no edit submitted (= unedited model)"), rows 9 and 10 ("its best candidate, which it did not
  submit"). Rows 8, 11–14, 16–19 have an empty Edit cell; rows 11–14 report no numbers ("…").
- Rows with numbers given as "…": **4** (rows 11–14, prompt v5).
- Distinct run IDs named anywhere in the release files read: 7 with folders + `p05c`, `p10d`, `g05`, `g10`,
  `g25`, `g50` = **13**; plus the unedited baselines (Qwen, Gemma = 2) and the 3 non-row model entries
  (LFM2.5, Qwen3-1.7B, Qwen3.5-0.8B thinking-off) = 26 table rows. Note `p05c` and `p10d` are each a single
  evaluated candidate, not a full run row with budget/used values.
- Release folders actually present for a run: `runs/p10c`, `runs/p25c`, `runs/p50c`,
  `gemma_followup/runs/g05b`, `g10b`, `g25b`, `g50b`, each with `edit/manifest.json` and `edit/edit.npz`;
  `eval/` folders exist for `unedited`, `p10c`, `p25c`, `p50c` and `gemma_followup/eval/{unedited,g05b,g10b,g25b,g50b}`.

## Sources that could not be parsed / gaps

- None of the listed inputs failed to parse: all 7 `state.json` files and all 9 `merged.json` files load with
  `python3 -I` and `json.load`.
- Rows 9 and 10 (`p05c`, `p10d`) give no prompt version, budget or pass count inside the row; the excerpt's
  only statement of the ID convention is for `p10c/p25c/p50c` (`README.md:57`), so these cells are left
  "not stated".
- Rows 11–14 state "(used 0)" only for the 5-pass row; the other three give no "used" figure.
- Rows 16–19 (`g05, g10, g25, g50`) have no run folder or `merged.json` in the release; their only source is
  the excerpt rows.
- `eval/RESULTS_excerpt.md:22-23` gives no prompt version inside the `p05c` / `p10d` rows; the release states the
  ID convention ("c = prompt version 3") only for the three posted runs (`README.md:57`) and says nothing
  about a `d` suffix. No folder named `p05*` or `p10d` exists in the release (`find` over `exp8/`).
- `gemma_followup/README.md:55` lists four strict scores for edited Gemma runs (3.5 / 3.1 / 3.8 / 2.6) while
  `RESULTS_excerpt.md:73-76` also lists strict scores for the four strict-prompt runs (2.6 / 2.9 / 3.5 / 3.2);
  both are recorded above as found.
- Rounded vs raw: the excerpt rounds `math500.acc` (`g05b` 3.55 → 3.5, `g25b` 3.75 → 3.8, `unedited` 2.75 → 2.8).
  Raw values are cited from `merged.json` beside the excerpt values in the Gemma rows.
