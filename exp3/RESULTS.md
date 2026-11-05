# Exp3 no-LLM baseline — results

Order followed: (1) PREREG.md written from README + prompt (runs/python_r0/message.txt) only, before opening eval/heads.json,
scores.json, key.json or any final answer; (2) baseline.py `predict` (never touches eval/) wrote predictions.json;
(3) baseline.py `score` read eval/ and wrote score_output.txt. The rule and X=3 were not changed after seeing truth.
Run: `/opt/homebrew/bin/python3 -I baseline.py predict|stats|score`. No forward pass; pure matrix algebra on blocks.npz.

## What is not available
No pos_emb (so no previous-token/positional signature), no tok_emb/W_out (so no W_U W_OV W_E copying score; the identity
is used in its place). LN weights folded in; ln1_b, mean-centering, 1/std and MLP paths ignored.

## Head prediction (pre-registered rules; full per-network output in score_output.txt)
| Rule | Predictions | Pooled P / R at 0.2 (5 true heads) | at 0.3 (2 true heads) |
|---|---|---|---|
| R1 K-composition, median+3 robust SD | none in any network | undefined / 0.00 | undefined / 0.00 |
| R2 R1 AND OV-positive | none | undefined / 0.00 | undefined / 0.00 |
| R3 OV positive-eigenvalue share (24 heads) | 16 heads, all layer 0 | 0.00 / 0.00 | 0.00 / 0.00 |
| Astra (both runs, summed) | 10 named / 10 true | 1.00 / 1.00 | 0.40 / 1.00 |

Truth (induct>0.2): python 2.1, 3.1, 3.5; romaji 3.3; english 3.1. At 0.3: python 3.5, english 3.1, romaji none.
Astra named exactly the 0.2 truth set in all six runs (so P=R=1 at 0.2; at 0.3 its precision drops to 0.40 because it
names heads below 0.3 — the discrepancy flagged in the audit).

The baselines score zero. Not even close: **post-hoc** (not pre-registered), the true copy heads have the *lowest* K-comp
scores among the 18 candidates (ranks 15-18 of 18; S 0.07-0.08 against medians 0.17-0.24; the highest are layer-1 heads),
i.e. this score is, if anything, anti-informative here. Likely reasons (hypotheses, untested): Frobenius-normalised
K-comp is dominated by large-norm shared residual directions rather than the induction-relevant ones; the induction
circuit needs the previous-token signal, which comes through positions (pos_emb, unavailable) and ln1_b/MLP paths that
the score ignores; LN scaling. So "standard weight-only composition scores do as well" is NOT supported — with these
inputs they do not find the heads at all. This is a negative result for this specific implementation and not evidence that
no weight-only method could work (e.g. one that had the embeddings, or an eigen-based QK/OV test of the full circuit).

## Language ID
Not testable: n=1 network per corpus (the two Astra runs share one blocks.npz), no classifier was pre-registered.
`baseline.py stats` prints descriptive per-layer norms; e.g. layer-0 attention norm python 20.9 / romaji 16.9 / english 23.5
and layer-0 MLP norm 48.7 / 52.7 / 42.8 differ, but three points cannot validate any separating statistic, and any rule
fitted on them would be overfit by construction. No claim made either way about Astra's language ID from this baseline;
note the prompt lists the three candidates, so Astra's task is a 3-way choice (chance 1/3).

## Caveats
One implementation of one family; thresholds fixed once; truth cutoffs from a single measurement per head (romaji 3.3 =
0.220 is near the 0.2 cutoff). Fine-tuning a threshold would not help: the truth heads rank at the bottom.
