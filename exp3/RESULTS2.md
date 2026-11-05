# Exp3 part 2 — can a fixed script find the previous-token head from weights alone? Results

## Order followed
1. Did not open eval/heads.json, eval/scores.json, Astra's answers or runs/*/turns until step 3 was finished.
2. PREREG2.md written first (one primary pick rule P1, two secondary S1/S2, one copy-set rule C1; frozen).
3. `/opt/homebrew/bin/python3 -I prevhead.py predict` (never touches eval/ or runs/) -> predictions2.json; then `prevhead.py score` -> score_output2.txt. Run once; no rule edited.
4. Only then read runs/<corpus>_r0 to describe Astra's method.
No tok_emb/pos_emb, no forward pass, pure weight algebra on blocks.npz (LN gains folded, as in baseline.py).

## Step 3a: previous-token head pick (truth = argmax of layer-0 `prev`; the true head scores 0.56-0.98, all other layer-0 heads <= 0.06)
| Rule | python (true L0H5) | romaji (true L0H3) | english (true L0H2) | right |
|---|---|---|---|---|
| **P1 (primary): K-side out-degree, mean Kc(A->later 18)** | L0H4 wrong (true rank 5/6) | L0H5 wrong (rank 3/6) | L0H5 wrong (rank 5/6) | **0/3** |
| S1 (secondary): antisymmetric fraction of W_QK, `\|\|(M-M^T)/2\|\|/\|\|M\|\|` | L0H5 right | L0H3 right | L0H2 right | **3/3** |
| S2 (secondary): lowest stable rank of W_QK | L0H0 wrong | L0H2 wrong | L0H5 wrong | 0/3 |

The pre-registered PRIMARY rule failed. My stated justification for demoting antisymmetry (it presumes sinusoidal/RoPE positions) was
wrong on the evidence: the secondary S1 is right in all three corpora. Chance for a 3-for-3 on a 1-in-6 pick is (1/6)^3 = 1/216, but
this is a secondary rule, one of three tried, and the margins are thin: S1 values python 0.784 vs next 0.741 (L0H1), romaji 0.730 vs 0.698,
english 0.724 vs 0.705. It is a lead, not a validated detector. n = 3 networks.

## Step 3b: copy heads from the picked head (C1 = Kc(A*->B) > median + 3 robust SD over the 18 heads B in layers 1-3)
Truth: induct > 0.2 (5 heads pooled: python 2.1, 3.1, 3.5; romaji 3.3; english 3.1); > 0.3 (2 heads: python 3.5, english 3.1).
| Rule | predicted heads (pooled) | P / R at 0.2 | P / R at 0.3 |
|---|---|---|---|
| **C1 after P1 (headline)** | none in any network | undefined / 0.00 | undefined / 0.00 |
| C1 after S1 (secondary) | romaji L1H2; english L1H5, L3H1 | 1/3 = 0.33 / 1/5 = 0.20 | 0.33 / 0.50 |
| C1 after S2 (secondary) | none | undefined / 0.00 | undefined / 0.00 |
| Astra, both runs summed | 10 named | 1.00 / 1.00 | 0.40 / 1.00 |

Headline: the fixed pipeline (P1 then C1) found nothing; P1 picked the wrong head every time. Even with the right previous-token head
(S1), the pre-registered threshold C1 recovers only english L3H1 (hit) plus two false positives (romaji L1H2, english L1H5), and misses
python entirely: with S1's head python's three true heads rank 1, 2, 4 of 18 (L3H5, L2H1, L3H1; L2H2 is rank 3), romaji L3H3 ranks 3, english L3H1 ranks 1. So the ranking is useful (matches the reviewer's post-hoc 1,2,4 / 3 / 1) but the
median+3*MAD cutoff is the wrong decision rule: the 18 scores are not heavy-tailed enough. A top-k rule would look better, but that is
post hoc and was not registered (python top-3 by S1 = L3H5, L2H1, L2H2, 2/3 right; romaji top-1 = L1H2 wrong, top-3 contains L3H3; english top-1 right). Not a result.

## Step 4: how Astra found the previous-token head (python_r0; also romaji_r0, english_r0 briefly)
python_r0 is exploratory, not a single algorithm, but the evidence it used is essentially my S1:
- runs/python_r0/tool_calls.jsonl:4 (turn t004 code, `symmetric energy` at turns/t004/answer.md:27) prints per head the symmetric-energy fraction
  of QK = ||(a+a^T)/2||^2/||a||^2 plus the top QK eigenvalues. Output (same file, `out`): L0H5 symmetric energy 0.386 (lowest of layer 0,
  i.e. most antisymmetric), L0H1 0.45, both with large complex-conjugate eigenvalue pairs (L0H5 (0.207, +-0.275), L0H1 (0.218, +-0.345)),
  while L0H0/H2/H3/H4 have 0.51-0.70 and mostly real eigenvalues. That equals my S1: 0.386 -> antisym fraction sqrt(0.614) = 0.784.
- tool_calls.jsonl:7 and turns/t009/answer.md:20-21 then test "phase(H1) approximately k*phase(H5)" via generalized eigenvalues of the QK pair, to
  call H5 one-back and H1 two-back (a rotation/shift reading). final_answer.md:8, 34: "strongest one-character-shift candidate". The "2:1 shift" claim is
  unverified (it states offsets are inferred).
- Copy heads then come from trace-based K-composition from L0H5 (and L2H2) with key-minus-query specificity (tool_calls.jsonl:8:
  `key-query` sorted), final_answer.md:9-13, 44. This differs from my Kc in two ways: signed trace(M B) rather than Frobenius norm, and subtracting the query-composition score; plus mean-centering P. That may be why Astra's copy heads are right and my Frobenius-normalised C1 is not (untested).
- romaji_r0/final_answer.md:18: "position-like rotational QK structure separated from its character-carrying value directions"; english_r0/final_answer.md:5: also cites unusually large L0H2 Q/K norms (30.8/28.3 vs 3-7), a different cue (norm outlier) plus complex eigenvalue pairs. So the cues vary by corpus.
Could a script copy it? The core cue (low symmetric energy / complex QK eigenpairs in layer 0) yes, that is S1, and it already works 3/3. The
later narrative (phase-ratio lag test, 2:1 shifts, choosing which pairs to inspect) is a judgement process, not a fixed algorithm. Astra's L0H1 also looks rotational in python but is labelled local, so the cue alone does not make the call unique; I did not check the other corpora's turns in depth.

## Caveats
n = 3 networks, one head each; S1 margin small; S1 was registered as secondary and the primary failed. The post-hoc top-k remark above is unregistered. This also does not test whether S1 generalises to other trained models. Truth `induct`
cutoffs are single measurements (romaji L3H3 0.220 near the 0.2 line). Two Astra runs per corpus share one network.
