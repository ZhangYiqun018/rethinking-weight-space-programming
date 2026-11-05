# Exp3 part 2 — previous-token-head-from-weights pre-registration

Written BEFORE opening eval/heads.json, eval/scores.json, Astra's answers or runs/*/turns (protocol step 1-2). Inputs: task file,
PREREG.md/RESULTS.md/baseline.py, array names/shapes of blocks.npz. Constraints as Astra: four blocks only (no tok_emb/pos_emb),
no forward pass on any input, pure weight algebra, float64, `python -I`, allow_pickle=False. Notation and head folding as in
baseline.py: Wq',Wk',Wv' = W * ln1_w (column scaling); W_QK(l,h) = Wq'^T Wk' (384x384); W_OV(l,h) = Wo[:,s] Wv'[s,:];
Kc(A->B) = ||W_QK_B W_OV_A||_F / (||W_QK_B||_F ||W_OV_A||_F) (identical to baseline.py).

## Step 1: previous-token head in layer 0 (6 candidates per network)
Justification (brief): a previous-token head's whole job is to write token i-1 into the residual at position i for later heads to
read through their K input; so it should be the layer-0 head whose OV output is the most strongly read by later K sides.
(The "antisymmetric / rotation" idea assumes sinusoidal/RoPE positions; with learned absolute positions there is no reason for W_QK
to be antisymmetric, so I demote it to a secondary rule.)

PRIMARY rule P1 (K-side out-degree): D(A) = mean over all 18 later heads B (layers 1-3) of Kc(A->B). Pick A* = argmax_A D(A) over
the 6 layer-0 heads. Tie-break: lowest head index (never expected with float64).
SECONDARY S1 (antisymmetry): f(A) = ||(M-M^T)/2||_F / ||M||_F with M = W_QK(0,A). Pick argmax f; tie -> lowest index.
SECONDARY S2 (low rank): r(A) = ||M||_F^2 / sigma_max(M)^2 (stable rank of W_QK(0,A)). Pick argmin r; tie -> lowest index.

## Step 2: copy-head set from the picked head
For a picked head A*, S_A*(B) = Kc(A*->B) for the 18 heads B in layers 1-3. PRIMARY set rule C1: B is predicted a copy head iff
S_A*(B) > median(S_A*) + 3*1.4826*MAD(S_A*) (over the 18, same robust-SD rule and X=3 as PREREG.md). If this predicts nothing,
that is a valid output. Reported: the C1 set for P1 (HEADLINE), and, labelled secondary, C1 applied to S1's and S2's pick.
Also reported descriptively (not a rule): ranking of the 18 heads by S_A*.

## Scoring (after predictions.json written; step 3)
Truth: eval/heads.json 'prev' (is the picked head the argmax of 'prev' in layer 0? also report its rank among 6) and 'induct' > c for
c = 0.2 and 0.3. Per-corpus and pooled precision/recall; compare with Astra's copy_named in eval/scores.json. Rules and X are
frozen now; any variant tested afterwards is labelled post hoc.
Step 4 (only after scoring): read runs/<corpus>_r0/turns to describe how Astra found the previous-token head.
