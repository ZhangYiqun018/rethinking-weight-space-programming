# Exp3 no-LLM baseline — pre-registration

Written BEFORE opening eval/heads.json, eval/scores.json, eval/key.json or any final_answer.md. Only the README and
runs/python_r0/message.txt (the prompt) were read, plus (after this file) array names/shapes of blocks.npz.

## What is and is not available (from README/prompt)
Available: per layer l=0..3: ln1_w, ln1_b, W_q, W_k, W_v, W_o (384x384, (out,in); head j = rows/cols 64j..64j+63),
ln2_w, ln2_b, W_fc, W_proj. NOT available: tok_emb, pos_emb, lnf, W_out. Hence: no previous-token/positional signature
(positions live only in pos_emb), no true W_U W_OV W_E copying score, no token labels. The residual basis is shared across
layers, so products of blocks from different layers are meaningful. No forward pass is run (pure matrix algebra).

## Scores (numpy, float64)
LayerNorm weights are folded into the readers: Wq' = Wq diag(ln1_w), likewise Wk', Wv'. ln1_b, the mean-centering and the
1/std rescale are ignored (documented approximations). Per head (l,h), with 64-row slices:
- W_OV(l,h) = Wo[:, slice] @ Wv'[slice, :]            (384x384)
- Q-side/K-side: Wq'(l,h), Wk'(l,h) are 64x384.
1. K-composition (Elhage et al. 2021) of later head B=(l2,h2) with earlier head A=(l1,h1), l1<l2:
   Kc(A->B) = || Wq'_B^T Wk'_B W_OV_A ||_F / ( || Wq'_B^T Wk'_B ||_F * || W_OV_A ||_F ).
   (Key input replaced by A's output; the Frobenius norm of W_QK_B W_OV_A over the product of norms.)
   Kc is computed for every pair; the head score is S(B) = max over all earlier-layer heads A of Kc(A->B).
2. OV copying proxy (Olsson et al. 2022 use the positive-eigenvalue share of W_U W_OV W_E; embeddings are missing, so
   the identity map is used for both ends): P(A) = Re(sum of eigenvalues of W_OV_A) / sum |eigenvalues of W_OV_A|
   (= trace / eigenvalue-magnitude sum; in [-1,1]; +1 = pure positive/copying).

## Decision rule (X fixed now, not tuned)
Candidates = heads in layers 1,2,3 (18 heads per network; layer-0 heads cannot be K-composed with any earlier head).
Primary rule R1 ("K-comp"): head B is predicted copy iff S(B) > median(S over the 18 candidates) + 3 * 1.4826 * MAD(S over
the 18 candidates). (X = 3 robust SDs.)
Secondary rule R2 ("K-comp + OV"): R1 AND P(B) > 0.
Rule R3 (OV only): P(B) > median(P over 24 heads) + 3 * 1.4826 * MAD(P over 24 heads).
Each rule is applied independently per corpus network. R1 is the headline rule; R2, R3 are reported as pre-specified
variants. No other rule/threshold will be reported as a headline; any post-hoc exploration will be labelled as such.
If a rule predicts no head, that is a valid output (zero recall).

## Scoring (after predictions are written to disk)
Truth: eval/heads.json `induct` > c for c = 0.2 (post) and c = 0.3 (authors' scorer). Precision/recall/F1 per corpus
and pooled over the three networks (24 head-slots x 3). Compare with Astra's copy_named in eval/scores.json (both runs).

## Language ID
Pre-registered: no decision rule. n = 1 network per corpus (the two Astra runs share one blocks.npz), so a weight statistic
that "separates" three points cannot be validated; I will print descriptive per-network statistics (per-layer weight norms,
LN weight means, MLP/attention norm ratios) with no classifier, and state that language ID is not testable weight-only here.
