# Claims vs evidence: the skeleton for the response blog

Blog source: github.com/RikiyaT/RikiyaT.github.io, `blog/weight-space-programming/index.html`. The current version is commit e35f074 (2026-11-03); the original is 6bfa7a6 (2026-11-02), archived in `notes/blog_src/index_6bfa7a6_original.html`. Line numbers below refer to `notes/blog_src/blog_text.md`, the text of e35f074.

Verdict categories:
- **Holds**: the claim survives our checks.
- **Overstated**: the numbers are real, but the conclusion does not follow.
- **Contradicted**: our data says otherwise.
- **Untested**: no evidence yet.

Every row cites the evidence file. Each one still needs a final re-check before publication.

## Headline / abstract

| # | Blog says | Our evidence | Verdict |
|---|---|---|---|
| H0 | Original abstract (6bfa7a6): LLMs read/write weights "**well past what we'd expect from humans or conventional algorithms**". | (1) A textbook difference-of-means edit with no LLM recovers 79% of the Exp8 gain (E8-5). (2) A 20-line, three-rule script names 3/4 Exp1 networks (E1-3). (3) On Nov 3 the authors softened the sentence to "...using a variety of recent tools and algorithms... **may** extend beyond what we'd expect..." (commit e35f074). | Contradicted in its original form; since walked back to "may" |
| H1 | "its fix improves instruction following too" (intro) | Only in the looping regime. With thinking off (B) the edit costs **−4.7** IFEval; with repetition penalty (D) it costs **−8.0**. Base with budget forcing (65.1) beats the edited model (62.9) (E8-3, E8-4). | Overstated |

## Exp1: which game (read)

| # | Blog says | Our evidence | Verdict |
|---|---|---|---|
| E1-1 | Astra names the right game for 4/4 networks from the weights alone. | Reproduced: 12/12 in each of orig/anon/swap12/perm, and randinit correctly gets "not identifiable" (`exp1/results/scores.csv`).<br>Open models on the same prompts: DeepSeek V4.1 orig 4/8, anon 3/8; Kimi K3 3/8, 3/8. Both confidently name a game for random-init weights (abstain 1/7, 3/8), whereas Astra abstains 12/12. | Holds |
| E1-2 | "so that it is not able to tell unless the weights are read" (l.141); "exciting particularly because there are so many games out there" (l.143) | No prompt-text leakage: in the anon condition the text is identical across the four games, and Astra still gets 12/12. | Holds (no leakage) |
| E1-3 | The cue is "which actions the output layer tells apart" (l.158), which the blog itself names. | The task is solvable from shallow action cues.
- A pre-registered three-rule action-axis script with no LLM gets 3/4 (`exp1/stats_baseline.py`).
- With channel names removed and W1 shuffled, Astra still names Freeway, Breakout and Pong 9/9, at confidence 25–65, and abstains 3/3 on Pac-Man (four directions). That is exactly the script's pattern.
- Shuffle alone (names kept): 10/12.
- n = 1 network per game (best of 3 seeds). | Overstated (identification comes from the action pattern) |

## Exp4: fix the car (write/edit)

| # | Blog says | Our evidence | Verdict |
|---|---|---|---|
| E4-1 | "In a single try with no tools, Astra's edit makes the car succeed in all 1,024 episodes" (l.254) | Reproduced: 1024/1024, mean length 73. It is robust to start shifts of ±0.5, observation noise, friction ×0.8–1.2 and body mass ×0.5–4 (`exp4/RESULTS.md`). | Holds |
| E4-2 | "Astra changes the policy to back up the car before driving forward"; "once the green wheel is past the red strip, the path switches off and the trained network drives the rest" (l.267) | A two-parameter open-loop controller that ignores observations (reverse 16 steps, then drive) matches the edit across the main perturbation grid: start shift ±0.5, friction ×0.8–1.2 and mass ×0.5–4 (256/256 each; a fine friction sweep ×0.7–1.8 is 16/16 per point), and observation noise (256/256 vs the edit's 252–253/256). They differ only at the edge: at friction ×2.0 and start shift −0.6 the edit succeeds (16/16) and open-loop fails (0/16). At friction ≥2.5 both fail while the unedited network succeeds (corrected 2026-11-04 after an independent check). The edited actor emits a single motor sequence (entry-0 logit margin about 600). The task never needs closed-loop control. | Overstated (nearly solvable open-loop; the closed-loop hand-off matters only at the edge of the range) |

## Exp5: MinAtar from scratch (write)

| # | Blog says | Our evidence | Verdict |
|---|---|---|---|
| E5-1 | Astra's one-shot networks score 517.4 (Breakout), 53.1 (Seaquest) and 53.2 (Asterix). | Reproduced on official MinAtar 1.0.15: 500.1 ± 7.8, 53.7 and 54.3 (`results/exp5_phase1b.md`). | Holds |
| E5-2 | "they score 20 times as much in Breakout, come close in Asterix, and reach 60% of its score in Seaquest" vs PPO at 500M steps | The authors' PPO uses CleanRL defaults (64 envs, lr 5e-4); in our sweep 64 envs is the worst setting in every game. Same network, same official evaluation, tuned (lr 1e-3, 512 envs), 500M steps, 5 seeds:<br>- **Breakout:** 56–440. Astra's 500 still leads, but by 1.1–9×, not 20×.<br>- **Seaquest:** 99–114, so tuned PPO is **about 2× Astra**, not Astra at 60% of PPO.<br>- **Asterix:** 36–43. The authors' baseline is fine here, and Astra about ties it.<br>Data: `results/runs/{exp5_ppo_breakout,exp5_ppo_seaquest,exp5_ppo_asterix}`. | Contradicted (Seaquest), overstated (Breakout), holds (Asterix) |
| E5-3 | "Astra reasons about the desired programs ... and converts them into neural form" (l.303); source code given (l.296) | Disclosed. Astra's Breakout net is close to a hand-written simulator-lookahead heuristic (546.4 here) and is a deterministic program: policy entropy 0.000 nats, 4% of the weights nonzero. | Holds (disclosed) |
| E5-4 | Implied: Astra can write such networks. | 10 fresh replies per game, effort xhigh, the authors' prompt (`exp5/astra_write/results/summary.csv`; all 60 replies parse, 0 skipped statements). Median [IQR] vs the authors' one network:<br>- **Breakout:** 490 [484, 543] vs 500; 9/10 reach 90% of it. Reliable.<br>- **Asterix:** 62.5 [48.8, 65.8] vs 54.3; 7/10. Reliable, and above tuned PPO (36–43).<br>- **Seaquest:** 33.5 [18.0, 37.7] vs 53.7; **1/10**. The published network is the luckiest-looking draw; the typical one is about 1/3 of tuned PPO (99–114).<br>Without the game's source code in the prompt (`nosrc`; about 5 KB instead of 10 / 18 / 11 KB for Breakout / Seaquest / Asterix): Seaquest 0.8, Asterix 17.4, Breakout 381 [60, 470] (3/10 collapse to about 21). **The networks are programs translated from the simulator's code, not policies found from the game.** | Holds for Breakout and Asterix; Seaquest overstated (n = 1 above the typical reply); depends on source code |

## Exp8: make Qwen3.5-0.8B better at math (edit)

| # | Blog says | Our evidence | Verdict |
|---|---|---|---|
| E8-1 | MATH-500 7.9 → 48.1, GSM8K 13.8 → 67.1, IFEval 21.0 → 62.3 (l.402–410) | Reproduced with full benchmarks: 8.3 → 50.6, 12.4 → 67.9, 20.5 → 62.9 (task exp8_config_A). | Holds |
| E8-2 | Section title "Make a language model better at math"; caveat "Fixing loops ... is not the same as making a model better at math" (l.417); "Qwen's model card notes that this model is prone to thinking loops" (l.396) | Partly disclosed. Not disclosed: the unedited model in its default non-thinking mode already scores **44.2 / 56.0 / 54.7**. The authors knew: their code comment records about 50–55% with thinking off before they switched to thinking on (`code/profiles.py`). | Overstated (baseline choice) |
| E8-3 | (implied) the edit is the way to get these scores | Edited A vs the best no-edit setting, paired: **+6.4** on MATH-500 (vs thinking off), **+11.5** on GSM8K (vs budget forcing), **−2.2** on IFEval (no gain) (`results/exp8_cross_config.md`). | Overstated |
| E8-4 | (not reported) | The edit only helps in the looping regime. With thinking off it is +0.4 on MATH-500 (n.s.) and −4.7 on IFEval; with repetition penalty it is −8.0 on IFEval. A single decoding knob (repetition penalty) lifts the unedited model from 7.9 to 23.1 on MATH-500. | New evidence against "improves instruction following" |
| E8-5 | "Astra traced the looping to one direction ... much like removing a behavior by deleting a single direction" (l.400) | **Holds, and that is the point.** The textbook difference-of-means recipe, with no LLM and k chosen on a practice split, reaches **41.9%** on MATH-500 vs Astra's 50.6%. Random-direction and random-column controls with the same k = 8 do nothing on the full MATH-500: 5.1% and 8.0%, vs base 8.3% (task exp8_steer). Our top-64 columns overlap Astra's in 31/64 and 22/64 (chance about 1; measured in the 16/16-window dry run). Astra adds +8.8 [+6.6, +10.8] over the recipe. | Overstated (mostly reachable without an LLM) |
| E8-6 | Table shows the 10, 25 and 50-pass runs. | The release lists 14 thinking-on runs; the post shows 3 edited ones. Only one run (p50c) gives a large gain. The v2 run made no edit; v4 runs gave no edit or an edit identical to p25c; v5 runs report no numbers. Logs are released (the blog says "complete logs of every experiment") but the post does not mention the other runs (`results/exp8_author_runs.md`). | Selective reporting in the text; transparent in the logs |
| E8-7 | Future: "Zero-data capability steering ... 'Can you add Capability X to the model?'" (l.449) | Their own Gemma run could not add ability (l.417), and our B config shows no math gain once looping is removed. | Overstated |

## Exp2, Exp3, Exp6, Exp7: static audit

Source: `results/audit_exp2367.md` by a Sonnet worker. I re-checked three items against the release files: the Exp6 unreleased runs (README l.51), the Exp3 cutoff (scores.json and README l.58–59) and the Exp7 base model (README l.33).

| # | Blog says | Our evidence | Verdict |
|---|---|---|---|
| E2-1 | Astra tells what shape each diffusion model generates (3/3); "it draws every shape as an outline ... This mistake kind of shows it does not rely on calculating the forward pass" | No-LLM read-outs (`exp2/RESULTS.md`; rules pre-registered and run blind to network ids):
- The output bias alone (no forward pass) gives the right family 3/3, by narrow margins.
- A one-step forward pass on noise gives 2/3 (square read as circle).
- Every read-out shows filled shapes, which is correct; Astra's outlines are wrong (the authors admit this).
- Caveats: our script knew the 3-shape candidate set, while Astra answered an open question; n = 3; the forward-pass read-outs are something Astra was not allowed to do. | Holds roughly; weak evidence either way. Astra's open-ended 3/3 is better than our candidate-aware script |
| E3-1 | Astra names the right language for all three networks in both attempts and finds every copying head. | **Holds. Astra's method decomposes into known weight-algebra steps that a script can copy only partly.**
- (1) The pre-registered K-composition baseline, taking the max over earlier heads, finds no copy heads. With LayerNorm centering added (post hoc) the true heads still rank 15–18 of 18 (`exp3/RESULTS.md`).
- (2) Pre-registered previous-token-head rules (`exp3/RESULTS2.md`):
  - The primary rule (K-side out-degree) is wrong 3/3.
  - The secondary rule S1 (antisymmetric fraction of W_QK) is right 3/3: chance 1/216, but margins are thin (0.784 vs 0.741 and similar). I re-checked this independently.
- (3) Given that head, K-composition ranks the true copy heads 1,2,4 / 3 / 1 of 18. The pre-registered median+3MAD cutoff still misses most of them.
- (4) Astra's own code uses the same cues: QK symmetric/antisymmetric energy and complex eigenpairs to find the previous-token head, then K- minus Q-composition.
- **Reading:** Astra correctly applies a known interpretability recipe without being told it, and makes the right judgement calls. Our fixed scripts recover the ranking but not the final decision. Caveats: n = 1 network per corpus, a 6-way candidate list in the prompt, language confidence 0.48–0.62, and the 0.3→0.2 cutoff (disclosed). | **Holds** (Astra better than our scripts; its cues are scriptable) |
- A pre-registered K-composition / OV-eigenvalue script finds no copy heads: P = R = 0 at cutoffs 0.2 and 0.3 (`exp3/RESULTS.md`). I reviewed the implementation against the prompt's forward pass (head slicing, LayerNorm, composition formula) and found no error.
- Astra worked from the middle blocks only, with no embeddings, and could run Python on the weights.
- Remaining caveats are minor: the prompt lists the candidate corpora (6 options with the shuffled variants); language calls are at 0.48–0.62 confidence; n = 1 network per corpus; the scorer cutoff went from 0.3 to 0.2, disclosed in the README. | **Holds** (our baseline failed) |
| E6-1 | Astra writes a Go network that beats PPO; win rate grows with budget. | The numbers recompute exactly. There is one run per budget; a 1-pass run and a v2 prompt were "made but are not in the post" (README l.51). The 50-pass run practised against a seed-1 PPO, a sibling of the two test opponents. The big-PPO baseline was tuned only on learning rate (3 values, 1 seed), and its seed spread is large. A plain-code heuristic baseline is missing.<br>**Rule-based baseline (`exp6/`, preregistered, our pgx engine reproduces all six published head-to-heads within 2.5 points):** a fixed 1-ply tactics script R1 (capture, save atari, threaten, avoid self-atari; no network, search or training) beats the same-size PPO s0 / s1 / s2 74.1 / 61.1 / 69.3 % (Astra b0: 42.6 / 26.3 / 36.3; b50: 74.2 / 70.0 / 78.5). Independent rerun with a new seed: 62.5 % vs s1. But Astra beats R1 head to head: b0 90 %, b50 95 %. | Holds as reported, but "beats PPO" is a low bar: a fixed script beats the same PPO by more than Astra b0 does. Astra is genuinely stronger than that script |
| E6-2 | Implied by the title ("LLMs can natively read ... weights"); not tested by the authors at this scale. | Our test (`exp6/readwt/`): fresh Astra sessions given the three Go networks blind (b0, b50, PPO), in a verified sandbox, vs a control without weights, 3 each. Identifies PPO 3/3 (dense vs sparse), but **swaps b0/b50 3/3 at ≥94 % confidence**. Behavioural predictions are no better than without weights for capture rate (MAE 31 vs 28 pts), atari escape (31 vs 31) and pairwise strength (36 vs 34). It predicts PPO tengen 14–30 % (true 99.9 %) and Astra-net win rate vs R1 32–65 % (true 90–95 %). Correct where the answer is visible in biases (Astra-net opening). | Not supported at 1.5M scale: structural facts yes, behaviour no |
| E7-1 | Weights alone: Astra cannot tell which model is sycophantic. With an attention tool it gets 3/3. | The authors report the weights-only failure themselves. The prompt states the exact behavioural contrast and the training-message format, and Astra's probes copy the training template almost word for word. One of the three pairs is called at only 0.55. The base model already follows 100% of wrong guesses, so the "sycophant" behaves like the base and the honest model is the one changed; distance-to-base and a scripted attention-to-guess probe are obvious baselines.<br>**No-LLM weight statistic (`exp7/`, independent reproduction):** we retrained the authors' recipe; 12 pairs pass the published behaviour gate. D = Σ‖W − W_base‖_F. Preregistered 'closer to base = sycophant': 0/6 (the sycophant moved farther in 6/6). The reversed rule, preregistered on 6 fresh pairs: **5/6, below the registered 6/6 bar (p = 0.11)**. Margins are small (ratio 0.97–1.05). So a weak, mostly consistent signal, not a reliable classifier. Our check does not overturn this claim. | Overstated (task framing gives the answer's structure) |

## Not yet covered

Exp2, 3, 6 and 7 have only a static audit, with no reruns. The cheapest decisive checks are listed in `results/audit_exp2367.md` §6.
