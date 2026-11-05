# Figure data and validation report

Completed the four placeholders and added five new data figures. The two existing Breakout figures remain. Captions are numbered in page order, with matching EN/ZH numbers; linked Figure references were updated. 

## Files and reproduction

- `figs/data/build.py`: data extraction and provenance; reads upstream only as data; executes only the two local scoring modules. No training, upstream code execution, or external requests. No array deserialization is needed.
- Nine figure JSON files plus `figs/data/bundle.js`, generated together. Each point has source path, line (document start for JSON, plus a field/key locator), and its derived values. This avoids fetch and works with file://.
- `css/figs.css` and `js/figs.js`: local SVG figures, responsive panel selection, keyboard-focusable point descriptions, EN/ZH rerendering, exact values/source tables, missing cells, and interval whiskers.
- The historical HTML installer is not distributed; the release includes the finished page.

```sh
python3 -I blog/figs/data/build.py
python3 -I blog/verify/figs/check_data.py
node blog/verify/figs/check_renderer.cjs
python3 -I blog/validate.py
node blog/verify/figs/render_figs.cjs
```

## Validation and limits

- PASS: HTML structure, local links/anchors/citations, bilingual sibling pairing and section number parity. Existing TODO markers elsewhere are unchanged.
- PASS: nine data files; finite values; CI bounds contain their point; shape-grid dimensions; Astra/tuned replicate counts; missing Go matchups and steering settings; caption text and figure numbers.
- PASS: real renderer executed in a minimal DOM harness across all 53 panels, three content widths, and both languages (318 panel executions), with expected mark counts and no non-finite SVG attributes. This is an execution check, not visual QA.
- This historical report does not establish browser rendering or pixel-layout correctness. Run the supplied browser checks on the target browser.
- Browser verification harness: `verify/figs/render_figs.cjs` tests every panel at 1440, 390 and 360 viewport widths in EN/ZH, asserts nonempty SVGs and exact expected mark counts, checks bounds and language toggling, saves each figure’s initial-panel screenshot (plus failure screenshots) under `verify/figs/`, and writes `render-results.json`. It defaults to file://; an HTTP URL is optional. The pre-existing page still uses its remote Distill template/fonts; the new figure assets themselves have no external dependency.

## Public-page cleanup

- Public source paths use the aliases in results/runs/. Model labels contain model names only. Raw reply provenance cites its scorer, reply directory, network/condition/repetition and model rather than provider-bearing filenames. The earlier GLM comparison is described as a second, earlier run. Public provenance uses repository-relative paths.
- PASS: verify/scan_public.py reports PUBLIC SCAN PASSED; bilingual HTML validation also passes after cleanup.

## Figure data

Values below summarize every displayed panel. Exact per-observation values, labels, intervals, denominators and source locators are in the corresponding JSON. Ranges exclude null/unmeasured cells; medians are counted as additional marks only where explicitly drawn.

### Figure 1: exp1-accuracy

Accuracy by condition; random initialization is scored as abstention.

Astra rows are filtered by model name; DeepSeek and Kimi use scores_open.csv only, so overlapping score tables are not double-counted. GLM raw replies are scored in memory with exp1/score.py:score_text, without writing to exp1/. Review-needed non-random replies are excluded and their counts retained. Random-init uses label==unknown as success. No-LLM anon/swap12 derive from orig and anonshuffle from shuffle because the rule ignores names; these are derived, not new runs. Missing open-model conditions remain null.

Data: [exp1-accuracy.json](exp1-accuracy.json).

| Panel | Marks | Missing cells | Value range |
|---|---:|---:|---|
| orig | 5 | 0 | 37.5 to 100 % |
| anon | 5 | 0 | 12.5 to 100 % |
| swap12 | 2 | 3 | 75 to 100 % |
| perm | 2 | 3 | 75 to 100 % |
| shuffle | 2 | 3 | 50 to 83.3333 % |
| anonshuffle | 2 | 3 | 50 to 75 % |
| randinit (abstention / 弃答) | 5 | 0 | 14.2857 to 100 % |

Source files (repo-relative):

- `blog/verify/exp1_baseline.txt`
- `exp1/results/raw`
- `exp1/results/scores.csv`
- `exp1/results/scores_open.csv`

### Figure 2: exp2-shapes

Released samples and literal Example grids: shape recognition does not establish fill recognition.

The first released individual sample is read from eval/samples/<id>.txt (twelve rows of doubled ASCII intensity characters). The renderer preserves one literal symbol per pixel; it does not invent grayscale values or threshold the sample. Astra grids are the original author answer.md Example, not a cherry-picked fresh rerun. The open comparator is fixed DeepSeek r0 for each network. exp2/score_replies.py is read and its local field/grid/flood-fill functions are used. All selected Astra grids are outlines; samples are individual filled shapes, not the misleading position-averaged square. These are illustrative drawings, not aggregate accuracy estimates.

Data: [exp2-shapes.json](exp2-shapes.json).

| Panel | Marks | Missing cells | Value range |
|---|---:|---:|---|
| triangle | 432 | 0 | Literal sample / Example grid cells |
| square | 432 | 0 | Literal sample / Example grid cells |
| circle | 432 | 0 | Literal sample / Example grid cells |

Source files (repo-relative):

- `exp2/results/raw`
- `upstream/lm_understand-logs/exp2_diffusion/eval/samples/246a5b58.txt`
- `upstream/lm_understand-logs/exp2_diffusion/eval/samples/a6a9dfb7.txt`
- `upstream/lm_understand-logs/exp2_diffusion/eval/samples/d88aba69.txt`
- `upstream/lm_understand-logs/exp2_diffusion/runs/circle/answer.md`
- `upstream/lm_understand-logs/exp2_diffusion/runs/square/answer.md`
- `upstream/lm_understand-logs/exp2_diffusion/runs/triangle/answer.md`

### Figure 3: exp4-openloop

Success rates across perturbation sweeps; use the selector to inspect each sweep.

All ten perturbation tables with success counts are included: nominal; coarse start; observation noise; coarse wheel friction; coarse uniform mass; fine start; body-only mass; wide friction; wide uniform mass; fine friction. Reverse-step schedule tuning is not a perturbation success-rate sweep and is omitted. The nominal perturbation table has its own episode count; the separate larger H1 determinism run is not substituted. Fine tables use their stated denominator. No base values are fabricated for fine start shift. Starred edge rows and generated notes identify the two requested boundary cases.

Data: [exp4-openloop.json](exp4-openloop.json).

| Panel | Marks | Missing cells | Value range |
|---|---:|---:|---|
| Fine wheel-friction sweep (16 episodes per point; successes out of 16) | 51 | 0 | 0 to 100 % |
| Nominal | 3 | 0 | 0 to 100 % |
| (a) car start x shifted by d (body and both wheels; scene units; car body is 0.66 wide, starts at x=2.8) | 18 | 0 | 0 to 100 % |
| (b) Gaussian observation noise, sigma x per-feature std; std taken over 64 rollouts of the named actor (live steps only); noise redrawn each step | 18 | 0 | 0 to 100 % |
| (c) wheel friction multiplier (both wheels) | 12 | 0 | 0 to 100 % |
| (c) car mass multiplier (body and both wheels, inverse mass and inverse inertia scaled) | 12 | 0 | 0 to 100 % |
| Fine x-shift sweep (16 episodes per point; successes out of 16) | 50 | 0 | 0 to 100 % |
| Body-only mass multiplier (256 episodes) | 21 | 0 | 0 to 100 % |
| Wheel friction, wider range (256 episodes) | 12 | 0 | 0 to 100 % |
| Uniform car mass, wider range (256 episodes) | 9 | 0 | 0 to 100 % |

Source files (repo-relative):

- `exp4/RESULTS.md`

### Figure 4: exp5-ppo-vs-astra

Individual seed and reply scores, with group medians.

Only released ppo500 seeds are shown for authors PPO; the upstream README states this is the final training budget and the original blog picks the best seed. Tuned PPO is restricted to s2_lr1e-3_e512_ent0.01_s* in the named artifacts. Fresh Astra orig/nosrc rows are explicitly model-filtered. summary.csv is never read. All available high-effort model runs are preserved separately. Median diamonds are computed from the plotted observations; random and author-Astra re-evaluation use phase1b.jsonl. The log(1+x) axis preserves zero. High-effort coverage is not equal across games; absent model runs are not imputed.

Data: [exp5-ppo-vs-astra.json](exp5-ppo-vs-astra.json).

| Panel | Marks | Missing cells | Value range |
|---|---:|---:|---|
| breakout | 37 | 0 | 0.512 to 596.327 score / 得分 |
| seaquest | 38 | 0 | 0.097 to 114.424 score / 得分 |
| asterix | 37 | 0 | 0.509 to 81.039 score / 得分 |

| Game | Group | Individual values | Median |
|---|---|---|---|
| breakout | Authors PPO / 作者 PPO | 25.344, 12.194, 19.82 | 19.82 |
| breakout | Tuned PPO / 调参 PPO | 307.823, 440.134, 407.647, 56.079, 195.069 | 307.823 |
| breakout | Astra no-source / 无源码 | 535.024, 476.286, 420.942, 341.913, 23.055, 541.135, 21.251, 21.251, 169.769, 451.061 | 381.428 |
| breakout | Fresh Astra / 新 Astra | 492.099, 487.772, 492.675, 483.697, 576.24, 446.086, 455.194, 559.59, 596.327, 484.492 | 489.935 |
| breakout | DeepSeek V4.1 · orig · high | 7.856 |  |
| breakout | Kimi K3 · orig · high | 7.815 |  |
| breakout | DeepSeek V4.1 Flash · orig · high | 11.296 |  |
| breakout | Random / 随机 | 0.512 |  |
| breakout | Author Astra re-eval / 作者 Astra 复测 | 500.071 |  |
| seaquest | Authors PPO / 作者 PPO | 7.493, 4.379, 88.884 | 7.493 |
| seaquest | Tuned PPO / 调参 PPO | 104.47, 114.424, 102.636, 103.359, 99.244 | 103.359 |
| seaquest | Astra no-source / 无源码 | 1.533, 1.616, 0.869, 1.135, 0.121, 0.74, 1.967, 0.371, 0.542, 0.097 | 0.8045 |
| seaquest | Fresh Astra / 新 Astra | 69.919, 35.6, 4.955, 33.475, 33.575, 14.677, 8.025, 41.498, 27.772, 38.42 | 33.525 |
| seaquest | DeepSeek V4.1 · orig · high | 0.564 |  |
| seaquest | Kimi K3 · orig · high | 0.644 |  |
| seaquest | DeepSeek V4.1 Flash · orig · high | 0.112 |  |
| seaquest | GLM 5.3 · orig · high | 0.18 |  |
| seaquest | Random / 随机 | 0.118 |  |
| seaquest | Author Astra re-eval / 作者 Astra 复测 | 53.725 |  |
| asterix | Authors PPO / 作者 PPO | 44.706, 55.204, 46.879 | 46.879 |
| asterix | Tuned PPO / 调参 PPO | 43.28, 35.742, 42.3, 39.519, 39.477 | 39.519 |
| asterix | Astra no-source / 无源码 | 15.203, 19.321, 17.655, 12.868, 17.167, 3.45, 42.893, 22.234, 10.972, 24.066 | 17.411 |
| asterix | Fresh Astra / 新 Astra | 58.171, 31.624, 64.922, 63.3, 72.084, 81.039, 61.747, 45.652, 41.308, 66.033 | 62.5235 |
| asterix | DeepSeek V4.1 · orig · high | 1.576 |  |
| asterix | DeepSeek V4.1 Flash · orig · high | 5.516 |  |
| asterix | GLM 5.3 · orig · high | 11.067 |  |
| asterix | Random / 随机 | 0.509 |  |
| asterix | Author Astra re-eval / 作者 Astra 复测 | 54.309 |  |

Source files (repo-relative):

- `exp5/astra_write/results/scores.csv`
- `exp5/astra_write/results_high/scores.csv`
- `exp5/results/phase1b.jsonl`
- `results/runs/exp5_ppo_asterix/runs/s2_lr1e-3_e512_ent0.01_s0/eval.json`
- `results/runs/exp5_ppo_asterix/runs/s2_lr1e-3_e512_ent0.01_s1/eval.json`
- `results/runs/exp5_ppo_asterix/runs/s2_lr1e-3_e512_ent0.01_s2/eval.json`
- `results/runs/exp5_ppo_asterix/runs/s2_lr1e-3_e512_ent0.01_s3/eval.json`
- `results/runs/exp5_ppo_asterix/runs/s2_lr1e-3_e512_ent0.01_s4/eval.json`
- `results/runs/exp5_ppo_breakout/runs/s2_lr1e-3_e512_ent0.01_s0/eval.json`
- `results/runs/exp5_ppo_breakout/runs/s2_lr1e-3_e512_ent0.01_s1/eval.json`
- `results/runs/exp5_ppo_breakout/runs/s2_lr1e-3_e512_ent0.01_s2/eval.json`
- `results/runs/exp5_ppo_breakout/runs/s2_lr1e-3_e512_ent0.01_s3/eval.json`
- `results/runs/exp5_ppo_breakout/runs/s2_lr1e-3_e512_ent0.01_s4/eval.json`
- `results/runs/exp5_ppo_seaquest/runs/s2_lr1e-3_e512_ent0.01_s0/eval.json`
- `results/runs/exp5_ppo_seaquest/runs/s2_lr1e-3_e512_ent0.01_s1/eval.json`
- `results/runs/exp5_ppo_seaquest/runs/s2_lr1e-3_e512_ent0.01_s2/eval.json`
- `results/runs/exp5_ppo_seaquest/runs/s2_lr1e-3_e512_ent0.01_s3/eval.json`
- `results/runs/exp5_ppo_seaquest/runs/s2_lr1e-3_e512_ent0.01_s4/eval.json`
- `upstream/lm_understand-logs/exp5_write_from_scratch/eval/minatar_eval.json`

### Figure 7: exp6-winrates

Go win rates with Wilson confidence intervals; missing matchups are marked.

Raw score and wilson95 fields are used without intermediate Markdown rounding. All requested rows and opponents are represented. b10/b25 versus PPO/random are unmeasured in the supplied results directory and remain explicit null cells. “Beats PPO” means the lower Wilson confidence bound exceeds equal wins, not merely a point estimate above it. Raw matches versus R0/R1 do not supply missing budget-versus-PPO results.

Data: [exp6-winrates.json](exp6-winrates.json).

| Panel | Marks | Missing cells | Value range |
|---|---:|---:|---|
| vs s0 | 4 | 2 | 14.75 to 75.5 % |
| vs s1 | 4 | 2 | 10.85 to 69.45 % |
| vs s2 | 4 | 2 | 13.25 to 80.95 % |
| vs random | 4 | 2 | 85.75 to 99.9 % |

Source files (repo-relative):

- `exp6/RESULTS.md`
- `exp6/results/eval_R0_vs_random.json`
- `exp6/results/eval_R0_vs_s0.json`
- `exp6/results/eval_R0_vs_s1.json`
- `exp6/results/eval_R0_vs_s2.json`
- `exp6/results/eval_R1_vs_random.json`
- `exp6/results/eval_R1_vs_s0.json`
- `exp6/results/eval_R1_vs_s1.json`
- `exp6/results/eval_R1_vs_s2.json`
- `exp6/results/eval_b0_vs_random.json`
- `exp6/results/eval_b50_vs_random.json`
- `exp6/results/gate_b0_vs_s0.json`
- `exp6/results/gate_b0_vs_s1.json`
- `exp6/results/gate_b0_vs_s2.json`
- `exp6/results/gate_b50_vs_s0.json`
- `exp6/results/gate_b50_vs_s1.json`
- `exp6/results/gate_b50_vs_s2.json`

### Figure 8: exp6-read-predictions

True quantities beside independent predictions with and without weights.

Only real W_r0..2/noW_r0..2 sessions are used; fake and void_emptyprompt trees are excluded. W X/Y/Z are remapped using each matching key.json; noW already uses b0/b50/ppo labels. Includes all behavioral scalar rates, mean length, tengen probability, the full first-move probability vector (board plus pass), and all pairwise wins. Pair orientation is complemented when required. First-move top-three coordinates are derived rankings of the displayed vector, not an independent numeric quantity. Qualitative strategy, predicted role identity, and self-reported role confidence are not behavioral truths and are not plotted as accuracy measurements.

Data: [exp6-read-predictions.json](exp6-read-predictions.json).

| Panel | Marks | Missing cells | Value range |
|---|---:|---:|---|
| capture_rate | 21 | 0 | 22.1378 to 92 % |
| atari_escape_rate | 21 | 0 | 15.0108 to 83 % |
| self_atari_rate | 21 | 0 | 0.379682 to 8 % |
| eye_fill_rate | 21 | 0 | 0 to 7.06388 % |
| pass_before_60_rate | 21 | 0 | 0 to 10 % |
| mean_game_length | 21 | 0 | 110.693 to 154 moves / 步 |
| winrate_vs_random | 21 | 0 | 78 to 99.9 % |
| winrate_vs_R1 | 21 | 0 | 16 to 95.1 % |
| p_tengen | 21 | 0 | 0.024229 to 99.8856 % |
| b0 first-move distribution / 首步分布 | 574 | 0 | 0.00226045 to 13.5 % |
| b50 first-move distribution / 首步分布 | 574 | 0 | 1.52767e-17 to 16 % |
| ppo first-move distribution / 首步分布 | 574 | 0 | 1.28336e-22 to 99.8856 % |
| Pairwise wins / 两两胜率 | 21 | 0 | 4 to 76 % |

Source files (repo-relative):

- `exp6/readwt/sessions/W_r0/key.json`
- `exp6/readwt/sessions/W_r0/predictions.json`
- `exp6/readwt/sessions/W_r1/key.json`
- `exp6/readwt/sessions/W_r1/predictions.json`
- `exp6/readwt/sessions/W_r2/key.json`
- `exp6/readwt/sessions/W_r2/predictions.json`
- `exp6/readwt/sessions/noW_r0/predictions.json`
- `exp6/readwt/sessions/noW_r1/predictions.json`
- `exp6/readwt/sessions/noW_r2/predictions.json`
- `exp6/readwt/truth/b0.json`
- `exp6/readwt/truth/b50.json`
- `exp6/readwt/truth/pairwise.json`
- `exp6/readwt/truth/ppo.json`

### Figure 9: exp7-distances

Distance ratios in discovery and confirmation, using the same orientation.

Discovery prints D_hon/D_syc; its printed ratios are reciprocated. Confirmation already prints D_syc/D_hon. Discovery is labelled with its registered smaller-distance call rule; confirmation with its separately registered larger-distance rule. Printed discovery ratio precision is preserved rather than claiming raw-checkpoint precision. The reference is equality. These are retrained pairs, not original author checkpoints.

Data: [exp7-distances.json](exp7-distances.json).

| Panel | Marks | Missing cells | Value range |
|---|---:|---:|---|
| Discovery: registered smaller-distance rule / 发现集：预注册较小距离规则 | 6 | 0 | 1.00945 to 1.03 D_syc / D_hon |
| Confirmation: registered larger-distance rule / 确认集：预注册较大距离规则 | 6 | 0 | 0.972561 to 1.04603 D_syc / D_hon |

Source files (repo-relative):

- `exp7/RESULTS.md`
- `exp7/RESULTS_CONFIRM.md`

### Figure 10: exp8-cross-config

Configuration comparisons with separate accuracy and paired-difference intervals.

Important source mismatch: results/exp8_cross_config.md gives confidence intervals for edited-A minus each row, not absolute accuracy. Source accuracy panels therefore have no invented whiskers; separate paired-delta panels retain the original intervals. Additional, explicitly labelled local-regrade panels read summary_local.json from the four artifacts and show absolute accuracy with its own CI. The default is the local-regrade MATH panel. IFEval scoring passes differ slightly, so original-table points are never combined with local-regrade intervals. All panels mark A and the best base configuration within their own data source. The edited-A paired-delta reference is labelled as a reference, not a missing accuracy.

Data: [exp8-cross-config.json](exp8-cross-config.json).

| Panel | Marks | Missing cells | Value range |
|---|---:|---:|---|
| math500 · local regrade + accuracy CI / 本地重评分及准确率区间 | 10 | 0 | 3 to 50.65 % |
| math500 · source accuracy / 原表准确率 | 10 | 0 | 3 to 50.6 % |
| math500 · paired delta / 配对差值 | 9 | 1 | 6 to 47.6 pp |
| gsm8k · source accuracy / 原表准确率 | 10 | 0 | 5.3 to 67.9 % |
| gsm8k · paired delta / 配对差值 | 9 | 1 | 3.1 to 62.6 pp |
| ifeval · source accuracy / 原表准确率 | 10 | 0 | 1.9 to 65.1 % |
| ifeval · paired delta / 配对差值 | 9 | 1 | -2.2 to 60.9 pp |
| gsm8k · local regrade + accuracy CI / 本地重评分及准确率区间 | 10 | 0 | 5.34496 to 67.9303 % |
| ifeval · local regrade + accuracy CI / 本地重评分及准确率区间 | 10 | 0 | 1.94085 to 65.0647 % |

Source files (repo-relative):

- `results/exp8_cross_config.md`
- `results/runs/exp8_config_A/summary_local.json`
- `results/runs/exp8_config_BE/summary_local.json`
- `results/runs/exp8_config_C/summary_local.json`
- `results/runs/exp8_config_D/summary_local.json`

### Figure 11: exp8-steer-sweep

Steering selection sweep and held-out full-benchmark scores.

The source contains steering k=1,2,3,4,6,8,10; k=5,7,9 remain explicitly unmeasured. Practice controls are included alongside the selection sweep. Full MATH uses randcols/randdir/steer from summary.txt and base/Astra from the requested cross-config table; the latter printed values have no absolute CIs in that source, so their bars have no whiskers. The full-benchmark and practice panels remain separate.

Data: [exp8-steer-sweep.json](exp8-steer-sweep.json).

| Panel | Marks | Missing cells | Value range |
|---|---:|---:|---|
| practice_select | 10 | 3 | 10 to 51.4 % |
| MATH-500 | 5 | 0 | 5.1 to 50.6 % |

Source files (repo-relative):

- `results/exp8_cross_config.md`
- `results/runs/exp8_steer/summary.txt`

