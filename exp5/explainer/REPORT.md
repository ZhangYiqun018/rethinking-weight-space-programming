# Report — Breakout Circuit Explainer Figure & Animation (`exp5/explainer/`)

## 1. Files Created & Updated

### Created in `exp5/explainer/` (data & build scripts)
- `exp5/explainer/make_frames.py` — Rolls out `upstream/lm_understand-logs/exp5_write_from_scratch/networks/breakout/hand.npz` in official MinAtar 1.0.15 Breakout (`seed=0`, sticky action probability `0.1`, actions sampled from `softmax(logits)`) by importing `observation()` and `Net` from `exp5/eval_minatar.py` by path. Records `380` consecutive steps (`0..379`) plus ground-truth future landing columns (`landing_col` at row 9 and direction-preserving `catch_col` at row 8) into `blog/figs/breakout_explainer/frames.json`.
- `exp5/explainer/unit_labels.py` — Inspects `hand.npz` (`np.load(..., allow_pickle=False)`), Astra's `answer.md`, and `frames.json` to label every hidden unit with nonzero weights (`h1[0..5]` and `h2[0..7]`) backed by three pieces of evidence: (a) what `answer.md` says, (b) the exact weight pattern in `W1/b1` or `W2/b2/W3`, and (c) empirical correlation / gate-agreement checks across all `380` recorded frames. Writes `blog/figs/breakout_explainer/units.json`.
- `exp5/explainer/check_consistency.py` — Runs the forward-pass reproduction, episode score, and landing-column movement checks on `frames.json` and `units.json`.
- `exp5/explainer/build_explainer_js.py` — Packages the interactive visualization script together with embedded fallback copies of `units.json` and `frames.json` into `blog/figs/breakout_explainer/explainer.js` so the page works both over `http(s)://` (via `fetch`) and directly from `file://` (where `fetch` is blocked by browser CORS policy).
- `exp5/explainer/REPORT.md` — This report.

### Created / Updated in `blog/`
- `blog/figs/breakout_explainer/frames.json` (`288.7 KB`, `< 1 MB`) — `380` recorded steps (including `32` paddle catches, `38` wall bounces, `31` brick hits, clearing all `30` bricks of the first rack at step `358` and refilling rack 2 at step `366`), all floats rounded to 3 decimals.
- `blog/figs/breakout_explainer/units.json` (`53.8 KB`) — Evidence-backed labels, weight patterns, empirical metrics, 10×10 `W1` channel heatmaps for `h1[0..5]`, and the `24` nonzero `W2` and `8` nonzero `W3` edges.
- `blog/figs/breakout_explainer/explainer.js` (`356.0 KB`) — Self-contained vanilla JS (zero external libraries/CDNs, `file://`-compatible, paired `lang="en"` / `lang="zh"` spans) rendering:
  1. **Figure 5 (`#fig-exp5-breakout-circuit`)**: Static inline SVG circuit diagram showing representative 10×10 `W1` channel heatmaps (`h1[0]`, `h1[4]`, `h1[5]`; clicking `h1[0..3]` switches the comparator heatmap), the `6` active Layer-1 units (`+26` zero units), the `8` active Layer-2 AND-gate units (`+24` zero units), the `6` output actions, and only the `24` nonzero `W2` and `8` nonzero `W3` connections colored by sign (CVD-validated blue `#2a78d6` vs red `#d93a4b`, protan $\Delta E = 21.5$) and scaled by magnitude.
  2. **Figure 6 (`#fig-exp5-breakout-anim`)**: Interactive rollout animation driven by `frames.json` with Play/Pause, `◀ Step`, `Step ▶`, timeline scrubber, event quick-jumps, and `0.5× / 1× / 2× / 4×` speed controls; left: 10×10 Breakout board with bricks, ball, trail, paddle, and ground-truth landing column markers (`landing_col` on row 9 and `catch_col` on row 8); middle: all `32 + 32` hidden units lit by $\tanh \in [-1, 1]$ with active units named; right: the `6` action probabilities and chosen action; bottom: live hover evidence inspector + collapsible full evidence table.
- `blog/css/style.css` — Added `.bx-*` layout and component styles matching the Distill/blog aesthetic.
- `blog/index.html` — Added the bilingual subsection **"How the Breakout network works" / "Breakout 网络是怎么工作的"** (`#exp5-breakout-circuit`) in Experiment 5 with 3 paired paragraphs, Figures 5 & 6 with bilingual captions, all data numbers wrapped in `<span class="todo TODO-VERIFY">...</span>`, and a reproducibility entry under `#methods`. `blog/validate.py` passes with 0 errors and 0 EN/ZH number mismatches.

---

## 2. Consistency Check Output

Running `python -I exp5/explainer/check_consistency.py`:

```text
=== CONSISTENCY CHECK ===
1. Forward-pass reproduction over 380 frames: max |h1_calc - h1_rec| = 0.000000, max |h2_calc - h2_rec| = 0.000000, max |logits_calc - logits_rec| (3-dec rounded) = 0.000000 (unrounded vs 3-dec stored: 0.000375)
2. Episode score (seed=0): 457.0 over 6016 steps (recorded 380-step window score: 31, 32 catches, 38 wall bounces, 31 brick hits)
3. Fraction of steps moving paddle toward true landing/catch column:
   - Among descending steps where paddle != catch_col: 36/36 = 1.000 (100.0%)
   - Among all steps where paddle moves (left/right): 166/180 = 0.922 (92.2%) toward direction-preserving catch_col (row 8), 147/180 = 0.817 (81.7%) toward row-9 landing_col
   - Across all 380 steps (moves toward catch_col or holds no-op at valid catch col): 344/380 = 0.905 (90.5%)
```

---

## 3. Unit Labels & Evidence (`units.json`)

Out of `32 + 32 = 64` hidden units (`20,486` total parameters), only **`6` Layer-1 units (`h1[0..5]`)** and **`8` Layer-2 units (`h2[0..7]`)** have nonzero weights (`815` nonzero parameters total: `W1` 769, `b1` 1, `W2` 24, `b2` 8, `W3` 8, `b3` 5). The remaining `26` Layer-1 units (`h1[6..31]`) and `24` Layer-2 units (`h2[8..31]`) have all-zero incoming weights, zero bias, and all-zero outgoing weights ($\text{activation} \equiv 0.000$). No unit required the fallback `"other"` label because every active unit's function is explicitly stated in `answer.md` and confirmed with $r \ge 0.87$ / $100\%$ gate agreement.

### Layer 1 (`h1[0..5]`)

| Unit | Short label (EN / ZH) | (a) `answer.md` | (b) Weight pattern (`W1`, `b1`) | (c) Empirical check (`380` frames) |
|---|---|---|---|---|
| `h1[0]` | `Target error: down-right` / `落点误差：右下落` (`err (↓→)`) | ll. 4–5, 11–69: *"Units 0..3: descending-right, descending-left, ascending-right, ascending-left. Subtract paddle position from each target."* | Ch 0 (paddle, row 9): negative column ramp $-0.02 \times \text{col}$ ($0$ to $-0.18$). Ch 3 (ball, 10×10): $+0.02 \times \text{target\_col}(r,c)$ with right-wall reflection and row 6–8 catch midpoints (`108` nonzeros, $b_1=0$). | $r = 1.000$ with $0.02(\text{target}-\text{paddle})$ across all 380 frames; in descending-right quadrant ($n=96$), $r = 0.934$ with $(\text{catch\_col}-\text{paddle\_col})$, $r = 0.745$ with $(\text{landing\_col}-\text{paddle\_col})$, sign match $1.000$ when $\|\text{catch\_col}-\text{paddle\_col}\|\ge 1$. |
| `h1[1]` | `Target error: down-left` / `落点误差：左下落` (`err (↓←)`) | ll. 4–5, 13–67: descending-left target minus paddle column. | Ch 0 (paddle, row 9): $-0.02 \times \text{col}$. Ch 3 (ball, 10×10): $+0.02 \times \text{target\_col}(r,c)$ with left-wall reflection and row 6–8 catch midpoints (`93` nonzeros, $b_1=0$). | $r = 1.000$ with $0.02(\text{target}-\text{paddle})$ overall; in descending-left quadrant ($n=95$), $r = 0.937$ with $(\text{catch\_col}-\text{paddle\_col})$, $r = 0.741$ with $(\text{landing\_col}-\text{paddle\_col})$, sign match $1.000$ when $\|\text{catch\_col}-\text{paddle\_col}\|\ge 1$. |
| `h1[2]` | `Target error: up-right` / `落点误差：右上升` (`err (↑→)`) | ll. 1–5, 14–68: *"ascending predictions anticipate the earliest possible brick bounce and revise as the ball climbs."* | Ch 0 (paddle, row 9): $-0.02 \times \text{col}$. Ch 3 (ball, 10×10): $+0.02 \times \text{target\_col}(r,c)$ assuming brick/top bounce then descent (`107` nonzeros, $b_1=0$). | $r = 1.000$ with $0.02(\text{target}-\text{paddle})$ overall; in ascending-right quadrant ($n=94$), $r = 0.874$ with $(\text{catch\_col}-\text{paddle\_col})$ and $r = 0.831$ with $(\text{landing\_col}-\text{paddle\_col})$. |
| `h1[3]` | `Target error: up-left` / `落点误差：左上升` (`err (↑←)`) | ll. 1–5, 15–69: ascending-left target minus paddle column. | Ch 0 (paddle, row 9): $-0.02 \times \text{col}$. Ch 3 (ball, 10×10): $+0.02 \times \text{target\_col}(r,c)$ assuming brick/top bounce then descent (`91` nonzeros, $b_1=0$). | $r = 1.000$ with $0.02(\text{target}-\text{paddle})$ overall; in ascending-left quadrant ($n=95$), $r = 0.876$ with $(\text{catch\_col}-\text{paddle\_col})$ and $r = 0.831$ with $(\text{landing\_col}-\text{paddle\_col})$. |
| `h1[4]` | `Horiz. direction (dx: right +1 / left -1)` / `水平方向开关（右 +1／左 -1）` (`dx (→/←)`) | ll. 71–90: *"Unit 4: horizontal direction, including stationary wall-reflection frames."* | Ch 3 (ball): column ramp $[10, 20, 40, \dots, 160, 170]$ on every row. Ch 4 (trail): negative column ramp $[0, -20, -40, \dots, -180]$ on every row (`190` nonzeros, $b_1=0$). At col 0/9 wall bounces where $\text{ball\_col}=\text{trail\_col}$, net sum is $+10 / -10$. | Saturates to $+1.000$ (right) or $-1.000$ (left) with **$100\%$ sign accuracy (`380/380` frames, $r = 1.000$)**, including all `38` stationary wall-reflection frames. |
| `h1[5]` | `Vert. direction (dy: down +1 / up -1)` / `垂直方向开关（下 +1／上 -1）` (`dy (↓/↑)`) | ll. 92–111: *"Unit 5: descending versus ascending; a stationary row-8 frame means paddle bounce."* | Ch 3 (ball): row ramp $[0, 10, 20, 30, 40, 50, 60, 70, 70, 90]$. Ch 4 (trail): negative row ramp $[0, -10, \dots, -80, -90]$, $b_1=+5$ (`180` nonzeros). At row-8 paddle bounce ($\text{ball\_row}=\text{trail\_row}=8$), $70-80+5=-5$. | Saturates to $+1.000$ (descending) or $-1.000$ (ascending) with **$100\%$ sign accuracy (`380/380` frames, $r = 1.000$)**, including all `63` stationary vertical bounce frames (bricks + paddle). |

### Layer 2 (`h2[0..7]`)

Each pair `(h2[2m], h2[2m+1])` is a 3-input `AND` gate with bias $b_2 = -415$, velocity-switch weights $\pm 200$ from `h1[4]` and `h1[5]`, comparator weight $\mp 1000$ from `h1[m]`, and output weight $W_3[u_2, \text{action}] = +30$:

| Unit | Short label (EN / ZH) | `W2` inputs & `b2` | `W3` output | Empirical check (`380` frames) |
|---|---|---|---|---|
| `h2[0]` | `Move Left gate (down-right)` / `左移门控（右下落）` | $-1000\,h_1[0] + 200\,h_1[4] + 200\,h_1[5] - 415$ | $+30 \to \text{left (1)}$ | Fires ($+1.000$) on `2/380` frames (otherwise $-1.000$); **$100\%$ (`380/380`)** agreement with $(dx=+1 \land dy=+1 \land h_1[0] < -0.015)$. |
| `h2[1]` | `Move Right gate (down-right)` / `右移门控（右下落）` | $+1000\,h_1[0] + 200\,h_1[4] + 200\,h_1[5] - 415$ | $+30 \to \text{right (3)}$ | Fires ($+1.000$) on `17/380` frames; **$100\%$ (`380/380`)** gate agreement. |
| `h2[2]` | `Move Left gate (down-left)` / `左移门控（左下落）` | $-1000\,h_1[1] - 200\,h_1[4] + 200\,h_1[5] - 415$ | $+30 \to \text{left (1)}$ | Fires ($+1.000$) on `16/380` frames; **$100\%$ (`380/380`)** gate agreement. |
| `h2[3]` | `Move Right gate (down-left)` / `右移门控（左下落）` | $+1000\,h_1[1] - 200\,h_1[4] + 200\,h_1[5] - 415$ | $+30 \to \text{right (3)}$ | Fires ($+1.000$) on `1/380` frames; **$100\%$ (`380/380`)** gate agreement. |
| `h2[4]` | `Move Left gate (up-right)` / `左移门控（右上升）` | $-1000\,h_1[2] + 200\,h_1[4] - 200\,h_1[5] - 415$ | $+30 \to \text{left (1)}$ | Fires ($+1.000$) on `14/380` frames; **$100\%$ (`380/380`)** gate agreement. |
| `h2[5]` | `Move Right gate (up-right)` / `右移门控（右上升）` | $+1000\,h_1[2] + 200\,h_1[4] - 200\,h_1[5] - 415$ | $+30 \to \text{right (3)}$ | Fires ($+1.000$) on `58/380` frames; **$100\%$ (`380/380`)** gate agreement. |
| `h2[6]` | `Move Left gate (up-left)` / `左移门控（左上升）` | $-1000\,h_1[3] - 200\,h_1[4] - 200\,h_1[5] - 415$ | $+30 \to \text{left (1)}$ | Fires ($+1.000$) on `57/380` frames; **$100\%$ (`380/380`)** gate agreement. |
| `h2[7]` | `Move Right gate (up-left)` / `右移门控（左上升）` | $+1000\,h_1[3] - 200\,h_1[4] - 200\,h_1[5] - 415$ | $+30 \to \text{right (3)}$ | Fires ($+1.000$) on `15/380` frames; **$100\%$ (`380/380`)** gate agreement. |

---

## 4. Notes & Nuances (What Required Care)

1. **Row-9 `landing_col` vs. Row-8 direction-preserving `catch_col` in MinAtar Breakout**:
   In `minatar/environments/breakout.py` (ll. 87–103), when a descending ball steps from row 8 (`ball_y == 8`) to `new_y == 9`, the paddle on row 9 catches the ball if `pos == ball_x` (the ball's column at row 8, which bounces the ball while **preserving** its horizontal direction) *or* if `pos == new_x` (the ball's column at row 9, which **reverses** horizontal direction). Moreover, when the ball is caught, `new_y` is reset to `8` before `state()` is emitted, while `self.ball_x` is updated to `new_x`.
   - Astra's design (`answer.md` l. 1) explicitly targets the row-8 direction-preserving column (`catch_col`) during early descent and sets the row 6–8 weights to the midpoint `(catch_col + landing_col) / 2` so that a paddle parked at *either* valid catch column has $1000\,|h_1[m]| \approx 10 < 15$ and outputs `no-op`.
   - To make this completely transparent in both data and visuals, `frames.json` records both `landing_col` (the exact column `new_x` where `new_y == 9` next occurs) and `catch_col` (the row-8 column on that same arrival), the board animation displays both markers (`landing_col` solid reticle + `catch_col` dashed bracket), and the consistency check reports movement fractions against both (`100.0%` during descent when not yet at `catch_col`; `92.2%` of all moves toward `catch_col` and `81.7%` toward `landing_col`).
2. **`file://` compatibility**:
   Standard `fetch("figs/breakout_explainer/frames.json")` fails under `file://` in modern browsers due to CORS restrictions on local files. `explainer.js` therefore detects `location.protocol === "file:"` (or a failed `fetch`) and falls back seamlessly to the embedded copies of `units.json` and `frames.json`, while using `fetch` when served over `http(s)://`.
