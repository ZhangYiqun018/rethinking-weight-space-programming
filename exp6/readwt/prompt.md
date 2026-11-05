Predict how three 1.5-million-weight Go policies behave. Make quantitative,
scorable predictions and a short strategy description for each. There is one
policy written by an LLM with zero forward passes and no training (role b0),
one written by the same kind of LLM after 50 rounds of games against practice
opponents and editing, without training (role b50), and one trained with PPO
self-play for 20.5 million games (role ppo). Do not assume an ordering of their
playing strengths.

The following game and network sections are quoted exactly from the task:

THE GAME: Go on a 9 x 9 board (the pgx implementation, "go_9x9").
Rows are numbered 0 (top) to 8 (bottom), columns 0 (left) to 8 (right). Black moves first; players alternate. A move either places a stone of the mover's colour on an empty point or passes. A group of stones with no adjacent empty point is captured and removed. Suicide is illegal, and so is retaking a single stone immediately (simple ko). A move that repeats an earlier whole-board position ends the game at once, and the player who made it loses. The game ends after two consecutive passes, or after 162 moves in total. Then it is scored by area: each player's stones on the board plus the empty points surrounded only by that player's stones; White receives a komi of 7.5 points. The higher score wins (no draws).

THE NETWORK. Input: 243 numbers, each 0 or 1: three planes over the 81 points, from the point of view of the player to move; input number (row * 9 + column) * 3 + plane, where plane 0 is 1 where the player to move has a stone, plane 1 is 1 where the opponent has a stone, and plane 2 is 1 everywhere when White is to move (0 when Black is). Output: 82 logits: output row * 9 + column = play at that point, output 81 = pass.
  h_1 = relu(x @ L1.W + L1.b)                          (512 units)
  h_k = relu(h_(k-1) @ Lk.W + Lk.b) for k = 2..6      (512 units each)
  logits = h_6 @ out.W + out.b                        (82)
Tensors: L1.W (243 x 512), L1.b (512), L2.W .. L6.W (512 x 512), L2.b .. L6.b (512), out.W (512 x 82), out.b (82); 1,480,274 numbers. Row i of a weight matrix belongs to input (or unit) i, column j to unit (or output) j. Only the values can be set; the architecture is fixed. Illegal moves (occupied points, suicide, simple ko) are removed before every move, and the move is sampled from the softmax of the remaining logits (temperature 1). There is no search: the network's output is the whole policy.


Measurement protocol (quoted verbatim; these measurements are made by the
coordinator, not by you):

For each of the three networks, measured with your engine, temperature 1, illegal moves masked:
Use 1,000 games against the uniform random player, the network Black in half, and count only the network's own
moves.
- `capture_rate`: among its moves where at least one legal capturing move exists, the fraction that capture.
- `atari_escape_rate`: among its moves where it has a group in atari that some legal move can bring to >= 2
  liberties, the fraction of moves that do so.
- `self_atari_rate`: fraction of its moves that leave the played stone's group with exactly 1 liberty without
  capturing.
- `eye_fill_rate`: fraction of its moves that fill its own single-point eye (R0's definition) while another legal
  non-pass move exists.
- `pass_before_60_rate`: fraction of games in which it passes before its 30th own move.
- `mean_game_length`: mean number of moves (both players) per game.
- `first_move`: from the empty board as Black, the probability of each point (from the softmax over legal moves);
  report the top 3 points and P(tengen = row 4, col 4).
- `winrate_vs_random`: 2,000 games.
- `winrate_vs_R1`: 2,000 games (R1 = your preregistered greedy-tactics player).
- Pairwise win rates: X-Y, X-Z, Y-Z (2,000 games each; reuse existing results where the same pair exists).

Operational details fixed before subject sessions:
- All rates and probabilities are numbers in [0,1], not percentages.
- Capture/escape rates pool network-own decisions over the 1,000 games.
  A pass counts as an own decision and as a miss when an opportunity exists.
  A capturing move removes at least one opponent stone.
- An escape saves at least one previously-atari own group to >=2 distinct
  liberties, including captures that free liberties without connecting the
  played stone to the saved group. It need not save every atari group.
- Self-atari and eye-fill rates divide by all own decisions, including passes.
  Passes contribute zero to their numerators.
- "Another legal non-pass move" means any other legal placement, even an eye.
  Own-eye definition: empty, all on-board orthogonal neighbors own stones,
  at most one opposing diagonal in the interior, zero on edges/corners.
- "Before its 30th own move" means own moves numbered 1 through 29 inclusive,
  regardless of color. pass_before_60_rate is the fraction of games with at
  least one such pass. This precise own-move definition controls the field
  name; it is not a test of a global move counter.
- mean_game_length is measured on the same 1,000 statistical games.
- first_move includes 82 probabilities: row-major 81 placements, then pass.
  top3 contains the three highest-probability board points (exclude pass),
  with their probabilities. Break equal-probability ties however you wish.
  p_tengen must equal probabilities[40].
- Rate and pairwise measurements use independent finite samples where new
  matches are required; existing 2,000-game results may be reused/inverted.
  Predict underlying expected values, not any particular random seed.

R1 is a fixed one-ply rule policy. It excludes illegal moves and own eyes
(defined above), and excludes pass whenever any other eligible placement
exists. Otherwise it passes. For each remaining placement its score is:
100 per captured opponent stone; +10 if at least one touching own group was
in atari and the newly formed own group has >=2 liberties; +1 if a touching
surviving opponent group changes from >=2 liberties to exactly one; -20 for
a new own group with exactly one liberty unless the move captures; +0.03
for distance 2 or 3 from the nearest edge; +0.02 for adjacency to an own
stone when total board occupancy is <40. Rescue and threat bonuses are
Boolean, not per-group sums. It samples softmax(10 * score), with no
lookahead, rollouts, training, or fitted coefficients.

<!-- W -->
CONDITION W: The three networks are blinded as X, Y, Z. Their weights are
in nets/X/, nets/Y/, nets/Z/ in your current directory, 14 .npy tensors per
network, named L1.W.npy, L1.b.npy, ..., L6.W.npy, L6.b.npy, out.W.npy,
out.b.npy. Load with numpy.load(path, allow_pickle=False).

You may analyze weights in any way EXCEPT computing a network's output for
any input. Do not propagate an input through all layers to logits, directly
or through split runs, and do not emulate play. Single-layer analysis,
products of weight matrices, and responses of a single layer to chosen
inputs are allowed. The coordinator will manually audit your code.
Predict per X/Y/Z and infer each role, with confidence in [0,1].
Pairwise keys: X_vs_Y, X_vs_Z, Y_vs_Z.
<!-- /W -->
<!-- noW -->
CONDITION noW: No network files are provided. Answer from what you know
about these kinds of networks. You may still use Python, for example to
analyze Go rules. Since X/Y/Z cannot be distinguished without information,
make the same predictions per role b0, b50, ppo instead. Set each role
field equal to its object key, and role_confidence to null; this condition
does not guess blinded identities.
Pairwise keys: b0_vs_b50, b0_vs_ppo, b50_vs_ppo.
<!-- /noW -->

To use Python, reply with exactly one fenced ```python block and nothing
else. Each block runs in a fresh isolated Python subprocess, with a
persistent current work directory. numpy, scipy and pandas are available;
no pgx, JAX, other Go library, network, or outside experiment files are
available. At most 40 code runs, 600 seconds each, combined stdout/stderr
truncated to 20,000 characters. Persist your own intermediate files if
needed. Python state does not persist. No other tools are available.

Finish with exactly one fenced ```json block and nothing else. The top-level
object has exactly two keys: "networks" and "pairwise".
"networks" maps the three condition-specific names to objects containing:
capture_rate, atari_escape_rate, self_atari_rate, eye_fill_rate,
pass_before_60_rate, mean_game_length, first_move, winrate_vs_random,
winrate_vs_R1, role, role_confidence, strategy.

Each first_move object has exactly:
  "probabilities": a list of 82 numbers summing to 1,
  "top3": a list of three {"row": integer 0..8, "col": integer 0..8,
                          "probability": number} objects, in decreasing order,
  "p_tengen": a number equal to probabilities[40].
"role" is one of "b0", "b50", "ppo". "strategy" is at most 200 words.
mean_game_length is a number in [2,162].
"pairwise" maps the three keys listed for your condition to probabilities
that the first named policy beats the second, with balanced colors.
Do not include additional JSON fields. Give a best estimate for every field.
