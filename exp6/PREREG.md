# Preregistration

Registered 2026-11-04T07:47:58Z (file creation time), after the network-only validation gate and
before implementing or running either rule player against any opponent.

Both players see only the current mover-relative board and legal-action mask.
They use no network, fitting, search over move sequences, or rollouts.
All sampling is categorical at temperature one over the specified logits.

## Fixed players

An own single-point eye is an empty point with all on-board orthogonal
neighbors occupied by own stones, and at most one opposing diagonal stone
in the interior, zero on an edge or corner. Empty diagonals do not count as
opposing stones. Both players exclude these eyes and all illegal placements.
They exclude pass whenever any remaining placement exists; otherwise pass
has logit zero and all placements have negative-infinite logits.

R0 random-noeye assigns zero logits to all remaining placements.

R1 greedy-tactics computes connected groups and their distinct liberties.
For each candidate placement, remove adjacent opposing groups whose sole
liberty is the candidate, and form the new own group from the candidate and
all touching own groups. Compute its distinct liberties after captures.
Also compute liberties of surviving opposing groups after this placement.
Its fixed score is:

- +100 per captured stone.
- +10 if at least one adjacent own group had exactly one liberty before
  the move and the new own group has at least two liberties afterward.
  This is a Boolean bonus, not a sum over rescued groups.
- +1 if at least one surviving opposing group adjacent to the candidate
  changes from at least two liberties to exactly one. Boolean bonus.
- -20 if the new own group has exactly one liberty and no stone is captured.
- +0.03 when the candidate's distance from the nearest edge is 2 or 3
  (third or fourth line).
- +0.02 if adjacent to an own stone and fewer than 40 stones total are
  currently on the board. Occupancy defines "early" because the player
  interface contains no turn history.

Eligible placement logits are exactly ten times this score. The capture
bonus dominates all other terms; rescue dominates threat and positional
bonuses; the self-atari penalty prevents those lesser bonuses from favoring
noncapturing self-atari. No coefficients will be tuned.

## Matches and seeds

Each match has 2,000 games, exactly 1,000 with the named player Black and
1,000 White. Batches have 250 games, split equally by color. Batch keys are
JAX fold_in(PRNGKey(seed), batch_index); each move splits the current key
and samples the whole batch with categorical sampling.

Opponent order: uniform random, PPO s0, PPO s1, PPO s2, Astra b0, Astra b10,
Astra b25, Astra b50 (all released networks).

- R0: seeds 61000 through 61007, respectively.
- R1: seeds 61100 through 61107, respectively.
- Reference Astra b0 vs uniform random: seed 61200.
- Reference Astra b50 vs uniform random: seed 61201.
- Network validation: b0 and b50 vs each PPO seed, 2,000 games each,
  seed 31337. These completed results will also be used in the main table.

Report wins/games and two-sided 95% Wilson intervals. Save per-game outcome,
color, length, and termination flags with each aggregate JSON.

## Decision rule

The claim is weakened if the best rule-based player's win rate vs each
same-size PPO seed is >= Astra budget 0's head-to-head on the same seed
(42.6 / 26.3 / 36.3 %); it is strongly weakened if it is >= Astra budget
50's (74.2 / 70.0 / 78.5 %).

Operationally, at least one single fixed player must meet all three
thresholds; we will not splice the best per-opponent result across players.
These are descriptive point-estimate decisions, not significance tests.
Wilson intervals describe binomial uncertainty within each match and are
not multiplicity-adjusted. No early stopping or additional seeds based on
outcomes. Any post-registration bug fix will be logged in CHANGES.md.
