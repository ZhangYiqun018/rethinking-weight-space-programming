# Exp6: fixed-rule audit of the released 9x9 Go opponents

Completed 2026-11-04, after all evaluations and checks. This report is the
last deliverable written.

## Finding

**The claim is weakened, but not strongly weakened, under the preregistered
decision rule.** A fixed tactical script with no network, training, or
search over move sequences beats all three released same-size PPO policies
and exceeds Astra budget 0's published results against each of them:

| Opponent | R1 wins / games | R1 win rate (95% Wilson) | Published Astra b0 | Published Astra b50 |
|---|---:|---:|---:|---:|
| PPO s0 | 1,482 / 2,000 | 74.10% [72.14%, 75.97%] | 42.6% | 74.2% |
| PPO s1 | 1,222 / 2,000 | 61.10% [58.94%, 63.21%] | 26.3% | 70.0% |
| PPO s2 | 1,385 / 2,000 | 69.25% [67.19%, 71.23%] | 36.3% | 78.5% |

R1 exceeds the b0 thresholds by 31.50, 34.80, and 32.95 percentage points.
It falls below all three b50 thresholds. Its 0.10-point shortfall against
the s0 b50 threshold is much smaller than sampling uncertainty; the
preregistered decision nevertheless uses the stated point estimates.
The s1 and s2 shortfalls are 8.90 and 9.25 points.

This is evidence that **beating these PPO opponents is achievable by a short,
fixed tactical policy**. It is not evidence that the script matches Astra
in general playing strength. R1 wins only 9.85% directly against Astra b0
and 4.90% against Astra b50. Results against the common PPO opponents are
nontransitive: they do not predict the direct Astra-versus-script result.

R0 random-noeye is insufficient: it scores 14.75%, 10.85%, and 13.25%
against the PPO seeds, despite beating uniform random 85.75%.
R1 beats uniform random 99.70%. The complete requested table, additional
b10/b25 results, and intervals are in [RESULTS.md](RESULTS.md).

## What was done

1. Read the released README first, followed by the exact budget-0 prompt,
   all three released evaluation scripts, network architecture metadata,
   and published result JSON. The evaluation scripts call an unreleased
   shared helper; their text establishes seed 31337 and color balancing.
   No upstream Python was executed, imported, or passed to `exec`.
2. Created `.venv/` and pinned every installed package in
   [requirements.txt](requirements.txt). The run used Python 3.12.13,
   pgx 2.6.0, JAX/jaxlib 0.12.0.1, and NumPy 2.5.4, on JAX CPU.
3. Implemented [match.py](match.py) independently. It uses the installed
   pgx Go game implementation with `vmap`/`jit`, komi 7.5, two passes,
   repetition loss, and a 162-move cap. Inputs are interleaved
   own/opponent/White-to-move planes; network layers use `x @ W + b`.
   Illegal actions are masked and actions are sampled at temperature one.
   Each 250-game batch assigns A Black in 125 games and White in 125.
   Terminal states are frozen. Rewards are indexed by color.
4. Passed all six required 2,000-game validation matches before rule-player
   evaluation. Results are in [VALIDATION.md](VALIDATION.md):
   b0 scored 42.10 / 27.60 / 34.40%; b50 scored
   75.50 / 69.45 / 80.95%. The maximum absolute deviation from the
   prescribed published values was 2.45 points, below the 3.5-point gate.
5. Wrote [PREREG.md](PREREG.md) before implementing and running the rule
   players, fixing coefficients, tie sampling, matchups, seeds, game
   counts, and the decision rule. Implemented [rules.py](rules.py).
   R1 computes groups and exact one-move liberties, with fixed bonuses
   for captures, atari rescue, threats, and positional preferences, plus
   a self-atari penalty. Both rules avoid own eyes and pass only when
   no eligible placement remains. No policy tuning or rule bug fixes
   occurred. [CHANGES.md](CHANGES.md) records only correction of the
   registration timestamp to the actual file creation time.
6. Ran [eval.py](eval.py): 16 rule/opponent matches, two Astra/random
   references, and the six validation matches, all with 2,000 games.
   This totals **24 matches and 48,000 evaluated games**, with 1,000 games
   per color in every match. The reference Astra b0/b50 scores against
   random were 99.65% / 99.90%; the released results were 99.55% / 100%.
7. Saved aggregate statistics, seeds, batch sizes, and each game's color,
   outcome, length, and termination flags in 24 JSON files under
   [results/](results/). Summed match runtimes were 415.56 seconds,
   excluding setup, independent checks, and reporting. No GPU, cluster job,
   or LLM API was used.

## What was verified

- [check_rules.py](check_rules.py) compared 3,190 legal candidate scores
  against a separate plain-Python flood-fill implementation. Capture,
  rescue, threat, and self-atari cases all occurred and agreed. Eye
  exclusions, illegal-move masks, pass fallback, and empty-board komi
  scoring also passed. Evidence: [checks.log](checks.log).
- [check_engine.py](check_engine.py) checked the direct game adapter against
  pgx's public `go_9x9` environment over 812 positions. Boards,
  mover-relative observations, legal masks, player/color mapping,
  termination, and rewards agreed. The fixtures exercised two-pass and
  move-limit endings. Evidence: [engine_checks.log](engine_checks.log).
- [audit_results.py](audit_results.py) verified all 24 expected matchups,
  preregistered seeds, 2,000 records per match, balanced colors, win totals,
  color-specific scores, mean lengths, Wilson intervals, and terminal
  flags. There were 46,858 two-pass endings, 476 repetition endings, and
  666 remaining move-limit endings. Evidence: [audit.log](audit.log).
- [manifest.json](manifest.json) records package/platform provenance and
  SHA-256 hashes of the Python sources, preregistration, requirements,
  raw results, and all 98 loaded network tensors. Every tensor was loaded
  with `np.load(..., allow_pickle=False)`.

## Reproduction

Run from this `exp6/` directory. All local Python execution uses isolated
mode; only this directory's own modules are explicitly loaded by path.

```sh
UV_CACHE_DIR="$PWD/.uv-cache" uv venv .venv
UV_CACHE_DIR="$PWD/.uv-cache" uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -I eval.py gate
.venv/bin/python -I check_rules.py
.venv/bin/python -I check_engine.py
.venv/bin/python -I eval.py evaluate
.venv/bin/python -I audit_results.py
```

The evaluation commands reuse existing result JSON files. For a fresh
reproduction, use a copy of the implementation without the existing
`results/` directory, with the same relative upstream network paths.
`evaluate` requires a passing `VALIDATION.md` and the preregistration.
The saved stdout logs are [gate.log](gate.log) and
[evaluation.log](evaluation.log). The saved report is the original
run's interpretation; evaluation regenerates RESULTS.md, not REPORT.md.

## What remains open

The hidden small PPO opponents and practice weak/strong policies were
not released. Their reported 54.1% / 54.9% budget-0 comparison cannot be
audited directly here. The conclusion applies to the three released,
tuned, same-size PPO networks.

The unreleased match helper and absent original environment lockfile
prevent bitwise reproduction of the original random streams. The six
required tolerance checks passed, and public-pgx adapter parity was
verified; this does not establish identity with every detail of the
original runtime.

Only these two preregistered scripts were tested. They neither exhaust
fixed-rule Go policies nor establish a general strength ranking. R1 has
access to exact one-ply liberty computation, whereas Astra's policy must
fit its network architecture. Per-match Wilson intervals are not
multiplicity-adjusted, and the published comparison rates also have
sampling uncertainty. The decisive direct wins by Astra over R1 remain
important evidence alongside the weak PPO comparison.

The optional conversion of R1 into network weights was not performed.
