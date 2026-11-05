"""Score Exp1 replies: parse `Game:` / `Confidence:`, map the game text to a label with explicit keyword rules.

    python score.py                       # all of results/raw/*.json -> results/scores.csv + table
    python score.py --authors             # the authors' four answer.md files (condition "authors")
    python score.py --raw DIR --out CSV

Rows whose game text matches no rule or rules of two labels get label "unmatched"/"ambiguous", needs_review=1 and
an empty `correct`; they are excluded from accuracy (counted separately in the table) until a human labels them.
"""
import argparse
import collections
import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "..", "upstream", "lm_understand-logs", "exp1_which_game")

# label -> regexes (case-insensitive, searched in the game text only). All keyword rules live here.
LABEL_RULES = {
    "breakout": [r"break-?out", r"brick", r"arkanoid"],
    "pong": [r"\bpong\b", r"ping-?pong"],
    "pacman": [r"pac-?\s?man", r"ms\.? pac", r"maze with (?:pellets|dots)", r"pellet"],
    "freeway": [r"freeway", r"frogger", r"road[- ]cross", r"cross(?:ing)? (?:the )?(?:road|highway|traffic)",
                r"traffic[- ]cross", r"traffic"],
    "other": [r"\bsnake\b", r"space invaders?", r"asterix", r"seaquest", r"\bcatch", r"dodg", r"coin collect",
              r"food collect", r"gridworld", r"\btetris\b", r"flappy", r"\bmaze\b", r"\bsokoban\b"],
    "unknown": [r"cannot (?:be )?determin", r"can'?t (?:be )?determin", r"unable to (?:determine|identify)",
                r"unclear", r"\brandom", r"\buntrained\b", r"no (?:clear|discernible|identifiable)", r"\bunknown\b",
                r"indeterminate", r"not possible to (?:tell|determine)"],
}
# "other"/"maze" alone must not beat a specific label, so specific labels are decided first; see classify().
SPECIFIC = ["breakout", "pong", "pacman", "freeway"]


def strip_md(s):
    return re.sub(r"[*_`#]+", "", s).strip(" \t:-–—")


def find_field(text, name):
    """Value of the last `Name:` line or `## Name` heading (value on the next non-empty line)."""
    lines = text.splitlines()
    val = None
    for i, line in enumerate(lines):
        m = re.match(rf"^\s*(?:#+\s*)?[*_`]*{name}[*_`]*\s*:?[*_`]*\s*:?\s*(.*)$", line, re.I)
        if not m:
            continue
        rest = strip_md(m.group(1))
        if not rest:
            nxt = next((strip_md(l) for l in lines[i + 1:] if l.strip()), "")
            rest = "" if nxt.startswith(("Description", "Confidence", "Game")) else nxt
        if rest:
            val = rest
    return val


def parse_confidence(s):
    if not s:
        return None
    m = re.search(r"\d+(?:\.\d+)?", s)
    if not m:
        return None
    v = float(m.group())
    if v <= 1 and "." in m.group() and "%" not in s:
        v *= 100
    return v if v <= 100 else None


# An answer that opens by declining to name a game is "unknown" even if it then lists games it cannot rule out
# ("Undetermined ... I cannot distinguish Sokoban, Snake, Pac-Man, ...").
ABSTAIN_LEAD = re.compile(r"^\W*(?:undetermined|not (?:reliably )?identifiable|unidentifiable|cannot be identified|"
                          r"no (?:specific )?(?:well-known )?game (?:is|can be) (?:reliably )?identif)", re.I)


def classify(game_text):
    """-> (label, needs_review)"""
    if not game_text:
        return "unmatched", True
    if ABSTAIN_LEAD.search(game_text):
        return "unknown", False
    hits = {lab for lab, pats in LABEL_RULES.items() if any(re.search(p, game_text, re.I) for p in pats)}
    if len(hits & set(SPECIFIC)) == 1 and not hits - set(SPECIFIC) - {"other"}:
        hits -= {"other"}  # e.g. "Pac-Man (maze)": the generic 'maze' keyword is subsumed by the specific game
    if len(hits) == 1:
        return hits.pop(), False
    return ("unmatched" if not hits else "ambiguous"), True


def score_text(text):
    game = find_field(text, "Game")
    label, review = classify(game)
    return game, label, review, parse_confidence(find_field(text, "Confidence"))


def rows_from_raw(raw_dir):
    for p in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        net, cond, rep, model = os.path.basename(p)[:-5].split("__", 3)
        yield net, cond, int(rep[1:]), model, json.load(open(p))["output_text"]


def rows_from_authors():
    key = json.load(open(os.path.join(HERE, "networks", "answer_key.json")))
    for net, info in key.items():
        p = os.path.join(UPSTREAM, "runs", info["game"], "answer.md")
        yield net, "authors", 0, "authors-astra", open(p).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "results", "raw"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "scores.csv"))
    ap.add_argument("--authors", action="store_true", help="score the authors' four answer.md files")
    args = ap.parse_args()
    key = json.load(open(os.path.join(HERE, "networks", "answer_key.json")))
    src = rows_from_authors() if args.authors else rows_from_raw(args.raw)
    rows = []
    for net, cond, rep, model, text in src:
        game, label, review, conf = score_text(text)
        truth = key[net]["game"]
        rows.append({"net": net, "game_true": truth, "condition": cond, "rep": rep, "model": model,
                     "game_text": game or "", "label": label, "correct": "" if review else int(label == truth),
                     "confidence": "" if conf is None else conf, "needs_review": int(review)})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["net"])
        w.writeheader()
        w.writerows(rows)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["model"], r["condition"])].append(r)
    print(f"{'model':36} {'condition':9} {'n':>3} {'scored':>6} {'acc':>6} {'mean_conf':>9} {'review':>6}")
    for (model, cond), g in sorted(groups.items()):
        sc = [r for r in g if r["correct"] != ""]
        cf = [r["confidence"] for r in g if r["confidence"] != ""]
        acc = f"{sum(r['correct'] for r in sc) / len(sc):.2f}" if sc else "-"
        print(f"{model:36} {cond:9} {len(g):>3} {len(sc):>6} {acc:>6} "
              f"{(sum(cf) / len(cf) if cf else float('nan')):>9.1f} {len(g) - len(sc):>6}")
    for r in rows:
        print(f"  {r['net']} {r['game_true']:8} {r['condition']:7} -> {r['label']:9} correct={r['correct']!s:2} "
              f"conf={r['confidence']!s:5} review={r['needs_review']} | {r['game_text'][:70]}")


if __name__ == "__main__":
    main()
