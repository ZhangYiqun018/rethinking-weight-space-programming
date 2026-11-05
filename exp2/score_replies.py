"""Score Exp2 replies (any model): shape named in the `Images:` line, and whether the `Example:` 12x12 drawing is
filled or an outline (an enclosed region of '.' that a flood fill from the border cannot reach = a hole = outline).

    python -I exp2/score_replies.py [raw_dir]
"""
import collections
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH = {"246a5b58": "triangle", "a6a9dfb7": "square", "d88aba69": "circle"}  # from the release's answer_key.json
# whole words only: "round" must not match "background"
SHAPES = {"circle": r"\b(?:circles?|circular|discs?|disks?|round|rings?|ellipses?|elliptical|ovals?|annulus)\b",
          "square": r"\b(?:squares?|rectangles?|rectangular|box(?:es)?|quadrilaterals?)\b",
          "triangle": r"\b(?:triangles?|triangular|wedges?|pyramids?)\b"}


def field(text, name):
    m = re.search(rf"\**{name}\**\s*:\s*(.*?)(?:\n\s*\**(?:Example|Confidence|Reasoning|Observations|Images)\**\s*:|\Z)",
                  text, re.S | re.I)
    return m.group(1).strip() if m else ""


def shape_of(images):
    hits = [s for s, p in SHAPES.items() if re.search(p, images, re.I)]
    return hits[0] if len(hits) == 1 else ("ambiguous:" + "+".join(hits) if hits else "unmatched")


def grid_of(text):
    rows = [ln.strip().strip("`") for ln in text.splitlines()]
    rows = [r for r in rows if re.fullmatch(r"[#.]{12}", r.replace(" ", ""))]
    return [r.replace(" ", "") for r in rows[:12]] if len(rows) >= 12 else None


def fill_type(g):
    if not g:
        return "no-grid"
    n = 12
    seen, stack = set(), [(r, c) for r in range(n) for c in range(n) if (r in (0, n - 1) or c in (0, n - 1)) and g[r][c] == "."]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen or not (0 <= r < n and 0 <= c < n) or g[r][c] != ".":
            continue
        seen.add((r, c))
        stack += [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    holes = sum(1 for r in range(n) for c in range(n) if g[r][c] == "." and (r, c) not in seen)
    return "outline" if holes else "filled"


def main(raw):
    agg = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(raw, "*.json"))):
        rec = json.load(open(p)); text = rec.get("output_text", "")
        net = os.path.basename(p).split("__")[0]; model = os.path.basename(p).split("__", 2)[2][:-5]
        shp = shape_of(field(text, "Images")); ft = fill_type(grid_of(field(text, "Example") or text))
        conf = re.search(r"Confidence\**\s*:\s*\**\s*([0-9]+)", text)
        ok = shp == TRUTH[net]
        agg[model].append((ok, ft))
        print(f"{model:28s} {net} truth={TRUTH[net]:8s} said={shp:22s} ok={int(ok)} drawing={ft:8s} conf={conf.group(1) if conf else '-'}")
    for m, v in agg.items():
        print(f"SUMMARY {m}: correct {sum(o for o, _ in v)}/{len(v)}; drawings {dict(collections.Counter(f for _, f in v))}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "raw"))
