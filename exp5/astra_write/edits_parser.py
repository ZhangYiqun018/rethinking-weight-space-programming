"""Parse the LAST ```edits block of a model reply and apply it to zero-initialised 600-32-32-6 weights.

Pure Python + numpy; reply text is untrusted, so nothing is eval()'d. Unreadable statements are skipped and counted.

    python -I edits_parser.py reply.md out.npz
"""
import re
import sys
from dataclasses import dataclass

import numpy as np

SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}
KEYS = {"W1": "actor/dense1/kernel", "b1": "actor/dense1/bias", "W2": "actor/dense2/kernel",
        "b2": "actor/dense2/bias", "W3": "actor/logits/kernel", "b3": "actor/logits/bias"}
MAX_ABS = 1e6
OPS = ("+=", "-=", "*=", "=")
NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\Z")
START = re.compile(r"\s*(W[123]|b[123])\s*\[")
FENCE_OPEN = re.compile(r"\s*```\s*edits\s*\Z")


@dataclass(frozen=True)
class EditResult:
    tensors: dict
    block_found: bool
    statements: int
    applied: int
    skipped: int
    errors: tuple


def last_edits_block(text):
    """Lines of the last ```edits block (an unterminated final block runs to the end of the text)."""
    lines, last, cur = text.splitlines(), None, None
    for line in lines:
        if cur is None:
            if FENCE_OPEN.match(line):
                cur = []
        elif line.strip().startswith("```"):
            last, cur = cur, None
        else:
            cur.append(line)
    return cur if cur is not None else last


def statements(block_lines):
    """Join continuation lines: a statement ends when its brackets balance and it does not end in an operator."""
    out, cur, depth = [], "", 0
    for raw in block_lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if cur and depth <= 0 and not cur.endswith(OPS) and START.match(line):
            out.append(cur)
            cur, depth = "", 0
        cur = f"{cur} {line}" if cur else line
        depth = cur.count("[") - cur.count("]")
    if cur:
        out.append(cur)
    return out


def parse_index(tok, size):
    tok = tok.strip()
    if ":" in tok:
        parts = tok.split(":")
        if len(parts) > 3:
            raise ValueError("bad slice")
        vals = []
        for p in parts:
            p = p.strip()
            if p and not re.fullmatch(r"[+-]?\d+", p):
                raise ValueError("bad slice")
            vals.append(int(p) if p else None)
        if len(vals) == 3 and vals[2] == 0:
            raise ValueError("zero step")
        return slice(*vals)
    if not re.fullmatch(r"[+-]?\d+", tok):
        raise ValueError("bad index")
    i = int(tok)
    if not -size <= i < size:
        raise ValueError("index out of range")
    return i


def parse_value(s):
    """Number or nested list of numbers -> float / list. Rejects anything else."""
    s = s.strip()
    if s.startswith("["):
        if not s.endswith("]"):
            raise ValueError("bad list")
        inner = s[1:-1].strip()
        if not inner:
            raise ValueError("empty list")
        items, depth, start = [], 0, 0
        for i, c in enumerate(inner):
            depth += (c == "[") - (c == "]")
            if depth < 0:
                raise ValueError("bad list")
            if c == "," and depth == 0:
                items.append(inner[start:i])
                start = i + 1
        items.append(inner[start:])
        if depth != 0:
            raise ValueError("bad list")
        if items[-1].strip() == "" and len(items) > 1:  # trailing comma
            items.pop()
        return [parse_value(it) for it in items]
    if not NUMBER.match(s):
        raise ValueError(f"not a number: {s[:20]!r}")
    v = float(s)
    if not abs(v) <= MAX_ABS:
        raise ValueError("value too large")
    return v


def apply_statement(tensors, stmt):
    m = START.match(stmt)
    if not m:
        raise ValueError("no tensor")
    name = m.group(1)
    close = stmt.find("]", m.end())
    if close < 0:
        raise ValueError("no ]")
    idx_toks = stmt[m.end():close].split(",")
    rest = stmt[close + 1:].strip()
    op = next((o for o in OPS if rest.startswith(o)), None)
    if op is None:
        raise ValueError("no operator")
    arr = tensors[name]
    if len(idx_toks) > arr.ndim:
        raise ValueError("too many indices")
    idx = tuple(parse_index(t, arr.shape[d]) for d, t in enumerate(idx_toks))
    value = parse_value(rest[len(op):])
    region = arr[idx]
    if isinstance(value, float):
        v = np.float64(value)
    else:
        v = np.array(value, dtype=np.float64)  # ragged lists raise ValueError
        if v.shape != np.shape(region):
            if v.size == np.size(region) and v.ndim == 1:
                v = v.reshape(np.shape(region))
            else:
                raise ValueError("shape mismatch")
    if op == "=":
        arr[idx] = v
    elif op == "+=":
        arr[idx] = region + v
    elif op == "-=":
        arr[idx] = region - v
    else:
        arr[idx] = region * v
    if not np.all(np.isfinite(arr[idx])):
        raise ValueError("non-finite")


def apply_reply(text):
    block = last_edits_block(text)
    tensors = {k: np.zeros(s, dtype=np.float64) for k, s in SHAPES.items()}
    if block is None:
        return EditResult(tensors, False, 0, 0, 0, ())
    stmts = statements(block)
    errors, applied = [], 0
    for st in stmts:
        backup = {k: v.copy() for k, v in tensors.items()}
        try:
            apply_statement(tensors, st)
            applied += 1
        except (ValueError, IndexError) as e:
            tensors = backup
            errors.append((st[:60], str(e)))
    return EditResult(tensors, True, len(stmts), applied, len(stmts) - applied, tuple(errors))


def to_npz_arrays(result):
    return {KEYS[k]: v.astype(np.float32) for k, v in result.tensors.items()}


def write_npz(result, path):
    np.savez(path, **to_npz_arrays(result))


if __name__ == "__main__":
    res = apply_reply(open(sys.argv[1], encoding="utf-8", errors="replace").read())
    write_npz(res, sys.argv[2])
    print(f"block={res.block_found} statements={res.statements} applied={res.applied} skipped={res.skipped}")
