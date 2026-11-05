"""Run on a host that can apply Seatbelt profiles to verify isolation. Exit 2 = unverified."""
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("subject_sandbox", ROOT / "sandbox_runner.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def check(work=None):
    work = Path(work or ROOT / "sandbox_test/work")
    work.mkdir(parents=True, exist_ok=True)
    truth = ROOT / "truth"
    truth.mkdir(exist_ok=True)
    sentinel = truth / "sandbox_sentinel.txt"
    sentinel.write_text("PRIVATE_TRUTH_SENTINEL\n")
    key = work.parent / "key_probe.json"
    key.write_text('{"secret":"PRIVATE_BLINDING_KEY"}\n')
    targets = [ROOT.parent / "RESULTS.md",
               ROOT.parent.parent / "upstream/lm_understand-logs/exp6_go_from_scratch/runs/budget_0/final_answer.md",
               key, work.parent / "key.json"] + sorted(truth.glob("*"))
    targets = [str(p.resolve()) for p in targets if p.is_file()]
    link = work / "forbidden_symlink"
    if not link.exists() and not link.is_symlink():
        link.symlink_to(sentinel)
    targets.append(str(link.absolute()))
    probes = {
        "allowed_runtime": """
import numpy, scipy, pandas, pathlib, importlib.util
assert all(importlib.util.find_spec(name) is None for name in ['jax','pgx','torch','go'])
assert numpy.dot([1,2],[3,4]) == 11
assert pandas.DataFrame({'x':[1,2]}).x.sum() == 3
p=pathlib.Path('allowed_write_probe.txt'); p.write_text('ok'); assert p.read_text()=='ok'; p.unlink()
print('PASS')
""",
        "forbidden_reads": f"""
import pathlib, errno
paths={targets!r}
for name in paths:
    try:
        pathlib.Path(name).read_bytes()
    except OSError as exc:
        assert exc.errno in (errno.EPERM, errno.EACCES), (name, repr(exc))
    else:
        raise AssertionError('READ LEAK: '+name)
print('PASS')
""",
        "network": """
import socket, errno
try:
    s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    s.bind(('127.0.0.1',0))
except OSError as exc:
    assert exc.errno in (errno.EPERM, errno.EACCES), repr(exc)
else:
    s.close()
    raise AssertionError('NETWORK BIND ALLOWED')
try:
    s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    s.connect(('127.0.0.1',8000))
except OSError as exc:
    assert exc.errno in (errno.EPERM, errno.EACCES), repr(exc)
else:
    s.close()
    raise AssertionError('NETWORK CONNECT ALLOWED')
print('PASS')
"""}
    results = {name: S.run(code, work, timeout=30) for name, code in probes.items()}
    blocked = any(r["launch_error"] for r in results.values())
    passed = all(r["returncode"] == 0 and r["output"].strip() == "PASS" for r in results.values())
    # Remove test-only entries before a subject receives its work directory.
    link.unlink(missing_ok=True)
    key.unlink(missing_ok=True)
    return {"status": "PASS" if passed else "UNVERIFIED_LAUNCH_BLOCKED" if blocked else "FAIL",
            "targets": targets, "probes": results, "profile": str(ROOT / "sandbox.sb")}


if __name__ == "__main__":
    result = check()
    (ROOT / "sandbox_test.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "PASS" else 2)
