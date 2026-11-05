"""Fail-closed subprocess runner. No unsandboxed execution fallback exists."""
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "subject_venv/bin/python"
LIMIT = 20000


def parameters(work):
    # This trusted probe runs no subject code and only discovers runtime paths.
    query = "import sysconfig,json; print(json.dumps({'stdlib':sysconfig.get_path('stdlib')}))"
    info = json.loads(subprocess.check_output([str(PYTHON), "-I", "-c", query], text=True))
    stdlib = Path(info["stdlib"]).resolve()
    return {"USER_HOME": str(Path.home()), "TEMP_ROOT": __import__("tempfile").gettempdir(), "WORK": str(Path(work).resolve()), "VENV": str((ROOT / "subject_venv").resolve()),
            "PYTHON": str(PYTHON.resolve()), "STDLIB": str(stdlib),
            "BASE_SITE": str(stdlib / "site-packages"),
            # The Python install (readable) and the symlinked home recorded in pyvenv.cfg (must resolve).
            "PYROOT": str(stdlib.parent.parent), "PYLINK": str(Path(_venv_home()).parent)}


def _venv_home():
    for line in (ROOT / "subject_venv/pyvenv.cfg").read_text().splitlines():
        if line.startswith("home"):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("pyvenv.cfg has no home")


def command(work):
    cmd = ["/usr/bin/sandbox-exec"]
    for key, value in parameters(work).items():
        cmd += ["-D", f"{key}={value}"]
    return cmd + ["-f", str(ROOT / "sandbox.sb"), str(PYTHON), "-I", "-B", "-u", "-"]


def run(code, work, timeout=600):
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    cmd = command(work)
    env = {"PATH": str(PYTHON.parent), "HOME": str(work), "TMPDIR": str(work),
           "LANG": "en_US.UTF-8", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    started = time.monotonic()
    try:
        process = subprocess.Popen(cmd, cwd=work, env=env, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    except OSError as exc:
        return {"returncode": None, "output": str(exc)[:LIMIT], "launch_error": True,
                "timed_out": False, "seconds": time.monotonic() - started}
    # Send code from an anonymous pipe, never from outside the allowed work dir.
    payload = (code + "\n").encode()
    os.set_blocking(process.stdin.fileno(), False)
    os.set_blocking(process.stdout.fileno(), False)
    selector = selectors.DefaultSelector()
    selector.register(process.stdin, selectors.EVENT_WRITE, "input")
    selector.register(process.stdout, selectors.EVENT_READ, "output")
    offset, chunks, kept, total, timed_out = 0, [], 0, 0, False
    try:
        while selector.get_map() or process.poll() is None:
            if time.monotonic() - started >= timeout:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            for key, _ in selector.select(timeout=0.2):
                if key.data == "input":
                    try:
                        offset += os.write(key.fd, payload[offset:offset + 65536])
                        if offset == len(payload):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                    except BrokenPipeError:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                else:
                    data = os.read(key.fd, 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    else:
                        total += len(data)
                        piece = data[:max(0, LIMIT * 4 - kept)]
                        if piece:
                            chunks.append(piece)
                        kept += len(piece)
        process.wait(timeout=5)
    finally:
        selector.close()
        # Kill descendants as well, even if the immediate Python process exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.wait()
    text = b"".join(chunks).decode("utf-8", "replace")
    truncated = len(text) > LIMIT or total > kept
    output = text[:LIMIT - 20] + "\n[output truncated]" if truncated else text[:LIMIT]
    launch_error = "sandbox_apply:" in output or "sandbox-exec:" in output
    return {"returncode": process.returncode, "output": output, "launch_error": launch_error,
            "timed_out": timed_out, "truncated": truncated, "seconds": time.monotonic() - started}
