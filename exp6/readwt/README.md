# Exp6-read: blinded weight-reading harness

This harness compares predictions about three Go policies with weights (W) and without weights (noW). W sessions receive randomly labelled X/Y/Z networks. The scorer uses the secret role key, compares predictions with the measured truth, and reports per-repetition and paired behavioral errors. Real predictions and scores are supplied separately from fake harness tests.

`prompt.md` is the canonical, verbatim subject prompt. `run_subject.py` reads it directly and selects the marked W/noW sections. The historical `build_prompt.py`, `audit_artifacts.py` and preparation report are excluded because they depend on internal or undistributed inputs. No prompt-generation step is needed.

The subject runner requires macOS Seatbelt (`sandbox-exec`) and the shipped `sandbox.sb`. It restricts file reads and writes, denies networking, runs code through a fresh isolated Python process, and has no unsandboxed fallback. Every real session must pass the sandbox preflight. The experimenter must also review subject code for prohibited input propagation; static warnings alone do not establish compliance.

## Run the tests

From the release root, create separate subject and trusted test environments:

```sh
python3.12 -m venv --copies exp6/readwt/subject_venv
exp6/readwt/subject_venv/bin/python -m pip install -r exp6/readwt/subject_requirements.txt
python3.12 -m venv exp6/readwt/truth_venv
exp6/readwt/truth_venv/bin/python -m pip install -r exp6/readwt/truth_requirements.txt
cd exp6/readwt
truth_venv/bin/python -I test_truth.py
truth_venv/bin/python -I test_harness.py
truth_venv/bin/python -I test_sandbox.py
```

The truth tests check behavioral definitions and aggregation. The harness tests check blinding, scoring, mocked requests and session recovery. The sandbox test checks allowed runtime access, forbidden reads and socket denial. Require status `PASS` and exit 0; `UNVERIFIED_LAUNCH_BLOCKED`/exit 2 means isolation was not verified. Do not bypass preflight.

For real sessions, obtain the author networks separately as described in the root README and configure `WSP_API_BASE`, `WSP_API_KEY` and `WSP_MODEL`. Run `run_subject.py --condition W --rep 0` and `--condition noW --rep 0` with the trusted interpreter; repeat for repetitions 1 and 2, then run `score.py`. `--model` and `--effort` select model settings. Keep truth files, keys, this README and prior replies out of subject inputs. Preserve the supplied prompt and measured truth across conditions.
