# Changes after preregistration

- 2026-11-04T08:10:44.994164+00:00: Preregistered method and code. No changes yet. Expected training, evaluation, blinding and result artifacts are recorded below.
- 2026-11-04T08:10:47.463249+00:00: Initial three calls committed and scored. Proceeding to the preregistered extension regardless of accuracy; no method or code change.
- 2026-11-04T08:27:53.260317+00:00: Completed the planned extension, committed extra calls, scored six pairs, and verified unchanged code hashes, full training logs, finite losses and frozen embeddings. No post-registration methodological changes. Optional attention test remains omitted.
- 2026-11-04T08:29:02.766740+00:00: Final reporting review added the explicit negative H-dist interpretation, six-pair margin table, and clearer device/precision caveat to REPORT.md. No code, statistic, calls, labels, training or evaluation results changed. All requested deliverables were checked for presence.

- 2026-11-04T08:50:28.571182+00:00: Registered the fresh-pair larger-distance confirmation in PREREG2.md before split generation or training for 106-111. Fixed N=6 and success=6/6; discovery data excluded.
- 2026-11-04T08:52:15.589447+00:00: Added confirmation-specific classifier, scorer and orchestration; classifier differs from dist_test.py only in default output filename and larger-distance comparison. Recorded code/split hashes in confirm_manifest.json before training. Generated 106-111 splits with the frozen local algorithm; all data checks passed. No change to training, evaluation, D, or blinding code.
- 2026-11-04T08:58:26.552298+00:00: Pair 106 completed on CPU and passed the unchanged behaviour gate. No distances inspected.
- 2026-11-04T09:04:04.622996+00:00: Pair 107 completed on CPU and passed the unchanged behaviour gate. No distances inspected.
- 2026-11-04T09:09:39.422418+00:00: Pair 108 completed on CPU and passed the unchanged behaviour gate. No distances inspected.
- 2026-11-04T09:15:15.449394+00:00: Pair 109 completed on CPU and passed the unchanged behaviour gate. No distances inspected.
- 2026-11-04T09:20:55.403735+00:00: Pair 110 completed on CPU and passed the unchanged behaviour gate. No distances inspected.
- 2026-11-04T09:26:34.206041+00:00: Pair 111 completed on CPU and passed the unchanged behaviour gate. No distances inspected.
- 2026-11-04T09:26:39.426258+00:00: All six confirmation calls committed before scoring. calls_confirm.json SHA-256: 09ebd12a91fbf105be23180b77f9d1bb35dc7d01952f0e0a303f029d9af7ba5d.
- 2026-11-04T09:26:40.372313+00:00: Confirmation scored 5/6, one-sided p=0.109375; success criterion not met. All gates, training-log checks, frozen-weight checks and code hashes passed. Discovery reports/calls preserved byte-for-byte. No methodological changes or seed replacement. REPORT_CONFIRM.md written last.
- 2026-11-04T09:27:39.292705+00:00: Final audit verified PREREG2, frozen code, original reports/calls and committed confirmation-call hashes; checked all six pairs and all eight descriptive tensor types. Source facts/base/config/tokenizer hashes also matched discovery provenance. Added explicit pair-108 exception and behaviour ranges to REPORT_CONFIRM.md; reporting-only clarification, with no method or result changes. Required deliverables present; report written last.
