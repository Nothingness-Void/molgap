# Authorized G1/G2 release receipt

Recorded 2026-10-01. Owner: server. This is execution evidence, not a result.

CPU recovery version 2 (kernel 136535008) passed the independent
`verification_recovery/acceptance_v2.json` gate: 150,000 rows, 150 path chunks,
source rederivation and exact degree-only initial-state transformation. Its
measured wall time was 739.464155918 seconds, with zero GPU allocation.

Both per-arm strict comparison plans passed the release-time real-reference
evidence verifier. Shared planning published distinct prospective trajectories
against `pcqm-gptrans-v5-audit-reference-s42`; no reference was retrained.
The first local preparation attempts stopped before any remote GPU request:
noncanonical Spec transport, absent closed-route context, then a semantic
target-transform digest incorrectly supplied as its serialized file SHA.
Zero prospective plans were published by the first plan failure; the later two
published plans were retained byte-for-byte while upload bindings were repaired.
No new plan or scientific identity was substituted during that recovery.

Qualified source package: `gptrans_author_dual_v3` in ignored platform storage.
`release.json` reported `LOCAL_RELEASE_INPUTS_VERIFIED`, no errors; the owning
Kaggle adapter repeated the input checks before its sole GPU POST.

Actual return: `kaseichou/molgap-gptrans-author-inputs-dual-s42`, kernel ID
136543794, version 1, `NvidiaTeslaT4`; all invalid-mount lists were empty and
`reconciliation_required=false`. Authority: `submission_v1.json`.
Pulled remote entry bytes matched the local release after transport-only CRLF
normalization; pulled metadata retained the two fixed/private dataset mounts,
completed CPU recovery source and T4 machine shape. The first observed API
state was RUNNING. This is not proof of completed preflight or the first epoch.

G1/G2 use independent one-device processes, unchanged FP32/BS128/seed42/60-epoch
reference recipe and per-arm native V5 traces. The notebook ceiling is six wall
hours/twelve allocated T4-hours. Per-arm optimizer calibration can stop an arm
before its training loop if throughput or memory does not fit the frozen budget.
The optional family-output profile remains disabled for the explicit target
digest compatibility reason in `recording_contract.md`.

Independent terminal acceptance, paired analysis and Replay-Ready publication
are future post-run gates; none was asserted by this submission record. The CPU
v2 input acceptance is retained, but its prospective NO_TRAIN RML terminal and
the earlier failed CPU-attempt terminals remain separate closure work. They
must not be called replay-ready training trajectories.

The request-to-submission five-minute target was not achieved: this release
required new owner-specific binding work. No precise fully-qualified-package
to-POST measurement was taken; file timestamps must not be mislabeled as a
measured end-to-end latency. Forty-four focused static/synthetic tests passed;
no local training, model inference, protected-role read or desktop modification
was performed.
