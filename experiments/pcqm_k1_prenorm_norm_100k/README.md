# K1 local normalization by residual placement

Two new100K architectures: pre-norm BN and pre-norm AdaRMSNorm. The retained
post-norm pair supplies contextual placement controls; neither old arm is rerun.
[Protocol](protocol.md) owns the question, [config](config.json) the frozen
recipe, [evidence review](evidence_review.md) the prior limits, and
[status](STATUS.md) the observed release state. Each arm has its own prospective
RML trajectory, native certificate, trace, predictions and resumable state.

Preparation uses `prepare_experiment.py --repo-root <checkout> --phase inputs`
with explicit initialization/reference inputs, then `--phase declarations`
after the executable source commit. The shared workflow stages acceptance
sidecars separately from executable source; they are not duplicated into the
Git source archive. The platform submission journal owns remote mutation.
