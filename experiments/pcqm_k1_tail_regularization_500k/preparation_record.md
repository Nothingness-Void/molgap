# Preparation and verification

2026-10-25 JST. Source873c3ac4, Spec
`f5404cb801a792bf57f56dfde0b496e8f21a7ad67b9f22f93a8721cc1d04a994`, package
`09efe4c0b754b0c7b7f2d1ea8e7a679be3b0a7b2763eb30a8afc9fff131c1e32`.
The existing platform preparation completed in76.28s; its release-input check
took13.42s. No native training or scientific result is established by this.

Interpretation boundary: versus the historical parent, both arms also share a
fresh optimizer/LR horizon and plain single-forward supervised loss instead of
the parent's two-forward consistency objective. Only B-versus-A isolates decay.
An improvement versus the retained starting predictor establishes a useful whole
tail recipe under this contract; it cannot uniquely credit weight decay or prove
the parent's plateau had one cause.

Used shared family training, atomic IO/state hashing, scientific-input compiler,
source-package/release/prospective owners and Kaggle prepared workflow. Added no
model, trainer loop, submitter, logger or RML framework. Small shared extensions:
frozen shadow development state reporting and trained-initialization prediction
reproduction before the first update. The declaration script uses the existing
immutable LF metadata writer; the first local attempt detected a raw-versus-LF
recipe hash mismatch before prospective publication and was corrected locally.

Luna executed four focused existing/new synthetic nodes with the primary Python
and this worktree's source: live-selected shadow-EMA serialized replay; ordinary
K1 cosine-epoch midpass replay; K1 standard logger state/RNG/evaluation invariance;
and actual shared session output/CPU mechanical inspection. All four passed.
These checks use synthetic CPU graphs; they do not exercise the full500K shard
set, cloud bootstrap, the new CUDA startup guard or native T4 qualification.
The registered cloud admission and exact retained-prediction gate remain required.
RML rebuild and `check --frozen` passed after both prospective records were staged.
No global historical artifact hydration or portable-readiness claim is made.

Review found an in-process cleanup limitation if the optional initialization
guard raises before `session.run()`: its loaders/kernel are not explicitly closed
by that early branch. This launch uses isolated one-shot workers; an uncaught
exception ends the worker, the pair runner cancels peers and reaps processes,
and no failed guard can start training or silently become random initialization.
The process-boundary cleanup limits this launch risk; the reusable explicit-close
follow-up must preserve the frozen package and goes to infrastructure maintenance.
The infrastructure owner subsequently fixed explicit exception cleanup and
sanitized HTTP failure observability in desktop3395ed0e/1a031ca1. This remote
package remains873c3ac4; no in-place payload mutation or experimental branch merge
was performed to adopt those fixes during training.

Kaggle1 read-only reconciliation by Luna observed DropPath140239597/v1 and RMS
position140063114/v3 COMPLETE,43.279aggregate GPU hours after reservations,
0reserved. Quota is not a separate concurrency-slot count. The submission adapter
must recheck account/quota/target identity before POST. No old job is modified.
