# K1 / GPTrans composed-recipe 500K transfer

Desktop-owned experiment on `codex/exp/k1-gptrans-500k-package`,
branched from verified desktop tip `3f194a932fe2da3d8ac14ecedffac709d90c3de7`.
Read [the frozen contract](protocol.md) for execution and acceptance,
and [the evidence review](plan.md) for selection rationale and limitations.
Physical execution and continuation are in [STATUS](STATUS.md).

Preparation: [prepare.py](prepare.py) freezes the Spec, two native recipes,
role/budget declarations and per-arm prospective planner inputs. It uses the
shared source package, prospective planner and release checker. Declare first
with `--source-commit <HEAD> --declare-only`, commit the reviewed source and
recipes, then prepare with the frozen HEAD and a fresh `--output` directory.
The planner publishes both trajectories before runtime diagnostics/training.

The registered `prepare-workflow` execution registry does not support these
500K families. [The platform bootstrap](../../platforms/kaggle/run_legacy_500k_pair.py)
calls the existing bounded500K owner, validates both arms on CPU, then completes
both isolated T4 preflights before starting either training worker. It retains
pair state, per-arm logs, native stage manifests and invocation allocation costs.
Publication, submission and continuation remain with the Kaggle workload skill;
local preparation grants no remote or scientific acceptance authority.
