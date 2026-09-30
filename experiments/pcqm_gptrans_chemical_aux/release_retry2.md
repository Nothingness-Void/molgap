# Source packaging reconciliation, 2026-10-01

Both complete train caches were accepted under `label_policy.md`: descriptors
100,000 rows and 1,115.8016178 seconds measured label-stage wall time;
fingerprints 100,000 rows and 54.3620658 seconds. This is not CPU busy time.

The first two prospective training plans froze source d813dfba. Package creation
then rejected `src/molgap/experiment_family_artifacts.py`, a required shared
executable module, because the generic path classifier treated its filename's
`artifacts` token as a retained-data path. No package was published and no
accelerator was allocated. Keep these original plans and their NO_TRAIN closure.

The existing source packager now permits storage-related words in Python module
filenames only below `src/molgap`. Directory restrictions, credential words,
non-source suffixes, traversal and tracked-byte checks remain enforced. This
does not allow retained data or credentials into source packages. Package only
the selected runtime code and explicit owning recipe/CLI inputs.

Freeze a second source commit and fresh v2 arm plans. Cache identities, model,
objectives, roles, exposure, comparison and budget remain those declared in
`training_protocol.md`. Use the same intended physical kernel reference; the
source/Spec attempt identity is separate and must be bound by the actual receipt.
