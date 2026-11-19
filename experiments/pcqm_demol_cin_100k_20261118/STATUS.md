# Status

- Prior run `pcqm-demol-cin-100k-20261118-s42` (`nothingnessvoid/molgap-demol-cin-100k-20261118-s42`, `kernel_id=145744975`) failed at Kaggle import (`KernelWorkerStatus.ERROR`) because `src/molgap/demol_cin.py` imported `.ogb_features`, which imports `rdkit`. Journal preserved untouched at `D:/molgap-artifacts/runs/pcqm-demol-cin-100k-20261118-s42/v1`.
- Fixed in `r2-s42` by replacing `.ogb_features` with `AtomEncoder` and `BondEncoder` from `ogb.graphproppred.mol_encoder` (zero `rdkit` dependency).
- Submitted `pcqm-demol-cin-100k-20261118-r2-s42`: `SUBMISSION_BOUND` (`nothingnessvoid/molgap-demol-cin-r2-100k-20261118-s42`, `kernel_id=145779299`, `version=1`, `KernelWorkerStatus.RUNNING`).

## 2026-11-19 continuation reconciliation

- Original kernel v1 is terminal ERROR; the original log reports `TimeoutError: Frozen allocation ceiling exhausted; retain complete-epoch resume state`. Latest observed version remains 1; no successor was submitted here.
- Original paused retrieval of all arms failed because this run mixes a completed reference and paused candidate. Reusing the primary checkout's existing `retrieve_family_outputs.py --all-arms --arm demol_cin --paused --execute` succeeded at pinned version 1.
- Retrieved original `selected_model_ema0.9999.pt` (22,304,077 bytes, SHA256 d1ce3097ea0f3849cd29c09e27060dc7bd007415e89db4e8259df9e9850de96f) and `development_predictions_ema0.9999.pt` (803,439 bytes, SHA256 0179ff39dc7b1ad51449e2117ebe647563c04a58ef4d0614473860b96609255f). No chunk endpoint substitution.
- Reviewed partial-arm continuation changes d73a0c20c and 734d7f8d9 were cherry-picked as 9a5561cf5 and bc409919d. Original frozen scientific package/source remains unchanged.
- Existing `build-resume` now stops at `Development target identity mismatch` in `family_training_runner.validate_screen_resume`: auxiliary prediction inspection hardcodes `target_identity=None`. Main and auxiliary predictions have identical 50K source rows and identical targets. Observed raw target digest 7b527c571baecbca050dd9fa8318fdde6b91d258e3709ad44fb49f869a9c606e differs from frozen expected a1c0b711815225f0bccee905419d124174e1bf01c76877f5c661b063ee170137.
- Shared recovery validation is not bypassed. No successor publication, new training, completed-reference retraining or terminal RML closure occurred. A reviewed target-identity-aware auxiliary resume validation path is required before preparation can proceed.

### Encoding diagnosis and recovery authorization boundary

The reported 7b527c57 digest is the existing `tensor_digest` encoding, not a raw FP32 byte digest. The main-thread independent comparison confirms all four original main/auxiliary endpoint targets have raw little-endian FP32 SHA256 a1c0b711815225f0bccee905419d124174e1bf01c76877f5c661b063ee170137, matching the frozen manifest. Original prediction files are not considered damaged.

Existing terminal acceptance has `TargetIdentityBinding.from_frozen_package`; the resume dispatcher/owner does not pass that binding. `experiment_workflow_resume._repair_payloads` checks `_runner_repair_ast` and explicitly permits only the source verifier, target-statistics tolerance and exact resume-probe deepcopy repair. A change to `validate_screen_resume` is rejected as `Runner repair permits only the verifier and exact resume-probe deepcopy`. No existing reviewed target-binding recovery overlay was found. Extending that narrow repair authority requires a reviewed change covering pinned acceptance plan/target-manifest transport, resume validation dispatch and the exact target-binding AST exception; scientific recipes, prediction bytes, hashes and optimizer state must remain unchanged.

### Authorized repair and continuation preparation

- User authorized the narrowly scoped target-identity recovery repair; implementation commits 580ac2ec1 and 8a7287beb retain original source/package and bind the original acceptance plan/target manifest in local, remote-launch and strict replay validation. The recovery overlay is source-hash-attested for exactly four declared files.
- Original GPTrans+development+auxEMA CPU continuation check passed. Missing binding, changed plan hash, changed target and unrelated runner AST modification are rejected. Infrastructure budget and diff checks passed.
- Original checkpoint SHA remains 7fff4e9002e70e3130ede56534161642563c9abeee9f16cdeb26a850a0a96397. The native cursor exporter now prioritizes checkpoint counters over the last complete-epoch trace observation: actual 42636 steps / 5457408 presentations, pass54 + batch462. Original state is unchanged.
- Fresh bundle manifest SHA fbb312e06eec85cfd927ebb9087a695bd90b8e8296c6feb2064b70ec310cc5df. Existing prepare-resume passed RESUME_INPUTS_PREPARED with errors=[] at D:/molgap-artifacts/resume/demol-cin-r2-20261119/prepared_v2; pair_update_norm is retained at original manifest 556072152d6f03808af65b2f21b8497ef5c063566e7bdfbc8024c0d4414cef7a.
- User requested no monitoring after submission; wait for their completion notification before terminal retrieval and acceptance.

### Continuation v2 bound

- Existing submitter first stopped before kernel submission because Kaggle still exposed source version1 after publication. The same SOURCE_PUBLISHED journal was re-entered after metadata propagation; no source was published twice.
- SUBMISSION_BOUND: original kernel145779299/version2, nothingnessvoid/molgap-demol-cin-r2-100k-20261118-s42; submit-time observed scheduler KernelWorkerStatus.RUNNING. Source version2 is pinned and verified by the existing submitter.
- Source/spec/scientific recipe/producer package remain frozen; only demol_cin resumes from original checkpoint42636 to46860. pair_update_norm remains its original version1 terminal evidence.
- Continuation preparation: D:/molgap-artifacts/resume/demol-cin-r2-20261119/prepared_v2. Continuation operation journal/receipt: D:/molgap-artifacts/resume/demol-cin-r2-20261119/operations_v2.
- No post-submit monitoring will run per the user's explicit instruction. Wait for user completion notification; then retrieve version2 with this continuation receipt/prepared/authorization and close through the existing strict acceptance/RML path. Native resume proof and final training completion are not yet claimed.
