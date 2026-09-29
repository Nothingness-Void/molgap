# GPTrans author/local matrix — prelaunch boundary

Decision date: 2026-09-29. The prelaunch evidence `results/prelaunch.json`
identified the accepted Kaggle2 100K cache and the completed synthetic/Cython
parity experiment. Those facts were sufficient to define a label-free real-graph
CPU input preflight, but not an accuracy conclusion or GPU training release.

The P0 preflight samples deterministic training-source rows, reads no Gap
labels, and is separately verifiable by fixed manifest/shard hashes. G1/G2
require their own prospective training identities and inputs after P0 and
code-level gates; O1 changes the optimizer/target contract and therefore
cannot use the old reference as a strict comparator. No result in this record
establishes an MAE contribution or author-score reproduction.

The first physical CPU attempt returned an infrastructure error at graph
deserialization, before any sampled row was read; its immutable receipt is
`results/submission_v1.json`. Its partial `summary.json` reported no labels,
checkpoint, or evaluation role. It is not a failed path hypothesis.

The source-layout-only repair in physical v2 passed independent
`results/acceptance.json` after both fixed train shards, deterministic row
samples, per-shard output chunks and native CPU cost were retrieved and
verified. Of 40,559 connected sampled atom pairs at distance 2–20, 38,905
belonged to within-molecule distance groups with multiple bond-path signatures
(95.9%); 26,917 sampled pairs' selected shortest path had a non-single first
bond category. This proves available input distinctions under a deterministic
BFS tie rule, not identity with the author's Cython tensors and not any MAE
benefit. G1/G2 were not released by this CPU result alone.

The subsequent no-training GPU prelaunch check is recorded in
`results/gpu_prelaunch_gaps.json`. The historical GPTrans 100K best checkpoint
and aligned development-prediction files remain locally present and match
their completion-manifest SHA-256 values, but no recovered V5 reusable
reference bundle was present. G1 lacked a frozen transformed initial state;
G2 lacked a full accepted path sidecar and path-tie/sentinel policy. Neither
T4 arm was submitted or treated as a failed scientific test.

The 2026-09-29 no-inference historical recovery subsequently produced
`recovered_reference/reference_bundle.json`, whose compact references pass V5
structural validation. The 50K aligned prediction payload reproduced the
accepted 0.1566272043 eV MAE, and its targets matched the previously recovered
fixed-data target manifest. This removed the missing-bundle blocker without
rerunning training. It did **not** establish strict-causal V5 readiness: the
retained 60-epoch trace contains EMA development MAE but no independently
evaluated live-model development MAE per epoch, a field required by V5 for an
EMA causal comparison. One final checkpoint cannot recover the absent history.
The old run can support paired endpoint or contextual analysis only under that
limitation; a new matched baseline with a full trace would require an explicit
compute decision. G1/G2 also remained blocked by their candidate-specific
prelaunch gaps and absent GPTrans T4 runtime calibration.

The separately submitted I0 CPU real-input scale check ended with one
mount-path-only startup failure (v1) followed by a SHA-addressed source-only
repair (v2). Kaggle2 reported COMPLETE for v2, its pulled runner matched the
submitted LF-normalized SHA-256, and independent `verify_initial_output.py`
acceptance reproduced both fixed-shard row samples and all aggregate energies.
The source dataset, retained GPTrans seed-42 initial state and fixed 100K cache
were distinct private Kaggle2 mounts. No label, model forward, optimizer step,
or protected role was used. The 512 sampled training molecules contributed
7153 atoms. Before the first LayerNorm, the atom-embedding mean square was
0.0690655650 and the two degree-embedding tables together 1.9114128556 per
channel; the cross term was -0.0025205970. Degree thus supplied 96.51% of the
atom-plus-degree component energy when the cross term is excluded by definition.
The two independently retrieved train shards gave 96.517% and 96.508%,
respectively; this is a consistency check, not a sampling confidence interval.
This is consistent with the earlier analytic 96.8% expectation and establishes
the input-scale imbalance on real train graphs, not a prediction-error cause.

Simply multiplying **only** the degree tables by 0.02 while retaining the
OGB atom tables would instead imply about 1.09% degree energy on this sample.
Matching the author's analytically expected 18.2% *relative* degree fraction
under this local atom initialization would correspond to a degree-table factor
of about 0.0897, calculated from the measured energies. Both are arithmetic
counterfactuals, not trained candidates, and the latter uses the same sampled
input distribution by construction. Neither G1 nor G2 was released by I0.
