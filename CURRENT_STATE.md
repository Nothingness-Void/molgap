# Current State

> Live truth only. Historical methods, metrics, and failures live in experiment
> decisions; task order lives in `ROADMAP.md`.

## Production

- Recommended Track A model: repaired-2M three-GPS dense pure 2D.
- Registry: `repaired_2m_dense_2d`; lower-cost preset:
  `repaired_2m_equal_2d`.
- Loader: `load_repaired_2m_2d` in `src/molgap/inference.py`.
- Authority: `production/04_evaluate/project_freeze/track_a_final_decision.md`.
- Track B experiments cannot change this registry.

## Track B candidate

Neural-Atom K1 and the adapted GPTrans-T core are the two frozen architectures
in the desktop-owned matched full-run comparison. K1 replaces dense per-layer global GPS attention with
three exchanges through one 64-channel molecular slot while retaining local
persistent real-bond EdgeState processing. Its PCQM-100K, once-read shadow, and
fixed 500K bridge passed; evidence is in:

- `experiments/pcqm_k1_shadow/decision.md`
- `experiments/pcqm_k1_scale500k/decision.md`

K1 is frozen at 3,658,817 parameters and architecture commit
`36215d9539acdd75542608637ec1e2db5341d3ff`. It is not a production promotion,
an official leaderboard result, or evidence against the separately configured
desktop 304-wide/pretrained model.

GPTrans-T is frozen at 5,246,817 parameters. Its 100K reference was weak, but
the same propagation core passed the accepted SCNet 500K gate and was
numerically slightly better than K1's separate 500K run. Because their
architecture-specific optimization contracts differ, the full matched study—not
the cross-job 500K scalar difference—owns the final comparison. Authority:
`experiments/pcqm_gptrans_t_500k/results/decision.md`.

The audited full-role implementation is `experiments/pcqm_k1_full/`. The
desktop-owned branch `codex/pcqm-k1-gptrans-full-fusion` has launched a matched
K1/GPTrans comparison at 20M presentations, seed 42, strict FP32/no-TF32, and
physical BS128. Its recoverable checkpoints and scheduler recovery remain
desktop-owned; server must not launch a duplicate or interfere. Official-role
use and any final submission remain governed by that experiment's frozen
contract.

## Discovery state

The extra PairToken+MoSE31 seed-42 round is complete and closed. Saved-artifact
analysis found an undeclared feature-identity change in its frozen prelaunch:
the candidate consumes RWSE16+MoSE31, while the architecture-only strict plan
copied K1's RWSE16 identity. Its aligned development endpoint is retained as
`PAIRED_ENDPOINT`, not a strict causal comparison or replay-ready win; no seed,
scale bridge, protected role, or successor follows. Authority:
`experiments/pcqm_k1_pair_token_mose_100k/decision.md`.

Round 1 of the newly authorized, at-most-three-round K1 sequence is closed.
The pure-2D sparse-triplet candidate was exactly K1 at initialization and
completed the fixed V5 screen, but reached `0.1424927413 eV`, worse than K1 by
`0.0011191070 eV`; its paired interval was entirely unfavorable. It fit the
training role more strongly and helped the hardest K1-error quintiles while
damaging easier rows, so full-depth triplet state is an allocation/overfit
failure rather than a missing-topology win. It is now a fourth complete
candidate in the RML replay pool. Authority:
`experiments/pcqm_k1_sparse_triplet_100k/decision.md`.

Round 2, Graphormer-SPD conditioning of the accepted layer-6 PairToken, is
closed. It reached `0.1397010982 eV`, a favorable but sub-threshold
`0.0016725361 eV` gain over K1, while regressing `0.0013710489 eV` against the
original PairToken with an entirely unfavorable paired interval. The learned
distance biases were non-trivial, so the result closes globally shared
shortest-path conditioning rather than an inactive implementation. It is the
fifth complete candidate in the RML replay pool. Authority:
`experiments/pcqm_k1_spd_pair_token_100k/decision.md`.

The final authorized round is closed. A zero-initialized, non-persistent
directed bond-to-bond triplet adapter at layer 6 beside original PairToken
reached `0.1387662739 eV`, improving K1 by `0.0026073605 eV` with a favorable
paired interval but missing the frozen `0.003 eV` gate. It was
`0.0004362366 eV` worse than original PairToken and that parent interval
crossed zero. The candidate again helped hard K1-error rows while damaging
easy rows. It is replay-ready but not promoted; original PairToken remains the
retained K1 relation mechanism. The three-round sparse-triplet/SPD/one-shot
sequence is exhausted. Authority:
`experiments/pcqm_k1_oneshot_triplet_pair_token_100k/decision.md`.

The multiplicative PairValue screen completed under the fixed V5 PCQM-100K,
FP32, BS128 contract and passed no-inference terminal acceptance. It reached
`0.1424374580 eV`, which was `0.0010638237 eV` worse than K1-v4 and
`0.0041074143 eV` worse than original PairToken; both paired intervals were
unfavorable. PairToken selector/value micro-architecture enrichment is closed.
Authority:
`experiments/pcqm_k1_multiplicative_pair_value_100k/decision.md`.

The chemistry-conditioned PairToken
completed under the fixed V5 PCQM-100K/50K, FP32, BS128 contract and passed
no-inference terminal acceptance. It reached `0.1385749578 eV`, improving K1 by
`0.0027986765 eV` with a favorable paired interval, but missed the frozen
`0.003 eV` gate. It was also `0.0002449146 eV` worse than original PairToken;
that paired interval crossed zero. Chemistry-conditioned selection is closed,
and original PairToken remains the strongest retained 100K relation-content
mechanism. Authority:
`experiments/pcqm_k1_chem_typed_pair_token_100k/decision.md`.

The standalone functional-group token screen completed at `0.1398314089 eV`
versus K1-v4 at `0.1413736343 eV`. Its favorable `0.0015422255 eV` gain missed
the prospectively frozen `0.003 eV` gate, so the standalone route is closed.
Its terminal evidence was ingested through the generic RML pipeline; replay
excludes it only because the historical K1 reference trace is not yet locally
canonicalized, not because candidate evidence is incomplete. Authority:
`experiments/pcqm_k1_functional_group_token_100k/decision.md`.

The node-adaptive PairToken-return screen
completed with strict V5 evidence. It improved immutable K1-v4 by
`0.0017446578 eV`, and its paired interval was entirely favorable, but it missed
the prospectively frozen `0.003 eV` material gate. Return redistribution is
therefore directionally useful but insufficient; extra seeds, scale-up and
further return-only variants are closed. Authority:
`experiments/pcqm_k1_pair_token_node_return_100k/decision.md`.

The retained K1-v4 100K artifacts have a repository-bound V5 reference bundle
and portable target-transform asset. Recovery verified the fixed cache,
row/target identities, aligned 50K predictions, model/checkpoint hashes,
runtime, roles, cost and trace without training or model inference. Training
stochasticity remains unavailable. Authority:
`experiments/v5_legacy_evidence_migration/k1_v4_100k_reference/reference_bundle.json`.

The K1 molecule-context MoSE gate
completed and passed mechanical acceptance, but reached `0.1419007480 eV`
versus immutable K1-v4 at `0.1413736343 eV` and the positive selective-MoSE
predecessor at `0.1393276304 eV`. Its paired interval crossed zero and its best
checkpoint occurred at epoch 39 of 40. The mechanism is therefore
`NEGATIVE_UNDER_CONTRACT`; contextual MoSE gate refinement is closed without a
retry, extra seed, scale bridge or successor. Authority:
`experiments/pcqm_k1_mose_context_gate_100k/decision.md`.

The predecessor selective MoSE residual improved immutable K1-v4 by
`0.0020460039 eV`, with an entirely favorable paired interval, but missed the
experiment-local, prospectively frozen `0.003 eV` gate. That value is not a
permanent V5 threshold. The candidate strongly improved K1-hard rows while
damaging K1-easy rows, and its late trajectory was effectively saturated. It
is retained as positive mechanism evidence but is not promoted under that
experiment's contract. Authority:
`experiments/pcqm_k1_mose_residual_gate_100k/decision.md`.

The prior K1-MoSE hidden-normalization screen completed and regressed versus
both K1-v4 and unnormalized MoSE. Its paired interval was entirely unfavorable,
and subgroup analysis found regression in every MoSE-magnitude quintile. Hidden
normalization and further normalization variants are closed. Authority:
`experiments/pcqm_k1_mose_hidden_bn_100k/decision.md`.

The V5 K1-MoSE information-source screen is complete. It replaced K1-v4's
RWSE16 input with 31 rooted motif-homomorphism counts while preserving the
EdgeState/K1 information flow and all non-intervention contract fields. The
accepted seed-42 candidate reached `0.1402545124 eV` versus K1-v4 at
`0.1413736343 eV`: a directional `0.0011191219 eV` gain with a favorable paired
row-bootstrap interval. It nevertheless failed the prospectively frozen
`0.003 eV` promotion gate, which accounts conservatively for training-level
variation that row bootstrap cannot measure. The outcome is
`NEGATIVE_UNDER_CONTRACT` / `DIRECTIONAL_SUBTHRESHOLD`; no extra seed, 500K
bridge, protected role, full training, or desktop handoff is authorized.
Authority: `experiments/pcqm_k1_mose_100k/decision.md`.

The `0.003 eV` value is a policy gate, not a permanently fixed V5 constant.
Any replacement must be calibrated from same-contract repeatability evidence
and frozen before another candidate is observed. The prior nominally identical
scratch-through-40 jobs differed by `0.0026921320 eV`, so the present MoSE
result does not justify lowering the gate retroactively.

The separately authorized three-round K1 architecture sequence is complete.
Round 1 closed RepSet final readout and cross-layer selector sharing. Round 2
found a directional but sub-threshold gain from removing length-one slot
attention. Round 3 found a similarly directional but sub-threshold gain from
uniform normalized return; inverse-score return regressed. No candidate cleared
the frozen `0.003 eV` material/run-variation gate, so K1-v4 remains unchanged.
Authority: `experiments/pcqm_k1_return_allocation_100k/decision.md`.

The explicitly reopened interaction screen combining the two directionally
positive K1 simplifications completed with a favorable but sub-threshold gain.
It is closed and does not alter the frozen K1 handoff. Authority:
`experiments/pcqm_k1_combined_simplification_100k/decision.md`.

The separate Kaggle2 discovery loop remains bounded to at most three
evidence-gated rounds. Round 1 completed a pure-2D GPTrans-T reference on the
accepted 100K/50K data identity. It was mechanically accepted but substantially
underperformed the contextual K1 reference. Reconciliation with the accepted
500K run does not isolate a causal scale effect: exposure, EMA time scale and
evaluation roles differ. The user explicitly reopened the two unused Kaggle2
rounds. The first completed: Pair PreNorm passed the mechanism-shortlist gate;
Centered Logits regressed and is closed. No K1/full promotion follows.
Authority: `experiments/pcqm_gptrans_relation_flow/decision.md`.
The last authorized round completed: both persistent-pair readback variants
failed to improve the reference and are closed. That two-round authorization
was exhausted without a full-scale promotion.
Pair PreNorm is the only retained seed42 mechanism shortlist, not a full-scale
or K1 promotion. Authority and consolidated evidence:
`experiments/pcqm_gptrans_memory_readback/decision.md`.

The user reopened three additional evidence-gated Kaggle2 rounds. Round 1
closed isolated K1 real-bond storage/read normalization: one arm regressed and
the other was directionally favorable but below the material gate. Round 2
tested that weak relation-recurrence signal with the two favorable global-slot
simplifications. Neither pair passed the material gate or retained both parent
effects, so triple stacking is closed. The final authorized round tested
GPS++-style sender-only and separate receiver/sender local aggregation while
preserving frozen K1 at initialization. Sender-only regressed; bidirectional
fit training more strongly but was slightly worse than K1 on development.
Neither passed the material gate, so all three rounds are exhausted with no
K1 change. Authorities:
`experiments/pcqm_k1_edge_memory_100k/decision.md` and
`experiments/pcqm_k1_edge_slot_interaction_100k/decision.md` and
`experiments/pcqm_k1_gpspp_local_100k/decision.md`.

The bounded three-attempt K1 mechanism sequence is complete. Paper-style
multi-slot grouping, channel-wise multi-head atom selection, and a
molecule-conditioned single query all fit training more tightly and generalized
worse under otherwise matched v4 contracts. Their decisions are
`experiments/pcqm_k1_paper_allocation_100k/decision.md`,
`experiments/pcqm_k1_multiview_pool_100k/decision.md`, and
`experiments/pcqm_k1_dynamic_query_100k/decision.md`. No attempt qualifies for
another seed, shadow access, scale-up, official evaluation, or submission.

The bounded K1-v4 selective-global screen completed with no winner. K1-G was
materially worse; K1-R was statistically indistinguishable and slightly worse.
Both are closed without extra seeds or scale-up. The accepted K1-v4 reference
remains reusable only for its exact benchmark contract. Authority:
`experiments/pcqm_k1_variants_100k/decision.md`.

GraphState and its derivatives failed full-scale transfer and are archive-only.
The local-hierarchy allocation, GAPE, adaptive denoising, cardinality channel,
multi-slot Neural-Atom, K1-G, K1-R, Fourier Edge, geometry, path, ring, PNA,
directed-bond, fragment, dual-stream, and attention variants are closed by
their own records.
The consolidated attribution is
`experiments/pcqm_gap_architecture/results/architecture_failure_attribution_2026-09-08/decision.md`;
archive indexes are `experiments/_closed/pcqm_server_archive_index.md` and
`experiments/_closed/qm9_top20_archive_index.md`.

Desktop continues to own SCNet full training, official evaluation, and final
submission. By explicit user authorization, server may use the new Kunshan
account for V4 architecture screens. Its reusable Python 3.10/DTK 23.10/PyG
runtime has passed the generic one-DCU FP32 BS128 determinism check; every
candidate still requires its own source/cache/memory/throughput V4 preflight.
Authority: `platforms/scnet/KUNSHAN_V4_RUNTIME.md`. This infrastructure result
does not modify the frozen K1 full handoff.

The coordinator explicitly reopened one fresh seed-42 K1 question: condition
the three sparse slot selectors on the incident persistent real-bond state.
The protocol is `experiments/pcqm_k1_edge_conditioned_slot_100k/protocol.md`.
Its Kaggle2 v1 submission terminated before training because the metadata-only
P100 request received a Tesla T4. The terminal diagnosis is
`experiments/pcqm_k1_edge_conditioned_slot_100k/results/failure_diagnosis.md`.
The user then authorized one infrastructure-only retry with an explicit CLI P100
accelerator override and the unchanged scientific contract. Version 2 also
received a Tesla T4 and terminated before candidate execution; its diagnosis is
`experiments/pcqm_k1_edge_conditioned_slot_100k/results/failure_diagnosis_v2.md`.
Version 3 completed with accelerator-flexible binding and passed saved-artifact
acceptance without model inference. It reached `0.1410829425 eV` on the fixed
development role, a `0.0002906919 eV` gain over K1-v4, while the paired
bootstrap interval crossed zero and the `0.003 eV` material gate failed.
`selected_candidate=null`; the mechanism is closed without shadow/official
roles, another seed, scale-up, or successor. The decision and attribution are
in `experiments/pcqm_k1_edge_conditioned_slot_100k/decision.md`.

The post-hoc K1 explainability Stage 1 completed without training or model
inference. Across 20 frozen same-contract payloads it found coherent K1
deficits at small/sparse/weakly conjugated and highly cyclic/high-RWSE topology
extremes. No old variant cleared the overall material-gain gate. A no-training
frozen-checkpoint Stage 2 is now informative, but no successor architecture is
released unless that intervention identifies a causal information-flow
bottleneck. Authority:
`experiments/pcqm_k1_explainability_audit/results/stage1_decision.md`.

Its frozen-checkpoint causal Round 1 also completed and reproduced K1-v4.
Deleting any global exchange was strongly harmful, but layer 6 showed the
largest topology-dependent update-magnitude and node-dispersion inflation.
The no-training Round-2 strength audit found coefficient `1.0` to be a sharp
optimum: every attenuation or amplification worsened all audited strata. The
scalar-strength hypothesis and its conditional Round 3 are closed; K1-v4 is
unchanged. Authority:
`experiments/pcqm_k1_explainability_audit/results/stage2_round2_decision.md`.

K1 PairToken is the only new seed-42 mechanism winner. It preserves K1-v4 and
adds one pre-normalized all-pair relation token at layer 6 without dense
atom-to-atom attention. It reached `0.1383300573 eV`, improving K1-v4 by
`0.0030435771 eV`, with a favorable paired interval and 22,848 added
parameters. The margin is only `0.0000435771 eV` above the material gate.
Frozen-checkpoint causal audit job
`122305552` showed that learned pair selection, cross-node pairs, and per-pair
normalization are all active; uniform averaging, diagonal-only pairs, or
removing normalization regressed. Its matched-V4 500K bridge completed at
`0.1043037325 eV`, only `0.0005561373 eV` better than the frozen K1 scalar and
below the `0.003 eV` gate. V5 execution/artifact acceptance passed, strict
paired comparison remained pending because the exact reference prediction
bundle was unavailable, and the route was closed as
`NEGATIVE_UNDER_CONTRACT`. Authority:
`experiments/pcqm_k1_pair_token_100k/decision.md` and
`experiments/pcqm_k1_pair_token_500k/results/decision_122312462.md`.

A frozen PairToken assignment audit rejected hard dynamic top-pair sparsity:
the largest 20% of ordered pairs carried only 42.9% median assignment mass,
including on K1-hard molecules. This was `NO_TRAIN` evidence, not a new model
candidate or replay-pool entry. No sparse successor was released. Authority:
`experiments/pcqm_k1_pair_selection_diagnostic/decision.md`.

The bounded PairToken scale-attribution Round 1 reference `122743291` is
terminal-accepted. Round 2 aligned all 50,000 fixed 500K development rows:
PairToken is `0.0006985342 eV` worse than the exact matched60-v4 K1 reference
(paired 95% row-bootstrap candidate-minus-reference interval
`[0.0001388103, 0.0012531244] eV`). The positive 100K gain did not retain.
The bounded PairToken scale-attribution sequence is closed. The sole Round-3
inference job `122843454` completed and passed independent acceptance: the
trained PairToken network relies on its relation branch, but PairToken's exact
matched 500K development MAE is `0.000699 eV` worse than K1, with a larger
train–development gap. This does not reject K1's own 100K-to-500K scale gain.
No PairToken successor, seed, full run, or protected-role access is released.
Authority: `experiments/pcqm_k1_pair_token_scale_attribution/results/round3_decision.md`.

Server acceptance now follows the V5 seven-state outcome model without
rewriting historical V4 decisions. A local audit found complete raw artifact
sets for all 21 inventoried V4 100K arms, and the PairToken 100K result has a
non-destructive V5 sidecar. The old K1 500K run remains paired-v3 evidence, not
a strict V4 reference. The PairToken profiling sweep reproduced BS128
throughput but was non-monotonic and only partially covered V5 timing stages;
it did not change the BS128 contract. Authorities:
`experiments/v5_legacy_evidence_migration/decision.md` and
`experiments/pcqm_k1_pair_token_batch_profile/results/decision_122388380.md`.

The server-owned V5 GPTrans-T/Pair-PreNorm 500K scale question is closed without
another continuation. Initial preflights failed before training, repaired
preflights passed, and training jobs `122432964`/`122432972` reached the
16-hour wall with atomic checkpoints through reference epoch 36 and candidate
epoch 38. A first continuation redundantly reran calibration; certified
continuations `122520066`/`122520074` then passed identity checks but failed
before adding an epoch because restored EMA tensors remained on CPU.

The preserved curve was already unfavorable: Pair PreNorm was worse at every
common epoch from 20 through 36 and would need a relative `0.0044580414 eV`
reversal to clear the material gate. Finishing would cost about 18.2 additional
DCU-hours, so the V5 outcome is `INCONCLUSIVE` science, `STOP_FOR_COST`, and no
desktop handoff. The experiment did not access protected roles. Authority:
`experiments/pcqm_gptrans_prenorm_500k_v5/decision.md`.

The separate train-only GPTrans shortest-path profile completed. Exact cached
distances accelerated the isolated path stage but improved representative
end-to-end throughput by only `1.0238x`, regressed the graph-size tail, and
missed the strict trajectory-equivalence tolerance. No implementation follow-up
or scientific-contract change was released. Authority:
`experiments/pcqm_gptrans_shortest_path_profile_v5/results/decision_122484011.md`.

A broader train-only V5 GPTrans execution profile completed on Kunshan. Forward
plus backward accounted for about 87% of synchronized step time; loader/H2D,
finite checks, checkpointing, AdamW and EMA did not yield an isolated,
strictly-equivalent material optimization. No active job or scientific
contract was changed. Authority:
`experiments/pcqm_gptrans_step_profile_v5/results/decision_122499114.md`.

## Data and comparison contract

- PCQM4Mv2 fixed 100K, 500K, 1M, and full identities are accepted under
  `platforms/_records/ims/pcqm_fixed_datasets_v1/`; Kaggle mirrors are indexed
  under `platforms/_records/kaggle/`.
- New screens use one immutable baseline per scientific contract. A candidate
  may compare directly across platforms without retraining that baseline when
  data, row order, seed, FP32 mode, BS128, optimizer, schedule, loss, selection,
  exposure, and role access match and each runtime has one reusable calibration
  certificate. Authority: `experiments/SCREENING_POLICY.md`.
- Historical paired-v3 results retain their original contracts. The K1 500K
  result remains valid but its 32-row epoch tail prevents use as a new v4
  reference.
- A reused development score is selection evidence, not an unbiased final
  estimate. Official validation and test-dev remain sealed.

## Hard boundaries

- Track B predicts Gap directly on official PCQM4Mv2; Track A remains on the
  repaired-2M PubChemQC corpus. No silent dataset replacement or augmentation.
- No teacher/distillation or privileged geometry on the OGB leaderboard line.
- Any geometry used in training and inference must use the same ETKDG contract.
- Molecular-research-server access is separately gated by
  `platforms/REMOTE_HANDOFF.md` and limited to `/lustre/home/users/sm2/chou/`.
- Remote jobs require immutable cache acceptance, atomic checkpoints,
  retrievable outputs, and a dated decision. Scientific failures close a route;
  only infrastructure failures may retry unchanged.

## Pointers

| Question | Authority |
|---|---|
| What happens next? | `ROADMAP.md` |
| What ships? | `production/README.md` |
| Track meanings | `TRACKS.md` |
| Experiment index | `experiments/README.md` |
| Remote operations | `platforms/README.md` and `platforms/REMOTE_HANDOFF.md` |
| Code ownership | `ARCHITECTURE.md` |
| Artifact inventory | `models/README.md` |
