# Muon cost and500K readiness

2026-11-06 JST. Supplementary saved-artifact analysis of job142927073/version1.
The accepted decision and finalized RML records are unchanged. Recommendation:
prioritize one matched500K AdamW/Muon pair; budget the actual implementation,
not language-model throughput claims. This document authorizes no new compute.

## Evidence added

Only two formal-training profile.json files and two train.log files were fetched
at exact version1, through existing selective transports. Native qualification's
two-update profile is distinct from these formal16-update profiles. No new model
execution, data export, checkpoint modification or training occurred.

Raw files and computed summary:
`D:/molgap-artifacts/reviews/k1-muon-cost-scale-20261106/`.
The retained final optimizer checkpoint contains116Muon matrix tensors and197
AdamW tensors. This inspection deserialized saved state, without constructing or
executing a model. The declared model and inference path remain unchanged.

## Why process time doubles

Formal training profile: warmup120updates, capture16updates, Torch2.11/cu130,
native Tesla T4. These are local profile windows, not full-run stage totals.

| Phase | AdamW host seconds/update | Muon host seconds/update |
| --- | ---: | ---: |
| forward, inclusive | .071816 | .063773 |
| backward | .100191 | .092047 |
| optimizer | .008444 | .228896 |
| train_update, inclusive | .191075 | .396326 |

Do not add inclusive parent and child spans or mix host timing with device
operator timing. The optimizer span grows27.11x while forward/backward do not
grow. This directly supports optimizer work as the dominant added cost.

The implementation loops over116matrices. Each of five Newton-Schulz iterations
uses one matrix product and two addmm calls:15matrix operations per tensor,
1740per update. Across the matched16-update capture, mm calls grow4416->13696
and addmm1280->19840; the differences equal116*5 and116*10 per update.
These include many small FP32 launches and serial Python dispatch. Device
operator records show SGEMM kernels. The evidence identifies the costly path;
it does not separately estimate launch latency, arithmetic and scheduling shares.

The T4 FP32 peak is8.1TFLOPS; its advertised65TFLOPS mixed-precision path is a
different arithmetic contract ([NVIDIA](https://www.nvidia.com/en-us/data-center/tesla-t4/)).
Our all-FP32/noTF32 update does not inherit a BF16-based implementation's cost.
The [Muon paper](https://arxiv.org/html/2502.16982v1) reports small optimizer
overhead in its large-model environment; that claim does not describe this small
graph model's optimizer-heavy measured trace.

Full-run process walls are5499.850894s/11177.999007s,2.0324x. Epoch logs provide
timing absent from canonical traces: pass11-40 median134.818865s/253.960267s,
1.8837x. First-ten-pass totals1401.767296s/3123.508170s account for the higher
whole-run ratio. Shared-host contention and changing execution conditions are
possible contributors; the retained telemetry does not uniquely attribute them.

Muon first reaches AdamW's final development quality at pass21,elapsed6319.157s;
AdamW's final epoch elapsed5456.543s. This is1.158x observed wall, so the measured
benefit is accuracy at equal presentations, not faster time to that target.

## Does the gain plausibly transfer?

The complete100K gate passes with8.773793meV gain (6.21%relative), conditional
row interval7.861417-9.708100meV in the improvement direction. Both models select
pass40. No additional encoder, teacher or output-state substitution explains
the contrast: only the fixed optimizer package changed.

Matched development gains decrease from28.112695meV at pass10 to17.119363 at20,
10.847360 at30 and8.773804 at40. The last-five mean gain is8.487302meV.
Thus faster early learning is established, and a large endpoint advantage remains;
additional exposure could still let AdamW narrow the difference further.

Saved-prediction post-hoc physical-Gap bins all improve: <4eV16.576meV,
4-6eV8.657meV,6-8eV6.982meV,>=8eV2.417meV. This is broader than one favorable
target slice, but does not qualify a new cohort. Development members change
between the accepted100K and500K contracts.

Prior PairToken500K and retained-pretraining500K results show that small or early
100K gains can fade or reverse; the [scale review](../pcqm_k1_scale_attribution_20261025/analysis.md)
and [initialization attribution](../pcqm_k1_initialization_forward_500k/terminal_acceptance/attribution.md)
own those facts. They do not answer Muon's distinct question.
[Fantastic Pretraining Optimizers](https://proceedings.iclr.cc/paper_files/paper/2026/file/ea4f0a6ad7088c9b93f22d74c0c9e8c2-Paper-Conference.pdf)
also observes horizon-dependent rankings and sensitivity to data/model ratio.
Those language-model results motivate the discriminator; they do not predict
a molecular500K MAE or a transferable fraction of8.77meV.

## Proposed next question and budget

Train two fresh identical original K1 models from the same original random state:
AdamW versus the accepted FP32 Muon/AdamW package. Pure2D, one supervised forward,
BS128, same initialization/seed/BN/roles, full60-pass cosine horizon, identical
selection of best development Live; track EMA separately. No SwiGLU, RnC or
pretraining combination. Use fixed500K TRAIN[0,500000)/development[500000,550000)
and its existing train-only transform/data/recipe owners. Historical500K references
are contextual checks unless their exact output-state/recipe qualification matches.

60passes mean234360updates/29998080presentations. Proposed material gate:
candidate improves selected MAE by>=3meV with paired95%upper delta<0, and final
and last-five differences remain favorable. Record selected, final and full curves,
native process/allocation costs and two independent replay entries. Freeze this
new question prospectively on a new desktop-owned branch before execution.

Two simple extrapolations (steady-pass scaling and full-run step scaling) give:

|500K horizon | AdamW wall estimate/h | Muon wall estimate/h |
| --- | ---: | ---: |
|30passes | 5.62-5.73 | 10.58-11.65 |
|40passes | 7.49-7.64 | 14.11-15.53 |
|60passes | 11.24-11.46 | 21.17-23.29 |

These are arithmetic planning estimates, not measured500K costs or confidence
intervals. They scale fixed development/IO work imperfectly; setup, platform
variance and resumption add uncertainty. A concurrent T4x2 allocation lasting
the slower60-pass arm implies roughly42-47allocated T4-device-hours before extra
setup/recovery. Pre-plan bounded durable continuation segments or use a faster
native-qualified device. A30-pass endpoint would answer a different horizon.

Equivalent FP32 batching of same-shaped matrices could reduce Python/launch cost,
but no speedup is measured and no such change is made here. Keep any execution
optimization separate and equivalence-qualified. The500K scientific question is
worth testing without making another infrastructure expansion a prerequisite.
