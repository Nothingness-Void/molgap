# K1 decay timescale: frozen protocol

Design dated2026-11-02. One same-run seed42 optimizer comparison; no architecture or objective change.

## Evidence and literature

RML `levers --family gptrans`, `levers --family k1` and `lookup --mechanism k1 decay --compact` were read before planning. The query found no indexed K1 decay comparison; this is not proof that all unindexed history is absent. GPTrans WD0.05-to0.01 was adverse at59/60 points: [decision](../pcqm_gptrans_decay_clock_100k/gpu/results/interpretation.md). It is not reopened. Triplet models selected before the final pass while train error kept falling: [analysis](../pcqm_gptrans_triplet_communication_100k/gpu/results/interpretation.md). These motivate testing regularization, not adding capacity; their behavior does not establish K1's cause.

Wang and Aitchison, ICML2025, [full text](https://arxiv.org/html/2405.13698v3), sections3.1 and4.1: AdamW retention timescale depends on LR times WD; its scale-invariant equivalence assumes initialization/architecture conditions not established here. Their image/language experiments are not molecular validation. With LR held at4e-4, WD1e-4 vs1e-3 changes the peak nominal decay timescale from25M to2.5M steps. Both exceed this31,240-update screen; cumulative shrink may therefore be weak. The study can reject a useful coefficient effect, not establish the paper's scaling law.

## Frozen contrast and budget

Use existing post-BN K1/3 preset, same frozen seed42 initialization, accepted fixed100K and CommonDev50K, pure2D, direct normalized Gap L1, FP32/noTF32, physicalBS128/drop_last. Keep the preset's40-pass/31,240-update/3,998,720-presentation horizon, LR4e-4 cosine and selected-live policy unchanged. Only decay differs: reference wd1e4=1e-4, candidate wd1e3=1e-3. No claim against an old K1 scalar or GPTrans. These are two previously unregistered coefficient arms, not a repeat of a completed baseline. One T4 per independent worker; maximum6 session hours/12 allocated physicalT4hours includes setup, evaluation and checkpoint reserve. No automatic retry/successor/scale-up. Atomic checkpoints and independently retrievable output chunks are mandatory.

## Outcomes and RML

Prospectively bind both arms as a same-run pair. Require observed roles/file-open provenance, target-transform asset, full trace, exact row/prediction alignment, native proofs and allocation ledger. At terminal use exact-version retrieval, draft-closure and strict accept_run; missing evidence stays pending, never fabricate Replay admission. Retain selected and final predictions/checkpoints and step/presentation/time curves. Candidate-minus-reference selected MAE must be<=-0.003eV, paired-row2000-bootstrap95% upper bound<0, final-live delta<0 and last-five observation mean delta<0 to retain a material lead. Other finite results are below-gate/negative/inconclusive as measured. Row intervals are not seed uncertainty. No protected roles, geometry, teacher, ensemble, extra seeds,500K or full. Stop on frozen wall budget without extending a stalled or incomplete arm.
