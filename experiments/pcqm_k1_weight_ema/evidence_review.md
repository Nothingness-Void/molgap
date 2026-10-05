# Prior evidence review - 2026-10-06

Desktop RML trajectory index and current canonical decisions/attributions were
read before this question. Original K1 width256 regressed; slot96 gained0.547meV
with bounds crossing zero; FLAG gained0.688meV at~2.92x step cost. Dropout
consistency gained1.529meV but missed3meV gate. Direct scalar-output distillation
is closed: strong gained0.433meV over consistency and lost3.630meV to fixed
fusion. [Dropout attribution](../pcqm_k1_dropout_consistency/terminal_acceptance/attribution.md)
and [distillation attribution](../pcqm_k1_fusion_distillation/terminal_acceptance/attribution.md)
do not identify underfitting/overfitting or need for extra exposure.

The accepted [slot/readout diagnostic](../pcqm_k1_slot_readout_diagnostic/terminal_decision.md)
does not establish negligible global return; widening is not repeated here.
Original retained clean reference custody and float32 target identity remain in
[reference binding](../pcqm_k1_slot_width96/reference_binding/reference_reuse_decision.json).
Its historical runtime is not substituted for the new paired T4 qualification.
The fresh reference is explicitly authorized and isolates the selection mechanism.

Cached server G1 evidence motivating the research report shows slow-EMA lag;
that is another family and source contract, not proof of K1 gain or live-weight
deficiency. Standalone K1 EMA .999 with copy-live BN buffers and EMA-only
40epoch selection is a new question. No teacher or archived negative module is
reopened. EMA arithmetic, buffer compatibility and stochasticity are alternatives
to a positive hypothesis, not pre-established causes. The cheapest falsifier is
the existing all-arm runtime barrier plus the small EMA resume mechanism check.

Primary research motivation: https://arxiv.org/html/2411.18704v1 (EMA recurrence,
normalization statistics and averaging-window interactions). No paper benchmark
score is used as a local promotion gate.
