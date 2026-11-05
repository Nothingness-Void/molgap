# A/B/C review — node FFN gating

Reviewed desktop `5e18aba3b023f75002ba8eddda69b3efbeab0a1d` and server
`cbc5bd80` on2026-11-06. Cards/lookup are navigation; decisions own outcomes.

## A — proposal

Replace K1's nine `192→384→192` SiLU node MLPs with two256-wide SwiGLU branches
and a256→192 projection. Both have147,456 primary matrix weights/MACs per node
per layer. Keep edge-state updates, local convolution, slot mixers, normalization,
residual order, prediction head, input roles and training recipe unchanged.
The hypothesis is better conditional channel mixing under a matched matrix budget.
It does not assume missing geometry, underfitting or proven FFN bottlenecks.

## B — paper-based objection

[GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202), Sections1–3,
studies T5 language pretraining/fine-tuning, omits FFN bias and reduces the GLU
hidden width by2/3. Pretraining has no dropout; fine-tuning uses dropout. This
supports an architectural hypothesis, not a PCQM accuracy or throughput promise.
Keeping K1 bias/dropout is a controlled project adaptation, not paper reproduction.

Dropout position must be explicit: apply hidden dropout after the gated product
and output dropout after the output projection. Adding independent dropout to
both branches would introduce a separate regularization change. The narrower
hidden representation also changes mask shapes; equal seeds do not imply equal
per-module training RNG trajectories across different architectures.

Different matrix shapes cannot generally preserve the original FFN function or
all original tensors. Preserve untouched modules' frozen initialization and define
new FFN initialization. Matrix budget equality does not determine wall time or
peak memory. The packed projection avoids an extra GEMM dispatch but still does
not establish an actual speedup.

## C — decision

Prepare the narrowly defined addon and matched recipe; defer model execution.
Choose the recent strictly qualified original K1 reference, not the older width192
custody-only reference. Retain exactly A/B as scientific arms and reuse A wherever
the eventual execution/comparison contract permits. Do not retrain a baseline or
invent another route to fill physical capacity. The parameter constraint is met
by source arithmetic; measured cost and runtime qualification remain pending.

## Prior evidence and non-duplication

RML `cards --ref origin/molgap-server --term swiglu`, `--term glu` and `--term ffn`,
plus targeted Markdown/trajectory/input and server source searches, found no
completed or declared K1 node SwiGLU intervention in the reviewed committed state.
This does not inspect another agent's uncommitted work or live scheduler state.

The server's `TC-gptrans-g1-ffn2-100k-s42` is INCONCLUSIVE and changes GPTrans
FFN expansion, not this K1 gate. Its decision remains at
`origin/molgap-server:experiments/pcqm_gptrans_capacity_nodes_100k/gpu/degree_ffn2_ema999/results/decision.md`.
Server `modern2d_model_integration/integration_cpu_report.json` mentions a SwiGLU
substitute inside ESA fidelity checks; that is not a K1 SwiGLU quality trial.

The [K1 message-DropPath terminal decision](../pcqm_k1_message_droppath_100k/terminal_acceptance/decision.md)
rejects graph-wise aggregate dropout and retains a qualified original control.
It is a source for the unmodified reference, not positive evidence for this gate.
The server identity-perturbation study
`origin/molgap-server:experiments/k1_perturbation_floor_100k/results/completed/decision.md`
shows a small architecture/initialization perturbation can move a single endpoint;
it supplies no transferable seed-noise threshold. Use the frozen material gate,
paired rows and honest single-seed limits, rather than declaring sub-meV changes
stable. Closed K1 norm/DropPath/tail-WD routes are not reopened by this proposal.
