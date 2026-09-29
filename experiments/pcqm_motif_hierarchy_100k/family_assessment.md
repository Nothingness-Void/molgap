# Two 2D family decisions — 2026-09-29

This compares the two server-owned directions discussed with the coordinator;
it is not a cross-family MAE ranking or a release of another experiment.

## Independent MetaGIN2D backbone

The accepted fixed-100K seed-42 adaptation was 4×256 with three-hop local
propagation, 5,268,481 parameters. Its row-aligned internal-development MAE
was 0.164187572 eV versus frozen K1-v4 0.141373641 eV (candidate minus
reference +0.022813930 eV, paired 95% interval entirely unfavorable).
[The terminal decision](../pcqm_metagin_2d_100k/attempt_v2/decision.md) closes
this exact adaptation. Its 43.472% row win rate does not rescue global MAE;
the post-hoc hardest K1 subset uses labels and cannot authorize routing.
Do not submit width/depth/hop, seed, scale, or full-run repeats under this
family label. Published MetaGIN is not generally falsified; a genuinely new
mechanism requires its own distinct evidence and prospective decision.

## K1 + topology-defined motif graph

The [CPU decision](decision.md) established a cheap, source-aligned pure-2D
sidecar over 150,000 fixed-cache graphs: 99,011/100,000 training molecules had
at least two motifs. This is coverage and execution feasibility only, not a
predictive win. The K1 layer-6 additive motif exchange is separately frozen
in [the GPU protocol](gpu_protocol.md); v1/v2 failed at runtime gates before
model work. T4-only v3 is the first attempt capable of answering the model
question. No extra motif variant, seed, or scale study is justified while its
terminal evidence is pending.

The reusable path already exists: deterministic motif construction and
acceptance in `src/molgap/pcqm_motif_partition.py` and
`src/molgap/pcqm_motif_sidecar.py`; the candidate mechanism in
`src/molgap/k1_motif_hierarchy.py`; the existing K1 trainer in
`src/molgap/pcqm_k1_variants_runner.py`; reference-bound prospective release
and source packaging via the shared experiment/RML components. The T4 runtime
selection accepts one or two actual T4s but exposes only one to this arm,
recording the entire allocation. Do not copy a new trainer, source packager,
or monitor for another motif variant. If v3 passes saved-artifact acceptance
and its declared 0.003 eV gate, the only eligible discussion is a separately
released frozen-checkpoint 500K NO_TRAIN portability audit. If it does not,
close this exact exchange and retain the sidecar as an infrastructure asset.
