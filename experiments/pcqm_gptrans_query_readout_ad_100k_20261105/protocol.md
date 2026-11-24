# Protocol — dynamic molecular relation readout / AD

## Question and authority

User authorized two T4x2 notebooks, four seed42 arms, fixed100K complete60 epochs.
This notebook compares full12 (GPU0) with full8_query4 (GPU1). The companion
notebook answers a different matched contrast. Cross-notebook scores are descriptive.
Both reference arms are explicitly authorized controls, not historical-reference repairs.
No extra experiment total time cap was requested. The declared43200s allocation
and24 T4-device-hour fields encode the12h platform session boundary. Pause/resume
uses original source, exact cursor and reconciliation; never duplicate unknown work.
No local queue trigger, persistent monitor, official validation or test access.

## Scientific delta

A=12 full BondLocal+fresh-pair-update-norm blocks. B=first6 full blocks and original head.
C=first6 full blocks then original blocks7–12 as query-only readout.
D=first8 full blocks then original blocks9–12 as query-only readout.
All start with the same frozen12-layer initialization, including identical head,
overlapping layer weights and original12-layer DropPath probabilities. No depth
rescaling, new random initialization, changed cache or graph schema.
Query memory M contains split-point graph token and real atoms, stays unchanged
within the forward, and is never detached. Each readout layer applies its own
original LayerNorm and K/V projection to M; Q and the query-to-memory pair row evolve.
Column0 remains the split-point graph-token key, not the evolving query. Preserve
the two node messages, node-to-pair update, pair-to-node readback and key padding.
PairNorm means pair += normalize(new_update) over channels at each pair, never
normalize(pair+new_update). Readout layers omit BondLocal; that omission is part
of the complete architecture intervention, not a single-attention causal claim.
The unchanged head receives concat(query[:,0], pair_row[:,:,0,0]). No TopK or K/V cache.

## Recipe, data and evidence

reference_recipe.json explicitly pins BS128 drop_last, FP32/noTF32, LR0.001,
WD0.05,4-pass warmup/cosine1e-6, gradient clip1, EMA0.999, seed42 pass-randperm,
46860 optimizer updates /5998080 presentations /60 complete epochs. No auxiliary
EMA; same choice for all four arms. Train source rows[0,100000), development
[100000,150000), selected best-development EMA; official validation/test sealed.
Epoch cadence uses existing shared logging. Existing family sessions own native
repeat/resume qualification, checkpoints, curves, cost ledgers and terminal outputs.
Existing bounded profiler captures4 updates after2 warmup updates, including shapes
and memory. Instrumented windows are excluded from any clean throughput claim.
Report actual mature full-pass wall time (median passes10–60), throughput and native
cost separately. Peak device memory comes from measured profiler/runtime records;
missing whole-run peak stays unknown. No inference latency claim without a separately
identified measurement on authorized inputs.

## Gates

Primary AD: D-A selected EMA paired-row95%CI upper<=+1meV and point<=+1meV,
plus >=15% mature full-pass training-time saving. This is a prospective engineering
non-inferiority margin, not measured seed variance. BC: C improves B >=3meV with
paired95%CI upper<0. Use2000 paired bootstrap resamples on aligned finite50K rows.
BC alone cannot establish non-inferiority to A across notebooks. One training seed
does not measure training stochasticity. Full60-epoch curves, strict per-arm native
qualification, completion/exposure, selected artifacts, resume state, actual native
T4 cost and observed role use are required before terminal RML. No automatic500K.

## Prior evidence and non-duplication

RML exact-term lookup found no completed dynamic relation readout /6+6 /8+4 trial.
The historical architecture audit closes multi-depth/query-head interventions;
it does not test removing late propagation in this GPTrans BondLocal recipe.
experiments/pcqm_gptrans_500k_frozen_readout/decision.md tests static heads on
frozen encoders and explicitly leaves end-to-end attentive readout unresolved.
Pair-memory/conditional-flow negatives do not implement a single evolving query
with differentiable split-point memory. Latest triplet-ordering closure remains
negative and unchanged; this is not its successor gain/WD/EMA grid.

Literature: GPTrans https://www.ijcai.org/proceedings/2023/0396.pdf ;
CaiT https://arxiv.org/abs/2103.17239 (representation then class-query attention);
HySparse2 https://arxiv.org/html/2609.26368v1 is motivation only: no borrowed LLM
cache speedup numbers are treated as molecular measurements. An unprovided
synthetic equivalence claim is not qualification; local checks must execute.
