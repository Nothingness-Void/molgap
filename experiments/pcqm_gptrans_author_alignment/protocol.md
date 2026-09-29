# GPTrans author/local attribution matrix — frozen 2026-09-29

## Question and evidence

Which verified differences between the published GPTrans-T recipe and the
adapted MolGap core are plausible contributors to the observed Gap error,
without changing the PCQM4Mv2 database or conflating implementation and MAE
evidence? The synthetic/Cython CPU parity result in
`../pcqm_gptrans_parity_cpu/decision.md` proves several non-equivalences, not
their accuracy effects. The author paper reports 300 epochs and total batch
1024; those are not the accepted MolGap 100K discovery contract.

## Matrix and release order

| Stage | Fixed intervention | Read set | Decision capability | Release gate |
|---|---|---|---|---|
| P0 | Label-free shortest-path bond-signature prevalence on deterministic train-only rows | Accepted Kaggle2 fixed PCQM 100K train shards; no labels or checkpoint | Establish whether the omitted path information actually varies among real molecules | Exact manifest and shard hashes, source-row identity, isolated CPU runtime, independently retrievable per-shard outputs |
| G1 | One author-inspired input-initialization scale intervention | Fixed 100K train/internal-dev, seed 42 | Single-factor paired accuracy effect against immutable GPTrans V4 reference | New initialized-state identity, otherwise exact V4 scientific fingerprint, optimizer-step calibration, T4 preflight and prospective RML |
| G2 | Path-edge input only, with no other model/optimizer changes | Same fixed 100K roles | Single-factor paired accuracy effect against that same immutable reference | Accepted CPU path sidecar and exact feature identity; separate model, RNG, optimizer, checkpoint, one visible T4, preflight and prospective RML |
| O1 | Author weight-decay grouping or raw-target loss | Same database, separately frozen contract | Optimization-contract effect | A new matched baseline is required for each changed contract; not released by G1/G2 |
| S1 | Frozen 500K inference or matched 500K training | Accepted fixed 500K assets only | Portability or scale response, respectively | Separate prospective role/cost decision after accepted 100K terminal; no automatic successor |

P0 is a preflight of input availability, not an independently selected model.
Its deterministic BFS signature is not represented as byte-identical author
Floyd-Warshall/Cython output when shortest-path ties exist. G1 and G2 are
independent *candidates*, not a factorial interaction estimate. Their combined
effect is not claimed. Architecture discovery uses one paired seed 42,
physical batch 128 per independent model/device, FP32/no TF32, no
accumulation, fixed 60 epochs/4 warmup, identical target transform and best
internal-development EMA selection. No official validation/test roles.

The CPU preflight only authorizes further engineering if its input checks pass;
it cannot release a GPU run by itself. Before G1/G2 submission, freeze the
exact source archive, reference bundle and comparison prelaunch, data/role
identities, native T4 budget, and per-arm RML trajectory. The author-path
sidecar must be made on CPU, then separately accepted before G2 uses it.
Only accepted candidate training can trigger a separate 500K NO_TRAIN
portability plan; no more seeds, desktop full run, or protected role follows
automatically. A positive 100K endpoint is a nomination, not official
reproduction of the paper's 0.0833 eV validation score.
