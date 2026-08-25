# PubChemQC 100K Architecture Transfer

**Question:** Do the architectures selected cheaply on QM9 transfer to
scaffold-disjoint PubChemQC molecules?

**Verdict:** The bounded two-SchNet precision fusion transferred at 100K scale.

Read `results/route_b_fusion_decision.md` first. Head and residual-bound A/B
decisions are adjacent; `results/experiment_manifest.json` owns artifact paths
and acceptance records.

## Isolated PairGPS2D question

The later pure-2D PairGPS2D branch used a separate matched validation-only
contract against GPS7 plus GPS9 equal. It passed that stage while the test role
remained sealed. Its A100 train-role throughput benchmark selected a bounded
configuration without authorizing long training. Read:

- `results/pair_gps_2d_fair_screen/decision.md`
- `results/pair_gps_2d_a100_benchmark/decision.md`

The complete immutable IMS logs and remote code snapshot are indexed under
`platforms/_records/ims/README.md`.
