# Final Pure-2D Sealed Set

`sealed_10k.csv` is a one-time final acceptance set for the pure-2D multi-expert
experiment. It contains 10,000 distinct scaffolds selected from hard-acquisition
rounds 04-05 after excluding all CID, canonical-SMILES, and scaffold overlap
with the original 1M data, repair candidate union, and broad/residual pool.

The set must never be used for training, fitting ensemble weights, choosing a
Router threshold, or selecting among new model variants. See
`selection_report.json` for counts, buckets, and SHA256. The decision that first
opened it is `../multi2d_final_eval/decision.md`.
