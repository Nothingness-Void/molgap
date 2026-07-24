# Original-1M Residual-Target Acquisition Plan

## Objective

Keep the original 1M dataset unchanged and acquire new PubChemQC rows around
the original dual-GPS model's largest fixed-external residual regions. This is
not a continuation of the rejected repair-v2 bucket design.

## Residual evidence

The original 1M common-eval Gap MAE is `0.11919` eV. The strongest intersecting
regions are:

| Region | N | Original 1M Gap MAE | Rejected 1.5M minus 1M |
|---|---:|---:|---:|
| high-sp3 and MW > 700 | 27 | 0.28771 | +0.01424 |
| macrocycle and MW > 700 | 23 | 0.28113 | +0.00342 |
| non-aromatic and MW > 700 | 25 | 0.27487 | +0.00188 |
| MW > 700 and Gap 2.5--4.0 | 64 | 0.17758 | +0.01492 |
| high-sp3 and non-aromatic | 145 | 0.17626 | +0.00939 |
| flexible and non-aromatic | 73 | 0.16967 | +0.00965 |

The worst 200 molecules enrich macrocycles `3.8x`, non-aromatic molecules
`2.2x`, high-sp3 molecules `1.8x`, and MW > 700 molecules `1.5x` relative to
the full diagnostic set. This differs from the previous aromatic-edge-heavy
acquisition hypothesis.

## Acquisition

Kaggle kernel `nothingnessvoid/molgap-residual-target-fetch` scans four durable
groups per 60K round:

- high-sp3/non-aromatic/flexible molecules above MW 700;
- very-large macrocycles and multi-amide molecules;
- flexible molecules with B3LYP Gap 2.5--4.0 eV;
- high-sp3 non-aromatic and moderate balanced controls.

Every group writes an independent CSV, log, and atomic progress JSON. It
excludes the complete 1.5M table and the prior repair candidate union by CID and
canonical SMILES.

## Gates

1. Build a candidate pool; reserve a scaffold-disjoint sealed subset first.
2. Select a 50K additive pilot on top of the original 1M.
3. Continue to 100K only if all development slices improve without a material
   HOMO/LUMO regression.
4. Consider 500K only after the 100K model passes a newly sealed scaffold set
   and the independent PCQM proxy.
5. Do not allocate 3D before the pure-2D gate passes.

The existing 1,977 common/OOD/P8-hard molecules have now informed acquisition
and are development data, not a sealed promotion test.
