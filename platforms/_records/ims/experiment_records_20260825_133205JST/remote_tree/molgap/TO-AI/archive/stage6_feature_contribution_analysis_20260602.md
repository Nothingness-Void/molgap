# Stage 6 Feature Contribution Analysis Archive

Archived at: 2026-06-02.

## Implemented file

```text
src/12_feature_contribution_analysis.py
```

## Run command

```bash
python src/12_feature_contribution_analysis.py
```

## Outputs

```text
results/feature_contribution/feature_importance_lightgbm_gain_split.csv
results/feature_contribution/feature_importance_summary_by_target.csv
results/feature_contribution/feature_group_importance.csv
results/feature_contribution/top_features_overall.csv
results/feature_contribution/top_rdkit_descriptors.csv
results/feature_contribution/top_morgan_bits.csv
results/feature_contribution/target_feature_overlap.csv
results/feature_contribution/permutation_importance_lightgbm_sample.csv
results/feature_contribution/ridge_coefficient_importance.csv
results/feature_contribution/top_features_homo.png
results/feature_contribution/top_features_lumo.png
results/feature_contribution/top_features_gap.png
results/feature_contribution/top_features_overall.png
results/feature_contribution/feature_group_importance.png
results/feature_contribution/rdkit_descriptor_importance.png
results/feature_contribution/permutation_importance_top.png
```

## Key result: feature group contribution

LightGBM normalized gain contribution averaged over HOMO/LUMO/gap:

```text
Morgan bits     : 0.0925  (~9.3%)
RDKit descriptors: 0.9075 (~90.7%)
```

Split-count contribution is less extreme but still favors RDKit descriptors:

```text
Morgan bits      split contribution: ~28.5%
RDKit descriptors split contribution: ~71.5%
```

Interpretation:

RDKit descriptors dominate the model's gain-based feature contribution. Morgan bits contribute, but they are not the main driver of predictive performance.

## Top overall LightGBM gain features

Top features by mean normalized gain across HOMO/LUMO/gap:

```text
1. desc_HallKierAlpha
2. desc_FractionCSP3
3. desc_BCUT2D_MRLOW
4. desc_SMR_VSA7
5. desc_VSA_EState2
6. desc_BertzCT
7. desc_BCUT2D_MRHI
8. desc_SlogP_VSA8
9. desc_VSA_EState4
10. desc_fr_nitro
```

All top overall features are RDKit descriptors.

## Permutation importance sanity check

Permutation importance was run on the top 50 LightGBM gain features, using a random test-set sample of 1000 molecules and 5 repeats.

Top permutation features by average R2 drop:

```text
1. desc_HallKierAlpha       drop=0.0811
2. desc_BCUT2D_MRHI         drop=0.0464
3. desc_VSA_EState2         drop=0.0438
4. desc_SlogP_VSA8          drop=0.0317
5. desc_FractionCSP3        drop=0.0261
6. desc_BertzCT             drop=0.0212
7. desc_SMR_VSA7            drop=0.0169
8. desc_MinPartialCharge    drop=0.0163
9. desc_BCUT2D_MRLOW        drop=0.0140
10. desc_SlogP_VSA1         drop=0.0110
```

Permutation importance supports the LightGBM gain conclusion: RDKit descriptors dominate. Only one Morgan bit appears near the top of the permutation list (`morgan_715`, rank 17 in the shown output).

## Target overlap

Top-30 feature overlap:

```text
HOMO vs LUMO: 13 overlapping features, Jaccard=0.2766
HOMO vs Gap : 6 overlapping features,  Jaccard=0.1111
LUMO vs Gap : 16 overlapping features, Jaccard=0.3636
All three targets: 4 common features
```

Common top features across HOMO/LUMO/gap:

```text
desc_BCUT2D_MRLOW
desc_SMR_VSA10
desc_SMR_VSA7
desc_VSA_EState2
```

Interpretation:

Gap shares more important features with LUMO than with HOMO, but it also has target-specific drivers. This is consistent with previous results showing gap is the hardest target to generalize.

## Ridge coefficient importance

`ridge_coefficient_importance.csv` was generated successfully. This mirrors the previous project's coefficient-based feature contribution analysis and provides a lightweight linear-model perspective.

## Main conclusion

The feature contribution analysis strongly supports the earlier benchmark:

```text
RDKit descriptors are the core predictive feature family.
Morgan fingerprints provide supplementary but smaller contribution.
```

This suggests the next model攻坚 step should prioritize descriptor selection and compact descriptor subsets before doing heavy genetic algorithms or embeddings.

## Recommended next step

Run descriptor-focused feature selection benchmark:

```text
src/13_descriptor_selection_benchmark.py
```

Suggested comparison:

```text
all_rdkit_descriptors
top_20_rdkit_descriptors
top_50_rdkit_descriptors
top_100_rdkit_descriptors
top_150_rdkit_descriptors
morgan_plus_top_50_rdkit
morgan_plus_top_100_rdkit
```

Use LightGBM and both random/scaffold splits. If top-k descriptors perform close to all descriptors, then a small GA over RDKit descriptors may be worthwhile later.
