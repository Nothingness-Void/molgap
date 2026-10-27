# Prediction-only routing of K1 and GPTrans PairNorm

Prospective 2026-10-27 CPU diagnostic. Old PCQM prediction-only routing
improved 0.279 meV, below its 1 meV nomination gate; its hard gate regressed.
The new question concerns the subsequently accepted GPTrans PairNorm 500K
predictions, not a retry of the earlier model pair. Fixed three-way fusion is
0.096301 eV on this already model-selection-used development role. The 50K
labels and three prediction files are exactly the hash-pinned artifacts in
`pcqm_k1_gptrans_pairnorm_fusion_500k/analysis.json`.

Check hashes, source rows 500000..549999, identical labels and finite values.
No backbone training, backbone inference, accelerator or protected roles.
Do not introduce graph metadata, embeddings, hyperparameter search or source
labels into inference features. Evaluate two predeclared groups: K1 versus
GPTrans 1e-3, and K1 versus the average of the two PairNorm GPTrans arms.

Reuse `pcqm_expert_oracle_feasibility`'s five source_idx modulo 5 cross-fits,
prediction-only features (both predictions, signed/absolute difference, mean),
weighted-median global L1 weight, logistic winner classifier, and the same
small absolute-error HistGradientBoosting convex gate. For group two, compare
with both the fitted global weight and the fixed equal three-model mean. Each
row's gate prediction is made by a gate trained on the other four folds.
Use existing paired row bootstrap, 2000 draws, seed 20261027. Report per-fold
effects, pooled MAE, winner AUC, discrete Oracle, mean model passes and CPU cost.

Diagnostic nomination: soft gate at least 1 meV better than cross-fitted
global weight, paired 95% interval favorable, and 4/5 folds improve. This is
not V5 promotion. No automatic next training or role access is authorized.
Cross-fitting does not undo backbone selection and extensive development-set
reuse. Oracle uses targets and is not deployable; both experts already run.
