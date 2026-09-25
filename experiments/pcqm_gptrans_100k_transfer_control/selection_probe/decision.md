# Raw-versus-EMA selection diagnostic

The prospective train-free probe completed on the accepted local GPTrans 100K
pair. Both raw and EMA states are from epoch 59, with unchanged source,
checkpoint identities, data, target transform, and model architecture. It read
only the already-consumed 100K and 500K internal development roles. The
result and input checks are in `result.json` and `acceptance.json`.

| Frozen evaluation cohort | Raw paired gain | EMA paired gain | EMA minus raw |
|---|---:|---:|---:|
| 100K internal development | 0.004532 eV | 0.006185 eV | 0.001653 eV |
| 500K internal development | 0.004348 eV | 0.005792 eV | 0.001444 eV |

Under the frozen rule, both selection effects fall between 0.001 and 0.003
eV: a weak but nonzero contribution, not a material explanation by itself.
Both raw gains remain above 0.004 eV. Thus the local candidate's advantage
does not depend on EMA selection, and changing evaluation weights alone does
not reproduce the historical 500K contraction.

The historical 100K/500K comparison also changes data size, optimization
recipe, initialization provenance, and selection semantics. The 500K
development cohort was previously used for model selection; its reuse here
is a diagnostic, not an independent test. Row-bootstrap intervals describe
row variation, not training-seed variation. No 500K training, promotion, or
protected-role access is authorized by this result. The exact 500K training
mechanism remains unresolved.

The existing RML traces give a second, no-training observation. Under the
matched 500K V4 recipe, the joint run's **live** development lead over the
GPTrans-T reference was 0.018467 eV at 39,060 optimizer steps (epoch 9),
but 0.002067 eV at 234,360 steps (epoch 59). The local matched 100K EMA lead
also fell from 0.013806 eV at epoch 32 to 0.006185 eV at epoch 59. These
within-contract curves support training-time erosion of an early advantage;
they do not isolate data size from optimization or prove a causal mechanism.
The authoritative traces are indexed by the RML replay pool under
`TB-matched-500k-v4-three-arm` and
`TB-gptrans-noisy-pair-norm-500k-s42`; the local curve is in
`../analysis_result.json`.
