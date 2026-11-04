# K1 same-init perturbation floor — 2026-11-04

The exact Kaggle2 kernel `kaseichou/molgap-k1-perturbation-floor-100k-s42`,
ID142629075/version1, completed both authorized arms. Both arms passed mechanical
verification with the observed protected-role records untouched. This record
reports a measurement defined in advance in `protocol.md`; it has no gate, no
promotion and no causal claim.

## Endpoint

Both arms trained the 40-pass horizon from the same frozen seed-42 initialization,
returned 0 from preflight and training, and wrote 40 trace observations each with
`learning_rate` and a distinct `checkpoint_identity`. The ledger records 6,840 s
wall and 13,680 allocated device-seconds (3.80 T4-hours).

| Arm | Selected development MAE, meV |
|---|---:|
| reference | 141.2945 |
| readout_scale (step-0 identity scalar gain) | 140.4866 |

The reference value equals the retained 141.2945 meV, so the rerun was
bit-identical and training is deterministic on this platform. Candidate minus
reference was -0.8079 meV with a 2,000-draw paired row bootstrap 95% interval of
[-1.7006, +0.0643] meV. The trained gain moved from 1.0 to 0.9558.

## Reading

The difference is below the 1.5 meV stop point named in the protocol. Because the
reference is bit-identical, the difference follows from the added gain parameter
and not from initialization noise. It is one sample, so it gives no spread
estimate; it is neither a confirmation nor a falsification of the claim that the
same-init floor is below 1.5 meV. The row interval measures molecule-level
uncertainty, not training-seed uncertainty, and the development cohort is reused
for selection, so absolute values are optimistic. For scale only: the 14 server
sub-gate K1 100K screens have mean -1.98 and SD 0.66 meV, and the init-only noise
has SD 1.074 and paired SD 1.519 meV.

## Outcome

The reference arm closes `CLOSED`. The readout_scale arm closes `INCONCLUSIVE`,
the label used for a finite candidate whose row interval includes 0 in a
single-seed screen; it is not evidence for or against the gain.

## Not claimed

No claim that the scalar gain helps, no floor estimate, no seed variance, no
successor, extra seed, 500K run or adoption. Protected roles were not used.
