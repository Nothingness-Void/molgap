# GPS++-style pure MPNN, 100K, seed 42: closure decision

Pre-registered rule (effect = candidate minus control, EMA dev MAE):
<= -4.6 meV nominate; -4.6 to -1.4 run seed 43; > -1.4 close.

Observed (same job, same dev role):
- EMA 0.9999 dev MAE: candidate 0.14382 eV, control 0.14337 eV, delta +0.46 meV.
- Point-selected dev MAE: candidate 0.14253 eV, control 0.13944 eV, delta +3.09 meV,
  row-bootstrap 95% CI [+2.00, +4.18] meV (2000 samples).
- Both readouts are above -1.4 meV, so the rule gives CLOSE. The candidate is not better.

Outcome: gpspp_mpnn = NEGATIVE_UNDER_CONTRACT. pair_update_norm is the same-job control: CLOSED.

Notes:
- Runtime certificate scope differs from the registered scope; closed on the pre-registered
  non-strict path 4.1(b).
- The init hash named in the candidate recipe was not used by the candidate.
- No seed 43 repeat, no no-background ablation, no 500K run: the gate was not met.
