# Kaggle1 GPTrans-T / centered-logits paired 100K terminal decision

Reference MAE: 0.156014488 eV. Centered-logits MAE: 0.155416209 eV. Gain: 0.000598279 eV. Paired row bootstrap 95% CI for candidate minus reference: [-0.0016655688993930816, 0.00046157600283622637]. Shortlist gate: FAIL (requires gain >= 0.003 eV and CI upper < 0).

Each arm completed 60 epochs with verified local artifacts and separate measured single-T4 epoch intervals. The reference is a same-job control. The row bootstrap does not measure training-seed variation. Replay eligibility remains false: `strict_replay_reference_binding_unavailable_under_frozen_prospective`; strict historical reference binding was not established under the frozen prospective trajectory. No 500K or full-scale release follows.

Evidence: `results/paired_comparison.json`, `results/reference/mechanical_acceptance.json`, and `results/centered_logits/mechanical_acceptance.json`.
