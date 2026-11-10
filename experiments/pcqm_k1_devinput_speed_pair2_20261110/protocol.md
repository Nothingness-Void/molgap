# Device-resident inputs: same-machine Kaggle T4x2 throughput

Infrastructure throughput measurement for the opt-in `device_resident_train` launch field, now a list of arm ids
(commit 49283cdf plus the per-arm fix). One kernel, two identical K1 Muon 100K arms from
../pcqm_k1_soap_100k_20261106 (recipe unchanged): `muon_off` on GPU 0 and `muon_on` (device-resident) on GPU 1,
running concurrently and paused after optimizer step 1562 (two epochs). Replaces the cancelled twin experiments
`..._off_20261110` / `..._on_20261110`, which compared two accounts.

Compare: per-arm allocation_costs.json wall seconds and the per-epoch train loss. Train loss must be identical at
matching epochs (same seed, same recipe); a mismatch invalidates the device-resident path and no speed claim is made.
Both arms share CPU cores and PCIe, so the absolute times are not solo times; the comparison is symmetric.
No scientific acceptance, no protected roles, no successors.

This replaces pcqm-k1-devinput-speed-pair-20261110 (attempt v1): muon_on was rejected at startup because the fixed PCQM graphs carry split_code, smiles_sanitized and mmff_converged, which the device-resident store did not list (fixed in 56ad2dae). muon_off of that run trained alone, so its timing is not comparable.
