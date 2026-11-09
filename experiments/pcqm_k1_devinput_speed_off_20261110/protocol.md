# Device-resident inputs: Kaggle T4 throughput (off)

Infrastructure throughput measurement for the opt-in `device_resident_train` launch field
(commit 49283cdf). The paired Muon/SOAP 100K recipe of ../pcqm_k1_soap_100k_20261106 is reused
unchanged; the job pauses after optimizer step 1562 (two epochs). This run has
`device_resident_train=false`; its twin run has the flag flipped.

Compare: allocation_costs.json wall seconds and the per-epoch train loss of the `muon` arm
across the two runs. Train loss must be identical at matching epochs; a mismatch invalidates the
device-resident path. No scientific acceptance, no protected roles, no successors.
