# Decision

2026-11-26. shared_bond_ring is **NEGATIVE_UNDER_CONTRACT**. The same-job control self_ring is **CLOSED**. The candidate missed the pre-registered nomination gate.

The primary EMA at pass 60 produced 132.449 meV for the control and 132.362 meV for the candidate. Candidate minus control was **-0.087 meV**, paired 95% interval **[-0.951, +0.699] meV** over the fixed 50,000 internal development rows with 2,000 row-bootstrap draws. The rescored endpoints matched the native trace within 0.0001 eV. The required material improvement was 3.0 meV.

The pre-registered shared-bond stratum had 9,784 rows. Candidate minus control was +1.278 meV, 95% [-0.557, +3.068]. The remaining 40,216 rows had -0.419 meV, 95% [-1.358, +0.504].

The best primary EMA MAE was 131.959 meV at pass 43 for the control and 132.033 meV at pass 47 for the candidate. The selected-point change was +0.074 meV. Native give-back from best to pass 60 was 0.490 meV for the control and 0.329 meV for the candidate.

Both arms completed 60 passes, 46,860 updates and 5,998,080 sample presentations. The paired ledger measured 5.568 h of allocation wall time and 11.136 T4-device-h including allocated idle time. Queue duration was unmeasured. Local endpoint rescoring used only the registered development role.

[Registered protocol](../protocol.md) · [Endpoint analysis](pass60_analysis.json) · [Attribution](attribution.md)
