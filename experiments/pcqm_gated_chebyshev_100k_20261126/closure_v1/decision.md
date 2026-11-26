# Decision

2026-11-26. chebyshev4 is **NEGATIVE_UNDER_CONTRACT**. The same-job control local_filter is **CLOSED**. The candidate missed the pre-registered nomination gate.

The primary EMA at pass 60 produced 161.629 meV for the control and 164.645 meV for the candidate. Candidate minus control was **+3.017 meV**, paired 95% interval **[+1.895, +4.111] meV** over the fixed 50,000 internal development rows with 2,000 row-bootstrap draws. The rescored endpoints matched the native trace within 0.0001 eV. The required material improvement was 3.0 meV.

The best primary EMA MAE was 159.074 meV at pass 38 for the control and 160.904 meV at pass 36 for the candidate. The selected-point change was +1.830 meV. Native give-back from best to pass 60 was 2.555 meV for the control and 3.741 meV for the candidate.

Both arms completed 60 passes, 46,860 updates and 5,998,080 sample presentations. The paired ledger measured 1.407 h of allocation wall time and 2.814 T4-device-h including allocated idle time. Queue duration was unmeasured. Local endpoint rescoring used only the registered development role.

[Registered protocol](../protocol.md) · [Endpoint analysis](pass60_analysis.json) · [Attribution](attribution.md)
