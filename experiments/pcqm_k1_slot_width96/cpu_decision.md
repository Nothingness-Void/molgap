# CPU qualification decision — 2026-10-02

The CPU qualification observation supports NO_TRAIN closure of this CPU trajectory. Removing only the final slot update increased mean absolute error by 0.14231514860875905 eV on the frozen 2048 development rows, from 0.14281156116339844 to 0.2851267097721575 eV. The paired row bootstrap 95% interval was [0.13232232011214365, 0.15158182376762852] eV. The original prediction reconstruction maximum deviation was 2.384185791015625e-6 eV, within the frozen 1e-4 eV tolerance.

The operational 1 meV eligibility gate passed. This establishes usefulness of the retained final-slot update on this subset; it does not prove a latent64 capacity bottleneck. Candidate initialization retained 3853793 finite parameters versus 3658817 reference parameters, saved-state roundtrip identity and a finite real pure2D forward. GPU qualification and actual runtime/software/source equivalence remain pending.

[Qualification observations](qualification_result.json) and [ablation observations](ablation_result.json) own the measured facts. The finalizer will bind mechanical acceptance and immutable RML publication separately. Candidate training requires a separate prospective trajectory and applicable release checks. CPU closure neither trains nor submits a candidate and releases no automatic successor, promotion, scale-up or protected-role evaluation.
