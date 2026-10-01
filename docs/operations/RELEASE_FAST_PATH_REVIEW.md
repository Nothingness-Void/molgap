# Release fast-path repair

On 2026-10-01, the G1/G2 submission review identified preparation failures that
were detected too late. The existing workflow was repaired rather than replaced.

| Failure | Owning repair |
|---|---|
| RML plans could be published before an upload/entry/source mistake surfaced | `prepare-release` runs the actual RML planner in validate-only mode and checks all upload inputs before any plan publication |
| Semantic tensor SHA and transport-file SHA could be confused | `UploadArtifact.from_file(...).to_workflow()` records the file digest; Spec tensor identity stays separate |
| Package sidecars were manually enumerated and a manifest could be omitted | Staging/restore reuse the package owner's complete sidecar inventory and verify exact bytes |
| Repeated checks gave no attributable timing | Monotonic per-phase timings in workflow output; preparation is not an end-to-end POST SLA |
| A structurally valid reference could make strict terminal binding impossible | New causal server releases reject shared role/cost/acceptance pointers before compute; historical plan/bundle validation is unchanged |

For a compatible addon, the shortest supported path is the
[standard workflow](EXPERIMENT_WORKFLOW.md): canonical Spec/workflow input,
one `prepare-release`, the owning adapter's required recheck and POST, then the
existing monitor binding. Do not routinely repeat `check-preparation` immediately
before `prepare-release`; the latter invokes the same checks.

The native GPTrans author-input outputs use `gptrans_author_acceptance` for
source/receipt/trace/hash/prediction checks and `gptrans_author_terminal` only for
metadata translation into `close_terminal_arm`. No replacement trainer,
submitter, RML validator or causal gate was added. This adapter is question-bound,
not a generic acceptance template for arbitrary GPTrans runs.

Real G1/G2 output acceptance and per-arm RML closure passed. The immutable
reference combined several evidence kinds in one metadata file and was not
replay-enrolled. Therefore the retained results are paired endpoints, not
STRICT_CAUSAL or Replay-Ready. No historical authority was rewritten and no
replacement reference training was submitted. A future reference needs distinct,
frozen evidence bindings and separately valid canonical replay enrollment.

The later metadata-only [reference qualification](../../experiments/pcqm_gptrans_v5_audit_reference/results/reference_qualification/decision.md)
recovered those bindings in a separate bundle and enrolled the accepted control
under its own evidence ID. Original G1/G2 releases and outcomes were preserved;
the qualified control alone does not create an eligible replay pair.

These repairs reduce repeated preparation work. They do not establish a measured
five-minute end-to-end submission time: addon design, upload, platform readiness
and monitor handoff remain outside local preparation timings.
