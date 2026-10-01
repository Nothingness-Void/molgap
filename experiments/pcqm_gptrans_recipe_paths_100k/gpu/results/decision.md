# Partial dual-arm terminal assessment

On 2026-10-02 JST, the server controller processed Luna's idempotent event `evt-121fbeeec1298eb7088a4094` for kernel136671627/version1. The parent ended ERROR; one complete child and one infrastructure failure were independently separated. No training or model inference was executed during acceptance.

## Completed endpoint-path arm

Saved predictions on the exact 50,000 internal-development rows gave MAE0.14561944287240505 eV against the accepted G1 EMA0.999 reference0.14423262914597987 eV. Candidate-minus-reference was +0.0013868137264251709 eV; paired-row bootstrap95% CI was [+0.0004906419098377228,+0.0022685762846469874]. Best epoch59; parameters5,246,817 unchanged. This failed the frozen material advancement criteria. Bootstrap quantifies row sampling, not seed stability.

The full60-epoch trace, selected model, last checkpoint, predictions, source/init/runtime identity and hashes passed saved-artifact acceptance. The candidate passed STRICT_CAUSAL qualification and was actually admitted as a complete candidate/reference replay pair. The existing terminal adapter's gate-failure label INCONCLUSIVE was preserved, not rewritten after finalization; the observed worse endpoint and failed promotion are explicit here and in paired analysis.

EMA and live weights were both worse near the endpoint. The final10-epoch EMA improvement was only0.0004714 eV; evidence does not support buying an extension merely because the selected epoch was59. This exact un-gated endpoint-contrast injection was closed, not all chemical path representations. The path is a restricted first/last-bond contrast, not full multi-hop sequence encoding.

48.88% of molecules had lower absolute error, but overall error worsened. Larger improvements in post-hoc high-reference-error quintiles do not establish an input-identifiable specialist region: grouping on reference residuals uses targets and suffers regression-to-the-mean bias. No deployable router or scale advantage follows. Possible interference/encoding scale and reduced representational suitability are hypotheses; no intermediate-state measurement establishes them causally.

## Failed grouped-decay arm

The diagnostic writer passed a list of parameter descriptors to the mapping-only `canonical_fingerprint()` API. This raised ValueError at the first epoch's diagnostic assembly. The retained best model and development payload prove evaluation was reached, but there were zero canonical observations and no optimizer/RNG continuation checkpoint. They cannot substitute for the original resumable state. No optimizer superiority/inferiority, terminal metric or exposure cursor was inferred.

The local repair wraps inventory under a `parameters` mapping and uses the same helper in trainer and acceptance. Regression tests cover the exact three-field inventory shape, partial acceptance fail-closed behavior and rejection of completed-worker failure closure. Existing source/package receipts and remote artifacts remain immutable.

The failed trajectory was finalized INFRASTRUCTURE_ONLY with actual allocation cost and role/failure evidence. It was deliberately excluded from training replay; no missing trace or checkpoint was fabricated. A failed-arm recovery would need a new qualified physical attempt and could not resume from `best_model.pt` alone. No completed arm or baseline should be retrained.

## Native budget and boundary

The parent reserved two Tesla T4s for8547.712011089001 seconds (2.3743644475 wall hours), totaling4.7487288950 allocated device-hours. Equal per-arm reservation is2.3743644475 device-hours and includes the failed worker's idle reservation; it is not GPU-busy time. A recovered single arm may still receive two T4s, whose idle allocation must also count.

No successor, seed confirmation, scale run, protected-role access or recovery POST was released by this receipt. The exact monitor was paused after delivery and the controller event closed pending a separately qualified recovery decision.
