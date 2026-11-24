# SwiGLU100K interrupted training decision

2026-11-06 JST. Kaggle3 job142921674/version1 ended ERROR. Outcome:
INCONCLUSIVE. Source/package/receipt remain those recorded in STATUS.md.

Native preflight exited0. Formal training completed10/40passes,7810updates and
999680presentations in the retained canonical trace and pass10 resume bundle.
Best retained Live development MAE is0.17512261867523193eV at pass10. The next
development evaluation raised ValueError("Development count/order/finite
predictions differ") in training/core.py evaluate_development; the parent
reported train_worker_exit_1. No terminal output_manifest/completion was produced.

This is interrupted training, not NO_TRAIN. The40-pass endpoint question remains
unresolved. Pass5/6 development MAEs reached479143168 and622525568eV before
returning to ordinary-range values. These are actual stability anomalies, not
completed-training scores. They motivate investigating evaluation/learned-state
stability; a unique BN, exposure or architecture explanation is unestablished.
No new model execution, recovery training or reference retraining was performed.

The failed allocation ledger records1722.620130591wall seconds and
3445.240261182allocated T4-device-seconds (0.957011T4-device-hours), including
the second idle device. Train worker wall1423.399984305s excludes preflight and
bootstrap. Queue, CPU and GPU-busy time remain unmeasured.

Only small diagnostics and role/cost/source metadata were retrieved to
`D:/molgap-artifacts/reconciliation/k1-swiglu-100k-20261106-failure/`.
The exact-version pass10 resume_bundle.json binds the durable checkpoint chunk
under experiment/checkpoint_chunks/swiglu/pass_10/; tensors stay remote.
Retained data-open provenance describes actual Python training-process role
access; official validation/test were not listed among opened dataset files.

Close this attempt with its observed failure evidence. Keep owner/recovery
pointers pending a user decision. No automatic retry, successor,500K release or
promotion. A recovery needs a specific stability discriminator and explicit
authority. Attribution is the development-evaluation guard and unstable recorded
development history; unique mechanism remains unresolved.
