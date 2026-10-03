# GPTrans CPU parity diagnostic

This is an execution-only, no-training diagnostic of the accepted MolGap
GPTrans-T implementation against the authors' pinned public source. It does
not read any PCQM graph cache, label, checkpoint, development role, or official
evaluation role. It cannot establish a validation-MAE contribution.

Traceability role: `diagnostic`. The technical interpretation is recorded in
[`decision.md`](decision.md); this directory has no scientific acceptance
bundle.

The Kaggle2 CPU script lives in `kaggle_cpu/run.py`. It fetches only immutable
GitHub commit URLs, checks every source SHA-256 before executing extracted
functions, and writes one atomic JSON per completed check plus `summary.json`.
Its checks cover synthetic shortest paths (including the node-zero boundary),
actual local shortest paths, official collator padding, the node/pair
propagation core with identical weights, initialization-scale arithmetic,
the authors' optimizer grouping, and a pinned source-level training-target
contrast. An offline static-only mode is available for local preflight.

The expected Kaggle owner is `kaseichou`; the kernel has no dataset mount,
internet access only for pinned source files, and GPU disabled. A completed
kernel is not verified merely because Kaggle says `COMPLETE`; run
`verify_output.py` against retrieved output after checking the actual remote
version. Passing this verifier establishes only technical execution of these
small checks, not scientific acceptance or an MAE explanation.
