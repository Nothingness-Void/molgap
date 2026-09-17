# Execution status

Kaggle3 kernel `nvoid912/molgap-gptrans-pair-norm-v5-s42` version 2 was
submitted from source commit `5f8d27e52d73dca782fad19cde63896e590e8245` on
2026-09-18. It remained `RUNNING` beyond the version-1 failure window during
the bounded startup check.

Version 1 confirmed two Tesla T4 devices, then failed before training because
the accepted graph cache referenced the missing `molgap.pcqm_wedge.WedgeData`
pickle class. Version 2 restores that compatibility module without changing
the graph cache, frozen initialization, model variants, or training contract.

No scientific result has been accepted. Reconcile the terminal kernel state
and artifacts when the desktop next resumes.
