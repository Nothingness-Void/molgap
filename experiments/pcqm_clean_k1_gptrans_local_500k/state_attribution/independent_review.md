# Independent CPU verification

2026-10-15 JST. Luna6/max, read-only, using the project virtual environment and
owner PYTHONPATH with CUDA/HIP/ROCR hidden. No model, checkpoint tensor, graph
cache, training or remote action was executed by the verifier.

- Focused regression:71 passed in11.71s, zero failures. Compileall/diff checks
  passed; added preparation/closure scripts also compiled after their addition.
- Saved-artifact acceptance:101/101 checks passed. Verified completion hashes,
  input/prospective/source pins, all18 prediction tensors' exact row membership,
  finite values, common targets,18 MAEs and20 central paired differences.
- All20 bootstrap JSON records pass sanity checks. Resampling was not repeated;
  uncertainty remains row-based, not training variability.
- Native selected-Clean reconstruction max difference2.86102294921875e-6eV,
  below1e-4. Raw/Clean selected MAEs recompute exactly from retained tensors.
- Checkpoint bytes/model internals were verified by the admitted worker and
  original artifact acceptance, not independently reopened by this verifier.

Passing saved-artifact check:3.0558 wall seconds/3.2031 process-CPU seconds.
Including a preliminary audit using an unprojected producer prediction record:
6.0995 wall seconds/6.4063 process-CPU seconds. That audit expected exactly three
fields; it was corrected to project the same prediction fields as the driver.
No experiment files or numerical results changed. Verification is excluded
from the numerical worker's reported467.4198346 wall/1798.234375 CPU seconds.
Other preparation/source-review/test costs remain outside that worker scope.
