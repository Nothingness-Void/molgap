# Phase 8 Archive

This tree preserves completed and superseded experiment entry points. Nothing
here is part of the supported active CLI surface, and active code must not
import from it.

| Path | Contents |
|---|---|
| `archive-r01-*` to `archive-r08-*` | Closed numbered experiment rounds retained with their exact reproduction payloads |
| `scaleup/` | Completed 1M, repair, residual, and exact-2M data/evaluation drivers |
| `remote/colab_1m/` | Historical Colab 1M fusion payload |
| `remote/kaggle/` | Completed Kaggle training, replay, repair, and fusion payloads |
| `remote/scnet/` | Completed SCNet experiment-specific jobs and environment probes |
| `legacy/` | Superseded exploratory scripts retained for provenance only |

Use result summaries rather than rerunning an archive entry. If a new task
needs old behavior, first check `src/molgap/`; only promote genuinely reusable
logic back into the package.
