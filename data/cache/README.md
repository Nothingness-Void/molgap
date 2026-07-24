# Local Regenerable Caches

This directory holds large graph and embedding payloads that can be regenerated
from tracked scripts and source data. Its binary contents are ignored by Git.

## Phase 8 expansion_1m

`phase8/expansion_1m/` contains the downloaded 1M continuation handoff:

| Location | Contents |
|---|---|
| `graphs/` | Full ETKDG graph cache and its separately-built 500K top-up |
| `embeddings/` | GPS7, GPS9, and SchNet frozen embeddings |

Use either the full graph cache or the base-500K plus top-up pair. Do not append
the top-up to the already-combined full cache.

## Generated Python caches

`cleanup_20260722/python_caches/` is a reversible holding area for generated
`__pycache__` and `.pytest_cache` directories removed from the readable source
tree. It is ignored by Git and safe to delete after backup review.
