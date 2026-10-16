# Modern2D feature and CPU lifecycle proposal

Proposed2026-10-16; approval for formal registration is pending. Scope is a
separate `modern2d_cpu_v1` synthetic-only profile, not a V1/V5 schema relaxation.
The accepted V2 master and old contiguous V1 assets remain immutable.

PPGT uses `ogb-atom9-bond3-rrwp-spe-v1`; ESA uses
`ogb-atom9-bond3-directed-edge-mask-v1`. Both derive from legal2D edge_index and
categorical atom/bond tensors. RWSE16 is not renamed RRWP or fed to ESA.

## Input rules

- Explicitly validate atom/bond categories, graph boundaries, source order and
  absence of geometry/unknown model inputs. Protected roles are rejected.
- PPGT coalesces repeated directed adjacency entries; inconsistent parallel
  bond attributes are rejected. W uses row-normalized binary adjacency.
  Isolated rows are zero for k>0, I for k=0; no hidden self-loop is added.
- ESA retains each directed PyG edge as a token. Shared endpoints or self give
  allowed keys within that graph; padding is always excluded. Empty-edge graphs
  are explicitly unsupported, not repaired with virtual chemistry.
- Pure derivation never reads y. An on-the-fly binding has base manifest, source
  row and topology SHA, feature schema, transform source/config SHA and role
  history. derived_feature_sha is null until actually materialized.
- CPU prototypes process each graph separately to bound memory and establish
  isolation. Future batching/cache choice requires real size/cost evidence.
  No full-data dense RRWP precomputation is authorized.

## Required minimal owner extensions

1. Add static new families and a versioned synthetic-only input/lifecycle
   profile. Keep every existing feature/role/recipe gate unchanged.
2. Extend existing Session's declarative family policy/model-forward hook only;
   do not add another loop/optimizer/sampler/checkpoint owner. CPU fixture uses
   the actual Session and exact uninterrupted/resumed state comparison.
3. Bind model/feature configuration in the static Spec/catalog, source closure
   and checkpoint identity. Unknown profiles/config keys fail closed.
4. Reject this CPU profile in platform preparation/submission and real dataset
   loading. No real training trajectory, runtime certificate or comparison
   eligibility follows from a synthetic lifecycle check.

Without approved safe owner extension, retain models/features and proposal but
stop before formal registration, Session claims or MODEL_CPU_INTEGRATED status.
No fallback to an old development role or to a K1/GPTrans family name is valid.
