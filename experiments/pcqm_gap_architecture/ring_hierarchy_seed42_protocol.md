# Deterministic ring-hierarchy seed-42 protocol

## Question

Does an explicit smallest-ring hierarchy improve the frozen
distance-plus-angle Sparse Triangle EdgeState GPS9 comparator on the official-
train-derived PCQM 100K/10K screen?

This is one architecture question. It does not authorize a descriptor model,
late fusion, graph token, BRICS vocabulary, conjugated-component variant,
optimizer search, extra seed, full training, official validation/test-dev
access, or molecular-research-server work.

## Why this is new information

The comparator represents atoms, real directed bonds, non-backtracking wedges,
ETKDG bond distances, and ETKDG wedge angles. OGB atom and bond categories
already include per-atom ring/aromatic flags and per-bond conjugation. Merely
adding those binary values again would be a duplicate.

The candidate instead adds relations that are absent from the flat graph:

- one node per deterministic symmetrized smallest ring;
- many-to-many atom--ring membership;
- ring--ring edges for spiro, fused, and directly bonded ring pairs;
- aggregate ring composition and internal-conjugation descriptors.

The design is motivated by RingFormer's atom, ring, and inter-level graph
factorization and its hierarchy ablations. It is deliberately smaller than
the published full model: the accepted atom GPS remains intact, the ring state
is narrow and recurrent, and prediction still pools only atoms. The official
paper and implementation are [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/31991)
and [TommyDzh/RingFormer](https://github.com/TommyDzh/RingFormer).

The closed archive-r06 topology audit is not a comparator or blocker. It only
tested whether scalar topology prevalence explained residuals from a different
frozen predictor and stopped before training. This protocol trains an explicit
hierarchical representation from scratch on the fixed official-PCQM screen.

## Immutable data contract

- Parent graph identity:
  `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`.
- Parent wedge identity:
  `dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406`.
- Parent distance/angle geometry identity:
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
- Roles: exactly 100,000 training graphs and 10,000 internal-validation
  graphs, selected from the first 3,378,606 official training rows.
- The cache may read canonical SMILES only for those selected row indices to
  recover the RDKit symmetrized smallest rings. It may not read the official
  validation role or test-dev.
- The scalar target remains direct official PCQM Gap in eV. No HOMO/LUMO,
  residual target, auxiliary target, pseudo-label, or prediction fusion is
  permitted.
- Ring preprocessing is CPU-only. A GPU task cannot start until all cache
  shards and hashes pass no-model acceptance.

## Deterministic hierarchy cache

For every accepted parent graph, RDKit `GetSymmSSSR` supplies ring atom sets.
Ring sets are canonicalized by sorted atom ids, deduplicated, and ordered by
ring size then atom tuple. The cache stores:

- `ring_features`: 12 finite channels containing normalized ring size;
  C/N/O/S/other composition fractions; aromatic-atom, internally conjugated-
  bond, and shared-atom fractions; spiro and fused indicators; and normalized
  ring-neighbor degree;
- `atom_ring_index`: atom-to-ring membership pairs;
- `ring_edge_index`: both directions of every ring relation;
- `ring_edge_attr`: four binary channels for spiro, fused, direct bond, and
  conjugated direct bond.

A molecule with no ring has zero rows in all hierarchy tensors and remains a
valid sample. Ring extraction failures remain visible with role, row index,
type, and message; no failed row may be silently dropped or replaced.

Cache acceptance must verify parent hashes, source commit, role counts, row
coverage, finite tensor statistics, batching increments, per-shard SHA-256,
one aggregate SHA-256, zero unresolved failures, and
`official_validation_role_read=false` / `test_dev_role_read=false`.

## Frozen architecture

Candidate name:
`ogb_distance_angle_ring_hierarchy_triangle_edge_state_gps9`.

The comparator's OGB encoders, RWSE16, node width 192, real-bond state width
64, wedge state width 16, nine GPS blocks, four attention heads, direct scalar
head, and mean atom pooling remain unchanged. Distance and angle geometry use
the already accepted ETKDGv3/MMFF94s single conformer.

The only addition is:

```text
smallest-ring descriptors + member atom mean
                    -> persistent ring state (64)
                    -> shared ring update
                       [membership + local ring relations]
                    -> rank-32 zero-initialized return to member atoms
```

The same ring update cell is reused after atom GPS blocks 2, 4, 6, and 8.
There is no independent ring Transformer stack, molecule token, ring-only
prediction head, ring pooling, or prediction-level blend. The ring-to-atom
value endpoint is zero initialized, so the candidate's initial predictions
must equal the comparator's initial predictions exactly while gradients remain
finite. Added parameters must keep the complete model at or below 5,000,000.

## Frozen training contract

- seed: 42;
- precision: FP32;
- batch size: 48;
- optimizer: AdamW;
- learning rate: `1.6e-4`;
- weight decay: `1e-6`;
- epochs: at most 40;
- early-stop patience: 8;
- target normalization, scheduler, data order, and loader settings: identical
  to the accepted distance-plus-angle comparator;
- GPU search budget: 14,400 seconds after dependency bootstrap;
- one atomic checkpoint after every epoch, including model, optimizer,
  scheduler, RNG, data-order, best state, and complete trace.

The GPU preflight must prove cache identity, exact baseline-key equality after
matched initialization, zero ring-to-atom injection, nonzero finite gradients
for the new path, the exact parameter count, finite loss, and memory headroom.
No model is executed locally.

## Decision rule

The candidate passes seed 42 only if its internal-validation Gap MAE is
strictly lower than the frozen comparator value
`0.1353926807641983 eV`, all artifacts pass independent no-inference
acceptance, and the parameter/time ceilings hold.

A pass grants only eligibility to plan fresh seeds 43/44. It does not submit
them automatically. A failure closes this exact hierarchy without width,
depth, ring-definition, learning-rate, dropout, schedule, or seed retries.
The next literature-ranked mechanism would then require its own protocol.
