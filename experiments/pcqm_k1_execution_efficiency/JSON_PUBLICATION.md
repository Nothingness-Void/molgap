# Long trace JSON publication

2026-10-10; synthetic CPU infrastructure qualification only. No molecular data,
checkpoint decoding, model construction, training, CUDA or RML learning evidence.
Uses the existing `training_reproducibility.atomic_json` and `sha256_file` owners,
not another logger or serializer. [Retained timing](json_publication_result.json)
owns the four alternating windows per fixture size.

## What is slow

The frozen sender rewrites both complete step histories every128 steps or120s.
Indented JSON selects Python-level formatting work; encoding is substantially
more expensive than local writing/hashing in the large synthetic fixtures.
This is a long-history finding, not an explanation of early-epoch A100/T4 parity.
Native histories, storage and total allocation remain unqualified.

| Synthetic rows per arm | Pretty encoding | Compact encoding | Pretty publish+hash | Compact publish+hash |
|---|---:|---:|---:|---:|
|8192|0.0716s|0.0168s|0.0816s|0.0229s|
|65536|0.6051s|0.1435s|0.6656s|0.1707s|
|234360|2.4116s|0.4754s|2.6620s|0.5903s|

At the largest size, pretty writing is0.1467s and hashing0.0624s. Compact output
reduces99,638,866 bytes to56,047,846, while publish+hash falls77.82%. These are
metadata publication costs, not77.82% faster training. Local page-cache IO is
not Drive/Kaggle throughput, fsync durability or native checkpoint allocation.
Neither metadata copying nor the binary checkpoint is included in these timers.

## Small opt-in change

`atomic_json(path, value, compact=True)` uses the same standard JSON encoder,
key order, escaping, float rendering, newline and atomic replacement. Its default
remains byte-identical pretty output. Compact metadata is explicitly opt-in;
human-facing config/receipts keep their formatting. The [sender patch](sender_compact_trace.patch)
changes only the `trace.json` call, not log frequency, fields, history retention,
guards, optimizer, loss or resumable checkpoint. It has not been applied to the
sender or a running/frozen package. Future release must bind the shared helper.

The file SHA necessarily changes. The unchanged sender hashes the actual trace
after publication and writes that observed digest into its receipt; no digest is
copied from the original format. CPU synthetic acceptance confirms identical
binary `last.pt`, logical trace/selection/exposure/RNG, and updated actual trace
SHA in `last.json`. Existing evidence bytes and historical receipts are unchanged.
Read-only patch applicability check passed on the hash-pinned source owner.

Shared-writer tests23 passed; focused writer/sender/cache/snapshot tests48 passed.
Combined execution/closure/sampler/JSON regression:355 passed, one upstream PyG
deprecation warning.
The adapter [test](../../tests/test_k1_sender_compact_trace.py) AST-isolates only
the frozen `_save` function and uses small invented metadata. This is not a real
model, complete training replay or GPU runtime certificate.

## Remaining bottleneck

Compact formatting reduces a constant factor; rewriting the complete cumulative
history still scales quadratically across publications. An incremental trace
format would need a separately reviewed, hash-bound retention/resume migration.
Do not remove records, reduce checkpoint frequency, convert canonical traces into
synthetic ones or edit an active package to claim that the full problem is solved.
Continue with the [host-path audit](HOST_OVERHEAD.md) and native phase timing.
