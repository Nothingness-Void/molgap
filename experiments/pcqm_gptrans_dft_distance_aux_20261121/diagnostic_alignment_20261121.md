# CPU alignment diagnostic (action A001): result and one amendment

2026-11-21 JST, local CPU, before any training. Builder:
`platforms/local/pcqm_dft_geometry_sidecar.py` (branch
`codex/infra/dft-geometry-sidecar-20261121`, commit 8cbf9cbb).

## Result: PASS

| Check | 500K rows 0..499,999 |
|---|---|
| SDF heavy atoms map one-to-one onto the fixed graph (elements and bond set) | 499,999 rows |
| of which identity order | 462 (0.09%); the rest need an isomorphism map |
| invalid (atom-count mismatch, row 140686: graph has 2 explicit H) | 1 row |
| bonds outside 0.75-1.25 x covalent-radius sum | 87 rows (0.017%) |
| parse + align + write wall | 35.3 s (budget about 7 min) |
| archive hash | 1.0 s |
| output | 92,066,561 bytes, manifest sha256 `370aae6b...a05fc` |

Invalid rows keep zero coordinates and `valid=False`; they get Gap loss only.

## Amendment to the distance falsifier (made before any training)

The protocol's proxy "bonded distances in 0.9-2.0 A for all but 0.1% of rows"
fails by letter: 3,201 rows (0.64%) have a bond outside it. A separate check of
those bonds by element pair shows they are real long bonds of heavy elements,
not misalignment: C-Br 2,428, Si-Cl 1,012, S-S 874, Si-Si 740, P-Cl 564, P-P
366, C-Cl 356, P-S 324 (directed counts), and so on. Among bonds of H/C/N/O
only, none falls outside; the only light-element case is B-C at 2.0 A (4).
The fixed range was a badly chosen proxy for "atoms are mapped wrongly". The
element-aware rule (0.75-1.25 x covalent-radius sum) measures the intended
thing and gives 87 rows (0.017%), under the 0.1% line. This amendment changes
only the alignment sanity proxy; no training gate, arm, constant or role
changes. The decision to proceed rests on the element-aware count and the
99.9998% topology match.

## Provenance of the source file

The official host timed out; the archive came from the mirror path in
protocol.md with sha256 equal to its LFS oid. Its identity is supported by the
row-by-row topology match above (499,999 of 500,000 molecules agree on elements
and bonds with the accepted fixed store). Coordinates themselves cannot be
checked against an official hash.

## Remote copy

Uploaded to the IMS fixed root as an additive subtree
`sidecars/pcqm-dft-train-geometry-500k-v1/` (no accepted file changed; files
made read-only). Local and remote sha256 of all 11 files match.

## Correction after pre-submit review (2026-11-21): v1 withdrawn, v3 used

The review found that the v1 mapping matched bare element skeletons and took
the first match, so symmetric groups were often swapped (C=O vs C-OH and
similar). A bond-order length check showed 44,837 rows (8.97%) with a graph
C=O longer than 1.30 A, a graph C-O shorter than 1.28 A, or a graph C=C longer
than 1.40 A. The v1 sidecar is withdrawn and never trained on.

The builder now matches on (element, H count) and requires SDF bond order to
equal the graph's for every non-aromatic bond (charged atoms excepted: OpenBabel
writes nitro and N-oxide groups uncharged). Builder commits 68f08e2da and
6beba08da on `codex/infra/dft-geometry-sidecar-20261121`.

| Check (500K) | v1 | v3 |
|---|---:|---:|
| graph C=O > 1.30 A | 25,103 | 4 |
| graph C-O < 1.28 A | 25,131 | 32 |
| graph C=C > 1.40 A (report only) | 20,504 | 12,505 |
| invalid rows | 1 | 1 |
| parse + align + write | 35 s | 60 s |

The C=C class does not gate in v3. Since the matcher already enforces the
non-aromatic bond order, a long C=C is real geometry: of 17,728 such bonds,
17,210 are ring bonds and 17,519 conjugated; graph C=C lengths over 230,497
bonds have median 1.344 A and p99 1.471 A. The gating classes (C=O, C-O) give
34 rows (0.007%), under the 0.5% line.

v3: `sidecars/pcqm-dft-train-geometry-500k-v3/` on the IMS fixed root, manifest
sha256 `5b1d8f47ee27b067a2d7c089b3ac38483cbe7e30407133bf2426ce24f39b5b51`
(shards byte-identical to v2). Uploaded, read-only, local and remote hashes
match. The v1 subtree remains on IMS, read-only and unused.
