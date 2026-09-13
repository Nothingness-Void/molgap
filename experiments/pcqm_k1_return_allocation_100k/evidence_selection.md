# Evidence selection — round 3

Decision date: 2026-09-13.

Earlier K1 tests added slots, heads, dynamic queries, molecule gates, relation
slots, readout capacity, or parameter sharing. They generally reduced training
error while worsening development error. Round 2 found the only favorable
direction so far: removing redundant length-one slot attention improved paired
errors, although its gain stayed below the frozen material threshold. The
linked authority is `../pcqm_k1_slot_processor_100k/decision.md`.

## Remaining bottleneck

K1 uses the same learned atom distribution twice: first to pool node values
into its molecular slot, then by transpose to return that slot to nodes. This
hard-codes source/recipient symmetry. An atom that contributes little to the
global summary consequently receives little global context, even though
underrepresented atoms may be the ones that need it most.

Round 3 keeps source pooling and slot processing exactly unchanged and varies
only the recipient distribution:

- uniform return tests whether every atom should receive equal normalized
  global context;
- inverse-score return sends more context to atoms least represented in the
  source slot.

Both return distributions sum to one per molecule, add no parameters, and are
exactly nested in K1 at initialization through its zero-initialized return
projection. This isolates information direction without reopening slot count,
attention depth, width, optimizer, seed, geometry, or readout.

## Stop condition

If neither candidate clears the frozen `0.003 eV` gain and paired-error gate,
source/recipient decoupling and the authorized three-round K1 sequence close.
No micro-variant or automatic extra seed follows.
