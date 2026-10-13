# CPU cycle-input v1 infrastructure failure

Recorded2026-10-14, Asia/Tokyo. Kaggle2 returned kernel138513598/version1 for
`kaseichou/molgap-gptrans-cycle-inputs-v1`. Source commit
`df3cbb6b66270066d1a1518ef667b6dda9605a68`; source archive SHA256
`e925003f6b43e1380a49c09bb074d5ff19c0123e229ffb1d4816c1a325287eab`.

The exact job ended ERROR at the first manifest lookup, `KeyError: assets`.
No graph was processed; no encoder, optimizer or GPU was executed. This is not
a negative scientific result. The independently downloaded accepted100K/500K
manifest metadata has `geometry_shards`, not the IMS topology-only
`assets/topology` layout. Both accepted byte identities matched their records.

The [compact failure receipt](failure_v1.json) retains observed cost and artifact
locations. Raw log, binding and cost remain in ignored platform record storage.
The pulled source was LF-normalized identical to the submitted entry; Kaggle
CLI introduced CRLF on local code pull. No duplicate version was submitted.

The bounded infrastructure repair adds exact-pin legacy-record compatibility,
with separate prospective recovery identity and independently tested guards.
Packed coordinates and labels are incidental deserialization only, not input
features or supervision. The recovery ceiling is7140wall seconds, retaining the
original37seconds within the total2hour CPU budget. No GPU release follows from
this failure or a queued recovery alone.
