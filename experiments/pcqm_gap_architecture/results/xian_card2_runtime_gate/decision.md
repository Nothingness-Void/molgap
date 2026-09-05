# Xi'an Card2 runtime qualification decision

The Xi'an 16 GB Card2 runtime passed the execution, cache, throughput, atomic
checkpoint, and artifact gates after rebuilding the HIP scatter kernel with a
256-thread launch width. The corrected 10K probe reached `567.19` graphs/s at
batch 96 and `733.89` graphs/s at batch 192, versus the pinned T4 reference of
`566.55` graphs/s. The repair accelerated the three tested batch sizes by
about 4.0--5.4 times.

Two unchanged full 100K/10K batch-48 runs both completed and passed mechanical
acceptance. Their mean throughputs were `322.17` and `347.84` graphs/s, and the
second run remained between `336.95` and `354.60` graphs/s. Runtime stability,
memory headroom, and artifact durability therefore passed.

Scientific repeatability did not pass an authoritative single-run ranking
gate. Despite identical source, cache, seed, precision, optimizer, and batch,
the two best validation MAEs differed by `0.02143 eV`; their epoch-3 validation
MAEs differed by `0.11723 eV`. Training MAE stayed close, and neither run had a
numerical or scheduler error. Together with the failed bitwise forward identity
preflight, this attributes the divergence to nondeterministic reductions in the
old vendor HIP/scatter path being amplified by optimization, rather than to a
broken cache or model implementation.

Xi'an is therefore conditionally qualified for high-throughput exploratory
pre-screening, not for final architecture selection. Every Xi'an comparison
must train a fresh comparator and candidate under the same batch-96 contract;
historical batch-48 metrics cannot serve as its comparator. A positive result
must reproduce in a second paired run with the same sign, then pass a canonical
batch-48 confirmation on Kunshan before it changes an architecture decision.
Xi'an alone cannot authorize extra seeds, full-data training, or official-role
access.

The qualification consumed approximately `1.58` accelerator-hours, below the
authorized 10-hour cap. Official validation and test-dev remained unread.

