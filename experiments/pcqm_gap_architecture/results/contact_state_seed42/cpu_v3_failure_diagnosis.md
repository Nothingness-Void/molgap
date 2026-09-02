# ContactState CPU cache version-3 infrastructure failure

On 2026-09-03 Kaggle2 version 3 reached terminal `ERROR` after locating the
source and beginning parent-shard deserialization. The accepted geometry cache
stores `molgap.pcqm_wedge.WedgeData` objects, but the minimal ContactState
source dataset contained only `pcqm_contact.py`; `torch.load` therefore raised
`ModuleNotFoundError: molgap.pcqm_wedge` before converting a graph.

The retained `failure.json` has SHA-256
`bb0020ab90c4aa2f62532bbd55172a17d470db4bd6bf3458bcc1b0afbb65a279`;
the retained log has SHA-256
`303e6cbbe7ea1a7cdcc3a68ce34acd3a464dc842cc89123dfd039fd887b3b45c`.
It records no conversion failure, cache shard, GPU use, model inference,
official validation access, or test-dev access.

Local data-only inspection confirmed that the parent shards contain exactly
`molgap.pcqm_wedge.WedgeData`. Source-dataset version 2 therefore adds only
the matching `pcqm_wedge.py` class definition from the frozen repository
source. It does not alter graph bytes, ContactState construction, labels, or
the scientific contract.
