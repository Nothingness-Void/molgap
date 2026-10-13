# Common-cycle CPU inputs accepted

Recorded2026-10-14, Asia/Tokyo. Kaggle2 kernel138515918/version1 completed the
source-bound CPU derivation. The [independent saved-output acceptance](acceptance.json)
verified all200,000selected rows in4chunks, containing7,504,482ordered cycle-pair
relations. Both fixed manifest hashes, source shard identities, complete row
coverage, chunk SHA256, decoded tensor schema/bounds/symmetry and frozen source
identity passed. The two enumeration algorithms agreed for every row remotely;
local acceptance verified saved bytes, not a third rederivation of source graphs.

The accepted roles were fixed100Ktrain100,000, its internal-dev50,000 and only
fixed500Kinternal-dev50,000. No500Ktraining or protected role was traversed.
Labels and coordinates were incidental trusted packed-file deserialization;
no labels were inspected/scored and no coordinate feature or model was used.
Role closure conservatively records incidental label deserialization, not
training membership, prediction, scoring or selection.

The final attempt measured261.824623793wall seconds. Including the two retained
infrastructure failures,339.297642152seconds was below the original2hour CPU
ceiling. GPU cost was0. Process CPU is recorded natively; full allocated CPU
and queue costs remain explicitly unavailable, not equated to process CPU.

The sidecar manifest SHA256 was
`1fd890bf91bc633c8323c17fe87d664f3106b26052f881e145d22b57f6606ac4`.
Payloads remain under ignored platform record storage; compact hashes and RML
terminal evidence are retained beside this decision. This is NO_TRAIN cache
evidence, not an accepted training trajectory or a model-performance result.
No further CPU retry or GPU successor was released. The [planning decision](../decision_prelaunch.md)
owns the remaining shared-contract/reference/audit release gaps.
