# Fixed OGB PCQM4Mv2 Datasets

This adapter creates immutable manifest views over already accepted official
PCQM4Mv2 assets on the molecular research server. It does not rebuild graphs or
copy the multi-gigabyte payloads. One hardlinked content store serves four
nested official-train scales:

| Identity | Train | Development | Kaggle1 |
|---|---:|---:|---|
| `ogb-train-100k` | `0:100000` | `100000:150000` | yes |
| `ogb-train-500k-scnet-v1` | `0:500000` | `500000:550000` | yes |
| `ogb-train-1m` | `0:1000000` | `1000000:1050000` | no |
| `ogb-train-full` | all 3,378,606 official train rows | none | no |

The 500K geometry view is byte-identical to the accepted SCNet cache and must
retain aggregate SHA256
`676a506c808402bc16a4437cc02286168239a8dd5de4451e992131eddb4f1b20`.
All graph views use OGB 9-field atom categories, OGB 3-field bond categories,
RWSE16, and the accepted ETKDGv3 plus MMFF94s geometry where geometry is used.

The full official archive is retained as provenance. Official validation is
not a development role, and test roles are never materialized by this adapter.

Submit `build_and_accept.pbs` from the code staging root. Final state is valid
only when `pcqm4mv2-fixed-v1/acceptance.json` reports `accepted` with
`verify_content: true`.

After acceptance, `export_kaggle1.py` creates hardlinked upload directories for
only the 100K and 500K identities. The export metadata attributes the derived
cache to OGB PCQM4Mv2 under CC BY 4.0 and defaults to private visibility; it
never exports the 1M or full identities. Raw row gzip files remain only in the
IMS canonical store because Kaggle rewrites compressed CSV paths; Kaggle graph
payloads retain source indices and labels directly.

## Separate Unified V2 identity preparation

The [Unified V2 contract](../../../docs/operations/PCQM_UNIFIED_ROLES_V2.md) and
[Si identity repair](../../../docs/operations/PCQM_UNIFIED_IDENTITY_REPAIR.md)
own the separate indexed-role CPU flow. `unified_roles_cpu.pbs` requires
`MOLGAP_V2_STAGE` and `MOLGAP_V2_OUTPUT`, immutable source/rule/ledger staging,
and defaults `MOLGAP_IDENTITY_ONLY=1`. Mode0 is only for the same accepted
identity inventory's authorized role derivation, not training. Historical
`prepare_unified_roles.pbs` remains the frozen r2 attempt, not a new template.
