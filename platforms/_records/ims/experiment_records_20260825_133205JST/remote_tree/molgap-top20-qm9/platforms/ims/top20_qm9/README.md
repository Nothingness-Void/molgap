# IMS adapter for the top-20 QM9 transfer screen

This adapter is deliberately separate from the frozen Route B outputs. The
approved root is `/lustre/home/users/sm2/chou/molgap-top20-qm9`.

Execution order:

1. `prepare_etkdg.pbs` runs on a high-memory CPU node. It downloads are staged
   before submission because IMS compute nodes have no external DNS, then
   builds the deployment-matched ETKDGv3+MMFF200 cache in atomic 2,000-row
   shards and writes `outputs/cache/cache_acceptance.json`.
2. `train_tgt_lite.pbs` verifies every accepted cache hash before requesting an
   A100. It resumes `checkpoint.pt`, writes per-epoch progress, and exports the
   model/embedding payload under `outputs/`.
3. Only after the seed-42 short screen clears the QM9 gate are seeds 43/44
   submitted. Only after the three-seed gate is a PCQM/production retraining
   workload authorized.

The ABI warning about the published `torch_cluster` extension is expected on
IMS. Both scripts put the accepted portable radius shim first on `PYTHONPATH`;
the TGT-lite pair/triplet model itself does not depend on the native radius
extension.
