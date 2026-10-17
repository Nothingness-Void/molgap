# Bounded native qualification

This server-owned execution-only task was authorized on 2026-10-17 JST by the
human's explicit reply. The [authority](authority.json) freezes resource and
role limits. It grants no formal training, ranking, scientific candidate
comparison, 500K run, extra seed, protected-role read or automatic successor.

## Reuse and acceptance

1. IMS CPU exports existing accepted train100K plus Common-Dev tensors using
   `pcqm_indexed_export`; the [request](cpu_export_request.json) pins the accepted
   parent independently. No graph construction or membership selection occurs.
   The thin owning [PBS adapter](../../../platforms/ims/pcqm_fixed_datasets/export_indexed_roles.pbs)
   requests16 CPUs, no accelerator, and at most7200 seconds. The larger CPU
   allocation provides memory headroom over the earlier4-CPU acceptance's limit;
   whole-job native accounting, not this ceiling, determines measured cost.
2. The same scheduled job independently reopens the completed exported chunks
   with `verify_indexed_export`. Durable receipt-first chunks allow recovery;
   existing accepted output is never overwritten. Pin acceptance, inventory and
   label-free graph census before private transfer/publication. Only selected
   train100K/Common-Dev payloads may be mirrored, never parent Full or Audit.
3. Kaggle2 T4x2 qualification is conditional on actual export acceptance, exact
   private published bytes, committed source, focused CPU checks and live quota.
   Each model has a separate visible device, RNG, optimizer and checkpoint root.
   The30-minute ceiling includes environment startup, feature preparation and
   idle devices. Scratch updates/numerics/resume prove execution only, not MAE.
   Any rejection or uncertainty preserves evidence and cannot trigger a blind
   retry or formal training. No qualification certificate releases a screen.

The [CPU integration record](../integration_attempt_2026-10-17.md) remains
historical CPU-only evidence. Live progress is owned by `CURRENT_STATE.md`.
The [2026-10-18 native acceptance](results/decision_2026-10-18.md) retained the
first GPU attempt's pre-runtime infrastructure failure, without a model result
or automatic retry.
