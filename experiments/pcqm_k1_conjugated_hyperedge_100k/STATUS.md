# Conjugated hyperedge screen status

CPU sidecar, GPU scheduler state and terminal scientific acceptance are
independent facts. Each arm requires its own prospective and terminal RML
record. No 500K/full or protected-role operation is released by this screen.

## CPU preparation receipt — 2026-09-23 UTC

- Kaggle2 CPU kernel: `kaseichou/molgap-k1-conjugated-sidecar-v1` v1,
  `COMPLETE`; source commit `b2340edd93bff46172959bdc2f9771da51fad3d6`.
- Local no-inference `accept_sidecar` passed for 100,000 train and 50,000
  internal-development rows, 30 ordered 5,000-row shards, with Gap labels and
  protected roles unread.
- Manifest SHA-256: `c3ec649721145ed99bd8fc9218eb7d23cfa7d86b2b15a2eadd7533a7622425c1`.
  Aggregate SHA-256: `d3e34ccba8c35184c25fd9c00d5de5989b4c2cc943fb76020d33ba9a87614cfe`.
- Published private Kaggle2 dataset:
  `kaseichou/molgap-k1-conjugated-cache-fixed100k` v1, `ready`, containing
  exactly the manifest, acceptance record, and 30 sidecar shards. Its ID is
  distinct from the CPU notebook ID because Kaggle reserves notebook slugs.
- The downloaded output and the published dataset are preparation artifacts;
  neither is a GPU result or a terminal scientific decision.

## GPU submission receipt — 2026-09-23 UTC

- Kaggle2 GPU kernel: `kaseichou/molgap-k1-conjugated-dual-s42` v1,
  submitted once; initial scheduler status `RUNNING` at 18:16 UTC. Startup
  logs had not yet exposed the two T4 workers or first epoch at this check.
- Input datasets: frozen source `kaseichou/molgap-k1-conjugated-source` v1,
  fixed graph asset `kaseichou/pcqm4mv2-ogb-fixed-100k-v1` v1, and accepted
  sidecar `kaseichou/molgap-k1-conjugated-cache-fixed100k` v1.
- Two arms use independent T4 workers with seed 42, FP32/no TF32, physical
  batch 128 each and atomic per-epoch checkpoints. No reference retraining,
  protected-role read, 500K/full submission, or result claim was made.
- Existing Luna task B monitors only this kernel via heartbeat
  `molgap-k1-conjugated-dual-kaggle2-monitor` every 30 minutes; healthy runs
  are silent, terminal outcomes are handed to the server controller. The
  separate IMS monitor remains bound to its own job.

## Terminal receipt — 2026-09-24 JST

- Kernel v1 ended `COMPLETE`; both T4 workers trained the full frozen 40
  epochs and exited 0. No second GPU submission or model inference occurred.
- Independent no-inference acceptance passed source/archive/sidecar identities,
  50,000 aligned internal-development targets and predictions, runtime
  contracts, all completion-manifest artifacts and final checkpoints. Official
  validation/test-dev/test-challenge remained untouched.
- The frozen K1-v4 reference was 0.1413736343 eV. One-shot reached
  0.1415904909 eV (epoch 37), a 0.0002168566 eV regression. Persistent
  reached 0.1401050985 eV (epoch 39), a 0.0012685359 eV gain whose paired
  row interval was favorable but below this screen's 0.003 eV gate. Neither
  arm was promoted. See the distinct arm-local `decision.md` files.
- Both arms finalized separately as `STRICT_CAUSAL` with complete replay
  capability and measured T4 training device time; RML validate and frozen
  derived check passed. The replay pool now has nine entries in one comparable
  group, including the historical-partial K1 reference.
- Saved-prediction component attribution is exploratory on the repeatedly
  used internal-development role. In particular, persistent communication
  regressed on molecules with multiple conjugated components; it did not
  justify a matched 500K release.
