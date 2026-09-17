# Reusable reference and evidence index

This is a compact asynchronous index for V5 reuse. It points to committed
contracts and accepted evidence; it is not a scheduler binding, a live
handoff, or a replacement for `CURRENT_STATE.md`. The production registry
remains authoritative for deployable model identity.

## PCQM4Mv2 Gap references

| Reference | Qualification and scope | Contract/source identity | Artifacts and provenance | Roles, runtime, and cost | Authority |
|---|---|---|---|---|---|
| GPTrans-T 100K V4, seed 42 | Accepted immutable 100K-train/50K-development reference; development Gap MAE `0.1566272043 eV`; one P100; 60 epochs, 46,860 optimizer steps, 5,998,080 presentations | `experiments/pcqm_gptrans_t_100k_v4/training_contract.json`; benchmark `ogb-lsc-pcqm4mv2-gap-internal-100k-v4`; manifest SHA256 `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d` | Accepted best model `f4da386ae1e32f6953b645c0bdb8e208aaba1f1d7b8ebec132776bd63c22d6ab`; aligned development predictions `4fa3d32f83b183503bfafeee33b7b44dc7ee5f396bef5b1b546c956669ce6d29`; frozen reference `c523e000925ce3bfab35e01c3e8ca6e7ce27a7876d4cdf7c012d6ccd398436aa` | Runtime certificate `9576f1e562f6ffda8341ea3370621d7af104811c9d72e787a8aa3687b153e983`; official validation, test-dev, and test-challenge all unread; native device-hours not recorded in the decision | V5 envelope `experiments/pcqm_gptrans_t_100k_v4/v5_evidence.json`; historical authority `decision.md` and `results/kaggle_acceptance.json` |
| Matched 500K V4 three-arm bridge | Accepted matched `pcqm-fixed500k-dev50k-matched60-v4`; 50,000 aligned development rows; K1 and GPTrans-T both beat EdgeState by the `0.003 eV` nomination floor; no material K1-vs-GPTrans-T superiority | `experiments/pcqm_500k_v4_evidence/protocol.md`; source and stage identities in `submission.json` and `stage5_*`; seed 42, FP32/no TF32, physical batch 128, 60 epochs, 29,998,080 presentations | Paired metrics and bootstrap identity in `final_comparison.json`; final EdgeState manifest SHA256 `01f3862ff557eb3d46cd8901f7512f8bc33f9a10b8e000ce4c69b098fac05c51`; per-arm stage manifests/checkpoints/predictions are indexed in `stage5_edge_k1_acceptance.md` and `stage5_gptrans_acceptance.md` | Per-arm runtime certificates and scope are in the stage acceptance records; official validation, test-dev, test-challenge, and shadow roles unread; target hardware was T4/Kaggle, but native device-hours are not consolidated | `experiments/pcqm_500k_v4_evidence/final_decision.md`; nomination evidence only, not a full-run authorization |
| Full K1 / GPTrans-T training and fusion | Mechanically accepted full official-train runs with 20,000,000 presentations; one-time official-validation fusion calibration/holdout is complete and the route was closed without promotion | `experiments/pcqm_k1_gptrans_full_fusion/training_contract.json`, `fusion_contract.json`, and accepted per-model contracts; official row manifest SHA256 `c329fbde935324118f4f62aa6e2b01b209d6f54642549637853419deb1761c4b` | Accepted K1 bundle `60b95d86736829fd38690fb33654199a753f693a333aa55f451d1983653fa3b7`; GPTrans-T bundle `0bc1c464f2e3cd64eb0926ae90e4f4f8c8094ce3a044405e3050f42518f1fb9e`; fusion predictions `b580d8295f159d7efc284d6bdb143b05452b4ce1c05bfac6a869ceba2ca7234b` | K1 runtime certificate `5a8aed2d0093d0aa4a806a3b61d0707ccfbc987cff661931a9483f456feff8d7`; full acceptance and recovery identities are in the linked JSON; official validation was read once by the fusion study, test-dev/challenge stayed sealed; native A100 cost is documented only in the recovery/submission records | V5 envelope `experiments/pcqm_k1_gptrans_full_fusion/v5_evidence.json`; historical authority `results/accepted_k1_gptrans_fusion_r3/decision.md` |

## V5 migration status

- The strict 100K reference and closed full K1/GPTrans-T/Fusion result have
  validated V5 envelopes in their owning experiment directories.
- The geometry-transfer 500K result is preserved as incompatible nomination
  evidence at `experiments/pcqm_geometry_transfer_500k/v5_evidence.json`; format
  migration does not upgrade its scientific authority.
- The matched 500K three-arm result remains indexed by its V4 authority. Its V5
  envelope stays pending until the accepted K1 artifacts have a second durable
  copy outside the desktop-only retrieval tree.

## Reuse rules

- Locate the referenced contract, acceptance record, and role history before
  comparing a new candidate. Missing payloads are comparison-pending; they do
  not justify retraining a baseline.
- Preserve `source_idx`/target alignment and the exact data, optimizer,
  schedule, precision, exposure, selection, and runtime identities from the
  owning record. Historical V3 results are not strict V4 comparators.
- Treat repeatedly consulted development or official-validation roles as
  consumed selection evidence. This index does not create a new holdout.
- Native hardware costs remain in their recorded units. Unknown or
  non-consolidated costs are deliberately marked as such.
- A positive Track B result remains Track B evidence until the separate Track A
  production gate changes the registry.
