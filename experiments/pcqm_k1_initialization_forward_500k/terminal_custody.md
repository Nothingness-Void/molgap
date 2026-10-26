# Terminal custody — 2026-10-27

Owner: `codex/exp/k1-initialization-forward-500k`. The pushed strict-acceptance
tip is `3f06556212e9451f45e3c9938c435c64cb61bdb1`. Frozen training source is
`64933302a449cd48961a610028a17bd6ae6d2e23`; original source archive SHA256 is
`da1ea4baf615cd642574d6904262cff958ab1c3e4cc404322f8baf611ae1ae88` (single)
and `d286174e03816a90280ddd32e6375f93dcd1c420963d971a57eab9a95fa2ff46`
(mean2).

`BRANCHES.md` owns the exact archive destination; retirement remains pending
until published reachability is recorded there. Restore the exact owner source
after fetching the published archive with:

```powershell
git worktree add --detach <restore-path> 3f06556212e9451f45e3c9938c435c64cb61bdb1
```

## Strict acceptance and D: artifacts

The successful reports are `D:/molgap-artifacts/reconciliation/k1-init-forward-500k-20261027/acceptance/single/strict_finalization.json` and `D:/molgap-artifacts/reconciliation/k1-init-forward-500k-20261027/acceptance/mean2/strict_finalization_reentry.json`. Both report `COMPLETE`, `closure_complete=true`, `strict_validation=VERIFIED`; per-arm outcomes and the exact acceptance route are in [qualification](terminal_acceptance/qualification.md), with science owned by the [decision](terminal_acceptance/decision.md) and [attribution](terminal_acceptance/attribution.md).

The four accepted output directories and SHA-bound output-manifest identities are:

| Slot / arm | External output directory | Existing mapping's `manifest_sha256` |
| --- | --- | --- |
| single / scratch | `D:/molgap-artifacts/retrieved/pcqm-k1-init-forward-500k/single/v2/experiment/scratch` | `8b1dbd0ff7a076ed0aa884e343951ff97a773a7025554390657327cf3923a088` |
| single / pretrained | `D:/molgap-artifacts/retrieved/pcqm-k1-init-forward-500k/single/v2/experiment/pretrained` | `d30469d98981014b80a7579ca2f70a5dfa9c12c6491a24cd781a6cadbc8ba251` |
| mean2 / scratch | `D:/molgap-artifacts/retrieved/pcqm-k1-init-forward-500k/mean2/v2/experiment/scratch` | `c7db76a600fdb2c240f9bfad3724ac7a2b6f98fbcd3144552d2fe841cc4877d0` |
| mean2 / pretrained | `D:/molgap-artifacts/retrieved/pcqm-k1-init-forward-500k/mean2/v2/experiment/pretrained` | `cd8a007da58d1df1f651c28eb398ccb66d673e4a535c63f1be0f5d41178e9606` |

Mappings are under each original run's `v1/operations/retrieved_outputs/2/`.
The strict reports bind the verified execution reports and native/actual-resume
proofs. Retrieval scope was 56 new files / 1,082,012,224 bytes (including eight
proof ZIPs / 723,812,922 bytes) plus 22 reused small files / 160,641 bytes; see
[qualification](terminal_acceptance/qualification.md).

Preserve the original v1 run roots under
`D:/molgap-artifacts/runs/pcqm-k1-init-forward-500k-single-s42/v1/` and `D:/molgap-artifacts/runs/pcqm-k1-init-forward-500k-mean2-s42/v1/`,
their partial outputs under
`D:/molgap-artifacts/retrieved/pcqm-k1-init-forward-500k/{single,mean2}/v1/experiment/`,
and the v2 prepared inputs, receipts and bundles under
`D:/molgap-artifacts/recovery/k1-init-forward-500k/{single,mean2}/v2/`.
Continuation depends on the v1 pass-10 checkpoint/trace prefix. The committed
[checkpoint verification](continuation_checkpoint_verification.json) records
all four bundle and checkpoint SHA256 values at pass 10, step 39,060 and
4,999,680 presentations. Retain that prefix and the original failure records;
a v2-only copy cannot restore the validated continuation chain. The v2 repair
overlay is commit `1feb1af0ba174d48f77916e4e1233a9eb6ce6082`.

## Owner checkout inventory

The owner-wide ignored inventory contains 94 Python bytecode files under
`src/molgap/**/__pycache__/`, totaling 1,226,017 bytes. They are regenerable
caches; no ignored model/replay payloads, symlinks or hardlink aliases were
listed (all ignored files have one hardlink). The only untracked file is this
custody note. No unique required payload is held only in the owner checkout;
required model/replay payloads are in the external D: locations above. Large
payloads were not rehashed for this record.
