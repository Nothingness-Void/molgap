# Cardinality-channel decision — 2026-09-10

## Question

Did a query-gated, unnormalized sum over exact three-hop support improve a
fresh EdgeState GPS9 on the frozen QM9-30K Gap screen, beyond both the baseline
and a parameter-matched support-size control?

## Acceptance

Kaggle2 kernel `kaseichou/molgap-qm9-cardinality-channel-s42`, version 3,
completed all three 40-epoch arms. Independent no-model acceptance passed with
source commit `9b0393bc9212671cc727a8e23391d7d882c472ff`, cache aggregate
`80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340`,
split `62f1cdefdaec6877`, seed 42, FP32, and physical batch 128. Artifact hashes
matched, and neither QM9 test nor official PCQM roles were read.

## Result

| Arm | Validation Gap MAE | Parameters | Mean epoch |
|---|---:|---:|---:|
| EdgeState GPS9 baseline | 0.1280647814 eV | 4,771,073 | 18.106 s |
| Support-size control | 0.1289295703 eV | 4,845,957 | 20.720 s |
| Cardinality-preserving channel | 0.1282242239 eV | 4,845,957 | 20.023 s |

The cardinality channel was `0.0001594424 eV` worse than the baseline and
`0.0007053465 eV` better than the size control. Its epoch-time ratio was
`1.1059`, within the resource limit, but it missed the frozen `0.003 eV`
baseline-gain and `0.001 eV` control-gain gates.

## Attribution

The size-only arm's regression shows that scaling target states by support
count was actively harmful. Replacing that shortcut with the summed source
content recovered most of the loss, so the channel did transmit nontrivial
neighborhood information. It nevertheless failed to add information that the
nine-layer real-bond EdgeState and RWSE16 backbone could exploit beyond its
existing local/global messages. The result is not explained by parameter,
memory, runtime, or incomplete-convergence failure: both candidates shared the
same size, retained about 97% memory reserve, met the timing gate, and completed
all 40 epochs.

## Decision

The exact K<=3 shared cardinality-channel mechanism was rejected and closed.
It received no PCQM-100K transfer, shadow audit, additional seed, width, hop,
placement, optimizer, schedule, desktop/full-scale, official-role, SCNet, or
IMS follow-up. The two preflight-only infrastructure failures remain separately
preserved in `terminal_report.md` and `terminal_report_v2.md`; they do not alter
the accepted scientific result in `terminal_report_v3.md`.

Machine provenance: `results/gpu_seed42_v3_launch.json`.
