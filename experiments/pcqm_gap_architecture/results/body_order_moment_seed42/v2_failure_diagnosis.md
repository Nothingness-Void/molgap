# Body-order moment seed-42 version-2 failure

## Disposition

Kaggle2 kernel `kaseichou/molgap-pcqm-body-order-graphstate-s42`, version 2,
ended in `ERROR`. This remains an implementation/preflight failure: the
body-order candidate completed no epoch, so no scientific comparison exists.

## Root cause

Version 2 replaced bitwise CUDA equality with a narrow numerical tolerance,
but a second full GraphState forward is itself not a reliable identity proof
when nine layers contain GPU scatter reductions. The correct proof is
structural: every shared parameter is bit-identical and the only candidate
path returning to the baseline is an exactly zero final projection. Both
conditions passed before the faulty forward comparison.

The runner is repaired to use that structural proof and to terminate sibling
workers immediately when any isolated worker fails. In version 2, GPU1 failed
at 132.39 seconds while GPU0 unnecessarily completed 40 baseline epochs and
consumed 8,113.68 seconds. The missing fail-fast behavior caused the avoidable
quota loss.

No retry is submitted from Kaggle2 because its reported GPU allowance is
exhausted. A later unchanged-contract retry requires a different authorized
account or a quota reset.

## Retained evidence

- Main failure SHA-256:
  `2593e9db4ad70142416b97fa6e4edc1dc7f10295f5a954fa71ec29a41841bf36`
- GPU1 failure SHA-256:
  `639db1a45c1327765aaaf65acbbf208622068f8cd048690cb1dc057585e2085f`
- Initialization preflight SHA-256:
  `5e94eef1c808002df093e86ab48e3805358d00c4e63cd185ea2635276663cf50`
- Terminal log SHA-256:
  `10605c6d1bdc7d89781b9c4cdd31655735270a4e7f62ac4d6bf271d90469`
- Completed-baseline metrics SHA-256:
  `ebfddbdb4a214e9b39a5aa453ba7452d1eab8fd89ccd9abae99e4722eb1f7804`
- Local record:
  `platforms/_records/kaggle/training/pcqm_gap100k_body_order_graphstate_seed42_v2`
