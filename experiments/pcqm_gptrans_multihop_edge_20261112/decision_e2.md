# E2 (500K) closure decision, 2026-11-14

## Scope
Covers E2 runs `pcqm-gptrans-multihop-edge-20261112-e2-s47` (kernel A: ref_s47, mh_s47) and `-e2-s48` (kernel B: ref_s48, mh_s48), including continuations kA_s47_v2 and kB_s48_v2. Each selected checkpoint was scored once on tail_holdout_v1 (rows 3,250,000..3,378,605 of the permanent holdout; official validation, test_dev, test_challenge not read). Tail scores below are the figures reported by the owner for this record; per-arm tail files are not copied into this directory. E1/N1/P1 rulings are unchanged and are not covered here.

## Evidence
Tail MAE in meV, final-EMA endpoint at 60 passes.

| Quantity | Value |
|---|---|
| Mean of candidate tails (mh_s47, mh_s48) | 107.801 |
| Gate (Amendment 2): mean candidate tail <= 107.94 - 3.0 | 104.94 |
| Shortfall against the gate | 2.86 (107.801 - 104.94 = 2.861) |
| ref_s47 tail | 107.849 |
| ref_s48 tail | 107.642 |
| Reference band (Amendment 2) | [105.3, 110.6] |
| Mean of candidates minus mean of the two references (107.7455) | +0.056 |

## Ruling
- Reference check: both references lie inside [105.3, 110.6]. The experiment is not infrastructure-inconclusive.
- Gate: 107.801 > 104.94. E2 FAILS. mh_s47 and mh_s48: NEGATIVE_UNDER_CONTRACT. ref_s47, ref_s48: CLOSED.
- Amendment 2: "E2 fails: the route closes, whatever E1 shows." The multi-hop path edge route for GPT-T closes at this contract. No adoption, full training or retry.
- W2 (ref_s49 + mh_s49) is only triggered by "E1 fails and E2 passes". E2 failed, so W2 is not triggered and not submitted.
- The candidate is not measurably different from the reference (+0.056 meV, far below the 3 meV gate and below the 1.3 meV 100K reference SD in decision_p1.md). The result reads as a null effect, not a harmful one.

## Provenance of kernel B v2
- The kernel B v2 package does carry the d73a0c20 partial-arm continuation overlay (`kB_s48_v2/recovery_repair.json`, files `kaggle_pair_runtime.py` and `family_training_runner.py` with original and repaired sha256). Its `source_commit` 873ee479 is the base archive, not the repaired source; the overlay is the repair on top of it.
- v2 resumed mh_s48 only from 57/60 to the frozen 60 passes; ref_s48 was retained from its v1 terminal output (pinned by output-manifest sha256). Recipe, init, data, gate and band did not change.

## Erratum
Protocol "Amendment 3 erratum" (kernel B v2 not submitted; mh_s48 PENDING at 57/60) is out of date. v2 was submitted and mh_s48 reached 60/60. See the protocol erratum 2 appended on this date.

## Procedural deviation (recorded, accepted on evidence)
kernel B v2 was submitted on 2026-11-13 17:50 UTC. `kB_s48_v2/continuation_authorization.md` sets the earliest submission after the 2026-11-14 00:00 UTC quota reset, so the submission was about 6 h early. At that time kaggle1 had used 150,343 of 162,000 GPU seconds; the run took about 1,740 s, which fits in the remaining 11,657 s, so no quota was exceeded and no other run was displaced. The result is accepted on that evidence and the deviation is stated here rather than hidden. Wall ceiling 5,400 s was not reached.

## Effect on other decisions
E1 verdict, N1, P1 and every frozen threshold are untouched. The route is closed under Amendment 2; reopening needs a new pre-registered question.
