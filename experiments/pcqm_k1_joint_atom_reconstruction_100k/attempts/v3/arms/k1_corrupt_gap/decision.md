# Corruption-only Gap: terminal decision

On 2026-09-28 JST, the accepted v3 seed42 arm was closed as
`NEGATIVE_UNDER_CONTRACT`. Original-dev MAE was 0.1417259396 eV versus frozen
K1 at 0.1413736414; the paired interval crossed zero and the material gate
failed. All 40 epochs completed; this was not an infrastructure failure.

The typed intervention was the corruption-bound training objective; the K1
inference architecture was unchanged. The separate frozen500K audit was
unfavorable, but was not 500K training. No retry, extra seed, scale-up or
successor was released. Full contrasts and limitations are in the
[study decision](../../../../decision.md); accepted native artifacts and
the observed strict comparison are retained beside this decision in `results/`.
