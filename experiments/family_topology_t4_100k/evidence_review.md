# Evidence Review - 2026-10-14

RML contains finalized unpaired complete operator trajectories for K1 and
GPTrans, with valid artifacts and explicit missing-reference replay exclusions.
Read the [terminal custody](../family_100k_operator_profile/terminal_acceptance/closure.md)
and [attribution](../family_100k_operator_profile/terminal_acceptance/attribution.md).

Baseline K1: internal Gap MAE0.1402572855657339, selected37/live, worker-entry
10763.485608378s. GPTrans: MAE0.14286130827367305, selected52/EMA.999,
11688.317988879s. Source/input/runtime/profile records are retained and hash-bound.
The16-step windows show little loader waiting and many CUDA launches. The
topology removals address real repeated expressions but may be too small to
yield a material whole-run improvement; no acceleration has yet been measured.

[Topology verification](../family_100k_operator_profile/terminal_acceptance/topology_verification.md)
retains207CPU passes,88package/release passes with1host symlink skip and a
separate successful short K1 transport check. The first incidental native
failure remains unexplained; no new T4 certificate or GPTrans native performance
qualification follows from CPU checks. This authorized run supplies native
qualification and complete matched-recipe endpoints, not a model promotion.

There is no same-physical-job old-source control. These baseline evidence IDs
are prior context, not new strict references. No closed architecture is reopened,
no official role is consumed and no broad threshold or stop policy is calibrated.
