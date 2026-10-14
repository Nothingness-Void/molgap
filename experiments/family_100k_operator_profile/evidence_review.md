# Selection review - 2026-10-14

K1 strongest established reusable100K recipe is mean2/live at0.140257285566eV:
[accepted fresh control](../pcqm_k1_clean_second_100k/terminal_acceptance/decision.md).
Dropout consistency has a lower0.138729point but missed the3meV material gate;
single/live0.139370 is numerically lower while its frozen clean-EMA endpoint
is0.143099 and formal qualification is incomplete. Do not call those adopted
best recipes. See [curve review](../../docs/operations/DESKTOP_RML_CURVE_REVIEW_20261013.md).

GPTrans's stronger retained numerical candidate is corrected degree-scaled
local-bond/EMA.999: server full60 endpoint0.1423582275eV, below its material
architecture gate; optimized40-of60 delivery endpoint0.1419957587eV passed
cost/quality delivery. These different exposures are not ranked as a causal pair.
[Frozen full60 snapshot](gptrans_selection.snapshot.md) is copied unchanged
from server `bdc4d238`, path
`experiments/pcqm_gptrans_capacity_relations_100k/gpu/degree_bond_local_ema999/results/decision.md`.
[Delivery decision](../../platforms/_records/integrations/gptrans_execution_tuning_v3/server_evidence/decision_terminal.md)
owns the numbers and limits. This is a user-selected profiling subject,
not an architecture promotion or a claim of an exhaustive global minimum.
The desktop registered core lacks this existing component: reuse only the
reviewed BondLocalBlock/BondLocalGPTrans from server `bdc4d238`, with identical
`gptrans.py`; do not merge server orchestration/history. Tensor hash of the
CPU-constructed frozen full initial state is independently checked against
`a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b`.

Prior [local shared-session diagnostic](../family_session_execution_speed/attribution.md)
did not establish a speed gain. Its small synthetic graphs and32-step passes
cannot locate native molecular operator cost. The new discriminator is a
bounded, real-training operator trace through the same shared wheel.
Historical references are context only; no strict reference is invented.
