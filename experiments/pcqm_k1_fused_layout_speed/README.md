# K1 local ten-epoch execution check

Entry: [frozen protocol](protocol.md); [planning decision](decision.md).
This is a TRAIN-only speed prefix, not a quality experiment or model promotion.
The local adapter reuses K1 factory/optimizer step/order/stats and atomic IO.
The registered T4/full-40ep trainer remains unchanged.

CPU prepare: `python experiments/pcqm_k1_fused_layout_speed/prepare.py --repo-root <root>`.
The receipt points to frozen source/config and the exact local command.
Only run under the explicit 10ep / shared 3600-second GPU authority.
No platform submission or recurring monitor is part of this question.
