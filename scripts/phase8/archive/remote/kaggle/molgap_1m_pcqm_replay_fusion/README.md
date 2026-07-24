# 1M Replay Fusion PCQM Check

This runner differs from `molgap_1m_pcqm_valid/` in exactly one input: it
loads the replay-weighted 1M `FusionHead` while retaining the frozen 1M GPS7,
GPS9, and SchNet weights. The target remains PCQM4Mv2 public-valid Gap only.
Submit it only after the replay-fusion kernel checkpoint is uploaded to the
private `molgap-1m-replay-fusion-model` Dataset.
