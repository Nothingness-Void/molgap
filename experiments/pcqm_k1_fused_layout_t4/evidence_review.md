# Evidence and missing discriminator

[Accepted local10ep result](../pcqm_k1_fused_layout_speed/terminal_decision.md)
retained24.13% lower joint arm-pipeline time on RTX5060. It read TRAIN only,
used an explicit unfused control, and does not qualify accuracy or native T4.
[A100 execution diagnostic](../pcqm_k1_colab_execution_profile/attribution.md)
found small fixture loader wait; it does not prove CPU dominates whole runs.
Single/mean2 quality work changes supervision/BN/dropout count and cannot
answer whether the same single-forward objective tolerates fused rounding.

New discriminator: same-job fixed100K original/native-default versus fused
plus CPU layout with unchanged scientific recipe, complete40ep, internal50K
selected paired predictions and measured native windows. Hypothesis: reducing
small optimizer launches/repeated GPU layout metadata reduces time without
material endpoint degradation. Alternatives: native foreach already closes
the gain; fused accumulation changes quality; shared CPU or development/IO
overhead hides training gain. Device placement/noise and one-seed limitations
remain explicit. No historical scientific conclusion is rewritten.
