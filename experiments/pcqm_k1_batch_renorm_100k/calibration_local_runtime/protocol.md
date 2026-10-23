# Selected LIVE Clean-BN: explicit local runtime qualification

Same scientific question and authority as the parent [protocol](../protocol.md)
and [original diagnostic](../calibration/protocol.md). Attempt001 was finalized
NO_TRAIN before any model forward: its local Torch2.7.1+cu128 differed from the
immutable training recipe's2.11.0. No primary environment or recipe is changed.

Attempt002 explicitly pins observed local Torch2.7.1+cu128, cuda:0 NVIDIA GeForce
RTX5060 and CPU4 threads. Original selected-LIVE predictions must first reproduce
the retained50K development predictions with identical rows/targets and maximum
absolute error at most1e-4eV. This qualifies only this local inference diagnostic;
failure stops, with no tolerance relaxation or calibration claim.

Then preserve the same four predeclared MAEs, original and calibrated
brn_scale/brn_affine; one cumulative Clean-BN pass on the same ordered first16384
training rows, BS128, dropout off, labels unused for calibration; evaluate the
same previously selection-used50K development rows. Report both affine-minus-scale
contrasts with2000 paired row bootstrap draws, seed42. Keep weights and non-BN
buffers fixed and preserve original artifacts. No state selection or promotion.

Total remaining process wall ceiling1780s, including identity/data verification,
model construction, reproduction, calibration, inference and analysis. Measure
wall, CPU and one local GPU allocation-window time separately. No optimizer,
gradients, training, remote allocation or protected roles. Data validation reads
100K training and50K development labels; record those actual reads separately.
Close NO_TRAIN with observed evidence or explicit incomplete blockers. Bind the
finalized prior failure trajectory/evidence/result in the new prospective.
