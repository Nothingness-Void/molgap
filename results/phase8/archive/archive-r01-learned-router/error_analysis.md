# archive-r01 Learned Router Error Analysis

| evaluation | actual gain | predicted gain | actual win | predicted win | gain Spearman | learned precision | fixed precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| common_all | +0.004311 | +0.006992 | 52.4% | 58.1% | +0.097 | 56.7% | 57.0% |
| common_ood1000 | +0.000616 | +0.006669 | 48.5% | 57.9% | +0.018 | 50.3% | 57.6% |
| common_p8_targeted_hard | +0.008085 | +0.007321 | 56.4% | 58.4% | +0.162 | 64.0% | 56.7% |
| pcqm_proxy | -0.004686 | +0.008150 | 44.5% | 58.5% | -0.023 | 43.8% | 47.4% |

**Conclusion:** the Router has weak useful ranking on the expansion500k held-out domain, but that ranking does not transfer to PCQM. It overestimates expert utility under the shifted expert-win prior. Do not tune on external tests; keep fixed routed-v4.
