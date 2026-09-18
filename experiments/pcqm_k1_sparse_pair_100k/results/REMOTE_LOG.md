# Remote profile log

## Kaggle profile version 3

- Kernel: `nothingnessvoid/molgap-k1-sparse-pair-profile-b6dd3b8`, version 3.
- Source commit: `fb35e8bfe7b4b729caa2341727c1ffb6ca90d3e1`.
- Source archive SHA-256:
  `c619a537ffe725eee9a42798e217a9b8dacd4d69205a1825608fab484a4f02b3`.
- Accelerator: one isolated Tesla T4 from a verified T4x2 allocation.
- Throughput: `1068.9732555864182 graphs/s`.
- Projected 40-epoch training time: `1.0390863847629344 h`.
- Peak reserved memory: `650117120` bytes; reserve fraction
  `0.9584218754584282`.
- Exact two-repeat optimizer-state replay passed.
- Profile artifact SHA-256:
  `163f2b1be04a02a452b63f01ae82c640c5357d5979f1b7995530b3014739bf93`.

The two earlier Kaggle versions were mechanical source-closure failures before
model construction. They are not scientific runs and consumed no training
exposure.

## Kaggle training version 1

- Kernel: `nothingnessvoid/molgap-k1-sparse-pair-train-s42-fb35e8b`.
- Completed epochs / optimizer steps: `40` / `31,240`.
- Best epoch: `36`.
- Development Gap MAE: `0.14110969007015228 eV`.
- Gain versus frozen K1: `0.00026394426822662354 eV`.
- Paired candidate-minus-K1 95% interval:
  `[-0.0011787418522238731, 0.0006223918405175207] eV`.
- Mean training throughput: `826.6882 graphs/s`.
- Best-model SHA-256:
  `42c360f7661e18ce2301a1503fd95dc0a1a24e514dc84edde6c7bda52dd6db80`.
- Prediction-payload SHA-256:
  `6ab5ae1a0b2eb574bd3849108473c739273c3f6187c0d7423b63b1103141b6bc`.
- Completion-manifest SHA-256:
  `7e4dae1a18a9377ba0935a5d45ca1929f63d3e97e2db8e139d5f93ed3354581a`.

Mechanical acceptance passed; the scientific gate failed.
