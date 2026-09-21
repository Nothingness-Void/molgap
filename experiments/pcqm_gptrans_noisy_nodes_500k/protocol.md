# GPTrans-T Noisy Nodes 500K Scale-Transfer Protocol

## Frozen Question

Under the matched `pcqm-fixed500k-dev50k-matched60-v4` contract on PCQM4Mv2, does the auxiliary Noisy Nodes denoising regularisation mechanism (which achieved a +0.004840 eV gain at 100K) effectively scale to 500K graphs and beat the frozen GPTrans-T 500K reference baseline (`0.106868 eV`) by at least the 0.003000 eV nomination threshold (development MAE < 0.103868 eV)?

## Contract Specifications

- **Contract**: `pcqm-fixed500k-dev50k-matched60-v4` under `MOLGAP-COMMON-V5-FINAL`
- **Data Cache**: `kaseichou/pcqm4mv2-ogb-fixed-500k-scnet-v1`
  - Manifest SHA256: `630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`
- **Roles**:
  - Training: source indices `0..499999` (500,000 graphs)
  - Internal Development: source indices `500000..549999` (50,000 graphs)
  - Sealed Roles: official validation, test-dev, test-challenge remain unread
- **Architecture**:
  - Backbone: GPTrans-T (12 blocks, node channels 256, pair channels 32, 8 heads)
  - Noisy Nodes: Gaussian embedding perturbation $\sigma=0.15$ during training, linear reconstruction head projecting atom embeddings to 119 OGB atom classes, cross-entropy auxiliary loss with weight $\alpha=0.10$
  - Parameter count: `5,277,400`
  - Evaluation mode: 0 inference noise, auxiliary head bypassed, 0 compute overhead
- **Training Hyperparameters**:
  - Precision: FP32 (no TF32, deterministic algorithms enabled)
  - Seed: 42
  - Physical batch size: 128 (drop_last 32 tail rows per epoch)
  - Epochs: 60
  - Optimizer: unfused AdamW (initial lr 4e-4, weight decay 1e-5, grad norm clip 1.0)
  - Schedule: cosine decay from 4e-4 to 1e-6 over 60 epochs
  - Steps per epoch: 3,906 (total optimizer steps: 234,360)
  - Sample presentations: 29,998,080

## Decision Gate & Falsifier

- **Comparator**: `pcqm-matched-500k-v4-three-arm` GPTrans-T baseline (`0.106868 eV`)
- **Falsification Threshold**: If best development MAE $\ge 0.103868\text{ eV}$ (i.e. fails to achieve $\ge 0.003\text{ eV}$ gain over the comparator), the 500K scale-transfer attempt is falsified and determined `NEGATIVE_UNDER_CONTRACT`.
- **Nomination**: If best development MAE $< 0.103868\text{ eV}$ with paired bootstrap interval clearing 0, the candidate is nominated as a superior 500K architecture.
