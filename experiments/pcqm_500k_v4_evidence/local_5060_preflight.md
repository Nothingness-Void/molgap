# Local RTX 5060 V4 preflight, 2026-09-16

The desktop environment passed a bounded physical-BS128 forward/backward and
optimizer-step probe against the accepted V4 development shard. No formal
training was started.

- GPU: NVIDIA GeForce RTX 5060, 8,151 MiB, compute capability 12.0.
- Driver: 616.92.
- Runtime: PyTorch 2.7.1+cu128 with native `sm_120` support.
- Input shard SHA256:
  `1a37dc5d458396b590044d978526822e6d1cb35df223de913a837dc7277d64c1`.
- Probe batch: 128 graphs, 1,811 nodes, 3,798 directed edges.

| Accepted arm | Steady optimizer step | Peak allocated GPU memory |
|---|---:|---:|
| EdgeState/full GPS | 0.103-0.124 s | 0.618 GiB |
| Neural-Atom K1 | 0.092-0.117 s | 0.520 GiB |
| GPTrans-T | 0.081-0.086 s | 0.803 GiB |

The first EdgeState step spent about 15.7 seconds initializing CUDA kernels;
subsequent steps were stable. The strict V4 physical batch 128 therefore fits
comfortably without AMP, TF32 or gradient accumulation.

Only the accepted 50K development shard was present locally at this preflight.
The ten accepted 50K training shards (about 2.0 GB total) still need to be
retrieved and hash-checked before formal local training. At observed steady
throughput, one 60-epoch arm is expected to require roughly 7-9 hours including
data loading and development evaluation; the two causal ablation arms must run
sequentially on the single GPU, roughly 14-18 hours total.
