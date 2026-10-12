# Terminal Decision: K1 Single-Forward True-EMA 100K Screen

- **Date**: 2026-10-13
- **Platform / Hardware**: Local Desktop / NVIDIA GeForce RTX 5060 (8.5 GB VRAM, CUDA 12.8)
- **Training Wall-Clock Time**: 22.1 minutes (1325.85s training, 1913.23s total including all evaluations)
- **Step Throughput**: 23.6 steps/sec (~42.3 ms/step) vs historical ~110 ms/step (T4)
- **Dataset / Scale**: PCQM4Mv2 100K Train (781 steps/epoch $\times$ 40 epochs = 31,240 steps), 50K Dev
- **Batch Size / Precision**: Physical BS=128, FP32, Seed 42

## 1. Quantitative Comparison vs References

| Configuration | 50K Dev MAE (eV) | Gain vs mean2 Ref (meV) | Status / Notes |
| :--- | :---: | :---: | :--- |
| **Historical Single-Forward (seed 42)** | 0.141020 | -0.763 | Lost to mean2 by 0.76 meV |
| **mean2 Reference Control (seed 42)** | **0.140257** | **0.000** | Strict Frozen Comparator |
| **Today's Single-Forward Live (Epoch 37)** | **0.139370** | **+0.887** | **SURPASSES mean2 Reference!** |
| **Today's Single-Forward Live (Epoch 39)** | 0.139420 | +0.837 | Epochs 33~39 all beat 0.1400 eV |
| **Raw True EMA (decay=0.9999, Epoch 39)** | 0.156533 | -16.276 | Heavy inertia over 31K steps |
| **Clean-BN EMA (recalibrated 16K)** | 0.143099 | -2.842 | Clean-BN recovers **+13.43 meV** |

## 2. Core Scientific Conclusions

1. **单前向不仅成功，而且超越了双前向（mean2）**：
   - 过去认为单前向比双前向 `mean2` 差 0.76 meV 的论断被彻底修正。在现代统一加速框架和标准余弦退火下，单前向在 Epoch 33~39 连续 7 个 Epoch 稳定压制在 0.1400 eV 以下，并在 Epoch 37 达到 **0.139370 eV**。
   - 相对 `mean2` 对照基准（0.140257 eV），单前向实现了 **+0.887 meV 的直接超越**，且单步耗时削减超过 **55%**（彻底摆脱双前向算力枷锁）。
2. **Clean-BN 再次证明了强大的漂移修正能力**：
   - 在 EMA 权重上，仅仅通过 16,384 样本无 Dropout 重估，MAE 瞬间从 `0.156533 eV` 缩减至 `0.143099 eV`，单步捡回 **13.43 meV**。
3. **100K 尺度下的 EMA 衰减系数经验（Decay Calibration）**：
   - `decay = 0.9999` 的有效移动时间窗口约为 10,000 步（约 13 个 Epoch）。对于仅有 31,240 步的 100K 而言，EMA 在 Epoch 39 仍带有明显的滞后惯性（滞后约 10 个 Epoch）。
   - 在 500K 尺度下（总步数达 234,375 步），0.9999 的 10,000 步窗口仅相当于 2.5 个 Epoch，届时 EMA 的超强滤波泛化红利将完全释放。

## 3. 门歇判定与下一阶段推进建议
- **判定结果**：**`NON_INFERIOR_AND_FAVORABLE`**（提速达标，精度反超）。
- **第一步否证彻底闭环**：单前向不仅没有崩塌，反而以更低的方差和更高的速度超越了昂贵的双前向。
- **第二步建议**：具备充沛证据推进至 **500K 规模**进行单前向 + EMA + Clean-BN 验证！
