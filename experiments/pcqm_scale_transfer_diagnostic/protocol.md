# Protocol: PCQM 100K-to-500K Scale-Transfer Diagnostic

## Question
Can cross-scale sample exposure mapping, empirical margin retention frontiers, and anti-collapse regularization metrics prospectively predict whether a 100K PCQM architecture winner retains material gain (>0.003 eV) at 500K scale?

## Evidence Boundary
- Zero training, zero model inference, zero accelerator hours.
- Evaluates existing accepted RML records (`replay_pool.json`, `canonical_trace.json`, and dated decisions).
- Evaluates both historical negative transfers (`K1 PairToken`, `GPTrans Pair PreNorm`) and new regularized candidates (`GPTrans Noisy Nodes`, `GPTrans Noisy Nodes + Pair Update Norm`).
- No protected evaluation roles (Official Validation, Test Dev/Challenge) are read or unsealed.

## Methodology & Formulations
1. **Cross-Scale Exposure Mapping**:
   - $x = \text{sample\_presentations} = \text{epoch} \times \text{dataset\_size}$.
   - Equivalence anchor: 100K 60 epochs ($5,998,080$ presentations) corresponds to exactly 500K Epoch 11.996 ($\approx 12.0$).
2. **Empirical Retention Frontier**:
   - Historical baseline decay: $81.7\%$ decay across $7.5\times$ optimization steps without anti-collapse regularization.
   - Regularized decay: anti-collapse node denoising ($\sigma=0.15, \alpha=0.10$) and pair update LayerNorm reduce decay to $\le 20-30\%$.
3. **Scale-Transfer Qualification Score (STQS)**:
   - Margin Subscore ($0-50$): $20 + (\Delta_{100K} - 0.003) / 0.005 \times 30$.
   - Node Anti-Collapse Subscore ($0-25$): $25$ if auxiliary denoising is present.
   - Pair Update Normalization Subscore ($0-20$): $20$ if relation LayerNorm is present.
   - Tail Stability Subscore ($0-5$): $5$ if tail learning curve has negative or zero slope.
   - Qualification Gate: $\text{STQS} \ge 60$ required to authorize 500K scale-transfer.
