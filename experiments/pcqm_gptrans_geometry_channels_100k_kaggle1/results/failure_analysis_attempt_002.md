# Post-hoc analysis of the angle increment

This analysis uses the accepted attempt-002 remote trace and aligned 50,000-row
internal-development predictions. It also uses the frozen development graph
shard `train/train_shard_0002.pt` (SHA256
`f8c0d054d4794ce9a8887e6c1799806d89533f6ed3aeb93f3e838b7319ac842a`)
for descriptive geometry groups. The shard's ordered `source_idx` equals the
prediction payload's ordered `source_idx`. No official validation or test role
was read. All subgroup and local live-checkpoint analyses below are post-hoc;
the frozen gate remains the remote EMA result in
[`paired_comparison_attempt_002.json`](paired_comparison_attempt_002.json).

## What happened during training

| Epoch | Distance-only EMA development MAE | Distance-angle EMA development MAE | Angle minus distance |
| ---: | ---: | ---: | ---: |
| 21 | 0.297322 eV | 0.297022 eV | -0.300 meV |
| 22 | 0.287209 eV | 0.290315 eV | +3.105 meV |
| 34 | 0.209908 eV | 0.233270 eV | +23.363 meV |
| 49 | 0.165034 eV | 0.174535 eV | +9.500 meV |
| 59 | 0.151979 eV | 0.156768 eV | +4.789 meV |

The angle arm was briefly ahead early, then fell behind after epoch 21. Its
relative deficit peaked near epoch 34 and narrowed thereafter. Both arms
selected epoch 59, and their EMA development MAEs fell over the last ten
epochs by 13.055 and 17.766 meV, respectively. The final learning rate was
the scheduled minimum `1e-6`. The endpoint is therefore not a flat learning
curve, but the frozen 60-epoch result does not show an angle advantage.

At epoch 59 the online training MAE was 0.102862 eV for distance-only and
0.101034 eV for distance-angle, a 1.828 meV angle advantage on training
batches. These online live-weight training losses are not directly comparable
to the final EMA development losses. Their opposite ranking is consistent
with poorer transfer of the learned angle pathway to the development rows;
it is not proof of a unique overfitting mechanism.

The retained final checkpoints permit a **supplemental local** live-weight
development evaluation on an RTX 5060. On the same 50,000 source indices,
distance-only gave 0.145197 eV and distance-angle gave 0.146996 eV: angle
remained worse by 1.799 meV. A 10,000-draw paired row bootstrap gave
`[+0.840, +2.774]` meV. The local EMA replay was not bitwise identical to
Kaggle T4 output (mean absolute prediction differences about 0.02 meV, with
larger isolated differences), so these live values are diagnostic only and
must not replace remote acceptance or fill its missing per-epoch live trace.
The checkpoint and local-prediction digests are recorded in
[`local_live_checkpoint_diagnostic_attempt_002.json`](local_live_checkpoint_diagnostic_attempt_002.json).

At the final checkpoint, local EMA minus local live MAE was approximately
6.783 meV for distance-only and 9.771 meV for distance-angle. The extra
2.988 meV angle EMA lag explains part of the remote 4.789 meV deficit, but
even the local live comparison retains a 1.799 meV angle deficit. With EMA
decay 0.9999, the weight half-life is about 6,931 optimizer steps, or 8.87
epochs at 781 steps per epoch. More EMA catch-up is plausible; a reversal and
the required 3.0 meV *advantage* are not established by this checkpoint.

## Which molecules carry the difference

The remote paired endpoint has 47.326% rows improved and 52.674% worsened
by the angle arm; the median per-row absolute-error change is +4.240 meV.
The mean prediction shift is -11.936 meV, and angle predictions have a
larger mean underprediction (-23.631 versus -11.695 meV). A constant shift
fitted on this same development role reduces, but does not eliminate, the
MAE gap (154.246 versus 151.191 meV after separately optimal median-bias
shifts). This is diagnostic calibration accounting, **not** a valid
post-calibration score or an authorized model change.

The result is not driven only by invalid geometry. There are 168 invalid
geometry rows (0.336%) with a +45.084 meV mean paired change; they
contribute about 0.152 meV of the total +4.789 meV. The 49,832 valid rows
still show +4.653 meV. Among the 44,340 rows marked MMFF-converged, the
angle arm is worse by +4.587 meV. Thus most of the aggregate loss occurs
where the generated geometry is marked usable.

Exploratory target deciles show a tradeoff: the common 4.81–5.52 eV region
loses roughly 8.8–9.9 meV, whereas the highest target decile (above
6.80 eV) gains 5.65 meV. The mean angle-minus-distance prediction shift is
negative in the middle target deciles and positive in the highest decile.
These bins were inspected after the result and are not a new selection gate.
Atom and wedge counts have weak correlations with per-row error change
(`r=0.019` and `r=0.026`), so molecule size alone does not explain the loss.

## Mechanism and limits

The frozen source adds a zero-initialized 16-basis cosine-angle projection,
averages its directed wedges at each center atom, and injects the result into
the initial node state. Distance is injected into real-bond pair state in both
arms. The final angle projection is nonzero (live weight L2 norm 8.131), while
the dormant reference projection remains zero, so the new path did learn.
The frozen source contains explicit geometry validity masking and wedge/edge
shape checks; the accepted run has no observed nonfinite state or role leak.

The evidence supports a **generalization/representation cost** of this
particular angle-to-node path at fixed 100K/60 epochs, with additional EMA
lag. Possible explanations include redundant or noisy single-conformer angle
information, loss of directional wedge detail from mean pooling, and the
un-gated injection changing the GPTrans node trajectory. The current pair
cannot isolate these explanations: the two trained cores differ after
optimization. A prospectively recorded, saved-checkpoint angle-path ablation
would separate direct inference contribution from training-path effects
without another training run. A longer or multi-seed training claim would
require a separate frozen contract. The historical 500K 2D-versus-combined
geometry result uses a different model/data/role contract and does not answer
the incremental 100K distance-versus-angle question.

Under the frozen shortlist gate, the observed angle arm would have needed
MAE at most 0.148979 eV against the observed 0.151979 eV reference; it was
0.156768 eV, a 7.789 meV shortfall to the gate. No scale-up or repeat is
supported by this analysis.
