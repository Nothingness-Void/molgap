# MsgNorm amplitude attribution

The frozen comparison tests whether nine learned message scales improve on
fixed unit scales. It does not test whether message normalization itself is
better than unmodified K1. The terminal decision preserves the3meV gate.

| Completed pass | Fixed Live MAE, eV | Learned Live MAE, eV | Learned minus fixed, meV |
|---|---:|---:|---:|
|10|0.1996065080|0.2009306848|+1.324177|
|20|0.1626469493|0.1629038453|+0.256896|
|30|0.1494006515|0.1479302645|-1.470387|
|40|0.1461331695|0.1447691172|-1.364052|

Both selected checkpoints are final pass40. The last10 passes improve fixed
by3.267482meV and learned by3.161147meV. Late relative advantage fluctuates;
these curves do not establish that extending the horizon would increase the
contrast. Trace float32 reduction and acceptance float64 recomputation differ
slightly; the decision uses the latter on retained aligned predictions.

Selected and final learned scales for layers1-9 are identical:
`[1.182488680,1.943668008,1.757683635,1.632215500,1.552553296,1.482280135,1.353539824,1.314676404,1.336131811]`.
All remain positive and exceed initialization1; fixed scales remain1. The
module receives optimization signal and increases relative message magnitude.
Scale movement alone neither proves an original scale pathology nor identifies
which layer causes the gain. Checkpoints were read safely on CPU; no inference
or training was performed for this inspection.

The candidate improves25234/50000 rows (50.468%). Tail error quantiles improve
slightly; mean signed error increases from0.01428546 to0.01952325eV. A post-hoc
same-role constant shift retains rather than removes the advantage. Thus a
simple global bias correction does not explain the full difference. All such
diagnostics are descriptive; same-development optimized blends/offsets are not
held-out performance or promotion evidence.

Online normalized training loss is not a fixed-cohort eV evaluation. Raw
per-layer message norms, matched fixed-training-cohort metrics and multi-seed
variability are absent. Underfitting, overfitting, exposure shortage and absolute
superiority to original K1 therefore remain **insufficient_evidence**. EMA
development evaluation is absent; selection is explicitly Live.

## Disposition

Learned amplitude offers a small favorable conditional effect, with no observed
added wall penalty in this one simultaneous pair. The sample is insufficient
for a causal speed claim. Do not scale this recipe to500K or repeat it solely
because pass40 is best. Keep MsgNorm as a subthreshold mechanism finding and
prioritize routes with a decision-relevant discriminator. Any future absolute
MsgNorm claim first needs an existing qualified original-K1 comparator with
matched recipe/exposure, or a separately justified prospective comparison;
this closure grants neither baseline retraining nor another job.
