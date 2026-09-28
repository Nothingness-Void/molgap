# Feature-denoising failure-mode audit

The [terminal decision](../decision.md) is `INCONCLUSIVE` for strict
qualification. Both arms trained under a source/optimizer identity different
from their prospective contract: executed AdamW LR/weight decay/end LR were
0.001/0.05/0.000001, while the frozen contract specified
0.0002/0.01/0.000002. The source commit also changed at retry without a new
prospective binding. These are observed identity failures, not an inference
from model accuracy. The exact checks are in
`platforms/_records/kaggle/training/pcqm_gptrans_feature_denoising_s42_v2/mechanical_acceptance/mechanical_acceptance.json`.

The two retained same-job traces each have 60 epochs and the same 46,860-step
endpoint. Under the *executed* recipe, adding bond reconstruction to full-atom
reconstruction improves development MAE by 1.302 meV at epoch 59. That
relative lead averaged 1.977 meV over epochs 50–59 and shrank from
3.005 meV at epoch 49. Both arms' development MAE still fell over the last
ten epochs (12.547 and 10.843 meV respectively); each selected epoch 59.
Their final auxiliary cross-entropies were about 0.00027, showing that the
auxiliary targets were learned on training batches. The bond arm's final
online normalized Gap training MAE was slightly *higher* (0.074881 versus
0.074226), despite its lower development MAE. These online and development
quantities have different semantics and units and cannot establish a
generalization mechanism.

This audit can rule out the claim that the run failed to execute or that the
auxiliary objective never learned. It **cannot** tell whether the intended
contracted recipe would help, whether more exposure would reverse the
reference comparison, or whether the auxiliary task harmed Gap
generalization. The observed arms both missed the historical reference, but
the contract/source mismatch bars a strict causal or promotion claim. A
repeat solely to repair bookkeeping is not justified. A future question
would need a new frozen contract and an independent reason why this
mechanism should beat the existing accepted models.

Inputs: the accepted `full_atom` and `full_atom_bond` raw `trace.json`
files under the same Kaggle record, plus the terminal decision. This
post-hoc calculation used no new training, checkpoint inference, role
access, or remote request.
