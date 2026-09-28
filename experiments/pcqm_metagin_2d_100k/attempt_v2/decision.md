# MetaGIN2D v2 terminal decision — 2026-09-29

The independent pure-2D MetaGIN-derived 4×256, three-hop backbone completed the
prospectively frozen seed-42 PCQM fixed-100K screen. The exact Kaggle2 physical
run was `kaseichou/molgap-metagin-2d-s42:v2` (kernel 136263563). The
[no-inference acceptance](results/acceptance.json) verified the frozen source,
sidecar, 40 observed epochs, 31,240 optimizer steps, 3,998,720 sample
presentations, model/checkpoint/prediction hashes, native cost and sealed
official-validation/test roles.

On the aligned 50,000-row internal-development role, the selected candidate
reached Gap MAE **0.164187572 eV** at epoch 34; the immutable K1-v4 reference
was **0.141373641 eV**. Candidate minus reference is **+0.022813930 eV**;
the paired row-bootstrap 95% interval is **[+0.021574525, +0.024081652] eV**.
This is a clear regression, not a `0.003 eV` nomination-gate pass. The
candidate has 5,268,481 parameters; native Kaggle cost was 5,606.23 wall
seconds and 11,212.45 allocated T4-device seconds (two visible GPUs, one used
for training). No budget overrun occurred.

The candidate has lower error on 43.472% of these rows and on the post-hoc
highest K1-error quintile, but that quintile is selected using targets. It is
diagnostic only, not evidence for a deployable router or promotion. The
observed whole-role loss and its paired interval close this **exact adaptation**
under its frozen contract. It does not falsify the published MetaGIN family.
No further seed, 500K/full run, protected-role access, or automatic successor
is released. Any new 2D architecture must have a distinct, evidence-backed
prospective hypothesis and compute decision.
