# Deep local-update control: terminal interpretation

On 2026-10-07, Kaggle2 kernel `kaseichou/molgap-gptrans-local-control-s42`,
ID137425845/version1, completed. The existing Luna B delivered one terminal
event; A claimed it, reconciled the physical version, and retrieved only exact
version-specific metadata and manifest-bound artifacts. The independent
[acceptance](acceptance.json) passed without local training or model inference.
Detailed observations and input hashes are in [interpretation.json](interpretation.json).

## Result and disposition

| Same-contract endpoint | Parameters | Best epoch, zero-based | Gap MAE, eV |
|---|---:|---:|---:|
| Frozen uncapped local-bond reference | 5,871,201 | 41 | 0.1423582275 |
| Parameter-free cap0.25, layers4–12 | 5,871,201 | 46 | 0.1433860061 |

The candidate worsened by0.0010277786 eV. Its candidate-minus-reference
paired-row95% interval was[0.0001212274,0.0019517564] eV;49.74% of rows
improved. The candidate did not pass the prospectively frozen0.003 eV material
gate. The shared terminal policy's INCONCLUSIVE/no-promotion label was retained;
the operational decision closed this exact amplitude-control route. Mechanical
acceptance and Replay admission did not imply a positive scientific result.

The comparator was the already accepted local-bond model, not the weaker G1
model. Both endpoints used identical initial tensors, data/targets/rows,
optimizer, exposure, LR, EMA and selection semantics. No baseline was retrained,
parameter added, checkpoint selection changed, or gate lowered after observing
the result. Single-seed row bootstrap does not establish seed stability.

## Why this was not an inactive or disconnected addon

The first scheduled training batch per epoch retained raw ratios, bounded
ratios and scales for128 molecules at all9 controlled layers. Every value was
finite and all bounded ratios satisfied0.25 within the declared floating-point
tolerance. All9 output branches had nonzero last-batch parameter gradients in
all60 epochs. These gradients and the amplitude samples came from different,
explicitly labeled batches; neither was a complete training-population survey.

The cap engaged on1.076% of sampled layer/molecule pairs in the first ten
epochs and9.262% in the last ten. Late engagement concentrated at layers4/5/6:
37.422%,21.641%,13.125%, respectively. Layer12 engaged only0.547% of sampled
pairs. The intervention genuinely suppressed some updates; it was not merely
an unused configuration flag. Uniformly applying the cap to layers4–12 mostly
affected mid-depth branches on this panel, not uniformly the deepest branches.

## Trajectory interpretation

Epoch0 training/EMA metrics reproduced the reference, as expected from
identical initialization and zero-initialized local outputs. At epoch9 the
candidate had a tiny0.0003044 eV EMA advantage, but it was already worse at
epoch19 and remained worse at the reference-selected epoch41. At epoch46 its
EMA score briefly beat the reference's same-step score, but not the reference's
frozen best endpoint. That same-step observation was not substituted for the
prospectively selected endpoint comparison.

Final training MAEs were almost equal:0.074706 versus0.074937 eV, while final
EMA development was0.143685 versus0.142773 eV. The selected-to-final candidate
development score worsened even as training error declined. Thus this cap did
not resolve late generalization erosion under equal exposure. The older
correlation between increasing update magnitude and eroding benefit did not
become evidence that amplitude suppression was a successful fix.

Removed useful local signal, gradients through the per-molecule norm, and
optimization-path divergence remain possible explanations; this single
intervention did not separate them. It did not prove that all residual control,
local-bond mechanisms, or pure2D architectures were ineffective. Target-derived
error-quintile observations in the acceptance were descriptive and subject to
regression-to-the-mean; they did not authorize routing or specialist training.

## Evidence, cost and closed authority

All60 canonical observations,46,860 updates,5,998,080 presentations, aligned
50,000-row saved predictions, selected/final state hashes and six independent
checkpoint chunks passed. The diagnostic log was retained with its SHA as an
analysis artifact; the sole canonical training trace remained the Replay owner.
The comparison was STRICT_CAUSAL with no blockers. The actual rebuilt pool
contained both candidate and immutable local-bond reference with complete
capabilities, no candidate exclusion and equal comparability keys; see the
[Replay-pair proof](replay_pair_proof.json).

Native worker wall time was11,487.902100 seconds, approximately3.191084 hours;
both allocated T4s counted as6.382168 device-hours although one model used one
device. Mean epoch time was188.972562 seconds. These were observed allocation
costs, not utilization or a complete platform-billing measurement. Provisioning,
queue, teardown, CPU and billed quota measurements remained unavailable. The
prospective4device-hour active-model estimate was not retroactively relabeled
as full allocation; the released budget had separately declared8estimated and
14maximum allocated T4-hours.

Two acceptance-infrastructure repairs preserved the scientific contract:
explicitly bounded retrieval of the6.25MB diagnostic JSON, and unambiguous
canonical-versus-analysis trace declaration. Twenty focused synthetic tests
passed; repository RML validation and frozen-derived checks passed after
closure. Mixed pre-existing user reconciliation changes were left unstaged.

No cap grid, extra seed, continuation,500K/full training, protected-role access,
desktop custody transfer or automatic successor was released. The local-control
chain and monitor binding closed, with the existing heartbeat paused.
