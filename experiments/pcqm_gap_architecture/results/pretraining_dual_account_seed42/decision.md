# Seed-42 same-database pretraining decision

On 2026-09-08 both paired T4x2 screens completed and passed no-model
acceptance. They used the same accepted 100K/10K PCQM train-derived cache, the
same initial GraphState9 encoder hash, and no external or sealed role.

The structure-source-task initialization reached `0.1293375633 eV` after its
40-epoch Gap fine-tune. This was `0.0026715487 eV` below its paired scratch run
through 40 epochs, but `0.0024720851 eV` above the same scratch run through 60
epochs. The apparent early advantage is not robust enough to override the
compute-normalized result: the other account's independent scratch-through-40
result was `0.1293169800 eV`, only about `0.000021 eV` better than the
pretrained result, while the two nominally identical scratch-through-40 runs
differed by about `0.002692 eV`. The current aggregate composition/hashed-
fragment source task therefore provides no accepted transfer gain.

The ETKDG geometry-histogram denoising initialization reached `0.1299758942
eV`. It was worse than its paired scratch control both through 40 epochs
(`+0.0006589142 eV`) and through 60 epochs (`+0.0036000067 eV`). The current
graph-level histogram/moment denoising target is scientifically negative.

Both exact objectives are closed without confirmation seeds. This decision
does not reject all pretraining: it specifically rejects graph-level aggregate
source tasks and graph-level ETKDG histogram denoising as implemented here.
Any successor must change the supervision locality or learning objective,
retain an equal-compute scratch control, and be admitted only after the
expanded literature audit. GraphState9 remains the downstream architecture
anchor. No full-data run, official validation/test-dev access, production
change, or molecular-research-server action is authorized by these results.
