# Kunshan projected-moment readout seed-42 decision

Job `121048634` completed the frozen paired K2 comparison on one Kunshan Hygon
DCU. Mechanical acceptance passed with 10,000 unique aligned internal-
validation identities, complete artifact hashes, finite recomputed metrics,
and no access to official validation or test-dev.

The fresh GraphState9 mean-readout control reached `0.1297868306 eV` Gap MAE
at epoch 39 with 3,665,809 parameters and `256.73 graphs/s`. The projected
nonlinear first/centered-second-moment readout reached `0.1294588245 eV` at
epoch 35 with 3,684,753 parameters and `273.98 graphs/s`. Peak reserved memory
was identical. The paired difference was `-0.0003280061 eV`, while measured
throughput was 1.067x the control; the speed difference is treated as run
variation because the candidate adds work rather than removing it.

This demonstrates that the bounded moment readout is inexpensive and does not
destabilize training, but its 0.25% relative MAE reduction is below the
discovery plan's `0.001 eV` promotion threshold. K2 is therefore closed as a
weak observation. Do not run seeds43/44 or full-data training for this
candidate. GraphState9 remains the frozen anchor, and K3 conjugated-component
communication is the next distinct information-flow question.
