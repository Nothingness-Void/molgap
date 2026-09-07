# Hop-path GraphState seed-42 decision

The mechanically accepted paired screen found a real but sub-threshold gain.
The exact-shortest 2/3-hop candidate reached `0.1291064769 eV`, compared with
`0.1300130934 eV` for its fresh GraphState9 control, a paired improvement of
`0.0009066164 eV`. The frozen protocol required at least `0.001 eV`, so this
result did not qualify for confirmation.

The mechanism was directionally consistent late in training: it beat the
control at all ten epochs from 30 through 39, with a mean late-epoch difference
of `-0.0011884198 eV`. It added only 31,728 parameters (0.87%), but reduced
throughput by 10.66% and increased epoch time by 11.93%. Its training MAE fell
more than validation MAE, increasing the generalization gap by
`0.0015462241 eV`. This combination supports the path-information hypothesis
but does not justify changing a precommitted acceptance threshold.

The route is therefore retained as weak positive evidence and closed without
seed 43/44, width, path-depth, optimizer or schedule variants. The confirmed
GraphState9 remains the server recommendation. No full-data training, official
PCQM validation/test-dev read, desktop handoff, or molecular-research-server
action was authorized by this result.
