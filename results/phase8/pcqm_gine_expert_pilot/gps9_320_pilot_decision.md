# PCQM GPS9-320 Architecture Pilot Decision

The pilot used the exact accepted GINE 250K sample, scaffold split, graph
cache, and fixed official-validation 5K. Its GPS encoder had nine layers, 320
hidden channels, eight heads, and 12,243,841 parameters.

The best development epoch was 11 at `0.462255 eV`. Training and development
MAE became non-finite from epoch 15, and the best checkpoint reached only
`0.491629 eV` on the fixed official-validation 5K.

Decision: reject this implementation and training configuration. The result
does not establish that GPS is intrinsically worse than GINE: the pilot lacked
the positional-encoding and optimization protocol used by competitive
PCQM4Mv2 GPS systems and became numerically unstable. Do not scale or resume
this checkpoint. Any future PCQM GPS experiment must first reproduce a
published bounded baseline with finite training before increasing sample size.
