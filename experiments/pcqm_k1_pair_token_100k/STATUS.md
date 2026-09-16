# Status

The protocol was submitted once to Kunshan as job `122272523`. The immutable
source commit is `396c2872b40e78c14bc588a0fe9985594236d1e1`; source archive
SHA-256 is
`61022724d7eb28bafb73ab342ae618a8cd753cb75707f2948939bf9ad0f2ade7`.
The job was first observed pending for priority with no other account job
queued or running. Terminal evidence must be accepted before interpretation.

Job `122272523` failed in preflight before training because the static contract
omitted the 64 affine parameters of the token LayerNorm. Runtime and cache
certification passed and no scientific evidence was produced. The corrected
identity is 3,681,665 parameters; the architecture and all training settings
remain unchanged for one infrastructure-only retry.

