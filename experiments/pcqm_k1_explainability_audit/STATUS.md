# Status

Stage 1 is authorized for the new Kunshan account. It reuses the accepted
fixed 100K/50K graph identity and 20 already-frozen same-contract K1 prediction
payloads. It performs no model inference or training and reads no sealed role.

Stage 2 and any successor architecture remain conditional on the Stage-1
decision required by `protocol.md`.

The first Stage-1 submission (`122243216`) failed before reading a payload:
the Kunshan CPU node could not import the vendor DTK PyTorch wheel because
`libhsakmt.so.1` is available only with a DCU allocation. The unchanged
analysis is therefore resubmitted on `kshdtest`; the accelerator is required
only to load PyTorch serialization and no model or training step is executed.
