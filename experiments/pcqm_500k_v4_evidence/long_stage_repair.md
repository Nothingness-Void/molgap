# Long-stage execution repair

The matched benchmark is a fixed 60-epoch V4 contract with best-development
selection. It is not a four-epoch or dynamic early-stop protocol. The first
four invocations were artificially capped at four epochs and three hours by the
Kaggle wrapper. That cap had no scientific purpose and incurred avoidable queue,
upload and idle time.

After the epoch-12 to epoch-16 stage, successors request every remaining epoch
in one invocation. The runner checkpoints best/last and trace after every epoch.
It exits at an epoch boundary only when the completed 60 epochs or a projected
next epoch would exceed the 41,400-second execution budget. This leaves about
30 minutes below Kaggle's nominal 12-hour ceiling. If the slow EdgeState/K1
pair cannot finish all remaining epochs, one final resume is expected rather
than repeated four-epoch stages.

Changing execution control changes the packaged source SHA but not the model,
data, seed, FP32/BS128 optimizer, schedule, loss, selection or sample-exposure
contract. The first long successor records both the accepted prior source SHA
and the new runtime source SHA; resume validation must match the former before
the new checkpoint adopts the latter. Source dataset:
`nothingnessvoid/molgap-500k-v4-source-raw-4218940805`, SHA256
`4218940805a84dbf69971452e3c113c7f5e8281df8be62371836b8bc3ff2b649`.

The wrapper now extracts mounted historical checkpoints into a temporary
directory outside `/kaggle/working`, avoiding duplicate publication. It writes
`kernel_status.json` from terminal stage manifests. Existing hashed evidence is
unchanged.
