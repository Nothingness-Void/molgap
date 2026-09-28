# 2026-09-28 MetaGIN2D bounded runtime decision

Kaggle2 `kaseichou/molgap-metagin-2d-runtime-profile:v2` completed and passed
the no-inference [acceptance](profile_v2_acceptance.json). It consumed only the
accepted fixed-100K train labels for 8 warm-up + 72 measured optimizer steps
and 32 train-role forward batches. No development metric, checkpoint selection,
official validation, test-dev or challenge role was produced/read. Its source
commit, immutable target transform, accepted CPU sidecar and output hashes are
bound by that receipt; raw files and logs remain in ignored platform records.

The actual hardware was Tesla T4 x2, though one candidate used only device 0.
Mean FP32/BS128 optimizer step was 0.134224 s and sampled forward step was
0.083803 s. The observed 40-epoch projection was 5,503.8 wall seconds;
including the predeclared 20% reserve gives 6,604.6 seconds (1.83 h), below
the six-hour worker ceiling. Peak reserved VRAM was 720 MiB of 14,911.7 MiB
on the active T4 (95.2% reserve). Counting both allocated T4s, the
projection is 3.67 GPU-device hours before approximately 0.14 device-hour
profile setup/diagnosis. The platform allocation inefficiency must remain
visible in any terminal cost.

The same 4 x 256 MetaGIN2D architecture is therefore **operationally
admissible** for one newly identified seed-42 100K screen under the unchanged
scientific contract. This is not evidence of scientific improvement and does
not authorize a second seed, 500K training, full training, or protected-role
access. The v1 cold-step preflight failure remains a separate infrastructure
attempt; its frozen prospective identity must not be overwritten. The next
attempt needs its own run ID, source package, validated V5 prelaunch and RML
trajectory before Kaggle submission.
