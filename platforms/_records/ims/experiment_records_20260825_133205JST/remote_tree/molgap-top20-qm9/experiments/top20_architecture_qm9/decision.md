# Top-20 Architecture Transfer Screen Decision

Status: implementation and remote short-screen preparation in progress.

## Predeclared decision rule

The old architecture is the promoted QM9 precision fusion from
`experiments/qm9_architecture/results/summary.json`. Its reference is
`0.0708138843 eV` average MAE and `0.084272936 eV` Gap MAE on the fixed
ETKDG-successful protocol.

The new `tgt_lite` candidate must, on the same requested split and successful
row intersection:

1. reach at least `0.003 eV` lower average MAE and `0.002 eV` lower Gap MAE;
2. repeat the improvement for encoder seeds 42, 43, and 44, or have a
   three-seed mean improvement with no target regression larger than `0.002 eV`;
3. remain within the remote resource budget and produce atomic checkpoints,
   independent metrics, and an embedding payload.

The first remote job is a bounded 30-epoch, seed-42 screen. It is evidence for
resource/accuracy triage only. If it clears both metric margins, the three-seed
confirmation is submitted. If it does not, the candidate is not promoted and
no PCQM training is launched from it.

## Current implementation

`src/molgap/tgt_lite.py` implements the candidate; the thin QM9 entrypoint
accepts `--candidate tgt_lite --geometry etkdg`. The model uses the existing
processed QM9 2D graph plus ETKDG coordinates, with no local execution in this
turn. Remote scripts and the submission record will be added after the IMS
input/environment preflight.
