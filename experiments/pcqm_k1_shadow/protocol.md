# Frozen protocol — K1 independent shadow audit

## Selection status

The one-slot latent-global model was a predeclared causal control in the
accepted PCQM-100K Fourier transfer. It beat fresh full GPS by
`0.009461689 eV`, with a paired bootstrap interval excluding zero and the same
direction in every target quartile. It was not the originally nominated
Fourier candidate, so the result is exploratory and may advance only through
one untouched shadow audit.

## Shadow construction

- source role: only the first 3,378,606 official PCQM4Mv2 training rows;
- size: 10,000 molecules, disjoint from all 100,000 training and 10,000
  internal-validation rows in the accepted parent cache;
- selection: NumPy `default_rng(2026091042)`, without replacement from the
  complement of the exact parent row identities; first 10,000 rows are shadow
  and 256 further rows are deterministic graph-construction reserves;
- cache construction reads only CSV columns `idx` and `smiles`;
- cache contains OGB categorical atoms/bonds, RWSE16, and row identity, with no
  `y` tensor or target column;
- every shard, split, failure, replacement, and aggregate identity is hashed;
- CPU only, no model inference, no shadow label read.

The candidate architecture and source checkpoint were selected before any
shadow label access. The original governance intended to freeze the shadow
before the first transfer; that did not happen. This protocol records the
deviation explicitly and restores independence by freezing the candidate and
index rule before label access. The audit must not be described as preregistered
before selection.

## One-time audit gate

After independent cache acceptance, one remote task may load the already
frozen `full_gps` and `neural_atom_k1` best checkpoints from the accepted
PCQM-100K task, perform inference on the unlabeled shadow graphs, then read the
10,000 target labels exactly once to compute paired MAE. No optimization,
checkpoint update, threshold selection, or architecture change is allowed.

K1 passes only if its shadow MAE is strictly lower than full GPS, the paired
bootstrap interval excludes zero in K1's favor, and inference cost remains
within the existing resource envelope. A reversal or interval including zero
closes K1 without another shadow, seed, or tuning run. Passing authorizes only
a desktop handoff for a separate scale-budget decision, not server full-scale
training or official validation/test-dev access.
