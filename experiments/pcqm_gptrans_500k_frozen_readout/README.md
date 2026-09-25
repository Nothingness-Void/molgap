# GPTrans 500K frozen-readout attribution

Question: does access to real-atom states, rather than only the virtual-token
and virtual-pair state, explain the weak retained 500K gains?

This is a bounded, post-hoc diagnostic on already trained 500K encoders. It
does not retrain the encoder, authorize full-scale work, or read protected
evaluation roles. The frozen question and role boundaries are in `protocol.md`;
the prospective RML record is `trajectory.json`. Read `decision.md` after the
run rather than interpreting intermediate metrics as a decision.

Entry points: `analyze_predictions.py` and `probe_readout.py`.

The small machine summaries are `results/prediction_attribution.json` and
`results/full/heads/head_probe.json`. Large feature tensors and fitted head
weights stay in ignored local `.pt` files; they are not portable RML evidence.
The scripts can regenerate them from the pinned accepted graph/checkpoint
inputs. This experiment does not claim replay-ready status.

Run from this checkout with `PYTHONPATH=src` and the project virtualenv. First
pass the accepted `platforms/_records/kaggle/training` tree and read-only
fixed-500K cache to `analyze_predictions.py`. Then call
`probe_readout.py extract` once per `baseline`/`pair_norm` with the SHA-pinned
selected checkpoint and prediction path, followed by `probe_readout.py fit`.
Both CLIs expose the exact required path flags in `--help`. No remote job or
protected role is involved.
