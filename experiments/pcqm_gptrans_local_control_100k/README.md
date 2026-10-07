# GPTrans deep local-update control

One parameter-free amplitude intervention on the accepted local-bond model.
The question, budget, interpretation limits and terminal gate are in
[protocol.md](protocol.md). Live execution belongs to `CURRENT_STATE.md`.

`freeze.py` declares the shared native GPTrans release; `accept.py` delegates
saved-artifact acceptance and terminal/RML closure. No second trainer, package,
platform submitter or monitor is introduced. At terminal closure verify the
actual candidate/reference Replay pair, not merely a valid envelope.
