# CPU/static feasibility contract, 2026-09-30

The user authorized feasibility analysis and low-cost validation preparation
while the independent geometry job finishes. This action uses fixed synthetic
SMILES fixtures only, no PCQM row, target, checkpoint, model execution, or GPU.
No dataset role is consumed. One CPU engineering test session is allowed;
no accuracy inference follows from it. GPU time and full-cache cost are unknown.

Reuse RDKit RDKFingerprint (minPath=1,maxPath=7,fpSize=512) and Descriptastorus
2.8.0 RDKit2DNormalized, preserving its 200 column names and library versions.
The leading success flag is metadata, not a 201st target. Reject incomplete
or nonfinite labels explicitly; never drop a source row or silently impute.
Library normalization uses external fixed distributions, not training-fitted
PCQM statistics. Do not claim exact equivalence with the paper's vendored code.

The source basis is the NVIDIA PCQM4Mv2 report, section 2.5:
https://ogb.stanford.edu/paper/neurips2022/pcqm4mv2_NVIDIA-PCQM4Mv2.pdf
Author repository: https://github.com/jfpuget/NVIDIA-PCQM4Mv2
Reviewed source snapshot: 82ee505cbaf26bd0830e2f39b6638b9f08a17de8.
The reported improvement is conditional on its Transformer-M recipe; it is
not an expected GPTrans gain. Our compact shared auxiliary head and loss
weighting differ and require independent validation.

Evidence review: desktop 22af38b; server remote-tracking snapshot is retained
in planning inputs. See pcqm_scale_transfer_reassessment/decision.md for
eroding early gains; pcqm_gptrans_input_init_100k for its negative matched pair;
server pcqm_k1_joint_atom_reconstruction_100k/attempts/v3/audit/decision.md
for unsuccessful clean-reference transfer of corrupted-input atom supervision.
These support caution, not a diagnosed missing chemical representation.
No exact clean-input descriptor+fingerprint graph-supervision pair was found
in the reviewed indexes. Novelty is bounded to those records.

Cheapest falsifiers: unavailable or unstable labels, hidden inference dependency,
or excessive measured training overhead. Local fixtures only address API and
failure semantics. A passed fixture does not qualify a full cache or a GPU run.

Reuse: GPTrans V4 family trainer, deterministic sampler, EMA/checkpoint machinery;
experiment CLI for future Spec/source/receipt binding; RML plan() for this
single non-model feasibility action. Do not copy the family training loop.

Budget: local CPU engineering checks only, no paid/remote allocation. No native
cost projection from fixture speed to PCQM. Protected roles remain untouched.
