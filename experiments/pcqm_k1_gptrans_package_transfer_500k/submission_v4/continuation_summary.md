# V4 continuation submission

Kaggle1 accepted version4 of kernel137144136; authenticated UI script version
355652612. Both source and checkpoint datasets are private and ready. Pulled
entrypoint bytes and the exact three input mounts match the release. The
existing immutable receipt binds the actual version and both frozen arms.

K1 resumes 30 complete epochs/117180 steps, GPTrans 45/175770, from verified
v3 source0d53f2ae5a46d51cff9730e00767fa16b80ffb27002fa50b52a67c12ac8ab65a.
Executable archive bytes are unchanged; new package identity binds source
commit7fd5073a. Only local preparation gained an explicit checkpoint dataset
parameter. Shared release checks passed and the existing submitter repeated
the required binding gate before POST. Reviewer py_compile passed.

The [training start observation](training_start_observation.json) confirms
Python3.11.17, Tesla T4x2 allocation, completed frozen-runtime installation,
both exact resume manifests at epochs30/45, CPU cache/model/recipe PASS and
both isolated T4 preflight PASS. Actual first batches advance K1 to epoch31,
step117181 at lr0.00019518921, and GPTrans to epoch46, step175771 at
lr0.00015237968. Batch/step and phase progress are forwarded live.
Final acceptance remains pending60complete epochs for each arm.

RML check --frozen --portable passed after the committed submission binding.
Both prospective trajectories remain ACTIVE; this startup is not replay-ready
terminal evidence or a promotion. No automatic monitor was created.
