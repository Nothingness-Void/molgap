# Kaggle1 pair launch decision — 2026-09-29

The P100 route closed `NO_TRAIN` after accelerator retirement. The later
Kaggle3 T4x2 single-arm prospective plan was published locally but no source
dataset, kernel, preflight or training action was submitted. The user directed
a Kaggle1 dual-arm experiment. Close that single-arm plan `NO_TRAIN` and run
one new paired T4x2 attempt under `protocol_kaggle1_pair.md`.

The second arm is the unmodified GPTrans-T seed-42 initialization, trained
concurrently as a same-allocation control. This resolves the cross-runtime
confound of comparing a new T4 candidate to the historical P100 endpoint.
The experiment tests only the input scale intervention, not another model
module. The accepted Kaggle1 fixed graph mirror, exact initial state, frozen
source, both arm trajectories, native T4 cost cap, and preflight are mandatory
release inputs. No protected role or scale-up is authorized.
