# Local release verification — 2026-10-25

The coordinator reused the scientific-input compiler, indexed-data family
recipes, frozen initialization reader, prospective planner, source packager
and owning Kaggle adapter. No new model or training loop was implemented.

An independent configured Luna worker ran `test_modern2d_scientific_input.py`
and `test_same_run_replay.py`:11 passed in251.75s on bounded CPU fixtures,
with network/native data/accelerators excluded. These exercised actual
Session/output/reference-first closure, retained-baseline reuse and actual
fixture replay-pool membership. They do not certify this new GPU runtime or
model quality. A CPU pin-memory warning was nonblocking.

Actual package preparation used the candidate's declarations and source,
verified both recipes, frozen state files, accepted input/target metadata,
strict comparison plan, packaged dependencies and final upload bindings.
The owning report returned `PREPARED_FOR_PLATFORM` and
`LOCAL_RELEASE_INPUTS_VERIFIED`; total local preparation was48.1573s.

- Frozen source: `9f0fe6222fdae7a2975c0497a15d8060be43662e`.
- Local storage attempt: `v2`; platform version remains independently assigned.
- Spec: `22a1ef0849c82646614a677e75aafe0abe0740206d9234b8b7905617f5e810b8`.
- Package: `948c135343390ddf7e45dd8abfe5fa190feeba1b7bd8952bfac6b0b7203d4f64`.
- Source archive: `790eb5cab8f885798e3ac2b78665ef996078a91e4035d7cda78963739889a1d7`.

External reports live below
`E:/MolGap/artifacts/runs/esa-count-persistence-k2-100k-s42-v1/v2/prepared/`.
The earlier unpublished local block is retained in
[`local_preparation.json`](local_preparation.json). The historical nested
`.gitattributes` correction changed transport rules only, not evidence bytes.

Fresh native all-arm numerical/repeat/resume qualification remains required
before the same allocation can train. Prospective closure capability is not
terminal acceptance or a claim that the task is already Replay-Ready.
