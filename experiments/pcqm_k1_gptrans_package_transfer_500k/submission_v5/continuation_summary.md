# V5 K1-only continuation - 2026-10-07

Kaggle1 accepted kernel137144136/version5. The authenticated active-event page
binds script355820538 and GPU T4x2. [Platform response](platform_response.json)
and [exact pulled entrypoint/mounts](remote_identity_verified.json) match the
82-check release binding: source01c561e967cb9cca4113a154eb6064cd915a716e,
archivec294a785f9d0b702f63152e4448511875f7124a830f1ad088c9acdd64203814a,
package5d5c70b732d44433733ecb26c9d10ee9266354b8928b6ccd659fe95c876858de.
The frozen scientific Spec and both prospective records remain unchanged.

The private checkpoint dataset12408227 contains all65 files required by the
original v4 stage manifests, checked before packaging/publication. Source
publication retains older versions. All three input datasets are READY.
[Continuation binding](continuation_binding.json) pins original per-arm
manifest, runtime and checkpoint hashes/cursors. [Launch receipt](receipts/3f191c34dabff9453b43c18ef8e0b8137f99ae8892ccba62d253b40b86087cab.json)
maps K1 execution to v5 and completed GPTrans evidence to v4; GPTrans does not
acquire a v5 training/source attribution.

Reuse: owning prepare.py, shared source/release helpers, credential_api,
explicit T4 submitter, source pull, immutable launch receipt and RML check.
Added code is limited to completed-peer retention in the existing legacy500K
platform runtime and an explicit bounded-stage option in the owning preparer.
No trainer/model/objective implementation was added or changed.

[Local qualification](local_qualification.json): syntax passed,82 release
checks passed, and7 completed-peer hook tests passed. The initial test batch
exposed local Python3.10 fixture incompatibility, fixed in tests only, and
three stale composed evidence tests targeting absent APIs. Those stale tests
remain recorded separately; this continuation does not change their trainer.
[RML frozen/portable check](rml_check.json) passed with existing evidence.

Budget and authority follow [the continuation plan](continuation_plan.md).
Only K1's remaining seven epochs execute; native cosine60 state, FP32 and
roles are preserved. Bound14400seconds/max8 allocated T4hours includes idle
GPTrans capacity. Native runtime cost is measured at termination; estimates
and unknown version1/external startup costs remain explicit.

Final terminal science, attribution, fixed50:50 fusion and two per-arm replay
qualifications remain pending. No automatic monitor or server handoff exists.

[Actual startup](training_start_observation.json) confirms K1 resume_epoch53,
epoch54/60 batch1/3906 global_step207019 at219.4s after bootstrap, carried
learning_rate1.1095177e-05 and only the K1 GPU worker. GPTrans COMPLETE
retention/no-execution and CPU/K1 T4 qualification passed visibly.
