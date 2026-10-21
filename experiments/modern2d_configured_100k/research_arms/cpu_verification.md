# Independent CPU verification before native release

Two Sol High workers implemented disjoint model modules. Two Luna6 Max workers
executed the targeted tests without recursive delegation or remote compute.
The coordinator reviewed changes and used the existing local packaging owner.

| Scope | Final result | Retained report |
|---|---|---|
| Existing release/bootstrap/budget/input regressions |114 passed|`E:/MolGap/artifacts/modern2d-research-campaign-20261022/audit/ba91f09f228b4c8b84f4571988c6ae4d/`|
| ESA research mechanism and formal session |85 passed|`E:/MolGap/artifacts/modern2d-research-campaign-20261022/esa/907fa0c0e47e4d1e91c4a9e4e2105b68/`|
| Shared configured outputs/session regressions |8 passed|`E:/MolGap/artifacts/modern2d-research-campaign-20261022/esa/399f470808c049e1a10fb8425796ba04/`|
| PPGT addon/input/release |63 passed|`E:/MolGap/artifacts/modern2d-research-campaign-20261022/ppgt/326b8e5bfa20424db91df1bea4265b84/`|
| All three PPGT addons and ESA addon through update/resume/output inspection |6 passed|`E:/MolGap/artifacts/modern2d-research-campaign-20261022/esa/369d2c3b6dce4c40832d4faec892f4c9/`|
| New workflow wire/authority/package-input regressions |6 passed plus2 corrected fixture cases passed|`E:/MolGap/artifacts/modern2d-research-campaign-20261022/ppgt/59bd46a5a7c44e5bbf8faba08275f771/`; `ppgt/2e6326b407df4c3da52366ab5a3b1e71/` under the same campaign root|

These batches overlap; their counts are not a unique-test total. Earlier fixture
failures were retained: a synthetic four-update recipe was incorrectly assigned
a full-horizon configured profile, and the output test incorrectly expected a
terminal trace event where the existing owner records a final observation.
Production validation was not weakened to make these tests pass.

Actual local packaging found a separate seam: the native qualification JSON must
be retained as a source input, but archived results are intentionally excluded
from runtime packages. `ppgt_native_qualification.json` is a byte-identical copy
of the accepted record (SHA256 `39bec22757b449a9463609136f3d66d051395fa8b4bc74c9ebdf85bf3ebed627`),
not a new acceptance or a relaxed source allowlist. The campaign is packaged too.

Fresh compact PPGT initialization had2,577,729 parameters/190 tensors. Its frozen
state SHA was `2574ae6e0079deb0e2ba3846fc07f14cdb464f47071d838164e0b7494f03987a`;
the retained artifact file SHA was `5a57dbb8f5abeb843683cc8efe1e5954a102ea8b1bf7d23075abc5a5b79c2c02`.
The initialization worker performed construction/save/reload only, with no real
data access or forward/backward pass. All model/session tests used bounded
generated CPU fixtures, not real molecular caches. Exact production configuration,
hardware, repeat and resume still require each notebook's native all-arm barrier.
No CPU result grants scientific acceptance or Replay eligibility.
