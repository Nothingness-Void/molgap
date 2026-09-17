# Kunshan V4 Account Bootstrap

## Staging Evidence

On 2026-09-16 JST, the user-uploaded bootstrap ZIP was extracted into a
dedicated directory under the authorized account home. The uploaded ZIP was
retained. No prior account environment or credentials entered the package.

- ZIP bytes: 102469148.
- ZIP SHA256: `779619f693119a355d7b1a043da86ed574958d9b4add82eee1b014ae93d0ee43`.
- Frozen source commit: `7f36a1d71ae0f063c53f07bd1e5b725dcc2ecebc`.
- Repacked source archive SHA256: `d27e97cb5e7e4c567d928983d689810cacc250528050b0f5b828b955c68846a1`.
- Fixed dataset manifest SHA256: `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
- Frozen initial artifact SHA256: `073fce25752f9fc5e15670177cd9e69286d681985a3f3e6bed757efef00b124a`.

All 24 package files passed remote size/SHA256 checks, and the bundled
`validate_standard_source_bundle` accepted the archive and its sidecars.
The three accepted graph shards contained 100K training and 50K internal
development rows. No official validation/test-dev/test-challenge input was
included or accessed. Matching inputs alone did not certify the runtime.

## Deployment Submission

The new account exposed `kshctest02` CPU and `kshdtest` Hygon DCU partitions.
DTK 25.04 existed, but the previously documented `sghpcdas/25.6` environment
did not. The centre-provided DTK 23.10 PyTorch 1.13.1 and PyG-extension wheels
were located instead. An offline Python 3.10 installation attempt failed
because the shared package cache did not contain Python 3.10.

- CPU setup `122210841`: 4 CPUs, 12G, maximum 1 hour; RUNNING in the submission snapshot.
- Single-DCU gate `122210845`: 8 CPUs, 27G, maximum 30 minutes; afterok dependency on setup.
- Setup adapter SHA256: `f6425f4e282d0f4a4d688aca9067044416e688c0f8f3aa235fd7eed18912b3bd`.
- Preflight adapter SHA256: `a171d1e93fae8c198b0379a37e7f200b3994c1530dae0eebc6a033c7b3264a25`.

Setup used account-local Conda/pip caches and a new Python 3.10 environment,
centre-adapted Torch, PyG 2.5.3, OGB 1.3.6 and NumPy 1.26.4. Both adapters
passed remote `bash -n`. Final dependency/import acceptance and actual V4
forward/backward certification were still pending at submission. Dedicated
logs and atomic acceptance JSON were configured on durable account storage.

No formal training was submitted. Live status belongs only in
`CURRENT_STATE.md`; this record describes the staging and submission event.
