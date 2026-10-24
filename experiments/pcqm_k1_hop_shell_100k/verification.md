# Focused implementation verification — 2026-10-24 JST

Executed tests/test_k1_hop_shell.py against this checkout's actual implementation
with the primary D:/文档/molgap/.venv Python and generated CPU PyG graphs only.

- Independent shortest-distance expectations on chain/cycle/disconnected and
  permuted batches; exact1..4 shells and mixed/ordered total-content identity.
- Both addon branches through real FamilyKernel with the retained random full
  K1 initialization: forward/backward, populated optimizer, all3 layer hooks,
  CPU context cleanup, eval, and serialized next-update model/optimizer/RNG equality.
- Registered composed-policy resolution and a fixed generated256-row TRAIN
  probe: source/target/prediction receipt, MAE/Gap/eV semantics, and preservation
  of weights, BN buffers, optimizer, modes and RNG.256 replaces formal4096 only
  in this synthetic implementation fixture.

Initial file run35.81s: two passed, one independent fixture expectation failed
for an isolated atom. Corrected only that expectation, then reran only the
affected test7.53s: passed. All three affected tests therefore passed; no broad
suite, actual molecular forward or native T4 qualification is claimed.
Log:D:/molgap-artifacts/reconciliation/k1-hop-shell-100k/local_checks.log.

Shared support extends existing CPU layout with optional substrate batch
contexts and existing family policies with registered composed-addon resource
and pass observers. No new trainer, scheduler, submission or RML validator.
Native resource cost and runtime/source qualification remain remote gates.
