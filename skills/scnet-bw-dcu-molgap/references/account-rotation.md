# SCNet Account Rotation

Use this procedure before an allocation expires or when moving MolGap to
another SCNet account or region.

## 1. Freeze the Handoff Boundary

Read `AGENTS.md` and `CURRENT_STATE.md`. Record running, pending, completed,
accepted, and rejected workloads separately. Do not cancel or duplicate a
quiet job only because another account is available.

Never migrate:

- private keys, usernames, tokens, or `known_hosts`;
- local `.venv`, pip caches, compiled extensions, or account-specific wheels;
- rejected checkpoints or regenerable caches without a documented dependency;
- a mutable directory while its producer is running.

## 2. Build a Content-Addressed Manifest

Include relative path, byte count, SHA256, role, producer job, and acceptance
status for each transferred file. At minimum consider:

- repository commit or bounded source patch;
- immutable raw CSV and split manifests;
- accepted 2D/3D graph shards plus sidecars and completion/acceptance JSON;
- active GPS/SchNet checkpoints;
- embedding parts and completion manifests required by the next head/fusion;
- fixed external evaluation inputs;
- metrics, decisions, logs, and resume checkpoints.

Live model status belongs in the repository, not in this skill.

## 3. Bootstrap Destination Storage

Query the new account instead of copying old assumptions:

```bash
sinfo -o '%P|%G|%c|%m|%l'
sacctmgr show assoc where user="$USER" format=Account,Partition,QOS
df -h "$HOME"
quota -s 2>/dev/null || true
```

Create three parameterized roots:

```bash
PROJECT_DIR="$HOME/molgap"
INPUT_ROOT="<large-project-storage>/molgap-inputs"
OUTPUT_ROOT="<large-project-storage>/molgap-results"
```

Keep only small source and launch files under `$HOME` when its quota is 50 GB.
Verify a write probe in every selected root.

## 4. Transfer and Verify

Prefer the platform's direct/cross-region transfer or fast uploader for large
files. Use `scp` only for bounded files. Recompute every destination SHA256;
file existence and byte count alone are insufficient.

Do not rewrite absolute paths inside accepted manifests. Treat them as evidence
from the source account and create a new destination handoff manifest.

## 5. Rebuild the Runtime

Install dependencies from the platform runtime on the destination account.
Never copy a virtual environment. Re-run:

1. compute-node Python and package report;
2. actual DCU tensor operation;
3. PyG pooling and the required graph operator;
4. one-batch model forward;
5. destination result-directory write and atomic rename.

Use the DTK/PyG and SchNetPack references in this skill for known compatibility
constraints.

## 6. Resume Safely

Submit the smallest prerequisite first. Use `afterok` only after the upstream
artifact contract is explicit. Before resuming training, verify that checkpoint
contract hashes match the destination graph, split, model configuration, and
source code.

The source account remains authoritative until the destination preflight and
first durable checkpoint are accepted.
