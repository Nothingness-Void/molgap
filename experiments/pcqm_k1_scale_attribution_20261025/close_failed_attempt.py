"""Bind one retained failed process receipt with the existing RML finalizer."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import struct
import subprocess

from molgap.research_memory.finalize import finalize
from molgap.training_reproducibility import atomic_json, sha256_file


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    rel = here.relative_to(root).as_posix()
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    inputs, trajectory = read(here / 'inputs.json'), read(here / 'rml/trajectory.json')
    receipt = read(here / 'execution_receipt.json')
    if receipt['status'] != 'failed' or receipt['worker_exitcode'] != 1 or receipt['timeout'] or (here / 'results').exists():
        raise ValueError('This closer only binds the observed pre-forward import failure')
    source = trajectory['state_at_start']['source_commit']
    preserved = []
    for name, digest in inputs['executed_source_files'].items():
        raw = subprocess.check_output(['git', 'show', source + ':' + name], cwd=root)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('Historical source bytes differ: ' + name)
        path = here / 'source_attempt_001' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != raw:
            raise ValueError('Historical snapshot already differs')
        path.write_bytes(raw)
        preserved.append(path)
    tid, action = trajectory['trajectory_id'], trajectory['actions'][0]
    run, attempt = action['run_ids'][0], action['attempt_ids'][0]
    evidence_id = 'pcqm-k1-scale-attribution-20261025-attempt001-failure'
    outcome = dict(execution_status='failed_no_training', artifact_status='partial_local_hash_verified',
        comparison_status='incomplete', scientific_status='NO_TRAIN', transfer_status='not_evaluated',
        budget_decision='infrastructure_attempt_closed', full_handoff_status='not_applicable')
    decision = dict(outcome='NO_TRAIN', decision_ref=rel + '/terminal_decision.md',
        next_allowed_actions=[], reopen_conditions=['Separate prospective authorized diagnostic'])
    cost = dict(schema='molgap-cost-event-v1', cost_event_id='cost-k1-scale-attribution-attempt001-observed',
        trajectory_id=tid, action_id=action['action_id'], run_id=run, attempt_id=attempt,
        platform='local-windows', hardware='CPU; no accelerator', category='other', evidence_ref=rel + '/acceptance.json',
        measurement={'wall_hours': {'value': receipt['parent_observed_wall_seconds']/3600, 'status': 'measured'},
                     'cpu_hours': {'value': None, 'status': 'measurement_missing'},
                     'device_hours': {'value': None, 'status': 'not_applicable'},
                     'queue_hours': {'value': None, 'status': 'not_applicable'}})
    roles = []
    for role, rows in [('train_descriptive', inputs['train_source_idx']), ('internal_development', inputs['development_source_idx'])]:
        digest = hashlib.sha256(b''.join(struct.pack('<q', i) for i in rows)).hexdigest()
        roles.append(dict(schema='molgap-role-event-v1', role_event_id='role-k1-scale-attribution-attempt001-' + role,
            trajectory_id=tid, action_id=action['action_id'], run_id=run,
            dataset_identity='pcqm4mv2-ogb-fixed-500k-scnet-v1', row_manifest_hash=digest,
            role_name=role, access_kind='labels_read', selection_used=False, evidence_ref=rel + '/acceptance.json'))
    role_use = {r: 'untouched' for r in ['official_validation', 'test_dev', 'test_challenge', 'common', 'ood']}
    role_use.update(train_descriptive='consumed', internal_development='consumed')
    acceptance = dict(format='molgap-local-frozen-diagnostic-acceptance-v1', evidence_id=evidence_id,
        run_id=run, outcome=outcome, trajectory_decision=decision, role_use=role_use, costs=[cost], roles=roles,
        comparison_class='CONTEXT_ONLY', checks={}, analysis_ref=rel + '/execution_receipt.json',
        blockers=['Frozen import guard failed before inference'],
        cost_scope='Measured outer process wall including spawn/import/loading; CPU unavailable; excludes preparation and closure',
        execution_scope='Graph decoding and source verification only; no predictions, optimizer, gradients or training',
        role_history='Development was selection-consumed by historical training; this failed attempt performed no new selection')
    atomic_json(here / 'acceptance.json', acceptance)
    names = ['inputs.json', 'roles.json', 'protocol.md', 'decision.md', 'execution_receipt.json',
             'terminal_decision.md', 'attribution.md', 'acceptance.json', 'close_failed_attempt.py']
    paths = [here / name for name in names] + preserved
    if (here / 'execution_tool_excerpt.txt').exists():
        paths.append(here / 'execution_tool_excerpt.txt')
    hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in paths}
    evidence = dict(format='molgap-v5-evidence-envelope-v1', contract='MOLGAP-COMMON-V5-FINAL',
        evidence_id=evidence_id, track='B', scope='pcqm-k1-scale-attribution-20261025',
        legacy_contract='pcqm-k1-scale-attribution-20261025-v1', outcome=outcome,
        authority={'pointers': [rel + '/' + n for n in ['protocol.md', 'terminal_decision.md', 'attribution.md', 'acceptance.json', 'execution_receipt.json']]},
        role_use=role_use,
        migration=dict(migrated_at=datetime.now(timezone.utc).date().isoformat(), training_executed=False,
            inference_executed=False, scientific_reinterpretation=False, verification_scope='Metadata closure of observed failed process'),
        observed_execution=dict(training_executed=False, inference_executed=False, training_replay_ready=False,
            execution_ref=rel + '/execution_receipt.json'),
        artifacts=[dict(name=n.removeprefix(rel + '/'), locator=n, sha256=h, availability='locally_retained_hash_verified') for n,h in hashes.items()])
    terminal = dict(format='molgap-rml-terminal-package-v1', trajectory_id=tid, run_id=run,
        action_id=action['action_id'], finalized_at=datetime.now(timezone.utc).isoformat(), acceptance_ref=rel + '/acceptance.json',
        artifact_hashes=hashes, evidence=evidence, decision=decision, costs=[cost], roles=roles)
    atomic_json(here / 'terminal.json', terminal)
    print(json.dumps(finalize(root, rel + '/rml', rel + '/terminal.json')))


if __name__ == '__main__':
    main()
