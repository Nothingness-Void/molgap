"""Prospective native terminal bundles. No scheduler, model, or data-role access."""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from molgap.comparison_readiness import assess_comparison_readiness, validate_comparison_readiness
from molgap.evidence_pointers import load_json_object
from .paths import repo_local_path, verify_bound_artifact
from .schemas import validate_trajectory, validate_id
from .trace import FIELDS, atomic_write, canonicalize_trace, file_digest, json_bytes

IDENTITY = ('trajectory_id', 'run_id', 'action_id', 'evidence_id')


def _load(root, path):
    return load_json_object(repo_local_path(root, path))


def _bind(root, path):
    return {'ref': path, 'sha256': file_digest(repo_local_path(root, path))}


def _put(root, path, value):
    target = repo_local_path(root, path)
    payload = json_bytes(value)
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError(f'immutable closure output differs: {path}')
    else:
        atomic_write(target, payload)


def _git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def _arms(root, spec):
    if spec.get('format') != 'molgap-native-terminal-bundle-v1':
        raise ValueError('unsupported native terminal bundle')
    arms = spec.get('arms')
    if not isinstance(arms, list) or not arms:
        raise ValueError('explicit nonempty arm roster required')
    seen = {key: set() for key in (*IDENTITY, 'arm', 'trajectory_ref', 'output_dir')}
    for arm in arms:
        for key in seen:
            value = arm.get(key)
            if not isinstance(value, str) or not value or value in seen[key]:
                raise ValueError(f'missing or duplicate arm {key}')
            seen[key].add(value)
        for key in ('trajectory_id', 'action_id', 'evidence_id'):
            validate_id(arm[key], key)
        path = repo_local_path(root, arm['trajectory_ref'])
        path.relative_to(Path(root).resolve() / 'experiments')
        if path.name != 'trajectory.json' or 'rml_finalized' in path.parts:
            raise ValueError('original prospective trajectory.json required')
        trajectory = validate_trajectory(_load(root, arm['trajectory_ref']))
        if trajectory['record_mode'] != 'prospective' or trajectory['owner'] != 'desktop':
            raise ValueError('independent Desktop prospective identity required')
        if trajectory['trajectory_id'] != arm['trajectory_id'] or not trajectory.get('decision_state'):
            raise ValueError('trajectory identity/frozen decision state missing')
        if trajectory['decision']['outcome'] != 'ACTIVE' or trajectory['result']['evidence_ids']:
            raise ValueError('original plan must be active and free of terminal evidence')
        action = next((a for a in trajectory['actions'] if a['action_id'] == arm['action_id']), None)
        if action is None or action['run_ids'] != [arm['run_id']]:
            raise ValueError('one separately frozen run per candidate action required')
        if arm['contract_ref'] not in trajectory['state_at_start']['contract_refs']:
            raise ValueError('contract not frozen in trajectory')
        reference = _load(root, arm['reference_bundle_ref'])
        if reference['reference_id'] != arm['reference_id'] or arm['reference_id'] not in trajectory['state_at_start']['reference_ids']:
            raise ValueError('reference not frozen in prospective state')
        if trajectory.get('reference_bundle_id') != reference['reference_bundle_id']:
            raise ValueError('reference bundle identity not frozen')
        output = repo_local_path(root, arm['output_dir'])
        output.relative_to(path.parent)
        if output == path.parent or 'rml_finalized' in output.parts:
            raise ValueError('closure output must be a distinct experiment subdirectory')
        mapping = arm['trace']['field_mapping']
        if set(mapping) - set(FIELDS) or len(set(mapping.values())) != len(mapping):
            raise ValueError('trace mapping must be one-to-one canonical fields')
        if any(not isinstance(v, str) or not v for v in mapping.values()):
            raise ValueError('explicit raw trace columns required')
        # Canonical axes must already be measured cumulative coordinates.
        for axis in ('optimizer_step', 'sample_presentations'):
            if axis in mapping and arm['trace']['axis_semantics'].get(axis) != 'observed_cumulative':
                raise ValueError('no inferred or per-epoch cumulative conversion in native closure')
        for field in ('device_time_seconds', 'cumulative_device_time_seconds'):
            if field in mapping and arm['trace'].get('field_semantics', {}).get(field) != 'observed_sum_over_devices':
                raise ValueError('device timing needs explicit observed sum-over-devices semantics')
        for generated in ('trace_manifest', 'role_history', 'cost_records'):
            if generated in arm['candidate_artifact_paths']:
                raise ValueError(f'{generated} is generated by shared closure')
    return arms


def freeze_native_bundle(repo_root, spec_ref, receipt_ref):
    """Freeze the complete roster before launch; never create a late trajectory."""
    root = Path(repo_root).resolve()
    spec = _load(root, spec_ref)
    arms = _arms(root, spec)
    bindings = {}
    pointers = {spec_ref}
    for arm in arms:
        if repo_local_path(root, arm['raw_trace_ref']).exists() or repo_local_path(root, arm['acceptance_ref']).exists():
            raise ValueError('terminal/raw output already exists: cannot freeze after execution')
        if (repo_local_path(root, arm['trajectory_ref']).parent / 'rml_finalized').exists():
            raise ValueError('candidate already finalized')
        pointers.update([arm['trajectory_ref'], arm['reference_bundle_ref'], arm['contract_ref'],
                         arm['trace']['semantics_ref'], arm['reference_side_ref']])
        reference_side = _load(root, arm['reference_side_ref'])
        for binding in reference_side['artifact_bindings'].values():
            verify_bound_artifact(root, binding['ref'], binding['sha256'])
            pointers.add(binding['ref'])
        trajectory = _load(root, arm['trajectory_ref'])
        for pointer, digest in trajectory['decision_state']['source_hashes'].items():
            verify_bound_artifact(root, pointer, digest)
            pointers.add(pointer)
    for pointer in sorted(pointers):
        bindings[pointer] = file_digest(repo_local_path(root, pointer))
    receipt = {'format': 'molgap-native-prelaunch-v1', 'spec_ref': spec_ref,
               'bindings': bindings, 'spec': spec, 'next_decision': 'SOL_REQUIRED',
               'launch_authorized': False}
    _put(root, receipt_ref, receipt)
    return receipt


def _receipt(root, receipt_ref, source_commit):
    receipt = _load(root, receipt_ref)
    if receipt.get('format') != 'molgap-native-prelaunch-v1':
        raise ValueError('native prelaunch receipt required')
    # The launch source must already contain the exact receipt and all plans.
    # A later trajectory cannot be grafted into a run's earlier source release.
    path = repo_local_path(root, receipt_ref).relative_to(root).as_posix()
    committed_receipt = _git(root, 'show', f'{source_commit}:{path}')
    local_receipt = repo_local_path(root, receipt_ref).read_bytes()
    if committed_receipt != local_receipt and committed_receipt.replace(b'\r\n', b'\n') != local_receipt.replace(b'\r\n', b'\n'):
        raise ValueError('receipt was not committed in the submitted source')
    for pointer, digest in receipt['bindings'].items():
        verify_bound_artifact(root, pointer, digest)
        relative = repo_local_path(root, pointer).relative_to(root).as_posix()
        committed_blob = _git(root, 'show', f'{source_commit}:{relative}')
        blob_digests = {
            hashlib.sha256(committed_blob).hexdigest(),
            hashlib.sha256(committed_blob.replace(b'\r\n', b'\n')).hexdigest(),
            hashlib.sha256(committed_blob.replace(b'\n', b'\r\n')).hexdigest(),
        }
        if digest not in blob_digests:
            raise ValueError(f'prelaunch source binding changed: {pointer}')
    _arms(root, receipt['spec'])
    return receipt


def guard_native_launch(repo_root, receipt_ref, source_commit):
    root = Path(repo_root).resolve()
    receipt = _receipt(root, receipt_ref, source_commit)
    for arm in receipt['spec']['arms']:
        if any(repo_local_path(root, arm[key]).exists() for key in ('raw_trace_ref', 'acceptance_ref')):
            raise ValueError('launch guard refuses an already observed run')
        if (repo_local_path(root, arm['trajectory_ref']).parent / 'rml_finalized').exists():
            raise ValueError('launch guard refuses a finalized candidate')
    return {'format': 'molgap-native-launch-binding-v1', 'receipt_ref': receipt_ref,
            'receipt_sha256': file_digest(repo_local_path(root, receipt_ref)),
            'source_commit': source_commit,
            'arms': [{k: arm[k] for k in ('arm', *IDENTITY)} for arm in receipt['spec']['arms']],
            'identity_guard': 'PASS', 'launch_authorized': False, 'next_decision': 'SOL_REQUIRED'}


def _prepare(root, arm, accepted, launch):
    """Convert accepted metadata; never synthesize acceptance or observed events."""
    if accepted.get('accepted') is not True or accepted.get('model_inference_executed') is not False:
        raise ValueError('retained no-inference acceptance required')
    for key in IDENTITY:
        if accepted.get(key) != arm[key]:
            raise ValueError(f'acceptance identity differs: {key}')
    if accepted.get('launch_binding') != launch:
        raise ValueError('acceptance must bind the original launch receipt/source/roster')
    hashes = accepted['artifact_hashes']
    for pointer, digest in hashes.items():
        verify_bound_artifact(root, pointer, digest)
    if arm['raw_trace_ref'] not in hashes:
        raise ValueError('raw trace not bound by acceptance')
    raw = _load(root, arm['raw_trace_ref'])
    if 'trajectory_id' in raw and raw['trajectory_id'] != arm['trajectory_id']:
        raise ValueError('native raw trace carries mismatched trajectory identity')
    if 'run_id' in raw and raw['run_id'] != arm['run_id']:
        raise ValueError('native raw trace carries mismatched run identity')
    rows = raw[arm['trace']['rows_key']]
    if not isinstance(rows, list):
        raise ValueError('raw observations must be an ordered list')
    observations = []
    for index, row in enumerate(rows):
        obs = {field: row.get(column) for field, column in arm['trace']['field_mapping'].items()}
        if 'event' in row:
            obs['event'] = row['event']
        elif index == len(rows) - 1 and accepted.get('outcome') and accepted['outcome'] != 'ACTIVE':
            obs['event'] = 'terminal'
        observations.append(obs)
    trace = canonicalize_trace({'trajectory_id': arm['trajectory_id'], 'run_id': arm['run_id'],
        'metric_semantics': arm['trace']['metric_semantics'],
        'device_time_semantics': arm['trace'].get('device_time_semantics'), 'observations': observations,
        'provenance': [{'source': arm['raw_trace_ref'], 'sha256': hashes[arm['raw_trace_ref']],
                        'field_mapping': arm['trace']['field_mapping'], 'launch_binding': launch}]})
    out = arm['output_dir'].rstrip('/')
    paths = {name: f'{out}/{name}.json' for name in ('canonical_trace', 'trace_manifest_terminal',
             'role_history', 'cost_records', 'comparison_readiness', 'terminal_evidence', 'terminal')}
    _put(root, paths['canonical_trace'], trace)
    costs, roles = accepted['costs'], accepted['roles']
    for event in [*costs, *roles]:
        if any(event[k] != arm[k] for k in ('trajectory_id', 'run_id', 'action_id')):
            raise ValueError('joint or foreign arm events cannot be allocated to a candidate')
        if event['evidence_ref'] not in hashes:
            raise ValueError('observed event source not acceptance-bound')
        kind = 'costs' if event in costs else 'roles'
        if event not in _load(root, event['evidence_ref']).get(kind, []):
            raise ValueError('event absent from original observed source')
    _put(root, paths['role_history'], {'format': 'molgap-role-history-index-v1', 'events': roles})
    _put(root, paths['cost_records'], {'format': 'molgap-cost-records-v1', 'costs': costs})
    fields = {key: any(row[key] is not None for row in trace['observations']) for key in
              ('optimizer_step', 'sample_presentations', 'epoch_or_pass', 'learning_rate',
               'live_train_metric', 'live_dev_metric', 'ema_dev_metric', 'checkpoint_identity')}
    manifest = copy.deepcopy(arm['trace_manifest'])
    manifest.update(schema='molgap-trace-manifest-v1', trajectory_id=arm['trajectory_id'], run_id=arm['run_id'],
                    contract_ref=arm['contract_ref'], reference_id=arm['reference_id'], comparison_role='candidate',
                    trace_artifact_ref=paths['canonical_trace'], trace_artifact_sha256=file_digest(repo_local_path(root, paths['canonical_trace'])),
                    terminal_evidence_ref=paths['terminal_evidence'], trace_fields=fields)
    # Comparison assessment does not read the trace's eligibility; freeze its
    # predeclared request and let finalize/replay apply their own strict gates.
    # The first assessment needs a real hash-bound manifest. This file is a
    # preparation artifact, not the atomic finalized publication.
    atomic_write(repo_local_path(root, paths['trace_manifest_terminal']), json_bytes(manifest))
    candidate = copy.deepcopy(accepted['candidate_side'])
    candidate_paths = {**arm['candidate_artifact_paths'], 'trace_manifest': paths['trace_manifest_terminal'],
                       'role_history': paths['role_history'], 'cost_records': paths['cost_records']}
    for pointer in arm['candidate_artifact_paths'].values():
        if pointer not in hashes:
            raise ValueError('candidate metadata not acceptance-bound')
    candidate['artifact_bindings'] = {name: _bind(root, path) for name, path in candidate_paths.items()}
    candidate['trace_field_availability'] = fields
    candidate['observed_role_event_kinds'] = sorted({r['access_kind'] for r in roles})
    reference = _load(root, arm['reference_side_ref'])
    readiness = assess_comparison_readiness(candidate_id=arm['evidence_id'], candidate=candidate,
        reference_id=arm['reference_id'], reference=reference, **arm['comparison'])
    if not readiness['strict_ready']:
        manifest['backtest_eligibility'] = {'eligible': False, 'exclusion_reasons':
            readiness['blocker_codes'] or ['comparison_not_strict_ready']}
        atomic_write(repo_local_path(root, paths['trace_manifest_terminal']), json_bytes(manifest))
        candidate['artifact_bindings']['trace_manifest'] = _bind(root, paths['trace_manifest_terminal'])
        readiness = assess_comparison_readiness(candidate_id=arm['evidence_id'], candidate=candidate,
            reference_id=arm['reference_id'], reference=reference, **arm['comparison'])
    validate_comparison_readiness(readiness, evidence_verifier=lambda p, h: verify_bound_artifact(root, p, h))
    _put(root, paths['comparison_readiness'], readiness)
    evidence = {'format': 'molgap-v5-evidence-envelope-v1', 'contract': 'MOLGAP-COMMON-V5-FINAL',
        'evidence_id': arm['evidence_id'], 'track': arm['track'], 'scope': arm['scope'],
        'legacy_contract': 'none-prospective-v5', 'outcome': accepted['outcome'], 'role_use': accepted['role_use'],
        'authority': {'pointers': [arm['contract_ref'], arm['trajectory_ref'], arm['acceptance_ref'], accepted['trajectory_decision']['decision_ref']]},
        'artifacts': [{'name': name, 'locator': paths[name], 'sha256': file_digest(repo_local_path(root, paths[name])),
                       'availability': 'repository_retained'} for name in ('canonical_trace', 'comparison_readiness')],
        'migration': {'migrated_at': accepted['finalized_at'], 'verification_scope': 'accepted native terminal metadata',
                      'training_executed': False, 'inference_executed': False, 'scientific_reinterpretation': False}}
    _put(root, paths['terminal_evidence'], evidence)
    bound = set(hashes) | set(evidence['authority']['pointers']) | {paths['canonical_trace'], paths['comparison_readiness']}
    terminal = {'format': 'molgap-rml-terminal-package-v1', **{k: arm[k] for k in ('trajectory_id', 'run_id', 'action_id')},
        'finalized_at': accepted['finalized_at'], 'acceptance_ref': arm['acceptance_ref'],
        'artifact_hashes': {p: file_digest(repo_local_path(root, p)) for p in sorted(bound)},
        'evidence': evidence, 'decision': accepted['trajectory_decision'], 'costs': costs, 'roles': roles,
        'role_use': accepted['role_use'], 'trace_manifest': manifest, 'comparison_readiness_ref': paths['comparison_readiness']}
    _put(root, paths['terminal'], terminal)
    return paths


def close_native_bundle(repo_root, receipt_ref, launch_ref):
    from .pipeline import finalize_rebuild_backtest
    from .compiler import frozen_differences
    root = Path(repo_root).resolve()
    launch = _load(root, launch_ref)
    receipt = _receipt(root, receipt_ref, launch['source_commit'])
    if launch['receipt_ref'] != receipt_ref or launch['receipt_sha256'] != file_digest(repo_local_path(root, receipt_ref)):
        raise ValueError('launch receipt binding mismatch')
    expected_roster = [{k: a[k] for k in ('arm', *IDENTITY)} for a in receipt['spec']['arms']]
    if launch.get('arms') != expected_roster or launch.get('identity_guard') != 'PASS':
        raise ValueError('launch roster mismatch')
    # Check all arms before publishing any finalization; shared job costs cannot
    # be counted twice by renaming a cost event or assigning it a second arm.
    accepted = [_load(root, a['acceptance_ref']) for a in receipt['spec']['arms']]
    seen_costs, seen_sources = set(), set()
    for record in accepted:
        for event in record['costs']:
            source_key = (event['evidence_ref'], event['category'])
            if event['cost_event_id'] in seen_costs or source_key in seen_sources:
                raise ValueError('shared/duplicate cost source: retain unknown arm costs instead')
            seen_costs.add(event['cost_event_id'])
            seen_sources.add(source_key)
    results = []
    for arm, record in zip(receipt['spec']['arms'], accepted):
        report = {'arm': arm['arm'], **{k: arm[k] for k in IDENTITY},
                  'replay_status': 'REPLAY_EXCLUDED', 'reasons': [], 'next_decision': 'SOL_REQUIRED'}
        try:
            paths = _prepare(root, arm, record, launch)
            result = finalize_rebuild_backtest(root, arm['trajectory_ref'], paths['terminal'], paths['canonical_trace'])
            report['pipeline'] = result
            if result['pipeline_status'] != 'COMPLETE':
                raise ValueError(result.get('error', 'terminal pipeline incomplete'))
            differences = frozen_differences(root)
            if differences:
                raise ValueError('frozen check differs: ' + ', '.join(differences))
            report['closure_complete'] = True
            pool = _load(root, 'research_memory/derived/replay_pool.json')
            match = lambda e: (e['trajectory_id'], e['run_id']) == (arm['trajectory_id'], arm['run_id'])
            if any(match(e) for e in pool['entries']):
                report['replay_status'] = 'REPLAY_READY'
            else:
                report['reasons'] = sorted({r for e in pool['exclusions'] if match(e) for r in e['reasons']}) or ['no_canonical_comparable_reference_pair']
        except (ValueError, OSError, KeyError, TypeError) as exc:
            report['reasons'] = [str(exc)]
        # Operational report is replaceable: retry after repairing missing
        # corpus/reference infrastructure without altering immutable finalization.
        atomic_write(repo_local_path(root, arm['output_dir'] + '/closure_status.json'), json_bytes(report))
        results.append(report)
    return {'candidates': results, 'next_decision': 'SOL_REQUIRED'}
