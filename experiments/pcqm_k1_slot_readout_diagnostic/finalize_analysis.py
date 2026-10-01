"""Accept retained observations and publish immutable terminal metadata."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.research_memory.finalize import finalize, verified_receipt

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
REL=HERE.relative_to(ROOT).as_posix()
TID='TB-k1-slot-readout-diagnostic-20261002'
RUN='local-k1-slot-readout-20261002'
EID='pcqm-k1-slot-readout-diagnostic-20261002'

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    if (HERE/'rml/rml_finalized').exists():
        print(json.dumps(verified_receipt(HERE/'rml/rml_finalized')));return
    frozen=read(HERE/'frozen_execution.json')
    for name,digest in frozen['source_hashes'].items():
        if sha256_file(HERE/name)!=digest:raise ValueError('Executed source/protocol changed')
    analysis=read(HERE/'results/analysis.json')
    for pointer,digest in analysis['inputs'].items():
        if sha256_file(HERE/pointer)!=digest:raise ValueError('Observed artifact changed')
    execution=read(HERE/'results/execution.json')
    measurement=read(HERE/'results/measurement/measurement.json')
    structure=read(HERE/'results/structure/structure_summary.json')
    if execution['status']!='complete' or measurement['status']!='complete' or structure['status']!='complete':
        raise ValueError('Incomplete execution')
    if execution['frozen_execution_sha256']!=sha256_file(HERE/'frozen_execution.json'):
        raise ValueError('Run/execution binding differs')
    if measurement['measure_sha256']!=frozen['source_hashes']['measure.py']:
        raise ValueError('Measurement source differs')
    if structure['script']['sha256']!=frozen['source_hashes']['structure.py']:
        raise ValueError('Structure source differs')
    full=np.load(HERE/'results/structure/structure_perrow.npz')
    if len(full['source_idx'])!=50000 or not np.array_equal(full['source_idx'],np.arange(100000,150000)):
        raise ValueError('Full row identity differs')
    if not all(np.isfinite(full[k]).all() for k in full.files):raise ValueError('Nonfinite retained structure')
    gain=full['reference_AE_eV']-full['candidate_AE_eV']
    if not np.array_equal(gain,full['paired_gain_eV']):raise ValueError('Retained gain does not reproduce')
    if abs(float(gain.mean())-structure['overall']['reference_minus_candidate_mean_AE_eV'])>1e-12:
        raise ValueError('Structure aggregate does not reproduce')
    for arm in ['reference','candidate']:
        a=np.load(HERE/f'results/measurement/{arm}_rows.npz')
        if len(a['source_idx'])!=2048 or not all(np.isfinite(a[k]).all() for k in a.files):
            raise ValueError('Invalid measurements')
        if np.max(np.abs(a['prediction_eV']-a['saved_prediction_eV']))>1e-4:
            raise ValueError('Prediction equivalence failed')
        for layer in [3,6,9]:
            if a[f'layer{layer}_forward_residual_max_abs_error'].max()!=0:
                raise ValueError('Observational hook changed forward')
            if a[f'layer{layer}_mean_update_u_over_n_max_abs_error'].max()>1e-4:
                raise ValueError('Slot numerical identity failed')
    outcome=dict(execution_status='complete_no_training',artifact_status='local_hash_verified',
        comparison_status='consumed_role_observational_diagnostic',scientific_status='NO_TRAIN',
        transfer_status='not_evaluated',budget_decision='no_training_release',full_handoff_status='not_applicable')
    decision=dict(outcome='NO_TRAIN',decision_ref=f'{REL}/terminal_decision.md',next_allowed_actions=[],
        reopen_conditions=['Separately authorized prospective slot utility or isolated capacity diagnostic'])
    role_use=dict(internal_development='selection_used',train_prefix='untouched',official_validation='untouched',
        test_dev='untouched',test_challenge='untouched')
    cost=dict(schema='molgap-cost-event-v1',cost_event_id='cost-k1-slot-observed',trajectory_id=TID,
        action_id='A001',run_id=RUN,attempt_id='attempt-001',platform='local-windows',
        hardware='CPU; inference4threads, structure1thread; no accelerator',category='inference',
        evidence_ref=f'{REL}/acceptance.json',measurement={
            'device_hours':dict(value=None,status='not_applicable'),
            'queue_hours':dict(value=None,status='not_applicable'),
            'wall_hours':dict(value=execution['wall_seconds']/3600,status='measured'),
            'cpu_hours':dict(value=(measurement['process_cpu_seconds']+structure['costs']['process_cpu_seconds'])/3600,status='measured')})
    roles=[dict(schema='molgap-role-event-v1',role_event_id=f'role-k1-slot-{kind}',trajectory_id=TID,
        action_id='A001',run_id=RUN,dataset_identity='pcqm4mv2-ogb-fixed-100k-v1',
        row_manifest_hash='3270b30071692d712c468fa52fa247d37f0a8e4b5444dc7c913aadb8ca0832c4',
        role_name='internal_development',access_kind=kind,selection_used=True,
        evidence_ref=f'{REL}/acceptance.json') for kind in ['prediction_input','labels_read','metric_computed','selection_used']]
    acceptance=dict(format='molgap-k1-slot-observational-acceptance-v1',evidence_id=EID,run_id=RUN,
        outcome=outcome,trajectory_decision=decision,role_use=role_use,costs=[cost],roles=roles,
        checks=dict(prospective_before_inference=True,source_archive_and_files_hash_verified=True,
            strict_checkpoint_loading=True,prediction_reconstruction=True,observational_forward_unchanged=True,
            slot_numerical_identity=True,aligned_graph_rows_targets=True,finite_artifacts=True,
            no_optimizer_updates=True,geometry_stripped=True,protected_roles_untouched=True),
        measured_molecule_forwards=4096,structure_rows=50000,comparison_class='CONTEXT_ONLY',
        scope='Observational slot magnitude and descriptive retained structure analysis; no causal utility claim',
        cost_scope='Launcher wall includes worker imports. Process CPU measured within worker timers only; imports/planning/summary/Git overhead outside CPU window unknown.',
        input_authority_snapshot_layout='reference_bundle.snapshot.json preserves frozen reference_bundle.json hash without importing a foreign canonical bundle',
        limitations=analysis['limitations'])
    atomic_json(HERE/'acceptance.json',acceptance)
    names=['protocol.md','inputs.json','roles.json','prepare.py','launch.py','measure.py','structure.py',
        'summarize.py','plot.py','finalize_analysis.py','frozen_execution.json','terminal_decision.md','attribution.md',
        'acceptance.json','results/execution.json','results/analysis.json','results/summary.png',
        'results/measurement/measurement.json','results/measurement/reference_rows.npz',
        'results/measurement/candidate_rows.npz','results/structure/structure_summary.json',
        'results/structure/structure_perrow.npz']
    names += ['input_authority/'+p.name for p in (HERE/'input_authority').iterdir() if p.is_file()]
    hashes={f'{REL}/{name}':sha256_file(HERE/name) for name in names}
    authority=[f'{REL}/{name}' for name in ['protocol.md','terminal_decision.md','attribution.md','acceptance.json','results/analysis.json']]
    retained=['results/analysis.json','acceptance.json','terminal_decision.md',
        'results/measurement/reference_rows.npz','results/measurement/candidate_rows.npz','results/structure/structure_perrow.npz']
    evidence=dict(format='molgap-v5-evidence-envelope-v1',contract='MOLGAP-COMMON-V5-FINAL',evidence_id=EID,
        track='B',scope='desktop_k1_retained_slot_readout_observational_diagnostic',
        legacy_contract='pcqm-k1-slot-readout-diagnostic-v1',outcome=outcome,authority=dict(pointers=authority),
        role_use=role_use,migration=dict(migrated_at='2026-10-02',training_executed=False,inference_executed=False,
            scientific_reinterpretation=False,verification_scope='Metadata export of separately executed prospective frozen inference; no inference during publication'),
        observed_execution=dict(training_executed=False,inference_executed=True,execution_ref=f'{REL}/results/execution.json'),
        artifacts=[dict(name=name,locator=f'{REL}/{name}',sha256=hashes[f'{REL}/{name}'],
            availability='locally_retained_hash_verified') for name in retained])
    terminal=dict(format='molgap-rml-terminal-package-v1',trajectory_id=TID,run_id=RUN,action_id='A001',
        finalized_at=datetime.now(timezone.utc).isoformat(),acceptance_ref=f'{REL}/acceptance.json',
        artifact_hashes=hashes,evidence=evidence,decision=decision,costs=[cost],roles=roles)
    atomic_json(HERE/'terminal.json',terminal)
    print(json.dumps(finalize(ROOT,f'{REL}/rml',f'{REL}/terminal.json')))

if __name__=='__main__':main()
