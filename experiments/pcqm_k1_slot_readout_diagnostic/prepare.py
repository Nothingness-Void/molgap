"""Publish a bounded observational plan before checkpoint inference."""
from pathlib import Path
import json
import hashlib
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = Path('D:/w/k1-width256')
REL = HERE.relative_to(ROOT).as_posix()
TID = 'TB-k1-slot-readout-diagnostic-20261002'
RUN = 'local-k1-slot-readout-20261002'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n', encoding='utf-8')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    from molgap.research_memory.plan import plan
    if (HERE / 'rml').exists():
        raise FileExistsError('Prospective already exists; do not rewrite')
    base = 'platforms/_records/kaggle/training/'
    inputs = {'format':'molgap-k1-slot-inputs-v1', 'checkpoints':{}, 'predictions':{}}
    paths = {'width192':OLD / base / 'pcqm_k1_width256_reference_kaggle3_v1/k1_pair/reference',
             'width256':OLD / base / 'pcqm_k1_node_width256_kaggle3_v1/experiment/width256'}
    pins = {'width192':('4a7dab43f50f5f3216b224374c213a01cc82cfb4160dcd6d54aab1ef614fce17',
                       '857fe31475d6728c3c9b0612571a0397a67db93815714c20ac107e06e44d4e82'),
            'width256':('bcdcfcd5c38b3a97d9c45bba67463492ee6ef28dcd79f0d9c3323640403750dd',
                       'a43b162b691ac4e69da97d70f71a97f471fde44759c6321adf60c0aa9b52581f')}
    for arm, directory in paths.items():
        for kind, name, digest in [('checkpoints','selected_model.pt',pins[arm][0]),
                                    ('predictions','development_predictions.pt',pins[arm][1])]:
            p = directory / name
            if sha(p) != digest:
                raise ValueError(f'Input hash mismatch {arm} {kind}')
            inputs[kind][arm] = {'path':p.as_posix(),'sha256':digest}
    cache = Path('D:/文档/molgap/data/pcqm_fixed_100k_v1')
    manifest = cache / 'manifest.json'
    if sha(manifest) != '1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d':
        raise ValueError('Manifest hash mismatch')
    shard = next(s for s in read(manifest)['geometry_shards'] if s['role']=='development')
    p = cache / shard['file']
    if sha(p) != shard['sha256']:
        raise ValueError('Development shard hash mismatch')
    inputs['cache'] = {'root':cache.as_posix(),'manifest_sha256':sha(manifest),
                       'development_shard':{'path':p.as_posix(),'sha256':sha(p)}}
    archive = OLD / 'platforms/_records/kaggle/staging/k1_width256_v1/package/source.tar.gz'
    if sha(archive) != '905c12f632605608886524ef38ecd96d2b60f176ace05d2a627827b61dbc0b1e':
        raise ValueError('Source archive mismatch')
    source = archive.parent.parent / 'cpu_source/src'
    inputs['source'] = {'archive':archive.as_posix(),'archive_sha256':sha(archive),
                        'pythonpath':source.as_posix(),
                        'files':{p.relative_to(source).as_posix():sha(p) for p in source.rglob('*.py')}}
    inputs['authorization'] = {'authority':'User 2026-10-02: 做分析吧, accepting frozen-checkpoint slot/readout diagnostic',
        'roles':'Existing internal development source_idx [100000,150000); no train/official-valid/test access',
        'training':False,'geometry_constructed':False,'parameter_updates':False}
    inputs['settings'] = {'sample_size':2048,'sample_rng_seed':20261002,'batch_size':64,
        'cpu_threads':4,'loader_workers':0,'precision':'fp32','mode':'eval',
        'inference_wall_ceiling_seconds':600,'structure_wall_ceiling_seconds':600,
        'prediction_reconstruction_max_abs_eV':0.0001,
        'source_indices_sha256':'3270b30071692d712c468fa52fa247d37f0a8e4b5444dc7c913aadb8ca0832c4',
        'target_sha256':'a1c0b711815225f0bccee905419d124174e1bf01c76877f5c661b063ee170137'}
    authority = HERE / 'input_authority'
    authority.mkdir(exist_ok=True)
    for name in ['terminal_decision.md','kaggle3_reconciliation_v1/acceptance.json',
                 'kaggle3_reconciliation_v1/scientific_metrics.json','reference_binding/reference_bundle.json']:
        origin = OLD / 'experiments/pcqm_k1_node_width256' / name
        shutil.copyfile(origin, authority / origin.name)
    inputs['accepted_owner'] = {'branch':'codex/exp/k1-node-width256','commit':'b6e6e7a5',
        'canonical_decision':(OLD / 'experiments/pcqm_k1_node_width256/terminal_decision.md').as_posix(),
        'retained_authority_hashes':{p.name:sha(p) for p in authority.iterdir()}}
    write(HERE / 'inputs.json',inputs)
    write(HERE / 'roles.json',{'role':'fixed50k_internal_development','source_idx_start':100000,
        'source_idx_stop':150000,'previously_selection_used':True,'current_access':
        ['prediction_input','labels_read','metric_computed','selection_used'],
        'selection_note':'posthoc diagnosis guides future research, not independent validation',
        'official_validation':'untouched','test_dev':'untouched','test_challenge':'untouched'})
    protocol = '''# K1 slot/readout diagnostic v1

Desktop-owned observational diagnostic of retained width192/256 checkpoints.
The accepted width256 bounded negative result remains immutable.
No training, optimizer, parameter update, geometry construction, or official validation/test.

## Question and falsifier

Does final K1 slot return become weak after mean pooling, and do width256
regressions concentrate by 2D graph size or cyclic/aromatic/conjugated structure?
The eval identity mean(delta_h)=u/N does not alone establish harmful attenuation:
u may adapt with N and learned head weights may compensate.
Measure norm ratios, assignment entropy/effective atom count, and size associations.
Width remains a cross-checkpoint observation with single-seed/runtime confounding.
No ablation or output-rescaling intervention is released by this contract.

## Inputs and roles

Hash-pinned selected_model, saved development predictions, source archive/extracted
Python files, and retained cache development shard in inputs.json.
Only source_idx [100000,150000), already consumed for selection. Strip geometry
through the existing pure2D loader; only original atom9/bond3/RWSE16 enter models.
No training-role rows loaded; frozen denormalization mean/std from accepted contract.
Checkpoint state strict loading and parameter counts 3658817/6035201 required.

## Observations

Uniform sample of 2048 offsets: numpy.default_rng(20261002).choice(50000,2048,
replace=False), sorted. CPU FP32 eval/inference_mode, four threads, batch64,
workers0. Read-only hooks at mixers3/6/9; no altered forward output.
Max CPU-vs-retained prediction difference must be <=1e-4 eV or model
interpretation stops. Measure pre-mixer atom-mean norm, mean-update norm,
their ratio, sum-update norm, entropy, effective atom count, max assignment,
and numerical single-slot mean-return identity. Report pergraph values.

Full50K graph metadata joins exact saved rows and targets. Size bins <=15,
16-25,26-35,>=36; cycle rank E-N+components bins0,1,2,>=3. OGB boolean aromatic,
in-ring and conjugated fractions only with verified feature mapping. Fraction
quartiles are descriptive posthoc boundaries. Paired row bootstrap1000 seeded
replicates; row uncertainty is not training stochasticity. No causal cohort claim.

## Cost and stop

Each worker wall ceiling600seconds; total scientific worker ceiling1200seconds.
Record process CPU and wall separately; local GPU/device and queue not applicable.
Atomic outputs per arm retain progress. Stop on changed hashes, missing/unaligned
rows, nonfinite results, reconstruction failure or ceiling; no retries/training.
Planning/implementation/git wall outside measured worker windows remains unknown.

## Interpretation

CONTEXT_ONLY; no performance promotion, strict causal comparison, replay-ready
trace, new protected role or automatic successor. A large learned slot contribution
falsifies the simple claim that it is numerically negligible. Small contribution
supports an intervention question but cannot prove that its information is useless.
Preserve terminal failure attribution and distinguish mechanism algebra from effect.
'''
    (HERE / 'protocol.md').write_text(protocol,encoding='utf-8')
    (HERE / 'decision.md').write_text('# Prospective decision\n\nACTIVE: perform authorized bounded observational diagnostic. No training release.\n',encoding='utf-8')
    cost = read(OLD / 'experiments/pcqm_scale_fit_retained/expected_cost.json')
    cost.update(cost_event_id='cost-k1-slot-expected',trajectory_id=TID,run_id=RUN,
        hardware='local Windows CPU, four intra-op threads',evidence_ref=f'{REL}/protocol.md')
    cost['measurement']={'device_hours':{'value':None,'status':'not_applicable'},
        'cpu_hours':{'value':1.3333333333333333,'status':'estimated'},
        'wall_hours':{'value':1/3,'status':'estimated'},'queue_hours':{'value':None,'status':'not_applicable'}}
    write(HERE / 'expected_cost.json',cost)
    policy = read(OLD / 'research_memory/policies/pcqm-scale-fit-retained-diagnostic.1.json')
    policy.update(policy_id='pcqm-k1-slot-readout-diagnostic',created_from_source_digest=sha(HERE/'protocol.md'))
    policy['comparability_selector']={'scientific_contract':'pcqm-k1-slot-readout-diagnostic-v1'}
    write(ROOT / 'research_memory/policies/pcqm-k1-slot-readout-diagnostic.1.json',policy)
    trajectory = read(OLD / 'experiments/pcqm_scale_fit_retained/trajectory_template.json')
    trajectory.update(trajectory_id=TID,family_id='k1-slot-readout-diagnostic',question=
        'Measure retained K1 slot contribution through mean readout and describe width256 regressions by pure2D structure.')
    trajectory['hypothesis'].update(hypothesis_id='H-k1-slot-readout-20261002',
        observed_deficiency='Width256 accepted negative bounded endpoint; final K1 update has eval u/N mean identity, but its learned magnitude and size association are unknown.',
        supporting_evidence_ids=['pcqm-matched-500k-v4-three-arm'],alternative_explanations=['Learned u scales with N or head compensates.',
            'Width regression reflects local representation/optimization or single-seed variance.',
            'Observed structure association is confounded by correlated graph features and repeated selection.'],
        changed_mechanism='No change: observational hooks and exact retained-prediction/graph joins only.',
        cheapest_falsifier='2048 retained graph forwards per checkpoint plus full50K graph metadata; fail closed on reconstruction or cost ceiling.',
        related_closed_family_ids=['neural-atom-k1','gptrans-500k-frozen-readout'],
        expected_native_cost_ref=cost['cost_event_id'],
        decision_changed_if_positive='Prioritize a separately frozen slot/readout intervention question, not training.',
        decision_changed_if_negative='Reject numerical-negligibility shortcut; prioritize unresolved conditional information or local-path diagnosis.',
        historical_unknowns=['Single-seed training variance.','Slot contribution causal utility without intervention.'])
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    trajectory['state_at_start'].update(source_commit=commit,contract_refs=[f'{REL}/protocol.md',f'{REL}/inputs.json'],
        reference_ids=[],parent_trajectory_ids=[],prior_evidence_ids=['pcqm-matched-500k-v4-three-arm'],role_snapshot_refs=[f'{REL}/roles.json'],
        budget_snapshot_ref=f'{REL}/protocol.md',source_config_identity='Verified desktop tip plus separately frozen worker/source file hashes; external accepted checkpoint identity in inputs.')
    trajectory['actions']=[{'action_id':'A001','type':'local_frozen_checkpoint_observational_diagnostic',
        'source_commit':commit,'run_ids':[RUN],'attempt_ids':['attempt-001'],'evidence_refs':[f'{REL}/inputs.json'],
        'cost_event_ids':[cost['cost_event_id']]}]
    trajectory['decision'].update(decision_ref=f'{REL}/decision.md',next_allowed_actions=['Bounded authorized observational diagnostic; no training'])
    trajectory.update(comparison_readiness_ref=f'{REL}/protocol.md',comparison_blockers=
        ['Consumed development role, single seed; observational not causal.','Reference external owner identity hash-bound, not a strict replay reference ID.'])
    action={'trajectory_id':TID,'state_timestamp':'2026-10-02','evidence_ids':['pcqm-matched-500k-v4-three-arm'],
        'user_requested_scope_reviewed':1,'authority':inputs['authorization']['authority']}
    write(HERE/'action_inputs.json',action)
    spec={'trajectory':trajectory,'costs':[cost],'action_inputs_ref':f'{REL}/action_inputs.json',
        'decision_state':{'known_trajectory_ids':[],'known_evidence_ids':[],'active_reference_ids':[],
        'available_actions':['RUN_DIAGNOSTIC','NO_TRAIN'],'chosen_action':'RUN_DIAGNOSTIC',
        'policy_id':policy['policy_id'],'policy_version':'1','role_snapshot_refs':[f'{REL}/roles.json'],
        'budget_snapshot_ref':f'{REL}/protocol.md','state_timestamp':'2026-10-02','source_commit':commit}}
    write(HERE/'plan_input.json',spec)
    receipt=plan(ROOT,spec,HERE/'rml')
    write(HERE/'plan_receipt.json',receipt)
    print(json.dumps(receipt))

if __name__=='__main__':
    main()
