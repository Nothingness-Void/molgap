"""Specific graph/input binding; shared RML planning, hashing and model owners."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
REL=HERE.relative_to(ROOT).as_posix()
RUN='k1-bn-mechanism-a100-20261008'
TID='TB-k1-bn-mechanism-a100-20261008'
POLICY='pcqm-k1-bn-mechanism-a100'

def main():
    from molgap.research_memory.plan import plan
    from molgap.training_reproducibility import atomic_json, sha256_file
    import numpy as np
    prior=json.loads((ROOT/'experiments/pcqm_k1_500k_bn_calibration/inputs.json').read_text())
    frozen=Path(prior['frozen_source_root']).parent
    cache=Path(prior['cache_root'])
    selected=prior['checkpoints']['best_model.pt']
    saved=prior['checkpoints']['best_predictions.pt']
    for item in (selected,saved):
        if sha256_file(Path(item['path']))!=item['sha256']: raise ValueError('Accepted input changed')
    if sha256_file(cache/'manifest.json')!=prior['manifest_sha256']: raise ValueError('Cache changed')
    names=('__init__','constants','screen_policy','qm9_neural_atom','pcqm_gap_architecture','gps',
           'k1_screen_training','training_reproducibility','v4_runtime','pcqm_wedge')
    source={f'src/molgap/{n}.py':sha256_file(frozen/f'src/molgap/{n}.py') for n in names}
    if any(prior['frozen_source_files'][n]!=h for n,h in source.items()): raise ValueError('Frozen source changed')
    rows=np.random.default_rng(20261008).choice(500000,16384,replace=False).tolist()
    dev=list(range(500000,550000))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    inputs={'source_commit':commit,'checkpoint':selected,'predictions':saved,'source_files':source,
            'cache_root':str(cache),'cache_manifest_sha256':prior['manifest_sha256'],
            'train_source_idx':rows,'development_source_idx':dev,'ceiling_seconds':1200}
    atomic_json(HERE/'inputs.json',inputs)
    atomic_json(HERE/'roles.json',{'train':{'bounds':[0,500000],'decoded_labels_read':500000,
        'prediction_input_rows':16384,'labels_used_for_calibration':False},
        'internal_development':{'bounds':[500000,550000],'labels_read':50000,'metric_computed':True,
            'previously_selection_used':True},'official_validation':'untouched','tests':'untouched'})
    template=json.loads((ROOT/'experiments/pcqm_k1_colab_execution_profile/plan_input.json').read_text())
    old=template['trajectory']['trajectory_id']
    text=json.dumps(template).replace(old,TID).replace('k1-profile-a100-20261007',RUN)
    text=text.replace('pcqm_k1_colab_execution_profile','pcqm_k1_bn_mechanism_a100')
    text=text.replace('pcqm-k1-colab-execution-profile',POLICY).replace('cost-k1-colab-profile-estimate','cost-k1-bn-mechanism-estimate')
    spec=json.loads(text)
    t=spec['trajectory']; t['family_id']='k1-bn-mechanism-a100'
    t['question']='Does dropout or duplicate BN calibration explain frozen K1 normalization error?'
    refs=['pcqm-k1-500k-bn-calibration-20261007','pcqm-k1-colab-execution-profile-20261007']
    t['hypothesis']={'hypothesis_id':'H-k1-bn-mechanism-20261008','supporting_evidence_ids':refs,
        'observed_deficiency':'BN calibration recovered1.245meV; source of normalization mismatch unresolved.',
        'alternative_explanations':['dropout activation mismatch','duplicate BN updates','historical EMA or trajectory effects'],
        'changed_mechanism':'Frozen BN statistics, dropout off/on x one/two repeated forwards',
        'cheapest_falsifier':'Four bounded buffer-only interventions on fixed training members and consumed development',
        'decision_changed_if_positive':'Nominate causal buffer-estimator effect for separate training qualification',
        'decision_changed_if_negative':'Deprioritize tested calibration mechanisms; preserve historical EMA uncertainty',
        'expected_native_cost_ref':'cost-k1-bn-mechanism-estimate','related_closed_family_ids':['k1-500k-bn-calibration']}
    t['state_at_start'].update(source_commit=commit,source_config_identity=sha256_file(HERE/'inputs.json'),
        contract_refs=[f'{REL}/protocol.md',f'{REL}/inputs.json'],role_snapshot_refs=[f'{REL}/roles.json'],prior_evidence_ids=refs)
    t['actions'][0].update(source_commit=commit,type='frozen_bn_mechanism_diagnostic')
    t['decision']['next_allowed_actions']=['Frozen A100 BN mechanism diagnostic']
    t['comparison_blockers']=['Consumed selected-development cohort','No training-time causality or promotion']
    s=spec['decision_state']; s.update(source_commit=commit,state_timestamp=datetime.now(timezone.utc).isoformat(),
        role_snapshot_refs=[f'{REL}/roles.json'])
    policy=json.loads((ROOT/'research_memory/policies/pcqm-k1-colab-execution-profile.1.json').read_text())
    policy.update(policy_id=POLICY,comparability_selector={'scientific_contract':POLICY+'-v1'},
        created_from_source_digest=sha256_file(HERE/'protocol.md'))
    atomic_json(ROOT/f'research_memory/policies/{POLICY}.1.json',policy)
    atomic_json(HERE/'plan_input.json',spec)
    atomic_json(HERE/'plan_receipt.json',plan(ROOT,spec,f'{REL}/rml'))
    # Trusted graph materialization occurs only after prospective publication.
    sys.path.insert(0,str(frozen/'src'))
    import molgap
    molgap.__path__.insert(0,str(frozen/'src/molgap'))
    import torch
    from molgap.k1_screen_training import _PackedGraphDatasetFactory
    manifest=json.loads((cache/'manifest.json').read_text())
    found={}; development=[]; shards=[]
    for shard in manifest['geometry_shards']:
        path=cache/shard['file']
        if sha256_file(path)!=shard['sha256']: raise ValueError('Shard changed')
        ds=_PackedGraphDatasetFactory.load(path)
        ids=ds._data.source_idx.view(-1).long()
        start=int(ids[0]); stop=int(ids[-1])+1
        if not torch.equal(ids,torch.arange(start,stop)): raise ValueError('Shard rows changed')
        if stop<=500000:
            found.update({i:ds[i-start].clone() for i in rows if start<=i<stop})
        elif start==500000 and stop==550000:
            development=[ds[i].clone() for i in range(len(ds))]
        else: raise ValueError('Unauthorized shard')
        shards.append(shard); del ds
    if len(found)!=16384 or len(development)!=50000: raise ValueError('Incomplete inputs')
    payload=ROOT/f'platforms/_records/colab/staging/{RUN}/payload'
    payload.mkdir(parents=True)
    for n in source:
        dest=payload/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(frozen/n,dest)
    for n in ('k1_frozen_inference','k1_bn_calibration','k1_bn_mechanism'):
        shutil.copyfile(ROOT/f'src/molgap/{n}.py',payload/f'src/molgap/{n}.py')
    shutil.copytree(HERE/'rml',payload/'prospective')
    shutil.copyfile(HERE/'protocol.md',payload/'protocol.md')
    shutil.copyfile(selected['path'],payload/'selected.pt')
    shutil.copyfile(saved['path'],payload/'original_prediction.pt')
    torch.save([found[i] for i in rows],payload/'train_probe.pt')
    torch.save(development,payload/'development_probe.pt')
    pm={'format':'molgap-k1-bn-mechanism-payload-v1','train_source_idx':rows,'development_source_idx':dev,
        'sample_source_idx':rows,'checkpoint':{'sha256':selected['sha256'],
        'source_sha256':'0d53f2ae5a46d51cff9730e00767fa16b80ffb27002fa50b52a67c12ac8ab65a'},
        'parent_manifest_sha256':prior['manifest_sha256'],'parent_shards':shards,
        'files':{p.relative_to(payload).as_posix():sha256_file(p) for p in payload.rglob('*') if p.is_file()}}
    atomic_json(payload/'payload_manifest.json',pm);atomic_json(HERE/'payload_manifest.json',pm)
    archive=payload.parent/(RUN+'.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in payload.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(payload))
    atomic_json(HERE/'upload_binding.json',{'file':str(archive),'sha256':sha256_file(archive),'bytes':archive.stat().st_size})
    print(archive,archive.stat().st_size,flush=True)

if __name__=='__main__':main()
