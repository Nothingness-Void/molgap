"""Saved-artifact SSMA comparison; reuses paired metrics and frozen bootstrap."""
from reconcile_accuracy import ROOT,HERE,RECORD,OUT,read,write,sha,load
import numpy as np
import torch
from molgap.experiment_family_workflow import tensor_digest
from molgap.k1_screen_training import validate_runtime_preflight
from molgap.training_reproducibility import assert_finite_state_dict
import hashlib
def main():
    helper_path=ROOT/'experiments/pcqm_gptrans_100k_transfer_control/analyze_pair.py'
    helper=load(helper_path,'saved_pair_owner')
    payloads={};arms={}
    for name in ('reference','ssma'):
        base=OUT/name/'family_outputs'; manifest=read(base/'output_manifest.json');contract=read(base/'training_contract.json')
        for item in manifest['artifacts'].values():
            if sha(base/item['path'])!=item['sha256']:raise ValueError('Artifact hash changed')
        pred=torch.load(base/'development_predictions.pt',map_location='cpu',weights_only=True);payloads[name]=pred
        trace=read(base/'canonical_trace.json');rows=trace['observations']
        if len(rows)!=40:raise ValueError('Exposure row count differs')
        for epoch,row in enumerate(rows,1):
            if (row['epoch_or_pass'],row['optimizer_step'],row['sample_presentations'])!=(epoch,epoch*781,epoch*99968):raise ValueError('Exposure differs')
        states={role:torch.load(base/manifest['artifacts'][role]['path'],map_location='cpu',weights_only=True) for role in ('selected_model','resume')}
        for role,state in states.items():
            assert_finite_state_dict(state['model'],label=role)
            if state['context']!=manifest['context']:raise ValueError('State context differs')
        selected=min(rows,key=lambda r:r['live_dev_metric'])
        if states['selected_model']['epoch']!=selected['epoch_or_pass']:raise ValueError('Selected epoch differs')
        qual=RECORD/'qualification'/name;provenance=read(qual/'runtime_provenance.json')
        certificate=validate_runtime_preflight(qual,provenance,overhead_policy='report_only')
        target=pred['target_eV'];f32=hashlib.sha256(target.float().numpy().astype('<f4').tobytes()).hexdigest()
        if f32!=contract['acceptance_requirements']['target_sha256']:raise ValueError('Frozen rawfloat32 target differs')
        arms[name]={'manifest_sha256':sha(base/'output_manifest.json'),'target_sha256_raw_float32':f32,'target_sha256_canonical_float64':tensor_digest(target,role='target'),'selected_epoch':selected['epoch_or_pass'],'selected_metric_eV':selected['live_dev_metric'],'trace':rows,'runtime_certificate':certificate,'architecture':read(qual/'architecture_preflight.json'),'allocation':read(qual/'allocation_cost.json')}
    endpoint=helper.paired_metrics(payloads['reference'],payloads['ssma'],100000,150000)
    target=payloads['reference']['target_eV'].double().numpy()
    gain=np.abs(payloads['reference']['prediction_eV'].double().numpy()-target)-np.abs(payloads['ssma']['prediction_eV'].double().numpy()-target)
    rng=np.random.default_rng(42);bootstrap=np.array([gain[rng.integers(0,len(gain),len(gain))].mean() for _ in range(10000)])
    endpoint.update(row_bootstrap_95pct_eV=np.quantile(bootstrap,[.025,.975]).tolist(),bootstrap_replicates=10000,bootstrap_seed=42)
    result={'scope':'Retained artifact tensors and metadata only; no model execution','endpoint':endpoint,'arms':arms,'helper_sha256':sha(helper_path),'alignment':{'rows':50000,'exact_source_idx':True,'identical_targets':True},'target_digest_disposition':'Frozen recipe matches rawfloat32-le; reviewed k1-screen-v1 profile validates this encoding. Original owner generic inspector predates this profile fix; raw metadata preserved.','limitations':['Single seed; row bootstrap is not training stochasticity','Repeatedly selection-used internal development','Online dropout training metric differs from fixed-cohort evaluation']}
    write(RECORD/'scientific_metrics.json',result)
    print({'endpoint':endpoint,'selection':{a:r['selected_epoch'] for a,r in arms.items()}})
if __name__=='__main__':main()
