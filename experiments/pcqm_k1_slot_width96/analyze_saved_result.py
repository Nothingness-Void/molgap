"""Thin retained-output comparison using the existing paired-metrics owner."""
from pathlib import Path
import json
import importlib.util
import numpy as np
import torch
from molgap.constants import REPO_ROOT as ROOT
from molgap.training_reproducibility import atomic_json,sha256_file
from molgap.experiment_spec import ExperimentSpec
from molgap.experiment_family_workflow import RunContext,inspect_output

HERE=ROOT/'experiments/pcqm_k1_slot_width96'
RECORDS=HERE/'reconciliation_v1'
REMOTE=ROOT/'platforms/_records/kaggle/training/pcqm_k1_slot_width96_kaggle3_v1/reconciliation_v1'
BASES={'reference':ROOT/'platforms/_records/kaggle/training/pcqm_k1_width256_reference_kaggle3_v1/k1_pair/reference',
       'candidate':REMOTE/'family_outputs/slot96'}

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def rel(path):return path.relative_to(ROOT).as_posix()

def main():
    RECORDS.mkdir(exist_ok=True)
    package=ROOT/'platforms/_records/kaggle/staging/slot96_workflow_v1b/package'
    manifest=read(package/'package_manifest.json')
    spec=ExperimentSpec.from_json((HERE/'experiment_spec_kaggle3_v1.json').read_text())
    receipt=next((HERE/'submission_v1/receipts').glob('*.json'))
    context=RunContext.from_launch(spec,receipt,package,expected_package_identity=manifest['package_identity'],arm_id='slot96')
    expected=read(HERE/'family_acceptance_plan.json')['arms'][0]['expected']
    mechanical=inspect_output(BASES['candidate'],context=context,expected=expected)
    atomic_json(RECORDS/'mechanical_acceptance.json',mechanical)
    if mechanical['status']!='MECHANICALLY_VERIFIED':raise RuntimeError('Mechanical output gate failed')
    helper_path=ROOT/'experiments/pcqm_gptrans_100k_transfer_control/analyze_pair.py'
    helper_spec=importlib.util.spec_from_file_location('retained_pair_owner',helper_path)
    helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
    predictions,traces,runtimes,outputs,artifacts={},{},{},{},{}
    for name,base in BASES.items():
        outputs[name]=read(base/'output_manifest.json')
        artifacts[name]={}
        for role,item in outputs[name]['artifacts'].items():
            path=base/item['path']
            if sha256_file(path)!=item['sha256']:raise RuntimeError('Retained artifact byte identity differs')
            artifacts[name][role]={'path':rel(path),'sha256':item['sha256']}
        predictions[name]=torch.load(base/'development_predictions.pt',map_location='cpu',weights_only=True)
        traces[name]=read(base/'canonical_trace.json')
        runtime_path=base/'runtime_manifest.json' if name=='reference' else REMOTE/'qualification/slot96/runtime_manifest.json'
        runtimes[name]=read(runtime_path)
        observations=traces[name]['observations']
        if len(observations)!=40:raise RuntimeError('Incomplete exposure')
        for epoch,row in enumerate(observations,1):
            if (row['epoch_or_pass'],row['optimizer_step'],row['sample_presentations'])!=(epoch,epoch*781,epoch*99968):
                raise RuntimeError('Exposure differs from fixed contract')
        if observations[-1]['checkpoint_identity']!='sha256:'+artifacts[name]['resume']['sha256']:
            raise RuntimeError('Trace final checkpoint differs')
    semantics={name:trace['metric_semantics'] for name,trace in traces.items()}
    # Preserve both literal role labels; rows/targets establish descriptive pairing.
    ref_sem=json.loads(json.dumps(semantics['reference']))
    cand_sem=json.loads(json.dumps(semantics['candidate']))
    labels={name:sem['live_dev_metric']['role_identity'] for name,sem in semantics.items()}
    if labels!={'reference':'pcqm4mv2-ogb-fixed-100k-v1:internal-development-100000-150000',
                'candidate':'pcqm4mv2-ogb-fixed-100k-v1:development-100000-150000'}:
        raise RuntimeError('Unreviewed development role identity')
    del ref_sem['live_dev_metric']['role_identity'];del cand_sem['live_dev_metric']['role_identity']
    if ref_sem!=cand_sem:raise RuntimeError('Other trace metric semantics differ')
    endpoint=helper.paired_metrics(predictions['reference'],predictions['candidate'],100000,150000)
    endpoint.update(gate_eV=.003,gain_definition='reference MAE minus candidate MAE',
       bootstrap_replicates=1000,bootstrap_seed=42,training_stochasticity='unknown_single_seed42')
    if abs(endpoint['candidate_mae_eV']-mechanical['observed']['development_mae_eV'])>1e-12:
        raise RuntimeError('Mechanical and paired endpoint differ')
    curve=[]
    for ref,cand in zip(traces['reference']['observations'],traces['candidate']['observations']):
        if ref['learning_rate']!=cand['learning_rate']:raise RuntimeError('Schedule differs')
        curve.append({'epoch':ref['epoch_or_pass'],'optimizer_step':ref['optimizer_step'],
          'sample_presentations':ref['sample_presentations'],'learning_rate':ref['learning_rate'],
          'reference_online_train_mae_eV':ref['live_train_metric'],'candidate_online_train_mae_eV':cand['live_train_metric'],
          'reference_development_mae_eV':ref['live_dev_metric'],'candidate_development_mae_eV':cand['live_dev_metric'],
          'development_gain_eV':ref['live_dev_metric']-cand['live_dev_metric']})
    selection={}
    for name,trace in traces.items():
        selected=min(trace['observations'],key=lambda row:row['live_dev_metric'])
        selection[name]={'epoch':selected['epoch_or_pass'],'development_mae_eV':selected['live_dev_metric'],
           'epochs':40,'optimizer_steps':31240,'sample_presentations':3998720}
    fields=['python','platform','torch','torch_cuda','cudnn','accelerator','determinism',
            'runtime_fingerprint','installed_distributions_sha256']
    distributions={name:set(runtime['installed_distributions']) for name,runtime in runtimes.items()}
    residuals={}
    for name,pred in predictions.items():
        residual=pred['prediction_eV'].double().numpy()-pred['target_eV'].double().numpy()
        residuals[name]={'mean_prediction_minus_target_eV':float(residual.mean()),
                        'residual_quantiles_eV':np.quantile(residual,[.05,.5,.95]).tolist()}
    result={'scope':'Retained50K predictions and40epoch traces; no model execution',
      'helper':{'path':rel(helper_path),'sha256':sha256_file(helper_path),'callable':'paired_metrics'},
      'endpoint':endpoint,'alignment':{'rows':50000,'source_idx_start':100000,'source_idx_stop':150000,
        'exact_order':True,'identical_targets':True,'finite_predictions':True},
      'selection':selection,'paired_curve':curve,'posthoc_descriptive_residuals':residuals,
      'trace_metric_semantics_as_recorded':semantics,
      'development_role_labels_as_recorded':labels,
      'role_label_reconciliation':'Literal labels differ; exact50K rows and targets justify descriptive pairing only. No frozen metadata relabelled.',
      'runtime_summary':{name:{key:runtime[key] for key in fields} for name,runtime in runtimes.items()},
      'installed_distribution_differences':{'reference_only':sorted(distributions['reference']-distributions['candidate']),
                                          'candidate_only':sorted(distributions['candidate']-distributions['reference'])},
      'native_costs':{name:output['costs'] for name,output in outputs.items()},
      'candidate_allocation':read(REMOTE/'qualification/slot96/allocation_cost.json'),
      'artifacts':artifacts,
      'limitations':['Row bootstrap excludes training stochasticity.',
         'Online pre-update dropout training MAE is not fixed-cohort evaluation.',
         'Cross-job software distributions and cost scopes differ; no causal architecture-speedup claim.',
         'Development was already selection-used; no protected role was read.']}
    atomic_json(RECORDS/'scientific_metrics.json',result)
    print(json.dumps({'mechanical':mechanical['status'],'endpoint':endpoint,'selection':selection,
                     'software_differences':result['installed_distribution_differences']}))

if __name__=='__main__':main()
