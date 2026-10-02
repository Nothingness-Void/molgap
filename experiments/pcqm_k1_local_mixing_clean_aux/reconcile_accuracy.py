"""Scoped skill-side retrieval and existing family inspection; no model execution."""
from pathlib import Path
import importlib.util
import json
import hashlib
from datetime import datetime, timezone
import requests
from molgap.experiment_spec import ExperimentSpec
from molgap.experiment_family_workflow import RunContext, inspect_output
from molgap.screen_policy import canonical_fingerprint

ROOT=Path('D:/w/k1-local-aux')
HERE=ROOT/'experiments/pcqm_k1_local_mixing_clean_aux'
RECORD=HERE/'kaggle3_accuracy_reconciliation_v1'
OUT=ROOT/'platforms/_records/kaggle/training/pcqm_k1_ssma_accuracy_kaggle3_v1'
KERNEL='nvoid912/molgap-k1-v4-ssma-accuracy-100k-s42-v1'
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
    api,scope=load(Path('D:/w/k1-slot96/platforms/_records/kaggle/staging/slot96_platform_ops.py'),'api_owner').api_pair()
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiGetKernelSessionStatusRequest,ApiListKernelSessionOutputRequest
    from kagglesdk.kaggle_object import KaggleObject
    with scope() as client:
        service=client.kernels.kernels_api_client
        req=ApiGetKernelRequest();req.user_name='nvoid912';req.kernel_slug=KERNEL.split('/')[1]
        kernel=service.get_kernel(req);metadata=KaggleObject.to_dict(kernel.metadata)
        req=ApiGetKernelSessionStatusRequest();req.user_name='nvoid912';req.kernel_slug=KERNEL.split('/')[1]
        status=KaggleObject.to_dict(service.get_kernel_session_status(req))
        req=ApiListKernelSessionOutputRequest();req.user_name='nvoid912';req.kernel_slug=KERNEL.split('/')[1];req.page_size=100
        listing=service.list_kernel_session_output(req)
        if listing.next_page_token: raise ValueError('Unexpected output pagination')
        urls={f.file_name:f.url for f in listing.files}
    actual=kernel.blob.source.encode('utf-8'); expected=(HERE/'kaggle3_accuracy_submission_v1/submitted_entry.py').read_bytes()
    if actual.replace(b'\r\n',b'\n')!=expected.replace(b'\r\n',b'\n'):raise ValueError('Remote source mismatch')
    if status['status']!='COMPLETE' or metadata['currentVersionNumber']!=1 or metadata['id']!=136643751:raise ValueError('Exact terminal identity mismatch')
    write(RECORD/'remote_observation.json',{'kernel':KERNEL,'observed_at_utc':datetime.now(timezone.utc).isoformat(),'status':status,'metadata':metadata,'remote_entry_sha256':hashlib.sha256(actual).hexdigest(),'submitted_entry_sha256':hashlib.sha256(expected).hexdigest(),'normalized_entry_match':True,'output_file_names':sorted(urls)})
    def small(remote,path):
        r=requests.get(urls[remote],timeout=(30,60))
        if r.status_code!=200 or len(r.content)>8*1024*1024:raise ValueError('Bounded metadata retrieval failed')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=r.content:raise ValueError('Retained metadata conflict')
        path.write_bytes(r.content);return sha(path)
    spec=ExperimentSpec.from_json((HERE/'experiment_spec_kaggle3_accuracy_v1.json').read_text())
    binding=read(HERE/'kaggle3_accuracy_submission_v1/submission-response.json')['release_binding']
    adapter=load(ROOT/'platforms/kaggle/retrieve_family_outputs.py','retrieval_owner')
    for arm in spec.to_dict()['arms']:
        name=arm['arm_id'];remote='k1_pair/'+name+'/output_manifest.json';pin=OUT/name/'pinned_output_manifest.json';digest=small(remote,pin)
        manifest=read(pin);ctx=dict(manifest['context']);ctx['platform_version']='1';context=RunContext(**ctx)
        for key in ('spec_identity','source_commit','source_archive_sha256','package_identity'):
            if ctx[key]!=binding[key]:raise ValueError('Frozen release mismatch: '+key)
        if ctx['arm_identity']!=canonical_fingerprint(arm) or ctx['training_recipe_sha256']!=arm['training']['recipe']['sha256'] or ctx['run_reference']!=KERNEL:raise ValueError('Arm context mismatch')
        receipt=adapter.retrieve_family_outputs(api,context=context,remote_manifest_path=remote,local_manifest=pin,manifest_sha256=digest,destination=OUT/name/'family_outputs')
        write(RECORD/(name+'_retrieval.json'),receipt)
        contract=read(OUT/name/'family_outputs'/manifest['artifacts']['contract']['path'])
        # Reviewed desktop inspector accepts the producer's optional certificate
        # digest; the retained owner inspector predates that metadata extension.
        import sys
        inspector_path=Path('D:/w/k1-slot96/src/molgap/experiment_family_workflow.py')
        profile_path=inspector_path.with_name('experiment_family_artifacts.py')
        profile_spec=importlib.util.spec_from_file_location('molgap.experiment_family_artifacts',profile_path)
        profile=importlib.util.module_from_spec(profile_spec);sys.modules[profile_spec.name]=profile;profile_spec.loader.exec_module(profile)
        module_spec=importlib.util.spec_from_file_location('molgap.desktop_output_inspector',inspector_path)
        inspector=importlib.util.module_from_spec(module_spec);sys.modules[module_spec.name]=inspector;module_spec.loader.exec_module(inspector)
        inspected_context=inspector.RunContext(**context.to_dict())
        report=inspector.inspect_output(OUT/name/'family_outputs',context=inspected_context,expected=contract['acceptance_requirements'])
        report['inspector_source']={'path':str(inspector_path),'sha256':sha(inspector_path),'profile_path':str(profile_path),'profile_sha256':sha(profile_path),'reason':'Reviewed k1-screen-v1 raw float32 target binding and optional runtime_certificate_id support; raw output unchanged'}
        write(RECORD/(name+'_mechanical.json'),report)
        extras=['architecture_preflight.json','runtime_certificate.json','runtime_manifest.json','runtime_provenance.json','allocation_cost.json','diagnostic_cost.json','canonical_trace.json.context.json']
        for filename in extras:
            small('k1_pair/'+name+'/'+filename,RECORD/'qualification'/name/filename)
        print(json.dumps({'arm':name,'mechanical':report['status'],'mae':report['observed'].get('development_mae_eV'),'blockers':report['blockers']}),flush=True)
    small('k1_pair/pair_state.json',RECORD/'qualification/pair_state.json')
if __name__=='__main__':
    try:main()
    except Exception as e:
        print(json.dumps({'status':'RECONCILIATION_BLOCKED','error_type':type(e).__name__,'reason':str(e) if isinstance(e,ValueError) else 'Remote operation failed; private URLs omitted'}));raise SystemExit(1)
