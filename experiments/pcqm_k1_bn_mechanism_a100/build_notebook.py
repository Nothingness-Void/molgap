"""Reuse the accepted Colab bootstrap; bind only this diagnostic's identity."""
import importlib.util
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    owner=HERE.parent/'pcqm_k1_colab_execution_profile/build_notebook.py'
    spec=importlib.util.spec_from_file_location('accepted_colab_bootstrap',owner)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.HERE=HERE
    module.main()
    old=HERE/'MolGap_K1_A100_Profile.ipynb'
    notebook=json.loads(old.read_text())
    for cell in notebook['cells']:
        text=''.join(cell['source']).replace('k1-profile-a100-20261007','k1-bn-mechanism-a100-20261008')
        text=text.replace('molgap.k1_execution_profile','molgap.k1_bn_mechanism')
        text=text.replace("log=RUN_ROOT/'worker.log'","log=RUN_ROOT/'liveworker.log'")
        text=text.replace('DURABLE_K1_PROFILE_COMPLETE','DURABLE_K1_BN_MECHANISM_COMPLETE')
        text=text.replace('V5/runs/k1-bn-mechanism-a100-20261008\'','V5/runs/k1-bn-mechanism-a100-20261008/attempt-001\'')
        cell['source']=text.splitlines(True)
    notebook['cells'][0]['source']=['# MolGap K1 frozen BN mechanism on A100\n',
        'Four buffer-only interventions; frozen weights; consumed internal development only.\n',
        'Results saved to private Drive. Release runtime after completion.\n']
    notebook['cells'][-1]['source']='''completion=json.loads((RUN_ROOT/'completion.json').read_text())
assert completion['complete']
for name,digest in completion['files'].items():
    assert hashlib.sha256((RUN_ROOT/name).read_bytes()).hexdigest()==digest,name
result=json.loads((RUN_ROOT/'result.json').read_text())
print('Original MAE:',result['original_mae_eV'])
for case in result['cases']:
    print(case['case'],'MAE:',case['mae_eV'],'seconds:',case['case_wall_seconds'])
with zipfile.ZipFile(DRIVE_ROOT/'k1-bn-mechanism-a100-results.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in RUN_ROOT.iterdir():
        if p.is_file():z.write(p,p.name)
print('DURABLE_BN_MECHANISM_RESULTS_COMPLETE',RUN_ROOT,flush=True)
'''.splitlines(True)
    notebook['metadata']['colab']['name']='MolGap_K1_BN_Mechanism_A100.ipynb'
    destination=HERE/'MolGap_K1_BN_Mechanism_A100.ipynb'
    destination.write_text(json.dumps(notebook,indent=2)+'\n',encoding='utf-8')
    old.unlink()
    print(destination)

if __name__=='__main__':main()
