"""Use the shared package-only bootstrap on actual serialized inputs."""
import json
from pathlib import Path
import pickletools
import subprocess
import sys
import sysconfig
import zipfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def main():
    from molgap.experiment_preflight import _atomic
    binding=json.loads((HERE/'upload_binding.json').read_text())
    payload=Path(binding['file']).parent/'payload'
    globals_found=set()
    for name in ('train_probe.pt','development_probe.pt','selected.pt','original_prediction.pt'):
        with zipfile.ZipFile(payload/name) as z:
            entry=next(n for n in z.namelist() if n.endswith('/data.pkl'))
            for op,arg,_ in pickletools.genops(z.read(entry)):
                if op.name=='GLOBAL':globals_found.add(tuple(arg.split(' ',1)))
    request=HERE/'import_request.json';response=HERE/'payload_import_check.json'
    _atomic(request,{'mode':'release-imports','source_root':str(payload),
        'modules':['molgap.k1_bn_mechanism','molgap.k1_bn_calibration','molgap.k1_frozen_inference'],
        'pickle_globals':sorted(globals_found),
        'dependency_paths':sorted({sysconfig.get_path('purelib'),sysconfig.get_path('platlib')})})
    subprocess.run([sys.executable,'-I','-S',str(ROOT/'src/molgap/experiment_preflight.py'),
        str(request),str(response)],check=True,timeout=60)
    report=json.loads(response.read_text())
    if report['errors']:raise ValueError(report['errors'])
    print('PACKAGE_ONLY_IMPORTS_VERIFIED',len(globals_found),flush=True)
if __name__=='__main__':main()
