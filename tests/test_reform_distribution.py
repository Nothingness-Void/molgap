"""Actual fixed source package in an isolated CPU interpreter, not a release."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from time import perf_counter

from molgap.experiment_package import build_experiment_source_package,verify_experiment_source_package
from molgap.experiment_source_inventory import runtime_source_files
from molgap.experiment_spec import ExperimentSpec
from molgap.v4_bundle import committed_payloads
from test_experiment_spec import payload

ROOT=Path(__file__).resolve().parents[1]


def test_committed_distribution_imports_actual_families_and_resumes_without_host_source(tmp_path,payload,record_property):
    names=runtime_source_files(ROOT)
    cache=tmp_path/"source-cache"
    measurements={}
    manifests=[]
    for label in ("cold","warm"):
        value=deepcopy(payload);value["logical_run_id"]="distribution-"+label
        started=perf_counter()
        target=tmp_path/label
        manifests.append(build_experiment_source_package(ExperimentSpec(value),ROOT,names,target,source_cache=cache))
        measurements[label+"_package_seconds"]=perf_counter()-started
        assert verify_experiment_source_package(target,ROOT)==manifests[-1]
    assert manifests[0]["archive_sha256"]==manifests[1]["archive_sha256"]
    assert manifests[0]["package_identity"]!=manifests[1]["package_identity"]
    # Scientific execution owners were not moved, rewritten or silently replaced.
    owners=["src/molgap/family_training_"+name+".py" for name in
            ("core","data","recipe","loaders","kernel","session","runner","outputs","state")]
    owners += ["src/molgap/family_model_ops.py","src/molgap/qm9_neural_atom.py","src/molgap/gptrans.py"]
    before=committed_payloads(ROOT,"11790547",owners)
    after=committed_payloads(ROOT,manifests[-1]["source_commit"],owners)
    assert before==after
    extracted=tmp_path/"isolated";extracted.mkdir()
    with tarfile.open(tmp_path/"warm/source.tar.gz") as archive:
        archive.extractall(extracted,filter="data")  # verified regular source members only
    driver=tmp_path/"driver";driver.mkdir()
    # Reuse existing tests instead of writing a second training/resume loop.
    files=["test_family_training_core.py","test_family_training_session.py","test_family_training_state.py"]
    for name in files:(driver/name).write_bytes((ROOT/"tests"/name).read_bytes())
    program=r'''
import json, pathlib, sys
source=pathlib.Path(sys.argv[1]);driver=pathlib.Path(sys.argv[2])
sys.path.insert(0,str(source))
import molgap, pytest
from molgap.experiment_execution import build_addon_declaration
from molgap.fast_sandbox import run_family_sandbox
results=[]
for family,prefix,version in (("neural_atom_k1","k1","3"),("gptrans_t","gptrans","2")):
    addon=build_addon_declaration((family,version),prefix+"_readout_scale",repo_root=source.parent)
    result=run_family_sandbox(family,addons=(addon,))
    assert result["status"]=="PASSED",result
    assert not result["frozen_initialization_loaded"]
    results.append({"family":family,"status":result["status"],"timing":result["timing"]})
code=pytest.main(["-q","--confcutdir",str(driver),str(driver),"--basetemp",str(driver/"scratch"),
                  "--junitxml",str(driver/"cpu-state.xml"),"--tb=short"])
assert code==0,code
for name,module in list(sys.modules.items()):
    if name=="molgap" or name.startswith("molgap."):
        file=getattr(module,"__file__",None)
        if file is not None:assert pathlib.Path(file).is_relative_to(source),(name,file)
print(json.dumps({"molgap_module":molgap.__file__,"sandbox":results,"native_qualification":"NOT_EVALUATED"}))
'''
    environment={**os.environ,"PYTHONPATH":str(ROOT/"src"),"CUDA_VISIBLE_DEVICES":"-1",
                 "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
    started=perf_counter()
    result=subprocess.run([sys.executable,"-I","-c",program,str(extracted/"src"),str(driver)],
        cwd=driver,env=environment,capture_output=True,text=True,timeout=180)
    measurements["isolated_cpu_seconds"]=perf_counter()-started
    assert result.returncode==0,result.stdout+result.stderr
    summary=json.loads(result.stdout.splitlines()[-1])
    assert [r["family"] for r in summary["sandbox"]]==["neural_atom_k1","gptrans_t"]
    evidence={"source_commit":manifests[-1]["source_commit"],"source_files":len(names),
        "archive_bytes":(tmp_path/"warm/source.tar.gz").stat().st_size,"timings":measurements,
        "package_identities":[m["package_identity"] for m in manifests],"child":summary}
    (tmp_path/"distribution_observation.json").write_text(json.dumps(evidence,indent=2)+"\n")
    record_property("distribution_observation",json.dumps(evidence))
