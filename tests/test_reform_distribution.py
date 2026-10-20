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
from test_experiment_spec import payload

ROOT=Path(__file__).resolve().parents[1]


def test_committed_distribution_imports_actual_families_and_resumes_without_host_source(tmp_path,payload,record_property):
    # Freeze this migration's active source in a separate fixture repository.
    # The shared checkout may have uncommitted migration edits during validation.
    repository=tmp_path/"fixture-source"
    for source in (ROOT/"src/molgap").rglob("*.py"):
        if "archive" in source.relative_to(ROOT/"src/molgap").parts:
            continue
        target=repository/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes().replace(b"\r\n",b"\n"))
    entry=repository/"platforms/kaggle/run_experiment.py"
    entry.parent.mkdir(parents=True,exist_ok=True)
    entry.write_bytes((ROOT/"platforms/kaggle/run_experiment.py").read_bytes().replace(b"\r\n",b"\n"))
    for arguments in (("init",),("config","user.email","fixture@example.invalid"),
                      ("config","user.name","Distribution fixture"),("config","core.autocrlf","false"),
                      ("add","."),("-c","commit.gpgsign=false","commit","-m","Frozen migration source")):
        subprocess.run(["git",*arguments],cwd=repository,check=True,capture_output=True)
    names=runtime_source_files(repository)
    cache=tmp_path/"source-cache"
    measurements={}
    manifests=[]
    for label in ("cold","warm"):
        value=deepcopy(payload);value["logical_run_id"]="distribution-"+label
        started=perf_counter()
        target=tmp_path/label
        manifests.append(build_experiment_source_package(ExperimentSpec(value),repository,names,target,source_cache=cache))
        measurements[label+"_package_seconds"]=perf_counter()-started
        assert verify_experiment_source_package(target,repository)==manifests[-1]
    assert manifests[0]["archive_sha256"]==manifests[1]["archive_sha256"]
    assert manifests[0]["package_identity"]!=manifests[1]["package_identity"]
    # The one-time pre-reform byte comparison is retained in migration evidence.
    # Regular tests check this distribution's integrity and runtime behavior,
    # not a permanent prohibition on repairing active scientific owners.
    extracted=tmp_path/"isolated";extracted.mkdir()
    from molgap.experiment_preflight import _unpack
    _unpack(tmp_path/"warm", extracted)
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
