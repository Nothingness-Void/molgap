"""New front doors compose old preparation and journal state machines."""
import json
from unittest.mock import Mock

import pytest

from molgap import kaggle_submission as owner
from test_experiment_workflow import (
    workflow_case,
    _git,
    test_prepare_workflow_packages_then_releases_then_plans_synthetic_candidates as prepare_fixture,
)
from test_kaggle_submission import release
from test_reform_input import platform_entry


def test_fixed_distribution_uses_original_complete_preparation_fixture(workflow_case,tmp_path,monkeypatch):
    # The owning fixture exercises real packaging/staging/launch/planning order.
    # Its synthetic scientific checks remain explicitly stubbed, not qualified.
    plan=workflow_case["plan"];repo=workflow_case["repo"]
    del plan["source_files"]
    plan.update(source_profile="runtime-v1",source_records=[])
    records=[plan["acceptance_plan"],*(b["plan_spec_ref"] for b in workflow_case["spec"].to_dict()["prospective"]["arms"])]
    _git(repo,"add","--",*records)
    _git(repo,"-c","commit.gpgsign=false","commit","-m","Frozen synthetic record inputs")
    monkeypatch.setattr("molgap.source_dependencies.discover_source_dependencies",
                        Mock(side_effect=AssertionError("AST in default preparation path")))
    prepare_fixture(workflow_case,tmp_path,monkeypatch)
    manifest=json.loads((tmp_path/"prepared/package/package_manifest.json").read_bytes())
    assert set(records)<=set(manifest["relative_allowlist"])


@pytest.mark.parametrize("lost_response",[False,True])
def test_editable_platform_entry_uses_same_journal_without_duplicate_post(
        release,platform_entry,monkeypatch,lost_response,capsys):
    monkeypatch.setattr("molgap.experiment_input.prepare_experiment_input",Mock(return_value={
        "status":"PREPARED_FOR_PLATFORM","prepared":str(release.prepared),"operations":str(release.output),
        "authority_ref":"authority.json","monitor_paths":["experiments/example/STATUS.md"]}))
    original=owner.submit_prepared_workflow
    monkeypatch.setattr(platform_entry,"submit_prepared_workflow",
        lambda *a,**kw:original(*a,**kw,api=release.api,http_session=release.http))
    if lost_response:
        def lost(**kwargs):
            release.api.pushed+=1;release.accepted()
            return {"status":"submission_unknown","reconciliation_required":True}
        monkeypatch.setattr(owner,"push_kernel_with_accelerator",lost)
    command=["--experiment","fixture.json","--credentials",str(release.credentials),"--execute"]
    assert platform_entry.main(command)==(1 if lost_response else 0)
    first=json.loads(capsys.readouterr().out)
    assert first["status"]==("RECONCILIATION_REQUIRED" if lost_response else "SUBMISSION_BOUND")
    assert platform_entry.main(command)==0
    assert json.loads(capsys.readouterr().out)["status"]=="SUBMISSION_BOUND"
    assert release.api.pushed==release.api.created==1
