"""Pinned optional profiling and the reviewed local-bond component."""
from dataclasses import asdict
import json
import pytest
from molgap.family_execution_profile import ProfileConfig
from molgap.family_training_runner import _recipe_configuration
from test_family_training_session import make_session, events


def test_profiler_configuration_is_frozen_and_rejects_unknown_fields():
    config = ProfileConfig.from_dict({'warmup_steps':120, 'active_steps':16, 'record_shapes':False, 'module_paths':['blocks.0']})
    assert config.module_paths == ('blocks.0',)
    assert ProfileConfig.from_dict(json.loads(json.dumps(asdict(config)))) == config
    with pytest.raises(ValueError):
        ProfileConfig.from_dict({'unbounded':True})
    session = make_session('gptrans_t',development=False,schedule_kind='cosine_step',events=events())
    try:
        record = session.recipe.to_dict()
        assert _recipe_configuration(record) == session.recipe
        record['profiling'] = asdict(config)
        record['profiling']['module_paths'] = list(config.module_paths)
        assert _recipe_configuration(record) == session.recipe
        record['profiling']['active_steps'] = 0
        with pytest.raises(ValueError):
            _recipe_configuration(record)
    finally:
        session.close()


@pytest.mark.parametrize('corruption',['missing','device','recipe','trace','report','path','updates'])
def test_native_profile_binding_rejects_rehashed_wrong_evidence(tmp_path,corruption):
    from types import SimpleNamespace
    from molgap.family_training_runner import _verify_profile_proof
    from molgap.training_reproducibility import atomic_json,sha256_file
    folder=tmp_path/'profile_qualification'
    folder.mkdir()
    atomic_json(folder/'trace.json',{'synthetic_fixture':True})
    report={'status':'COMPLETE','device_events_observed':True,'captured_updates':2,
        'recipe_identity':'a'*64,'trace_sha256':sha256_file(folder/'trace.json')}
    atomic_json(folder/'profile.json',report)
    proof={'report':'profile_qualification/profile.json','sha256':sha256_file(folder/'profile.json')}
    probe=SimpleNamespace(identity='a'*64)
    _verify_profile_proof(tmp_path,probe,proof,required=True)
    if corruption=='missing': proof=None
    elif corruption=='device': report['device_events_observed']=False
    elif corruption=='recipe': report['recipe_identity']='b'*64
    elif corruption=='trace': atomic_json(folder/'trace.json',{'changed_fixture':True})
    elif corruption=='report': proof['sha256']='c'*64
    elif corruption=='path': proof['report']='../profile.json'
    else: report['captured_updates']=1
    if corruption in ('device','recipe','updates'):
        atomic_json(folder/'profile.json',report)
        proof['sha256']=sha256_file(folder/'profile.json')
    with pytest.raises(ValueError):
        _verify_profile_proof(tmp_path,probe,proof,required=True)
