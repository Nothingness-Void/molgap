import math
from molgap.pcqm_500k_v4_evidence import scientific_contract, schedule


def test_exposure_excludes_every_tail_batch():
    contract = scientific_contract()
    assert contract['sample_exposure'] == 29998080
    assert contract['steps_per_epoch'] == 3906
    assert contract['sample_exposure'] % 128 == 0
    assert 500000 - contract['steps_per_epoch'] * 128 == 32


def test_schedule_survives_stage_boundaries():
    whole = [schedule(epoch) for epoch in range(60)]
    staged = [schedule(epoch) for start in range(0, 60, 4) for epoch in range(start, start+4)]
    assert whole == staged
    assert math.isclose(whole[0], 4e-4)
    assert math.isclose(whole[-1], 1e-6)
    assert all(a >= b for a, b in zip(whole, whole[1:]))
