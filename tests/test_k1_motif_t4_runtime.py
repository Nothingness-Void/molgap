"""The single-candidate GPU selection is independent of Kaggle's T4 count."""
import pytest

from molgap.k1_motif_study_runtime import select_t4_device


@pytest.mark.parametrize("names", [["Tesla T4"], ["Tesla T4", "Tesla T4"]])
def test_accepts_one_or_two_t4_devices(names):
    assert select_t4_device(names) == 0


@pytest.mark.parametrize("names", [[], ["Tesla P100"], ["Tesla T4", "Tesla P100"], ["Tesla T4"] * 3])
def test_rejects_unqualified_allocations(names):
    with pytest.raises(RuntimeError, match="Expected one or two T4"):
        select_t4_device(names)
