import numpy as np
import pandas as pd

from molgap.hierarchical_external_eval import _aligned_table


def test_aligned_table_restores_reference_order():
    reference = pd.DataFrame(
        {
            "eval_set": ["ood1000", "p8_targeted_hard"],
            "cid": [10, 20],
            "smiles": ["CC", "CO"],
            "homo": [-5.0, -6.0],
            "lumo": [-1.0, -2.0],
            "gap": [4.0, 4.0],
        }
    )
    candidate = reference.iloc[::-1].reset_index(drop=True)
    aligned = _aligned_table(reference, candidate, "candidate")
    assert np.array_equal(aligned.cid.to_numpy(), np.array([10, 20]))
