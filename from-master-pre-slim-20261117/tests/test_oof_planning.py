import pandas as pd

from molgap.oof_planning import assign_scaffold_folds


def test_scaffold_folds_are_disjoint_and_balanced():
    scaffolds = pd.Series(["a"] * 4 + ["b"] * 3 + ["c"] * 2 + ["d"])
    folds = assign_scaffold_folds(scaffolds, 2)
    for scaffold in scaffolds.unique():
        assert len(set(folds[scaffolds.eq(scaffold)])) == 1
    counts = pd.Series(folds).value_counts()
    assert abs(int(counts.iloc[0]) - int(counts.iloc[1])) <= 2
