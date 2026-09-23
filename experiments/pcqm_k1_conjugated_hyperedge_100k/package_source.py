"""Package committed source through the existing hash-inventory adapter."""
from __future__ import annotations

from experiments.pcqm_k1_sparse_triplet_100k import package_source as packager


def main():
    packager.EXPERIMENT = "experiments/pcqm_k1_conjugated_hyperedge_100k"
    packager.main()


if __name__ == "__main__":
    main()
