"""Print the schema of one accepted fixed-dataset shard without model work."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    args = parser.parse_args()
    data, slices = torch.load(args.shard, map_location="cpu")
    print(type(data).__name__)
    print(data)
    print(sorted(data.keys()))
    print(
        {
            name: {
                "dtype": str(getattr(data, name).dtype),
                "shape": list(getattr(data, name).shape),
            }
            for name in data.keys()
        }
    )
    print({name: value[:5].tolist() for name, value in slices.items()})


if __name__ == "__main__":
    main()
