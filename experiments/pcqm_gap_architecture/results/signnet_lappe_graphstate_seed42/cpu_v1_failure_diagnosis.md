# SignNet-LapPE CPU cache v1 failure diagnosis

Kaggle2 kernel `kaseichou/molgap-pcqm-signnet-lappe-cache-s42` version 1
terminated before the first cache shard. `torch.load` needed the frozen
`molgap` package while unpickling the accepted geometry graphs, but the kernel
had not yet extracted and inserted the source archive into `sys.path`.

This is an implementation-only packaging failure. The retained `failure.json`
reports `ModuleNotFoundError: No module named 'molgap'`, 12.34 seconds elapsed,
`gpu_used=false`, `model_inference_executed=false`, and both sealed-role flags
false. Its SHA-256 is
`3efb2a231e403b1c01ad09ba6d8f1d25ef3ea4358022cd9c8a9cd382aad5b596`;
the downloaded log SHA-256 is
`c9cb5a6a06d504d758a962cf91d6959201889aad159f683efca75310a4829542`.

The permitted repair extracts the already pinned `src.zip` and inserts its
`src` root before any graph deserialization. It does not change the parent
cache, Laplacian/eigenpair policy, roles, seed, model, or later GPU contract.
