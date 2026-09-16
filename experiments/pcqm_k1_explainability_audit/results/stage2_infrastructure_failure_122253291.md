# Stage-2 infrastructure failure 122253291

Job `122253291` reached the accepted DCU runtime and loaded the frozen model,
then failed after 2 minutes 16 seconds before producing an attempt directory.
The diagnostic descriptor path called CUDA `torch.bincount` while PyTorch
deterministic algorithms were enabled:

```text
RuntimeError: _bincount_cuda does not have a deterministic implementation
```

This is an instrumentation failure, not a model or scientific result. No
training ran and no counterfactual metrics were written. The unchanged retry
keeps model inference on the DCU but transfers descriptor and diagnostic
reductions to CPU, where deterministic `bincount` and `index_add_` are
available. Data, checkpoint, weights, ablations, strata, FP32 mode, and batch
size remain unchanged.
