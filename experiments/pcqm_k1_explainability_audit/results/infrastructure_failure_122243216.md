# Stage-1 infrastructure failure 122243216

Job `122243216` terminated after 12 seconds before loading the fixed graph
cache or any prediction payload. Importing the accepted DTK PyTorch wheel on a
Kunshan CPU-only node failed with:

```text
ImportError: libhsakmt.so.1: cannot open shared object file
```

The runtime had previously passed on `kshdtest`, where the vendor HSA runtime
is exposed with a DCU allocation. This is an infrastructure-only failure: no
model was constructed, no training ran, no metrics were produced, and no
scientific contract changed. The retry uses the same source, graph cache,
payload set, and analysis code on one `dcu:Hygon` allocation.
