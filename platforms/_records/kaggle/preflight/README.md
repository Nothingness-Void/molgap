# Kaggle accelerator preflight evidence

## Kaggle3 TPU v5e-8

Account `nvoid912` was tested without scientific training or protected roles.
The official Kaggle CLI stored `enable_tpu=true` and
`machine_shape=TpuV5E8`, but actual batch execution did not allocate a TPU.

| Probe | Kernel form | Actual JAX device | Result |
|---|---|---|---|
| `molgap_v5_tpu_kaggle3_v1` | script, legacy CLI | `TFRT_CPU_0` | rejected |
| `molgap_v5_tpu_kaggle3_v2` | script, classic REST | `TFRT_CPU_0` | rejected |
| `molgap_v5_tpu_kaggle3_v3` | script, official CLI | `TFRT_CPU_0` | rejected |
| `molgap_v5_tpu_notebook_kaggle3_v1` | notebook, official CLI | `TFRT_CPU_0` | rejected |

The accepted mounts in the full probe prove the Kaggle3 V5 runtime and fixed
100K dataset are available. TPU quota remained unused. Direct batch TPU use is
closed until a web-created interactive session exposes a TPU and separately
passes a PyTorch/XLA/PyG operator test. Remote metadata alone is not accepted
as hardware evidence.
