# Split version-1 failure diagnosis

Both isolated seed-42 width kernels received one `Tesla P100-PCIE-16GB`
instead of the requested T4 allocation. Input discovery and deterministic
initialization preflight completed, but the first CUDA forward failed before
epoch 0 because the Kaggle image's bundled PyTorch did not contain an `sm_60`
kernel image.

No candidate completed, no checkpoint was accepted, and no official
validation or test-dev role was read. This was a runtime compatibility failure,
not a model-quality result.

Version 2 keeps the scientific contract unchanged and repairs only the runtime:
on P100 it installs the repository's previously verified PyTorch 2.7.1/cu126
stack, restarts the process, verifies `sm_60`, and only then loads the model.

Ignored raw evidence SHA-256:

- W64 log: `1ed2450791c8a05d011f38532fc1c98c79ec79212fbb90b6e43a20bd52e50b66`
- W64 initialization preflight: `4feddd03fb6396f263fdce02915d2030f19f11ac28db4a9b8b5be04bee09fd39`
- W64 failure record: `6e0502c7fa1124590410e6021589d490167da73c16feff1652e9e9a5f13535a8`
- W128 log: `4c1efeea498427e5b14c5caebee745f686464d63d95345d0fbb202192603fdcf`
- W128 initialization preflight: `fa26c741b0e17bf70faa08d2772a26d5f0c860ff3b882f69fe196c754c685a9d`
- W128 failure record: `6489bf7d4ead44aaefab87544d44c553c87fe9cd08df126bb33ebf4497b4fce6`
