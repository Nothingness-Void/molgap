# K1 target recovery, attempt 1 reconciliation

## Actual run

- Kaggle returned `/code/nothingnessvoid/molgap-k1-target-proof-500k-20261025`, version `1`, kernel ID `140726588`. The exact status poll ended `ERROR`.
- The published source is the submitted `entry.py`: its CRLF-normalized bytes equal the frozen local file and its normalized SHA-256 is `170759b37a81f3ee16562e96e9e04b46efaac0ec1611f5b1316b6db4aeeaf0b9`. The runtime entry receipt reports `170759b37a81f3ee16562e96e9e04b46efaac0ec1611f5b1316b6db4aeeaf0b9` and worker `094ae2ae2c954348b064e160b36ffe09cc8753aea52f4dfb481b6c162e8fea6d`, matching the frozen package.
- Pulled metadata identifies kernel ID `140726588`, private, CPU only (`enable_gpu=false`, `enable_tpu=false`). Its dataset source list contains the two expected dataset slugs but omits version suffixes. The pre-POST metadata file hash and requested refs are in `request_frozen.json`; the remote exact version listings are in `preflight_exact.json`.

## Failure and role evidence

- Worker observation: stage `train_label_read_started`, state `reading_unknown_subset`, decoded membership `unknown`. Exact error: `ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject`.
- The saved traceback is [diagnostic_abi_traceback_minimal.txt](diagnostic_abi_traceback_minimal.txt). It fails while importing `torch_geometric`, through `torch._dynamo.utils` access to `np.random`, at `numpy.random.mtrand` ABI initialization.
- NumPy was loaded before `pip install numpy<2`: worker.py line 97 imports `molgap.experiment_package`; experiment_package.py line 19 imports v4_bundle; v4_bundle.py line 13 imports training_reproducibility; training_reproducibility.py line 19 unconditionally imports NumPy. The install runs later at worker.py line 103. The failure occurs still later in `load_training_data`.
- The likely mechanism is a same-process import followed by an environment mutation: pip may replace NumPy on disk while the original module remains loaded. The retained log contains no exact before/after NumPy versions, so the mechanism is strongly consistent, not fully version-proven.

## Verified input content and limits

The worker reaches `train_label_read_started` only after checking the mounted source payload digest and source package identity, the original recipe's data manifest SHA `630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`, exact manifest resolution, and exactly ten TRAIN shards covering 500,000 rows. Those guards therefore passed before the import failure. The remote preflight listing verified requested source version 1 and data version 3; the submitted metadata used those exact refs. The server-pulled metadata snapshot strips numeric suffixes, so it is not an independent numeric-version receipt.

No completion manifest, labels, indices, or copied data manifest was produced. Only `entry_receipt.json` and `worker_observation.json` were retrieved with exact-version metadata calls; the diagnostic stream was read directly and only the ABI traceback excerpt was retained. No `kernels_output` bulk retrieval was used.

- Wall: 257.768537s, 2026-10-25T06:26:42.752193+00:00 to 2026-10-25T06:31:00.520705+00:00.
- Entry reports process-tree CPU 784.43983s; CPU-hours remains `measurement_missing` because the run failed. Queue is `measurement_missing`; GPU is `not_applicable`.
- Membership stays unknown. This failed runtime proof is not scientific acceptance or a successful target recovery.
