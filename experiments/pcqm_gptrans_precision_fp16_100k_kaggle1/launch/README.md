# Kaggle1 precision-pair launch

The existing Kaggle accelerator adapter submitted one T4x2 script. Its push
response reported no invalid data sources. Kaggle listed the canonical kernel
as `nothingnessvoid/molgap-gptrans-fp32-fp16-precision-100k-s42-v1`; the
subsequent authoritative status was `RUNNING`. The requested metadata slug was
different; [kaggle_observation.json](kaggle_observation.json) preserves both.

[platform_response.json](platform_response.json) binds that observation to the
frozen local source package and both arm identities. The local experiment CLI
reconciled it as `ACCEPTED` in [receipts/](receipts/). This records a launch,
not a scientific result or replay-ready acceptance.
