# Prepared overnight handoff - 2026-10-06

PREPARED_FOR_PLATFORM; remote submission remains root/platform owner.
Kernel nvoid912/molgap-k1-spectral-gaussian-100k-s42-v1.
Private source nvoid912/molgap-k1-spectral-source-s42-v1.
Fixed input nvoid912/pcqm4mv2-ogb-fixed-100k-v1.
Pair reference GPU0 / spectral GPU1;40epochs/100K/seed42/FP32/BS128.

Frozen executable source b1000d3ff20e3bde00bcc1828d825899e1893ab3.
Spec76883e0acbb7ef62059e608a80f1bbf5de1588a5720d1ed152fdc8b3d7f3e105.
Package7e486762cd5a6ae366966b3c70df4a93d62ed97ec09d01c6e84afe7bcebe136b.
Archive13b22dfc0e71a86f64f9dc5879bc97bc4866f05eda334d9526ae3c869d96a544.
Prepared local root platforms/_records/local/k1_spectral_100k_night_20261006/release_v1/
contains source_dataset/, kernel/, release_report.json and workflow_report.json.

CPU prospective published first; full EVD completed on CPU,24.06s wall/23.39s
process.150K graphs,maximum18nodes,cache159MBtensor; all finite and orthonormal
maxerror1.03e-7. Exact cache and manifest pins are in Spec addon config; shared
stage_input_artifacts copies them to private source_dataset/spectral_cache/.
No GPU-side full-cache construction. Addon calculated25,088parameter increment,
3,683,905total; actual construction budget checked in assigned-T4 preflight.

Reused K1loader/trainer/resume/FamilyOutputSession,execution registry,source
inventory,Kagglepair barrier,workflowpackaging/prospective/release and RML.
Only mechanism,CPUcache datasetadapter,mode/config and inputstaging hooks new.
Remote qualifier requires nonzero2-step addon gradients,sign/repeatedspace/node
invariance,repeatability,resume/selectedreload,<=25%step overhead and<=12GiBpeak.
No local formal training/model forward or tests were performed.

Preparation published both prospective records then stopped on absent historical
ignoredbytes during RMLrebuild. Exactly two known main-checkout blobs restored,
unchangedhash validators passed; see local_artifact_custody_repair.json.
The same existing source/package/prospects were finalized using shared freeze_inputs,
bind_release,validate_staged_trajectory and build_launch_receipt; no replan.
RMLrebuild/check--frozen passed. Remote state is UNKNOWN until root reconciles
returned exact kernel/version and startup qualification. No promotion/scale-up.
