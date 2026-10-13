# Input-resolution failure attribution - 2026-10-13 JST

The exact discriminator is shared by both arms: `_artifact_path` rejects
the configured `pcqm4mv2-ogb-fixed-500k-scnet-v1/manifest.json` under the
bootstrap's `/kaggle/input` root. This occurs before manifest validation,
graph deserialization, model construction and formal training. The pair
barrier correctly prevents either arm from training after failed qualification.

The local source-dataset roundtrip and package/import/init checks verified
source transport. They did not verify the actual remote graph mount at the
recipe's assumed root. This is the release coverage gap. Dataset membership
in kernel metadata alone does not prove filesystem layout. A newer nested
mount layout is a possible explanation, but no retained directory inventory
establishes it; missing publication/mount is not excluded. Dependency warnings
are present, but are not the throwing failure.

Underfitting, exposure shortage during training, overfitting, EMA lag, BN
mismatch and second-view benefit are all insufficient_evidence: there are
zero formal updates and no predictions. Do not infer a scientific loss from
this attempt. The cheapest necessary implementation evidence before any
separately authorized retry is an exact mount-location/manifest-hash check
through the platform input resolver, plus a focused synthetic nested/flat
mount test. Preserve source, data, initialization and training semantics;
record any corrected source/package as a new physical attempt. No retry was
submitted during this acceptance.
