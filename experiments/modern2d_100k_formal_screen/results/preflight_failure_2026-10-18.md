# V1 preflight failure and bounded repair

The exact Kaggle2 kernel139364627/version1 errored before either formal trainer
started. The retained execution report recorded both preflight workers exiting1
and `training_started:false` for both arms. Neither a model-quality result nor a
negative architecture conclusion followed.

Both worker logs reported train standard deviation1.2750902063230574 instead
of the frozen1.275090206323082. The common bootstrap applied its declared CPU
thread count after label reduction. An independent Luna CPU replay reproduced
the frozen value at1 thread and the observed value at2 threads. The repair moves
the existing thread policy before data loading and exports OMP/MKL settings
before worker imports. Frozen statistics, inputs, recipes and strict equality
checks were preserved.

The existing all-arm barrier prevented formal training. Common-Dev and protected
roles were not opened by this attempt. Metadata, logs, receipt, source check and
overlapping allocation ledgers were retained under
`E:/MolGap/artifacts/modern2d-formal-screen/inspection_v1` and the original
immutable run's `operations` directory. The bootstrap observed71.427264596 wall
seconds across2 T4 devices; the nested allocation ledger overlaps that interval
and must not be added to it. Full per-job billing remained unobserved.

The separately approved retry retained the original two scientific arms and
60-pass endpoint. [The budget record](retry_budget.json) charged the entire
settled account increment, rounded up to77 seconds at2 devices, against the
original9 T4-hour aggregate ceiling. Account quota was not reinterpreted as an
exact per-job invoice. This left16123 observed wall seconds for the retry,
including600 seconds reserved for checkpoint retention. No additional retry,
budget, seed, model or protected-role authority was created.

Frozen V1 declarations and initial submission observations were not overwritten.
The retry uses a distinct attempt and source package, not a mutation of the old
prepared directory. The known retained dataset-role manifest format is excluded
from RML access-event discovery; ambiguous and unknown event records still fail
closed. No derived user view was rebuilt.
