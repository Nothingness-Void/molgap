# Frozen normalization mechanism protocol

User approved this four-arm design and Kaggle1 submission on2026-10-22 JST.
Source/initialization/input and role pins in the generated Spec are authoritative.
A=post_node_rms; B=pre_node_rms; C=post_graph_rms; D=post_node_ada.
Slot1 trains A/B; slot2 trains C/D. One arm per T4, no DDP or changed batch size.

## Scientific interventions

Replace exactly18 local norm1/norm3 sites in9 K1 ResGatedGraphConv blocks.
Retain mixers/attention LayerNorm, edge updates, pooling/head and all shared weights.
Post: h=N1(x+Dropout(Local(x))); y=N3(h+MLP(h)).
Pre: h=x+Dropout(Local(N1(x))); y=h+MLP(N3(h)). No added final norm.
Node RMS: each node over its192 feature channels, uncentered, epsilon1e-5.
Graph RMS: each graph/channel over its nodes, uncentered, no running statistics.
This is an axis probe, not full GraphNorm with learned centering.
Ada RMS: common outer affine plus rms(alpha*x+beta), alpha0/beta1, same epsilon.
The initial adaptive multiplier is sqrt(1+eps), approximately5ppm above unity;
alpha1/beta0 can recover amplitude. Extra learned parameters6912 are disclosed.
Changing scope intrinsically changes which relative amplitudes are preserved;
these are conditional architecture contrasts, not independent universal factors.

## Common training

Fixed100K train[0,100000), next50K development[100000,150000), pure2D.
Full transported seed42 state, identical epoch shuffle,128/drop-last/no accumulation.
FP32/noTF32, single-forward normalized Gap L1, AdamW2e-4, WD1e-5, clip1,
betas(.9,.999),eps1e-8, dropout.05, warmup8 then cosine160 to1e-6.
160passes,124960 updates,15994880 gradient sample presentations per arm.
Track EMA.999 but select best-development Live only; no Clean-BN or teacher.
No early stopping inferred from partial curves, no500K/full or production change.

## Attribution and decisions

Predeclare B-A placement (same job), C-A statistics axis and D-A adaptive gain
(cross job, exploratory). D-C is only a same-job package contrast and does not
isolate either axis or amplitude. The four cells do not estimate interactions.
Report all contrasts, selected endpoints plus matched40/80/120/160 endpoints;
row bootstrap is conditional uncertainty, never training-seed variance.
A material point gain>=3meV and row95% interval below0 only nominates further
review, not automatic adoption/scale-up. Smaller effects remain reportable.
No universal BN/LN ranking or causal claim from one endpoint.

Retain preclip gradient norms and clipping frequency for the first1000 formal
updates in100-step aggregates, no added batch logging. At40/80/120/160 evaluate
fixed first4096 training members using the exact Live eval state and target
transform, plus layer magnitude/residual/Ada-gain observations. This cohort is
not representative of all training; diagnostic inference exposures are separate
from gradient sample presentations. Preserve state/RNG/modes and exact probe
identity; no label outside the existing train/development roles is consumed.
These observations can test new-run instability, not reconstruct past gradients.

## Resources and durability

Each job ceiling28800 entry-wall seconds including setup/diagnostics/native
qualification,180seconds cleanup reserve,16 allocated native T4-device hours;
two jobs total32 ceiling. Expected total24-30 native T4-device hours is estimated,
not measured. Account GPU-quota units are a separate platform observation.
All-arm native repeat/serialized-resume preflight precedes any formal training.
Use shared atomic best/last/resume state and bounded two retained10-pass chunks.
Freeze explicit epoch logging through standard TRAIN_LOG, no custom metric printer.

Per arm preserve runtime certificate,160 observations including actual LR and
checkpoint SHA, selected model, aligned finite50K Live predictions, full last
resume state, observed costs/roles and independent terminal RML. Frozen trace
manifest semantics must survive to terminal; strict comparison/readiness remains
an independent fail-closed gate. Same-run declarations are not historical strict
reference bundles; absent qualification stays excluded rather than invented.
Only acceptance-required small metadata/model/prediction/resume/diagnostic files
may be retrieved to D: using exact-version allowlists. No whole output/cache or
all epoch chunks. No new monitor, retry, successor or server handoff.
After both actual submissions are reconciled and startup evidence retained,
push the reviewed owner branch, then shut down as user requested.
