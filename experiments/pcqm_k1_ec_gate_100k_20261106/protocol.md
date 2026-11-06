# Early directed expansion guidance: prospective100K contract

User authorized Kaggle3 submissions and autonomous implementation on2026-11-06.
This owner starts from committed desktop da1fbb4c. Two actual fresh-training
arms test topology guidance,without SOAP or changes to existing jobs.

## Evidence and question

RML local/server lookup found no SOAP or exact expansion-contribution experiment.
Adjacent negatives remain binding context: SSMA+2.067meV at+48.98percent measured
step cost; extremal graph readout+3.333meV; ordered hop-shell below material gate;
degree-balance scale reversal; current-layer linear-attention replacement's late
regression. See the original decisions in pcqm_k1_local_mixing_clean_aux,
pcqm_k1_extremal_readout_100k,pcqm_k1_hop_shell_100k,and server
experiments/pcqm_k1_linear_attention_100k/decision.md. Running message-statistics
and history-read are different frozen questions,not accepted positive evidence.
Accepted original K1/Muon100K0.132520668eV is the optimization context;500K is
still unresolved. No evidence asserts that K1 has diagnosed over-squashing.

EC-Gate final paper:
https://openreview.net/pdf/7f71cc869930dfaec7208371c6d0d7d2775ef9e6.pdf.
Its Lipo/AqSol/PCBA results are not PCQM results. Section4.2 warns of compounded
gating; its full method is not reproduced here. The public-paper definition,
not an unverified author-code claim,is the mechanism authority. The displayed
Algorithm1 does not explicitly initialize C(1); use Definition1's cumulative
within-hop fields,including self,implemented by CPU integer bitsets.

Question: does directed early expansion coverage improve terminal Gap accuracy
beyond a matched source-degree-guided residual message gate? An improvement
over degree does not by itself prove general sufficiency or paper reproduction.

## Intervention and controls

For real directed bond u→v,define c_l=|R_(l-1)(u) intersect
(R_l(v) minus R_(l-1)(v))|. Compute c2,c3 from original bond topology on CPU.
Batch graphs remain disjoint; no geometry,new edges or protected data.
`degree_gate` uses log1p(max(source_degree-1,0)); `expansion_gate` uses
log1p(c_l). Only local layers2,3 receive the new path,using their corresponding
c2/c3. Avoid deep saturated contributions rather than silently reusing historical
statistics at every layer. Both arms compute identical CPU statistics for fair cost.

Preserve original gated message m exactly as the direct path. Add
0.1*tanh(MLP(feature)),8 groups,applied to a learned Linear192→192(m):
returned message=m+group_gate*projection(m). This is an EC-informed bounded
residual modulation,not the paper's0..1 replacement gate. Projection starts at
identity; MLP is1→16→8 with SiLU; final weight starts at1e-3 times its normal
Torch initialization and final bias zero. Identical seed42 RNG isolation in both
arms. Both add74064 learned parameters total. Original edge-state updates,BN,
local residuals,slot mixers and readout stay unchanged. No whole-model baseline
retraining. No claim of function-identical initial outputs between distinct guides.

Existing substrate PRE_MESSAGE and PyG message hooks carry the helper; one CPU
topology calculation per forward is reused at both layers,cleared after layer3.
There is no persistent learned/cache state across batches. Only the scientific
addon module/catalog and experiment records are new; reuse all training,
checkpoint,source,receipt,output,role/cost,acceptance and RML owners.

## Cheapest falsifier before training

After standard prospective publication,inspect1024 evenly spaced training graphs
on CPU through load_training_data with development=False. Do not inspect labels
or development quality. The diagnostic reuses the exact addon edge_statistics.
Check hand-derived chain/triangle examples to distinguish cumulative reachability
from exact-hop walk counts; no new test framework. Record actual CPU wall/process
time and topology timing for8batches of128; GPU time is not applicable.

Release only if at least20percent of sampled graphs contain a receiver with
nonconstant incoming(c2,c3),at least3 observed(source_degree,target_degree)groups
with>=20sample edges contain multiple EC values,and median batch topology
time<=0.05s. These are prospective informativeness/cost screens,not accuracy or
early-stop claims. If they fail,record NO_TRAIN/STOP_FOR_COST rather than fill a
slot. Do not tune layers or thresholds after observing this diagnostic.

## Recipe,budget and terminal gate

Frozen original K1 random initialization,pure2D OGB atom9/bond3/RWSE16,
100K train/50K internal development,40passes,31240optimizer calls,
3998720presentations,BS128/drop_last,single supervised forward,seed42,
FP32/noTF32,native Torch2.11,k1-muon-adamw-fp32-v1,LR4e-4/WD1e-5/clip1,
cosine40→1e-6,no warmup,best-development-Live selection,EMA.999 tracking.
Keep inherited checkpoint/resume,short profiler and epoch-only logging.
Official validation,test-dev and test-challenge remain unauthorized.
One worker per T4,one T4x2 session,18000wall-second/36000T4-device-second cap,
180s cleanup reserve. Expect3.5–4.5h from retained Muon runtime plus headroom;
this is an estimate,not measured T4 EC performance. Luna owns status checks.

Primary paired gate: expansion-minus-degree selected MAE<=-0.003eV,
paired2000-bootstrap95percent upper<0,negative final and last-five deltas.
Both models must stay below2x original parameter count. Compare complete finite
aligned50000-row artifacts,full trace/exposure,native repeat/resume,actual
cost and observed roles. Each arm must independently reach complete replay
with no exclusions. Cross-run Muon comparisons need their own actual runtime
qualification; observed environment drift stays visible rather than fabricated
as matched. Endpoint,row intervals and strict causality remain separate.

Failure to outperform the degree control closes this exact EC recipe. If both
trail existing Muon,do not promote a relative win. Derive fit/exposure attribution
only from comparable evidence; otherwise insufficient_evidence. No automatic
successor,new seed,500K/full or protected-role release. Write attribution,then
use the existing per-arm terminal/RML closure and one derived-index rebuild.
