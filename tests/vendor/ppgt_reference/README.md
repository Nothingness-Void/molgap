# PPGT author reference slices

Source: [LiamMa/PPGT, commit 62cafd3abbb8d72e9a8b3146f947a192cf0aacaf](https://github.com/LiamMa/PPGT/tree/62cafd3abbb8d72e9a8b3146f947a192cf0aacaf).
The MIT notice is retained in `LICENSE`. `manifest.json` binds original source
hashes, inclusive line ranges and retained slice hashes. Files are raw author
text, not a rewritten oracle. Retrieval used public pinned raw HTTP only.

`ppgt_author_reference_harness.py` executes selected raw classes/functions after
removing imports and registration decorators in memory. It supplies PyTorch/PyG
and small layout/registry adapters, without installing the upstream project,
importing its trainer, reading a dataset or using MolGap forward/feature code.
The complete-graph padding adapter scatters author RRWP into all ordered pairs.
Its einops adapter only reshapes/transposes; einsum delegates to torch.einsum.
The optional timm class-token path is rejected. Embedding tables use the fixed
OGB atom9/bond3 category sizes and sum semantics; this is the explicitly bounded
MolGap categorical front end, not a pinned upstream dependency/environment.

## Source-compatible scope

Only the released PCQM pre-norm AdaRMSNorm/GELU branch is ported. Attention uses
separate full-width Q, KV, O and a joint per-head positional map P. QK is
unscaled; the negative half key norm is divided by sqrt(head width). P's first
half is additive theta and its second half plus one multiplies the softmax.
The gate is unconstrained and is not renormalized. There is no softmax clamp.
Norm epsilon is 1e-6. Both residuals have independent learned channel scales
initialized to 1e-4. Each block has two pre-norms; the backbone also has pre/post
norms. Source drop-path is uniform over depth, graph-wise, with keep scaling.

RRWP includes identity plus powers 1 through ksteps (eight steps means nine
channels). The PCQM YAML uses symmetric normalization and an undirected union
if adjacency is asymmetric. Parallel edges sum before normalization. Isolates
have zero nonidentity powers. Node stems add diagonal RRWP and log1p(N), with
separate biased linear projections. Pair stems append inverse N only, put all
raw channels first, then feature/frequency/(sin,cos), add encoded bonds, and
apply pre-norm, residual pre-norm FFNs and post-norm. Attention reads pair
features as query=target/key=source. Backbone/stem/head linears use truncated
normal std=.02 and zero biases; node projections and pair MLP output use
Kaiming uniform a=1, and the hidden pair projection uses a=0. OGB embedding
tables use Xavier uniform. Exact full-constructor RNG/bitwise initialization
parity is not claimed. The three-layer linear_graph head keeps hidden width,
with GELU between linears, after the declared sum/mean pooling.

## MolGap adaptation scope and limits

`src/molgap/ppgt_author.py` is a new version, independent of frozen v1 math.
It accepts validated device-local topology and produces normalized Gap [B].
Trainer-owned normalization is neither applied nor inferred here. It processes
each graph independently, allocates no full-corpus RRWP, never reads y/RWSE,
does not mutate the batch and holds no persistent per-batch feature cache.
Training dropout has the same graph/branch distribution, but graph-wise execution
does not promise the author's batched RNG draw order. Memory caps reject an
entire over-limit request; no molecule is dropped or reparsed. CUDA allocation
is implemented code only; runtime/AMP/TF32/memory/BS128/native qualification,
real graph acceptance, source-package release and scientific training remain
outside these tests.

Default caps are 256 nodes/1024 directed edges per graph; configurable bounds
cannot exceed 1024 nodes/4096 directed edges. These are rejection limits, not
measured guarantees of accelerator memory feasibility. The pinned PCQM branch
requires use_phi=True; the source's broken non-URPE self-attention layout is not
silently repaired or included in the parity scope. Norm epsilon overrides are
constructor adaptations; the raw PCQM oracle checks only its released 1e-6.

Default budget structure is 16x256, eight heads, pair width 128, pair MLP width
512, eight stem blocks, FFN ratios two, 24 walk steps (25 channels), ten Fourier
bases and sum pooling. These are a disclosed MolGap architecture adaptation,
not the author's small model recipe. `PPGTAuthorConfig.author_pcqm_base()` maps
the sole released PCQM YAML: 12x768, 12 heads, pair128, MLP512, two stem blocks,
ratios four, eight walk steps, five bases, mean pooling, attention dropout .1,
drop-path .2. It does not copy the trainer's batch size, optimizer or schedule.
No PCQM small YAML was released at this pin. Table 9's 16x512/ratio2/17.6M
cannot all describe independent dense blocks: backbone weight matrices alone
contain 33,554,432 parameters. No sharing/removal/count-fitting is introduced.

The authored tests are bounded synthetic CPU fixed-weight forward/gradient
oracles, including padded multi-graph author execution and feature orientation.
They must be executed by the main agent's Luna leaf worker before reporting
scoped numerical parity. They do not establish benchmark reproduction, MAE,
real-data support, GPU qualification, optimizer fidelity or training authority.
