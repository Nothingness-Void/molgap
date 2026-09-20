# K1 module analysis and selected breakthrough question

## Evidence boundary

This review uses only existing 100K development-role evidence. It does not run
model inference, read official validation/test roles, or reinterpret an Oracle
or label-selected blend as deployable evidence.

The main module findings are:

| Module family | Accepted observation | Disposition |
|---|---|---|
| K1 local persistent-edge backbone | K1 beat the matched Full-GPS 500K reference while using less compute | retain |
| Molecular slot at layers 3/6/9 | Deleting any exchange caused a large regression; layer 9 was most necessary | retain the exchanges |
| Slot self-attention | A length-one slot cannot perform meaningful token selection; deleting it improved K1 directionally | remove |
| Return allocation | Uniform normalized return improved K1 directionally; inverse allocation regressed | use uniform return |
| Scalar slot strength | Every tested non-unit layer-6 coefficient regressed | closed |
| Readout/selector freedom | learned query, multi-depth, tied/untied selectors, and richer readouts failed | closed |
| Pair/relation expansion | PairToken failed 500K transfer; InducedPair, SparsePair-SPSE, recurrent and stateless bridges all failed the 100K gate | closed |
| MoSE structural information | selective zero-initialized MoSE residual improved K1 by 0.002046 eV with a favorable paired interval | retain as the only open structural signal |
| Molecule-context MoSE gate | regressed versus both K1 and the ungated MoSE residual | closed |

The no-slot-attention plus uniform-return combination improved the immutable K1
reference by 0.002162 eV and removed 49,920 parameters. The selective MoSE
residual independently improved it by 0.002046 eV while operating on the input
structural path. These mechanisms have not been trained together. Their paths
are distinct enough to pose one interaction question, but their gains must not
be assumed additive.

## Residual-diversity caution

Aligned no-inference analysis of the same 50,000 development rows found that
50:50 averages between K1 and several candidates improved by roughly
0.0055-0.0065 eV, including candidates whose standalone MAE was worse than
K1. Three- and four-arm averages improved further. Because this behavior was
generic across both successful and failed mechanisms, it is evidence of
prediction diversity, not proof that any one added module is better. It may
also include optimization/runtime trajectory diversity.

Consequently, this trajectory does not authorize an ensemble, a learned
router, or a multi-pass production model. Architecture merit must come from a
single candidate trained under the frozen comparison contract.

## Selected question

The highest-probability bounded candidate is `K1 StructuralLite Slot`:

1. preserve the complete K1 local and persistent-edge backbone;
2. preserve RWSE16;
3. add the selective zero-initialized MoSE residual already accepted as
   `POSITIVE_BELOW_GATE`;
4. remove self-attention from the single molecular slot;
5. return the slot uniformly to valid atoms;
6. keep the layer 3/6/9 exchange locations and all training semantics fixed.

The estimated parameter count is about 3,623,474: 3,608,897 for the combined
slot simplification plus 14,577 for the MoSE residual. The exact count remains
a mandatory preflight assertion, not an accepted fact until implementation.

This candidate is preferred over another pair module because every tested pair
compression, recurrence, and sparse-relative route has already failed its
gate. It is preferred over a new selector because selector and scalar-gating
families are also closed. Published work independently supports motif
structural encodings and keeping structural/positional information distinct;
that literature is motivation, not project evidence.

## Stop rule

Run at most one fixed seed-42 PCQM-100K V5 screen. Promotion requires at least
0.003 eV gain over the immutable aligned K1 reference and an entirely favorable
paired interval. A miss closes the composition without another seed, wider
MoSE branch, alternate gate, 500K bridge, or full-scale run.

