# Frozen-checkpoint portability: NO_TRAIN terminal decision

On 2026-09-28 JST, both accepted frozen100K checkpoints reproduced their
original-dev predictions within 1.91e-6 eV and produced aligned predictions
on the fixed500K internal-development rows 500000..549999. No optimizer step,
checkpoint selection, 500K training or protected-role access occurred.

Neither arm passed the frozen portability gate. Relative to clean K1, A's
mean error increased by 0.0031896144 eV with an unfavorable paired interval.
B increased by 0.0006199697 eV but its interval crossed zero. B improved upon
A by 0.0025696448 eV with a favorable interval. These findings preserve useful
conditional mechanism evidence without promoting the complete recipe.

The action was closed as `NO_TRAIN`, separately from both training
trajectories. Observed audit device time was recorded once; joint parallel
wall/CPU intervals remained unavailable. No training replay trace was invented.
The [study decision](../../../decision.md) owns the complete attribution,
cost accounting and reused-role/row-bootstrap caveats. No successor was released.
