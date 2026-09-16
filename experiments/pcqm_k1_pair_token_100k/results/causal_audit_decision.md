# PairToken causal-audit decision

Kunshan job `122305552` completed frozen-checkpoint inference on the fixed
50,000-row development role. It executed no training and read no sealed role.
The accepted PairToken payload reproduced exactly at prediction level and to
`1.73e-08 eV` at MAE level.

## Counterfactual evidence

| Intervention | Overall MAE | Delta vs trained PairToken |
|---|---:|---:|
| Trained PairToken | `0.1383300 eV` | — |
| Disable relation residual | `0.3672131 eV` | `-0.2288831 eV` |
| Uniform all-pair averaging | `0.1656279 eV` | `-0.0272979 eV` |
| Diagonal/self pairs only | `0.4188089 eV` | `-0.2804788 eV` |
| Off-diagonal/cross-node pairs only | `0.1390511 eV` | `-0.0007211 eV` |
| Remove per-pair channel normalization | `0.2580297 eV` | `-0.1196997 eV` |

Every paired 95% interval was strictly unfavorable. The same directions held
for the topology-extreme union and middle control.

## Interpretation

The trained branch is strongly co-adapted with K1, so disabling it does not
estimate a freshly trained K1 baseline. The controlled comparisons nevertheless
identify the branch's active mechanism:

1. learned pair selection is necessary; a uniform global second-order mean is
   insufficient;
2. cross-node pairs carry almost all useful relation content; self pairs add a
   smaller but measurable contribution;
3. per-pair channel normalization is essential to keep the learned query and
   return path in a usable scale regime.

The result supports a normalized, selective cross-node relation bottleneck.
It does not support more atom slots, another scalar gate, or an unnormalized
relation path. Because the current implementation explicitly materializes all
ordered pairs and reduced throughput, the next architecture question should
preserve these three proven properties while approximating pair selection more
efficiently. No successor is submitted by this decision.

