# Bond-type local color decision — 2026-09-28

The 40-epoch seed-42 V5 screen completed on one Tesla T4. Saved-artifact
acceptance measured 0.1429534107 eV on the fixed 50,000-row internal
development role versus frozen K1-v4's 0.1413736343 eV: a -0.0015797764 eV
gain (regression). The paired candidate-minus-K1 row-bootstrap 95% interval
was [+0.000684922, +0.002492793] eV. Model and checkpoint SHA256, observed
roles, runtime certificate, trace and cost are bound in `results/`.

The prospectively frozen +0.003 eV gate failed. The equal-capacity bond-type
control outperformed the atom-pair arm numerically, but neither beat K1. No
fixed500K inference audit, extra seed or scaled training was authorized.
