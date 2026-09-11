# Desktop handoff — frozen K1 candidate

The server-side architecture funnel has completed for Neural-Atom K1. The
authoritative decision is `decision.md`; mechanical evidence and artifact
hashes are in `terminal_report_fixed_s42_v3.md` and
`terminal_handoff_marker_fixed_s42_v3.json`.

Desktop may consider exactly one full-scale execution after recording:

- an explicit compute budget;
- the accepted fixed full-data identity on IMS;
- the unchanged K1 architecture from source commit
  `36215d9539acdd75542608637ec1e2db5341d3ff`;
- physical batch and precision chosen once for the full stage;
- an optimizer-step/sample-exposure schedule derived before training;
- one-time official validation and OGB submission boundaries.

This handoff does not authorize the server agent to launch full training. It
does not authorize K1 architecture tuning, more discovery seeds, reuse of the
consumed shadow role, or access to official validation/test-dev during model
selection.
