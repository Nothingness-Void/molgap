# Stage-1 infrastructure retry 122250328

- SCNet account: `changfeng2006`
- Partition/resource: `kshdtest`, one `dcu:Hygon`
- Source commit: `73ace53`
- Source archive SHA-256:
  `b44caec37d4afc0535df95770879732e6fbbb91976f9301f8269d631c8ce6618`
- Remote immutable source:
  `/public/home/changfeng2006/molgap-v4-bootstrap/k1-xai-stage1/source-73ace53`
- Submission: `122250328`
- First observation: `RUNNING` on `e06r4n13`

The archive contains the complete tracked `src/molgap/` tree and the frozen
Stage-1 entry points. The job must pass its in-process graph/payload preflight
before writing attribution output. This retry changes packaging and mechanical
acceptance only; it does not change data, predictions, metrics, or analysis.
