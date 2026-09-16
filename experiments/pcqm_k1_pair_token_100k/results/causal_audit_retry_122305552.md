# Causal-audit compatibility retry

- Job: `122305552`
- Source commit: `9d6cfabdffbce45ab0450500cc75bbb71906954d`
- Source archive SHA-256:
  `f7901a54ed1c31ddffe27aab1686ea4f43bda0a7323d272b5d38b83b2e183830`
- Scientific changes from job `122303028`: none
- Training: prohibited

The source changes only the unsupported tuple-axis boolean reduction to its
flattened equivalent. All frozen interventions and identities remain unchanged.

