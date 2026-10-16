# Cross-scale comparison

Question: SAME_SCALE_MODEL

Grade: CONTEXT_ONLY

Automatic early stop: NOT_RELEASED.

Missing evidence: left:source_sha256, left:checkpoint_sha256, left:weights, left:bn_state, left:seed, left:cost, right:source_sha256, right:checkpoint_sha256, right:weights, right:bn_state, right:seed, right:cost

## Observed evidence

```json
{
  "format": "molgap-cross-scale-comparison-report-v1",
  "comparison_type": "SAME_SCALE_MODEL",
  "strictness": "CONTEXT_ONLY",
  "curve_comparison": {
    "status": "MATCHED_PREFIX",
    "comparison_class": "MATCHED_PREFIX",
    "comparison_type": "SAME_SCALE_MODEL",
    "history_issues": [],
    "strictness": "STRICT_SAME_SCALE",
    "reasons": [],
    "axis": "optimizer_step",
    "aligned": [
      {
        "optimizer_step": 1,
        "left": 0.3,
        "right": 0.4,
        "left_observation": {
          "optimizer_step": 1,
          "sample_presentations": 128,
          "complete_passes": 0,
          "learning_rate": 0.001,
          "value": 0.3,
          "metric": "live_dev_metric",
          "unit": "eV",
          "checkpoint_identity": "3d2f0a2b152dee4bfa14db9f022f62a494e243c06a769a57f9cf8068eef8a96f",
          "weights": "live",
          "evaluation_role": "internal_development",
          "metric_kind": "development"
        },
        "right_observation": {
          "optimizer_step": 1,
          "sample_presentations": 128,
          "complete_passes": 0,
          "learning_rate": 0.001,
          "value": 0.4,
          "metric": "live_dev_metric",
          "unit": "eV",
          "checkpoint_identity": "3d2f0a2b152dee4bfa14db9f022f62a494e243c06a769a57f9cf8068eef8a96f",
          "weights": "live",
          "evaluation_role": "internal_development",
          "metric_kind": "development"
        },
        "left_minus_right_eV": -0.10000000000000003,
        "comparison_class": "MATCHED_PREFIX",
        "reasons": []
      },
      {
        "optimizer_step": 2,
        "left": 0.2,
        "right": 0.25,
        "left_observation": {
          "optimizer_step": 2,
          "sample_presentations": 256,
          "complete_passes": 0,
          "learning_rate": 0.001,
          "value": 0.2,
          "metric": "live_dev_metric",
          "unit": "eV",
          "checkpoint_identity": "1246f895ff5deaafc1b2b5b90fa9f902662a35042da3e5962fa8e43f738b3818",
          "weights": "live",
          "evaluation_role": "internal_development",
          "metric_kind": "development"
        },
        "right_observation": {
          "optimizer_step": 2,
          "sample_presentations": 256,
          "complete_passes": 0,
          "learning_rate": 0.001,
          "value": 0.25,
          "metric": "live_dev_metric",
          "unit": "eV",
          "checkpoint_identity": "1246f895ff5deaafc1b2b5b90fa9f902662a35042da3e5962fa8e43f738b3818",
          "weights": "live",
          "evaluation_role": "internal_development",
          "metric_kind": "development"
        },
        "left_minus_right_eV": -0.04999999999999999,
        "comparison_class": "MATCHED_PREFIX",
        "reasons": []
      },
      {
        "optimizer_step": 3,
        "left": 0.1,
        "right": 0.09,
        "left_observation": {
          "optimizer_step": 3,
          "sample_presentations": 384,
          "complete_passes": 0,
          "learning_rate": 0.001,
          "value": 0.1,
          "metric": "live_dev_metric",
          "unit": "eV",
          "checkpoint_identity": "318ac0ae4f83e9f457597dd577e1886da5f48845b688401be446abfcd9cb6765",
          "weights": "live",
          "evaluation_role": "internal_development",
          "metric_kind": "development"
        },
        "right_observation": {
          "optimizer_step": 3,
          "sample_presentations": 384,
          "complete_passes": 0,
          "learning_rate": 0.001,
          "value": 0.09,
          "metric": "live_dev_metric",
          "unit": "eV",
          "checkpoint_identity": "318ac0ae4f83e9f457597dd577e1886da5f48845b688401be446abfcd9cb6765",
          "weights": "live",
          "evaluation_role": "internal_development",
          "metric_kind": "development"
        },
        "left_minus_right_eV": 0.010000000000000009,
        "comparison_class": "MATCHED_PREFIX",
        "reasons": []
      }
    ],
    "extrapolated": false,
    "automatic_training_action": false
  },
  "runs": {
    "left": {
      "model": "CPU_SYNTHETIC_FIXTURE",
      "source_sha256": null,
      "identity": {
        "benchmark_identity": "synthetic:benchmark_identity",
        "dataset_identity": "synthetic:dataset_identity",
        "data_role_identity": "synthetic:data_role_identity",
        "row_membership_identity": "synthetic:row_membership_identity",
        "row_order_identity": "synthetic:row_order_identity",
        "feature_identity": "synthetic:feature_identity",
        "target_identity": "synthetic:target_identity",
        "seed": 42,
        "precision": "synthetic:precision",
        "tf32_enabled": false,
        "deterministic_algorithms": true,
        "physical_batch_per_device": 128,
        "gradient_accumulation_steps": 1,
        "tail_batch_policy": "drop_last",
        "optimizer_identity": "synthetic:optimizer_identity",
        "optimizer_mode": "synthetic:optimizer_mode",
        "optimizer_fused": "synthetic:optimizer_fused",
        "schedule_identity": "synthetic:schedule_identity",
        "loss_identity": "synthetic:loss_identity",
        "target_transform_identity": "synthetic:target_transform_identity",
        "target_transform_asset_sha256": "synthetic:target_transform_asset_sha256",
        "sample_presentations": 29998080,
        "optimizer_steps": 234360,
        "checkpoint_selection_identity": "synthetic:checkpoint_selection_identity",
        "evaluation_role_identity": "synthetic:evaluation_role_identity",
        "selection_role_identity": "synthetic:selection_role_identity",
        "architecture_config_identity": "synthetic:architecture_config_identity",
        "runtime_certificate_scope": "synthetic:runtime_certificate_scope",
        "weight_semantics": "synthetic:weight_semantics",
        "ema_enabled": false,
        "ema_decay": "none",
        "ema_update_frequency": "none",
        "evaluation_weight_source": "synthetic:evaluation_weight_source",
        "sampler_identity": "sampler42",
        "schedule_axis": "optimizer_step",
        "target_unit": "eV",
        "evaluation_row_sha256": "58f12e93c4bbfdd519e1d1e232c1da5af96370971010e26568d3968d88a656a0",
        "evaluation_target_sha256": "62c263f2e35b975fb78e3a462e74dcab585896d18ef90a1f2e33176c2378645c",
        "evaluation_row_count": 50000,
        "bn_calibration_identity": "not_applicable",
        "drop_last": true
      },
      "checkpoint_sha256": null,
      "weights": null,
      "bn_state": null,
      "seed": null,
      "uncertainty": null,
      "curve": [
        {
          "optimizer_step": 1,
          "sample_presentations": 128,
          "complete_passes": 0,
          "learning_rate": 0.001,
          "value": 0.3,
          "metric": "live_dev_metric",
          "unit": "eV",
          "checkpoint_identity": "3d2f0a2b152dee4bfa14db9f022f62a494e243c06a769a57f9cf8068eef8a96f",
          "weights": "live",
          "evaluation_role": "internal_development",
          "metric_kind": "development"
        },
        {
          "optimizer_step": 2,
          "sample_presentations": 256,
          "complete_passes": 0,
          "learning_rate": 0.001,
          "value": 0.2,
          "metric": "live_dev_metric",
          "unit": "eV",
          "checkpoint_identity": "1246f895ff5deaafc1b2b5b90fa9f902662a35042da3e5962fa8e43f738b3818",
          "weights": "live",
          "evaluation_role": "internal_development",
          "metric_kind": "development"
        },
        {
          "optimizer_step": 3,
          "sample_presentations": 384,
          "complete_passes": 0,
          "learning_rate": 0.001,
          "value": 0.1,
          "metric": "live_dev_metric",
          "unit": "eV",
          "checkpoint_identity": "318ac0ae4f83e9f457597dd577e1886da5f48845b688401be446abfcd9cb6765",
          "weights": "live",
          "evaluation_role": "internal_development",
          "metric_kind": "development"
        }
      ],
      "cost": null
    },
    "right": {
      "model": "CPU_SYNTHETIC_FIXTURE",
      "source_sha256": null,
      "identity": {
        "benchmark_identity": "synthetic:benchmark_identity",
        "dataset_identity": "synthetic:dataset_identity",
        "data_role_identity": "synthetic:data_role_identity",
        "row_membership_identity": "synthetic:row_membership_identity",
        "row_order_identity": "synthetic:row_order_identity",
        "feature_identity": "synthetic:feature_identity",
        "target_identity": "synthetic:target_identity",
        "seed": 42,
        "precision": "synthetic:precision",
        "tf32_enabled": false,
        "deterministic_algorithms": true,
        "physical_batch_per_device": 128,
        "gradient_accumulation_steps": 1,
        "tail_batch_policy": "drop_last",
        "optimizer_identity": "synthetic:optimizer_identity",
        "optimizer_mode": "synthetic:optimizer_mode",
        "optimizer_fused": "synthetic:optimizer_fused",
        "schedule_identity": "synthetic:schedule_identity",
        "loss_identity": "synthetic:loss_identity",
        "target_transform_identity": "synthetic:target_transform_identity",
        "target_transform_asset_sha256": "synthetic:target_transform_asset_sha256",
        "sample_presentations": 29998080,
        "optimizer_steps": 234360,
        "checkpoint_selection_identity": "synthetic:checkpoint_selection_identity",
        "evaluation_role_identity": "synthetic:evaluation_role_identity",
        "selection_role_identity": "synthetic:selection_role_identity",
        "architecture_config_identity": "synthetic:architecture_config_identity",
        "runtime_certificate_scope": "synthetic:runtime_certificate_scope",
        "weight_semantics": "synthetic:weight_semantics",
        "ema_enabled": false,
        "ema_decay": "none",
        "ema_update_frequency": "none",
        "evaluation_weight_source": "synthetic:evaluation_weight_source",
        "sampler_identity": "sampler42",
        "schedule_axis": "optimizer_step",
        "target_unit": "eV",
        "evaluation_row_sha256": "58f12e93c4bbfdd519e1d1e232c1da5af96370971010e26568d3968d88a656a0",
        "evaluation_target_sha256": "62c263f2e35b975fb78e3a462e74dcab585896d18ef90a1f2e33176c2378645c",
        "evaluation_row_count": 50000,
        "bn_calibration_identity": "not_applicable",
        "drop_last": true
      },
      "checkpoint_sha256": null,
      "weights": null,
      "bn_state": null,
      "seed": null,
      "uncertainty": null,
      "curve": [
        {
          "optimizer_step": 1,
          "sample_presentations": 128,
          "complete_passes": 0,
          "learning_rate": 0.001,
          "value": 0.4,
          "metric": "live_dev_metric",
          "unit": "eV",
          "checkpoint_identity": "3d2f0a2b152dee4bfa14db9f022f62a494e243c06a769a57f9cf8068eef8a96f",
          "weights": "live",
          "evaluation_role": "internal_development",
          "metric_kind": "development"
        },
        {
          "optimizer_step": 2,
          "sample_presentations": 256,
          "complete_passes": 0,
          "learning_rate": 0.001,
          "value": 0.25,
          "metric": "live_dev_metric",
          "unit": "eV",
          "checkpoint_identity": "1246f895ff5deaafc1b2b5b90fa9f902662a35042da3e5962fa8e43f738b3818",
          "weights": "live",
          "evaluation_role": "internal_development",
          "metric_kind": "development"
        },
        {
          "optimizer_step": 3,
          "sample_presentations": 384,
          "complete_passes": 0,
          "learning_rate": 0.001,
          "value": 0.09,
          "metric": "live_dev_metric",
          "unit": "eV",
          "checkpoint_identity": "318ac0ae4f83e9f457597dd577e1886da5f48845b688401be446abfcd9cb6765",
          "weights": "live",
          "evaluation_role": "internal_development",
          "metric_kind": "development"
        }
      ],
      "cost": null
    }
  },
  "missing_evidence": [
    "left:source_sha256",
    "left:checkpoint_sha256",
    "left:weights",
    "left:bn_state",
    "left:seed",
    "left:cost",
    "right:source_sha256",
    "right:checkpoint_sha256",
    "right:weights",
    "right:bn_state",
    "right:seed",
    "right:cost"
  ],
  "cohort_proof": null,
  "limitations": [
    "Observed points only; no terminal projection",
    "Row bootstrap is not training variance",
    "Report grades do not grant V5 STRICT_CAUSAL or role/training authority"
  ],
  "next_gates": [
    "Bound complete outputs, role history, native cost and separate prospective authority"
  ],
  "early_stop_status": "NOT_RELEASED",
  "automatic_training_action": false,
  "training_authority": "NOT_AUTHORIZED_FOR_SCIENTIFIC_TRAINING",
  "scope": "CPU_SYNTHETIC_ONLY"
}
```
