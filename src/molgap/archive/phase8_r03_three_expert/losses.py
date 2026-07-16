"""Loss terms for archive-r03 heterogeneous MoE training."""

from __future__ import annotations

import torch
import torch.nn as nn


class HeteroMoELoss(nn.Module):
    def __init__(
        self,
        aux_weight: float = 0.30,
        balance_weight: float = 0.02,
        consistency_weight: float = 0.0,
        conformer_weight: float = 0.0,
        beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.aux_weight = aux_weight
        self.balance_weight = balance_weight
        self.consistency_weight = consistency_weight
        self.conformer_weight = conformer_weight
        self.target_weights = (0.25, 0.25, 0.50)
        self.criterion = nn.SmoothL1Loss(reduction="none", beta=beta)

    def _weighted(self, prediction, target):
        weights = prediction.new_tensor(self.target_weights)
        return (self.criterion(prediction, target) * weights).sum(dim=-1).mean()

    def forward(self, outputs, target, paired_prediction=None):
        main = self._weighted(outputs["prediction"], target)
        experts = outputs["expert_predictions"].permute(2, 0, 1)
        auxiliary = torch.stack([self._weighted(value, target) for value in experts]).mean()
        mean_usage = outputs["router_weights"].mean(dim=0)
        balance = ((mean_usage - 1.0 / mean_usage.size(-1)) ** 2).sum()
        prediction = outputs["prediction"]
        consistency = torch.abs(prediction[:, 2] - (prediction[:, 1] - prediction[:, 0])).mean()
        conformer = prediction.new_zeros(())
        if paired_prediction is not None:
            conformer = torch.abs(prediction - paired_prediction).mean()
        total = (
            main
            + self.aux_weight * auxiliary
            + self.balance_weight * balance
            + self.consistency_weight * consistency
            + self.conformer_weight * conformer
        )
        return {
            "loss": total,
            "main": main.detach(),
            "auxiliary": auxiliary.detach(),
            "balance": balance.detach(),
            "consistency": consistency.detach(),
            "conformer": conformer.detach(),
        }
