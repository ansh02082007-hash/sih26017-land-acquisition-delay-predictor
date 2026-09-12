"""Loss functions for Multi-Task Land Acquisition Delay Prediction.

This module provides:
1. FocalLoss: Implemented from scratch to address class imbalance in delayed cases.
2. MultiTaskDelayLoss: Combines Focal Loss (classification) and masked Huber Loss (regression).
Regression loss is strictly masked to rows where delay occurred (target == 1), avoiding
corrupting gradient updates with non-delayed cases.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from config import TrainingConfig


class FocalLoss(nn.Module):
    """Focal Loss for binary classification implemented from scratch.

    Focal Loss reshapes the standard cross-entropy loss by multiplying it by a modulating
    factor (1 - p_t)^gamma. This dynamically downweights well-classified / easy examples and
    focuses optimization on difficult, boundary-case land acquisition disputes.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where:
        p_t = p if y == 1 else (1 - p)
        alpha_t = alpha if y == 1 else (1 - alpha)
    """

    def __init__(
        self,
        alpha: float = TrainingConfig.focal_alpha,
        gamma: float = TrainingConfig.focal_gamma,
        eps: float = 1e-7,
        reduction: str = "mean",
    ) -> None:
        """Initialize Focal Loss hyperparameters.

        Args:
            alpha: Weighting factor for the rare/delayed class (typically 0.25).
            gamma: Focusing parameter (typically 2.0) that attenuates loss on easy cases.
            eps: Epsilon value to prevent log(0) numerical instability.
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss between predicted probabilities and ground truth targets.

        Args:
            pred_probs: Predicted delay probabilities of shape (B, 1) or (B,) in [0, 1].
            targets: Binary ground truth targets of shape (B, 1) or (B,) in {0, 1}.

        Returns:
            Computed scalar tensor loss (or per-sample loss if reduction is 'none').
        """
        pred_probs = pred_probs.view(-1, 1)
        targets = targets.view(-1, 1)

        # Numerical clamping prevents numerical underflow or NaNs when computing log(p)
        p = torch.clamp(pred_probs, min=self.eps, max=1.0 - self.eps)

        # Compute p_t and alpha_t depending on binary target class
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        # Focal modulating factor: (1 - p_t)^gamma vanishes for high-confidence correct predictions
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        loss = -alpha_t * modulating_factor * torch.log(p_t)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiTaskDelayLoss(nn.Module):
    """Combined multi-task loss with masked regression for land acquisition delay prediction.

    Loss Formulation:
        Total Loss = lambda_focal * FocalLoss(p, y_prob) + lambda_huber * HuberLoss_masked(m, y_months)

    Crucial Engineering Justification:
    Delay duration is only meaningful/defined when a case is actually delayed (y_prob == 1).
    Including non-delayed cases (which have 0 delay months) in regression loss would falsely penalize
    the model for high duration predictions on borderline cases or force the duration head to learn
    a discontinuous zero-spike. Masking strictly isolates the regression head to delayed cases.
    """

    def __init__(
        self,
        focal_alpha: float = TrainingConfig.focal_alpha,
        focal_gamma: float = TrainingConfig.focal_gamma,
        huber_beta: float = TrainingConfig.huber_beta,
        lambda_focal: float = TrainingConfig.lambda_focal,
        lambda_huber: float = TrainingConfig.lambda_huber,
    ) -> None:
        """Initialize multi-task loss modules and balancing weights."""
        super().__init__()
        self.focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="mean")
        # Huber Loss (SmoothL1Loss with beta=1.0) provides quadratic error penalty for small errors
        # and linear penalty for large outlier delays (e.g. 5-year prolonged litigation), preventing
        # gradient explosions from rare extreme events.
        self.huber_loss_fn = nn.SmoothL1Loss(beta=huber_beta, reduction="none")
        self.lambda_focal = lambda_focal
        self.lambda_huber = lambda_huber

    def forward(
        self,
        pred_prob: torch.Tensor,
        pred_months: torch.Tensor,
        target_prob: torch.Tensor,
        target_months: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the combined multi-task loss.

        Args:
            pred_prob: Model predicted delay probability tensor of shape (B, 1).
            pred_months: Model predicted delay duration tensor in months of shape (B, 1).
            target_prob: Ground truth delay binary indicator tensor of shape (B, 1).
            target_months: Ground truth delay duration tensor in months of shape (B, 1).

        Returns:
            Tuple of (total_loss, focal_loss, huber_loss) as PyTorch scalar tensors.
        """
        pred_prob = pred_prob.view(-1, 1)
        pred_months = pred_months.view(-1, 1)
        target_prob = target_prob.view(-1, 1)
        target_months = target_months.view(-1, 1)

        # 1. Classification Loss across all cases in the batch
        focal_loss = self.focal_loss_fn(pred_prob, target_prob)

        # 2. Regression Loss strictly masked to delayed cases (target_prob == 1)
        # Create boolean mask for delayed cases
        delayed_mask = (target_prob >= 0.5).squeeze(-1)
        num_delayed = delayed_mask.sum()

        if num_delayed > 0:
            # Only compute Huber loss over cases where delay actually occurred
            raw_huber = self.huber_loss_fn(pred_months[delayed_mask], target_months[delayed_mask])
            huber_loss = raw_huber.mean()
        else:
            # If batch has no delayed cases, regression loss is zero with valid gradient hook
            huber_loss = torch.tensor(0.0, device=pred_prob.device, dtype=pred_prob.dtype, requires_grad=True)

        # 3. Weighted Combined Loss
        total_loss = (self.lambda_focal * focal_loss) + (self.lambda_huber * huber_loss)

        return total_loss, focal_loss, huber_loss
