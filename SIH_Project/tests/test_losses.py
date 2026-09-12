"""Unit tests for FocalLoss, Huber regression loss, and masked multi-task loss."""

import pytest
import torch

from losses import FocalLoss, MultiTaskDelayLoss


def test_focal_loss_properties() -> None:
    """Verify focal loss correctly penalizes misclassifications while downweighting easy instances."""
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    # High confidence correct predictions
    easy_targets = torch.tensor([[1.0], [0.0]])
    easy_preds = torch.tensor([[0.99], [0.01]])
    easy_loss = loss_fn(easy_preds, easy_targets)

    # Confident wrong predictions (hard mistakes)
    hard_targets = torch.tensor([[1.0], [0.0]])
    hard_preds = torch.tensor([[0.01], [0.99]])
    hard_loss = loss_fn(hard_preds, hard_targets)

    assert hard_loss.item() > easy_loss.item() * 10
    assert not torch.isnan(easy_loss)
    assert not torch.isnan(hard_loss)


def test_focal_loss_gradients() -> None:
    """Verify FocalLoss supports continuous gradient backpropagation."""
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    preds = torch.tensor([[0.6], [0.4]], requires_grad=True)
    targets = torch.tensor([[1.0], [0.0]])

    loss = loss_fn(preds, targets)
    loss.backward()

    assert preds.grad is not None
    assert not torch.isnan(preds.grad).any()


def test_multi_task_loss_masking_delayed_only() -> None:
    """Verify Huber loss is strictly evaluated on delayed cases and ignores non-delayed cases."""
    loss_fn = MultiTaskDelayLoss(lambda_focal=1.0, lambda_huber=1.0)

    # Case 1: Delayed (y=1), Duration target = 20.0 months, Prediction = 10.0 months (error = 10.0)
    # Case 2: NOT Delayed (y=0), Duration target = 0.0 months, Prediction = 50.0 months
    # Even though prediction for Case 2 is completely wrong (50 vs 0), it MUST BE MASKED OUT!
    pred_prob = torch.tensor([[0.9], [0.1]])
    target_prob = torch.tensor([[1.0], [0.0]])

    pred_months = torch.tensor([[10.0], [50.0]])
    target_months = torch.tensor([[20.0], [0.0]])

    tot_loss, foc_loss, hub_loss = loss_fn(pred_prob, pred_months, target_prob, target_months)

    # Now compute Huber loss ONLY on Case 1 manually (delta=10, Huber smooth L1 with beta=1 is 10 - 0.5 = 9.5)
    expected_huber = 10.0 - 0.5
    assert abs(hub_loss.item() - expected_huber) < 1e-3, (
        f"Huber loss was {hub_loss.item()}, expected {expected_huber}. "
        "Non-delayed case was falsely included in regression loss calculation!"
    )


def test_multi_task_loss_zero_delayed_in_batch() -> None:
    """Verify loss cleanly handles batches where no cases experienced delays."""
    loss_fn = MultiTaskDelayLoss(lambda_focal=1.0, lambda_huber=1.0)

    # All non-delayed
    pred_prob = torch.tensor([[0.2], [0.1], [0.3]])
    target_prob = torch.tensor([[0.0], [0.0], [0.0]])
    pred_months = torch.tensor([[5.0], [8.0], [12.0]])
    target_months = torch.tensor([[0.0], [0.0], [0.0]])

    tot_loss, foc_loss, hub_loss = loss_fn(pred_prob, pred_months, target_prob, target_months)

    assert hub_loss.item() == 0.0
    assert abs(tot_loss.item() - foc_loss.item()) < 1e-5


def test_loss_weight_scaling() -> None:
    """Verify lambda_focal and lambda_huber scale their respective components."""
    loss_fn_1 = MultiTaskDelayLoss(lambda_focal=1.0, lambda_huber=1.0)
    loss_fn_2 = MultiTaskDelayLoss(lambda_focal=2.0, lambda_huber=0.5)

    pred_prob = torch.tensor([[0.8]])
    target_prob = torch.tensor([[1.0]])
    pred_months = torch.tensor([[15.0]])
    target_months = torch.tensor([[10.0]])

    _, f1, h1 = loss_fn_1(pred_prob, pred_months, target_prob, target_months)
    tot2, f2, h2 = loss_fn_2(pred_prob, pred_months, target_prob, target_months)

    assert abs(f1.item() - f2.item()) < 1e-5
    assert abs(h1.item() - h2.item()) < 1e-5
    expected_tot2 = 2.0 * f1.item() + 0.5 * h1.item()
    assert abs(tot2.item() - expected_tot2) < 1e-4
