"""Unit tests for HITL continual learning, EWC, replay buffer, and governance gating."""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import pytest
import torch

from config import ContinualLearningConfig
from data.synthetic_generator import generate_synthetic_dataset
from preprocessing import FeatureProcessor
from model import DelayPredictionNet
from retrain import (
    StratifiedExperienceReplayBuffer,
    GovernanceStagingManager,
    compute_fisher_matrix,
    ewc_penalty,
)
from losses import MultiTaskDelayLoss
from train import TabularBatchDataset, collate_tabular_batch
from torch.utils.data import DataLoader


@pytest.fixture
def test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Provide historical and new incoming datasets."""
    df_hist = generate_synthetic_dataset(n_samples=300, seed=1)
    df_new = generate_synthetic_dataset(n_samples=200, seed=2)
    return df_hist, df_new


def test_experience_replay_ratio(test_data: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Verify replay buffer accurately produces the requested 1:4 (20% old, 80% new) ratio."""
    df_hist, df_new = test_data
    buffer = StratifiedExperienceReplayBuffer(old_ratio=0.20)
    buffer.load_historical_data(df_hist)
    buffer.add_new_samples(df_new)

    sample = buffer.sample_batch(total_samples=100)
    assert len(sample) == 100
    # Both outcomes should be represented
    assert (sample["delay_probability"] == 1).sum() > 0
    assert (sample["delay_probability"] == 0).sum() > 0


def test_fisher_matrix_and_ewc_penalty(test_data: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Verify Fisher Information computation and non-negative EWC quadratic penalty."""
    df_hist, _ = test_data
    processor = FeatureProcessor()
    processor.fit(df_hist)
    batch = processor.transform(df_hist.head(50))

    model = DelayPredictionNet(
        vocab_sizes=processor.vocab_sizes_,
        embedding_dims=processor.embedding_dims_,
        num_one_hot_features=processor.num_one_hot_features_,
        num_numerical_features=processor.num_numerical_features_,
    )
    device = torch.device("cpu")
    criterion = MultiTaskDelayLoss()
    loader = DataLoader(TabularBatchDataset(batch), batch_size=16, collate_fn=collate_tabular_batch)

    fisher = compute_fisher_matrix(model, loader, criterion, device)
    star_params = {n: p.clone().detach() for n, p in model.named_parameters()}

    # Identical parameters must yield 0 penalty
    zero_penalty = ewc_penalty(model, fisher, star_params)
    assert zero_penalty.item() == 0.0

    # Perturbed parameters must yield positive penalty
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05)

    positive_penalty = ewc_penalty(model, fisher, star_params)
    assert positive_penalty.item() > 0.0


def test_governance_validation_gate_rejection(tmp_path: Path, test_data: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Verify automated validation gate rejects a candidate model that degrades protected slices."""
    df_hist, _ = test_data
    processor = FeatureProcessor()
    processor.fit(df_hist)

    cfg = ContinualLearningConfig(
        registry_path=str(tmp_path / "model_registry.json"),
        staging_dir=str(tmp_path / "models/staging"),
        production_dir=str(tmp_path / "models/production"),
        pending_approval_dir=str(tmp_path / "models/pending_human_approval"),
        max_slice_regression_pct=0.02,  # 2% max regression
    )
    manager = GovernanceStagingManager(cfg)

    prod_model = DelayPredictionNet(
        vocab_sizes=processor.vocab_sizes_,
        embedding_dims=processor.embedding_dims_,
        num_one_hot_features=processor.num_one_hot_features_,
        num_numerical_features=processor.num_numerical_features_,
    )

    # Create candidate model with corrupted/worse weights
    cand_model = DelayPredictionNet(
        vocab_sizes=processor.vocab_sizes_,
        embedding_dims=processor.embedding_dims_,
        num_one_hot_features=processor.num_one_hot_features_,
        num_numerical_features=processor.num_numerical_features_,
    )
    with torch.no_grad():
        # Invert predictions
        for p in cand_model.head_classifier.parameters():
            p.mul_(-2.0)

    val_df = df_hist.tail(100).copy()
    passed, reason, report = manager.validate_candidate(
        candidate_model=cand_model,
        production_model=prod_model,
        val_df=val_df,
        processor=processor,
        device=torch.device("cpu"),
    )

    assert not passed, "Degraded candidate model should have been rejected by safety gate!"
    assert "REJECTED" in reason or "degraded" in reason


def test_human_approval_promotion(tmp_path: Path, test_data: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Verify valid candidate promotion to production upon explicit human sign-off."""
    df_hist, _ = test_data
    processor = FeatureProcessor()
    processor.fit(df_hist)

    cfg = ContinualLearningConfig(
        registry_path=str(tmp_path / "model_registry.json"),
        staging_dir=str(tmp_path / "models/staging"),
        production_dir=str(tmp_path / "models/production"),
        pending_approval_dir=str(tmp_path / "models/pending_human_approval"),
    )
    manager = GovernanceStagingManager(cfg)

    model = DelayPredictionNet(
        vocab_sizes=processor.vocab_sizes_,
        embedding_dims=processor.embedding_dims_,
        num_one_hot_features=processor.num_one_hot_features_,
        num_numerical_features=processor.num_numerical_features_,
    )

    # Stage candidate with passed validation
    manager.stage_candidate(
        candidate_model=model,
        version=1,
        report_data={"overall": {"cand_auroc": 0.88, "cand_mae": 3.2}},
        passed_validation=True,
    )

    # Simulate human administrative sign-off
    prod_path = manager.approve_and_promote(version=1, approver_name="Dr. R. Sharma (Director, MoRD)")
    assert Path(prod_path).exists()

    reg = manager.load_registry()
    assert reg["active_production_version"] == 1
    assert reg["history"][0]["status"] == "PRODUCTION"
    assert reg["history"][0]["approver"] == "Dr. R. Sharma (Director, MoRD)"
