"""Unit tests for DelayPredictionNet tensor shapes, outputs, and layer freezing."""

import pytest
import torch

from model import DelayPredictionNet, ModelOutput
from config import ModelConfig
from preprocessing import ProcessedBatch


@pytest.fixture
def dummy_model() -> DelayPredictionNet:
    """Create a DelayPredictionNet with synthetic dimensions for testing."""
    vocab_sizes = {"pin_code": 50, "district": 25}
    embedding_dims = {"pin_code": 8, "district": 6}
    num_one_hot = 14
    num_numerical = 7
    return DelayPredictionNet(
        vocab_sizes=vocab_sizes,
        embedding_dims=embedding_dims,
        num_one_hot_features=num_one_hot,
        num_numerical_features=num_numerical,
        config=ModelConfig(),
    )


def test_forward_pass_shapes_and_ranges(dummy_model: DelayPredictionNet) -> None:
    """Verify output shapes and bounded numerical ranges for both heads."""
    batch_size = 16
    batch = ProcessedBatch(
        embedding_indices={
            "pin_code": torch.randint(0, 50, (batch_size,)),
            "district": torch.randint(0, 25, (batch_size,)),
        },
        one_hot_features=torch.randn(batch_size, 14),
        numerical_features=torch.randn(batch_size, 7),
    )

    output: ModelOutput = dummy_model(batch)

    # 1. Output shapes
    assert output.delay_probability.shape == (batch_size, 1)
    assert output.delay_months.shape == (batch_size, 1)
    assert output.fused_representation.shape == (batch_size, 32)  # 16 wide + 16 deep
    assert output.deep_features.shape == (batch_size, 16)
    assert output.wide_features.shape == (batch_size, 16)

    # 2. Probability Head range [0, 1]
    assert (output.delay_probability >= 0.0).all()
    assert (output.delay_probability <= 1.0).all()

    # 3. Regression Head non-negativity guarantee (Softplus)
    assert (output.delay_months >= 0.0).all()


def test_single_case_inference_shape(dummy_model: DelayPredictionNet) -> None:
    """Verify forward inference executes cleanly on a single row (B=1) in eval mode."""
    dummy_model.eval()
    batch = ProcessedBatch(
        embedding_indices={
            "pin_code": torch.tensor([5]),
            "district": torch.tensor([3]),
        },
        one_hot_features=torch.randn(1, 14),
        numerical_features=torch.randn(1, 7),
    )

    with torch.no_grad():
        output: ModelOutput = dummy_model(batch)

    assert output.delay_probability.shape == (1, 1)
    assert output.delay_months.shape == (1, 1)


def test_early_layer_freezing(dummy_model: DelayPredictionNet) -> None:
    """Verify freeze_early_layers locks embedding and Layer 1 weights while keeping heads trainable."""
    dummy_model.freeze_early_layers()

    # Early layers must NOT track gradients
    for param in dummy_model.embeddings.parameters():
        assert not param.requires_grad
    for param in dummy_model.deep_fc1.parameters():
        assert not param.requires_grad
    for param in dummy_model.bn1.parameters():
        assert not param.requires_grad

    # Subsequent layers and heads MUST remain trainable
    for param in dummy_model.deep_fc2.parameters():
        assert param.requires_grad
    for param in dummy_model.deep_fc3.parameters():
        assert param.requires_grad
    for param in dummy_model.head_classifier.parameters():
        assert param.requires_grad
    for param in dummy_model.head_regressor.parameters():
        assert param.requires_grad
    for param in dummy_model.wide_linear.parameters():
        assert param.requires_grad

    # Unfreeze check
    dummy_model.unfreeze_all_layers()
    for param in dummy_model.parameters():
        assert param.requires_grad
