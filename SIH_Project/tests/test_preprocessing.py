"""Unit tests for the data preprocessing pipeline and FeatureProcessor."""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from preprocessing import FeatureProcessor, ProcessedBatch
from data.synthetic_generator import generate_synthetic_dataset


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Generate a small realistic dataset for unit testing."""
    return generate_synthetic_dataset(n_samples=200, seed=42)


def test_feature_processor_fit_transform(sample_data: pd.DataFrame) -> None:
    """Test standard fit_transform producing ProcessedBatch with expected tensor shapes."""
    processor = FeatureProcessor(embedding_dim_max=12)
    batch: ProcessedBatch = processor.fit_transform(sample_data)

    assert isinstance(batch, ProcessedBatch)
    assert "pin_code" in batch.embedding_indices
    assert "district" in batch.embedding_indices

    # Verify tensor shapes
    n_rows = len(sample_data)
    assert batch.embedding_indices["pin_code"].shape == (n_rows,)
    assert batch.embedding_indices["district"].shape == (n_rows,)
    assert batch.one_hot_features.shape[0] == n_rows
    assert batch.numerical_features.shape[0] == n_rows
    assert batch.target_prob.shape == (n_rows, 1)
    assert batch.target_months.shape == (n_rows, 1)

    # Verify embedding dimensions are bounded by embedding_dim_max
    assert processor.embedding_dims_["pin_code"] <= 12
    assert processor.embedding_dims_["district"] <= 12


def test_unknown_category_mapping(sample_data: pd.DataFrame) -> None:
    """Verify that unseen categorical values cleanly map to index 0 (<UNK>) without crashing."""
    processor = FeatureProcessor(embedding_dim_max=12)
    processor.fit(sample_data)

    # Test case with previously unseen PIN and District
    unseen_df = pd.DataFrame([{
        "case_id": "TEST-UNKNOWN-001",
        "pin_code": "999999",  # Non-existent PIN
        "district": "Unseen_Fictional_District",
        "land_type": "Agricultural",
        "rfctlarr_stage": "Preliminary Notification (Sec 11)",
        "land_parcel_area_acres": 10.0,
        "compensation_amount_lakhs": 50.0,
        "days_since_notification": 100,
        "number_of_displaced_families": 2,
        "litigation_cases_count": 0,
        "social_vulnerability_index": 0.3,
        "official_processing_speed_score": 7.0,
    }])

    batch = processor.transform(unseen_df)
    assert batch.embedding_indices["pin_code"][0].item() == 0  # <UNK>
    assert batch.embedding_indices["district"][0].item() == 0  # <UNK>


def test_missing_value_handling_and_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test median imputation and verification that >15% missingness triggers audit warning."""
    df_with_high_missing = generate_synthetic_dataset(n_samples=100, seed=123, inject_high_missing_col=True)

    processor = FeatureProcessor()
    with caplog.at_level(logging.WARNING):
        batch = processor.fit_transform(df_with_high_missing)

    # Verify no NaNs remain in transformed continuous or one-hot tensors
    assert not torch.isnan(batch.numerical_features).any()
    assert not torch.isnan(batch.one_hot_features).any()

    # Verify data quality warning was logged
    warning_found = any("DATA QUALITY WARNING" in record.message for record in caplog.records)
    assert warning_found, "Expected data quality warning for column exceeding 15% missingness."


def test_feature_processor_serialization(tmp_path: Path, sample_data: pd.DataFrame) -> None:
    """Verify joblib save and load restores exact identical transform outputs."""
    processor = FeatureProcessor()
    processor.fit(sample_data)
    save_file = tmp_path / "test_processor.joblib"
    processor.save(save_file)

    loaded_processor = FeatureProcessor.load(save_file)
    batch_orig = processor.transform(sample_data)
    batch_loaded = loaded_processor.transform(sample_data)

    assert torch.equal(batch_orig.numerical_features, batch_loaded.numerical_features)
    assert torch.equal(batch_orig.one_hot_features, batch_loaded.one_hot_features)
    assert torch.equal(batch_orig.embedding_indices["pin_code"], batch_loaded.embedding_indices["pin_code"])
