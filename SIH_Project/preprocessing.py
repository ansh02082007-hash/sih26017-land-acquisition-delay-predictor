"""Data preprocessing and feature engineering pipeline for Land Acquisition Delay Prediction.

This module provides the FeatureProcessor class, compatible with scikit-learn's Pipeline
architecture, capable of preparing heterogeneous tabular data for the dual-head Wide & Deep
neural network. It produces entity embedding indices, one-hot encoded categories, and scaled
numerical features packed in a PyTorch-ready ProcessedBatch.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch

from config import (
    HIGH_CARDINALITY_COLS,
    LOW_CARDINALITY_COLS,
    RFCTLARR_STAGES,
    SKEWED_NUMERICAL_COLS,
    PLAIN_NUMERICAL_COLS,
    ALL_NUMERICAL_COLS,
    ModelConfig,
)

logger = logging.getLogger("preprocessing")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ProcessedBatch:
    """Dataclass holding PyTorch tensor batches for model training and inference.

    Attributes:
        embedding_indices: Dictionary mapping high-cardinality column names (e.g., 'pin_code')
            to 1D LongTensors of categorical entity IDs (index 0 is reserved for <UNK>).
        one_hot_features: 2D FloatTensor containing one-hot encoded low-cardinality features.
        numerical_features: 2D FloatTensor containing scaled skewed and non-skewed features.
        target_prob: Optional 2D FloatTensor of shape (B, 1) indicating delay occurrence (0 or 1).
        target_months: Optional 2D FloatTensor of shape (B, 1) indicating delay duration in months.
        case_ids: Optional list of case identifiers for tracking predictions.
    """

    embedding_indices: Dict[str, torch.Tensor]
    one_hot_features: torch.Tensor
    numerical_features: torch.Tensor
    target_prob: Optional[torch.Tensor] = None
    target_months: Optional[torch.Tensor] = None
    case_ids: Optional[List[str]] = None

    def to(self, device: torch.device) -> "ProcessedBatch":
        """Move all contained tensors to the specified PyTorch compute device."""
        return ProcessedBatch(
            embedding_indices={k: v.to(device) for k, v in self.embedding_indices.items()},
            one_hot_features=self.one_hot_features.to(device),
            numerical_features=self.numerical_features.to(device),
            target_prob=self.target_prob.to(device) if self.target_prob is not None else None,
            target_months=self.target_months.to(device) if self.target_months is not None else None,
            case_ids=self.case_ids,
        )

    def __len__(self) -> int:
        """Return the batch size."""
        return self.numerical_features.shape[0]


class FeatureProcessor(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible processor for land acquisition tabular data.

    Transforms raw administrative and project features into neural-network ready batches:
    - Entity embeddings for PIN codes and districts (maps rare/unseen values to index 0).
    - One-hot encoding for statutory RFCTLARR Act stages and land classifications.
    - Log1p transformation + StandardScaler for heavy-tailed financial/temporal features.
    - StandardScaler for bounded demographic/operational metrics.
    - Missing value detection with automatic >15% threshold data-quality warnings.
    """

    def __init__(
        self,
        high_cardinality_cols: Optional[List[str]] = None,
        low_cardinality_cols: Optional[List[str]] = None,
        skewed_numerical_cols: Optional[List[str]] = None,
        plain_numerical_cols: Optional[List[str]] = None,
        embedding_dim_max: int = ModelConfig.embedding_dim_max,
        missing_threshold_warning: float = 0.15,
    ) -> None:
        """Initialize the feature processor with column definitions and hyperparameter limits."""
        self.high_cardinality_cols = high_cardinality_cols or list(HIGH_CARDINALITY_COLS)
        self.low_cardinality_cols = low_cardinality_cols or list(LOW_CARDINALITY_COLS)
        self.skewed_numerical_cols = skewed_numerical_cols or list(SKEWED_NUMERICAL_COLS)
        self.plain_numerical_cols = plain_numerical_cols or list(PLAIN_NUMERICAL_COLS)
        self.embedding_dim_max = embedding_dim_max
        self.missing_threshold_warning = missing_threshold_warning

        # Fitted vocabulary mappings: col -> {category: index}, where index 0 is <UNK>
        self.category_maps_: Dict[str, Dict[str, int]] = {}
        # Calculated embedding dimensions per column: col -> dim
        self.embedding_dims_: Dict[str, int] = {}
        # Total vocabulary sizes per column (including <UNK>): col -> size
        self.vocab_sizes_: Dict[str, int] = {}
        # Numerical medians for imputation
        self.numerical_medians_: Dict[str, float] = {}
        # Scalers
        self.skewed_scaler_: Optional[StandardScaler] = None
        self.plain_scaler_: Optional[StandardScaler] = None
        # One-hot encoder
        self.one_hot_encoder_: Optional[OneHotEncoder] = None
        # Output feature dimension tracking for downstream network instantiation
        self.num_one_hot_features_: int = 0
        self.num_numerical_features_: int = 0
        self.is_fitted_: bool = False

    def _check_missingness(self, df: pd.DataFrame) -> None:
        """Check column missingness and surface data quality warnings for officials/judges."""
        for col in df.columns:
            missing_ratio = df[col].isna().mean()
            # Surfacing missingness alerts directly reflects e-governance data audit standards
            if missing_ratio > self.missing_threshold_warning:
                logger.warning(
                    f"DATA QUALITY WARNING: Column '{col}' has {missing_ratio * 100:.1f}% missing values, "
                    f"exceeding the {self.missing_threshold_warning * 100:.0f}% tolerance threshold! "
                    f"Imputation will be applied, but administrative data audit is recommended."
                )

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "FeatureProcessor":
        """Fit vocabulary maps, imputation medians, one-hot encoders, and scalers on training data.

        Args:
            X: Training pandas DataFrame containing structured case attributes.
            y: Ignored for unsupervised feature processing.

        Returns:
            Fitted instance of FeatureProcessor.
        """
        df = X.copy()
        self._check_missingness(df)

        # 1. Fit High-Cardinality Categoricals (PIN Code, District)
        self.category_maps_ = {}
        self.embedding_dims_ = {}
        self.vocab_sizes_ = {}
        for col in self.high_cardinality_cols:
            if col in df.columns:
                series = df[col].fillna("Unknown").astype(str)
                unique_categories = sorted(series.unique())
                # Index 0 is reserved for unseen/unknown categories (<UNK>)
                mapping = {cat: idx + 1 for idx, cat in enumerate(unique_categories)}
                mapping["<UNK>"] = 0
                self.category_maps_[col] = mapping
                n_cats = len(unique_categories) + 1  # include UNK
                self.vocab_sizes_[col] = n_cats

                # Rule-of-thumb embedding dimension: min(50, (n_cats + 1) // 2) capped at embedding_dim_max
                # Using dense embeddings captures spatial/administrative similarities without one-hot sparsity
                calc_dim = (n_cats + 1) // 2
                self.embedding_dims_[col] = max(2, min(self.embedding_dim_max, calc_dim))

        # 2. Fit Low-Cardinality Categoricals (Land Type, RFCTLARR Stage)
        for col in self.low_cardinality_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str)

        # Explicitly configure known categories for RFCTLARR Act statutory stages
        # to ensure stages preserve statutory meaning across any arbitrary batch
        categories_list = []
        for col in self.low_cardinality_cols:
            if col == "rfctlarr_stage":
                # Ensure canonical stages are always represented in order
                known_stages = list(RFCTLARR_STAGES)
                if "Unknown" not in known_stages:
                    known_stages.append("Unknown")
                categories_list.append(known_stages)
            elif col in df.columns:
                cats = sorted(df[col].unique().tolist())
                if "Unknown" not in cats:
                    cats.append("Unknown")
                categories_list.append(cats)
            else:
                categories_list.append(["Unknown"])

        self.one_hot_encoder_ = OneHotEncoder(
            categories=categories_list,
            handle_unknown="ignore",
            sparse_output=False,
        )
        self.one_hot_encoder_.fit(df[self.low_cardinality_cols])
        self.num_one_hot_features_ = int(sum(len(c) for c in self.one_hot_encoder_.categories_))

        # 3. Fit Numerical Features (Medians, Log1p, StandardScaler)
        self.numerical_medians_ = {}
        for col in self.skewed_numerical_cols + self.plain_numerical_cols:
            if col in df.columns:
                median_val = float(df[col].median(skipna=True))
                # Fallback to zero if entire column is NaN
                self.numerical_medians_[col] = 0.0 if np.isnan(median_val) else median_val
            else:
                self.numerical_medians_[col] = 0.0

        # Skewed features: log1p then standard scale
        # Monetary amounts (crores/lakhs) and delay days have extreme right tails;
        # log1p stabilizes variance and compresses outliers before entering gradients
        skewed_data = df[self.skewed_numerical_cols].fillna(
            {col: self.numerical_medians_[col] for col in self.skewed_numerical_cols}
        ).values.astype(np.float32)
        skewed_log = np.log1p(np.maximum(0.0, skewed_data))
        self.skewed_scaler_ = StandardScaler()
        self.skewed_scaler_.fit(skewed_log)

        # Plain numericals: standard scale directly
        plain_data = df[self.plain_numerical_cols].fillna(
            {col: self.numerical_medians_[col] for col in self.plain_numerical_cols}
        ).values.astype(np.float32)
        self.plain_scaler_ = StandardScaler()
        self.plain_scaler_.fit(plain_data)

        self.num_numerical_features_ = len(self.skewed_numerical_cols) + len(self.plain_numerical_cols)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> ProcessedBatch:
        """Transform input DataFrame into a PyTorch-ready ProcessedBatch.

        Args:
            X: Input DataFrame of land acquisition cases.

        Returns:
            ProcessedBatch containing embedding IDs, one-hot vectors, and scaled continuous features.
        """
        if not self.is_fitted_:
            raise RuntimeError("FeatureProcessor must be fitted before calling transform().")

        df = X.copy()
        self._check_missingness(df)
        batch_size = len(df)

        # 1. Transform High-Cardinality Categoricals into Embedding Index Tensors
        embedding_indices: Dict[str, torch.Tensor] = {}
        for col in self.high_cardinality_cols:
            if col in df.columns:
                series = df[col].fillna("Unknown").astype(str)
                mapping = self.category_maps_.get(col, {})
                # Unseen categories or NaNs map cleanly to index 0 (<UNK>)
                indices = [mapping.get(val, 0) for val in series]
            else:
                indices = [0] * batch_size
            embedding_indices[col] = torch.tensor(indices, dtype=torch.long)

        # 2. Transform Low-Cardinality Categoricals into One-Hot Tensor
        for col in self.low_cardinality_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str)
            else:
                df[col] = "Unknown"
        one_hot_arr = self.one_hot_encoder_.transform(df[self.low_cardinality_cols])
        one_hot_tensor = torch.tensor(one_hot_arr, dtype=torch.float32)

        # 3. Transform Skewed and Plain Numericals
        skewed_data = df[self.skewed_numerical_cols].fillna(
            {col: self.numerical_medians_[col] for col in self.skewed_numerical_cols}
        ).values.astype(np.float32)
        skewed_log = np.log1p(np.maximum(0.0, skewed_data))
        skewed_scaled = self.skewed_scaler_.transform(skewed_log)

        plain_data = df[self.plain_numerical_cols].fillna(
            {col: self.numerical_medians_[col] for col in self.plain_numerical_cols}
        ).values.astype(np.float32)
        plain_scaled = self.plain_scaler_.transform(plain_data)

        numerical_arr = np.concatenate([skewed_scaled, plain_scaled], axis=1)
        numerical_tensor = torch.tensor(numerical_arr, dtype=torch.float32)

        # 4. Extract Targets if present (Delay Probability & Delay Duration Months)
        target_prob_tensor: Optional[torch.Tensor] = None
        target_months_tensor: Optional[torch.Tensor] = None
        if "delay_probability" in df.columns:
            target_prob_tensor = torch.tensor(
                df["delay_probability"].values.astype(np.float32).reshape(-1, 1),
                dtype=torch.float32,
            )
        elif "is_delayed" in df.columns:
            target_prob_tensor = torch.tensor(
                df["is_delayed"].values.astype(np.float32).reshape(-1, 1),
                dtype=torch.float32,
            )

        if "delay_months" in df.columns:
            # Mask or fill NaN delay months with 0.0 (masked in loss function for non-delayed cases)
            months_arr = df["delay_months"].fillna(0.0).values.astype(np.float32).reshape(-1, 1)
            target_months_tensor = torch.tensor(months_arr, dtype=torch.float32)

        # Extract case IDs if available for traceable audit reports
        case_ids = df["case_id"].astype(str).tolist() if "case_id" in df.columns else None

        return ProcessedBatch(
            embedding_indices=embedding_indices,
            one_hot_features=one_hot_tensor,
            numerical_features=numerical_tensor,
            target_prob=target_prob_tensor,
            target_months=target_months_tensor,
            case_ids=case_ids,
        )

    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> ProcessedBatch:
        """Fit to data, then transform it into a ProcessedBatch."""
        return self.fit(X, y).transform(X)

    def save(self, path: Union[str, Any]) -> None:
        """Serialize and persist the fitted FeatureProcessor to disk using joblib."""
        joblib.dump(self, path)
        logger.info(f"Fitted FeatureProcessor successfully persisted to: {path}")

    @classmethod
    def load(cls, path: Union[str, Any]) -> "FeatureProcessor":
        """Load a persisted FeatureProcessor instance from disk."""
        processor = joblib.load(path)
        if not isinstance(processor, cls):
            raise TypeError(f"Loaded object is of type {type(processor)}, expected {cls}.")
        logger.info(f"FeatureProcessor successfully restored from: {path}")
        return processor
