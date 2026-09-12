"""Configuration module for Land Acquisition Delay Prediction Dual-Head Neural Network.

This module defines all hyperparameters and configuration dataclasses, eliminating magic
numbers across the codebase. Every choice includes rationale comments suitable for defense
to Smart India Hackathon (SIH26017) technical evaluators and MoRD officials.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path

# Canonical RFCTLARR Act (Right to Fair Compensation and Transparency in Land Acquisition,
# Rehabilitation and Resettlement Act, 2013) statutory stages in sequence.
RFCTLARR_STAGES: List[str] = [
    "Preliminary Notification (Sec 11)",
    "Survey & SIA (Social Impact Assessment)",
    "Draft Declaration (Sec 19)",
    "Final Declaration",
    "Award Enquiry (Sec 21-23)",
    "Award Declaration",
    "Compensation Disbursement",
    "Possession",
]

# High-cardinality geographic/administrative categorical columns requiring entity embeddings
HIGH_CARDINALITY_COLS: List[str] = ["pin_code", "district"]

# Low-cardinality categorical columns suitable for one-hot encoding
LOW_CARDINALITY_COLS: List[str] = ["land_type", "rfctlarr_stage"]

# Numerical columns with heavy right-skew (monetary compensation, area, days elapsed)
# Transformed with log1p followed by standard scaling
SKEWED_NUMERICAL_COLS: List[str] = [
    "land_parcel_area_acres",
    "compensation_amount_lakhs",
    "days_since_notification",
]

# Numerical columns with bounded or near-normal distributions
# Scaled directly using standard scaling
PLAIN_NUMERICAL_COLS: List[str] = [
    "number_of_displaced_families",
    "litigation_cases_count",
    "social_vulnerability_index",
    "official_processing_speed_score",
]

# All numerical columns combined
ALL_NUMERICAL_COLS: List[str] = SKEWED_NUMERICAL_COLS + PLAIN_NUMERICAL_COLS


@dataclass
class ModelConfig:
    """Hyperparameters defining the Wide & Deep Dual-Head Neural Network."""

    # Maximum embedding dimension cap (prevents parameter explosion on 1000s of PINs)
    embedding_dim_max: int = 12
    # Post-embedding dropout to prevent co-adaptation in spatial lookup tables
    embedding_dropout: float = 0.1
    # Hidden layer tapering dimensions: 64 -> 32 -> 16 to funnel multi-feature interactions
    hidden_dims: Tuple[int, ...] = (64, 32, 16)
    # Tapering dropout matching depth to prevent over-regularizing smaller bottleneck layers
    dropouts: Tuple[float, ...] = (0.3, 0.2, 0.1)
    # Wide component projection dimension (memorization path for linear interactions)
    wide_output_dim: int = 16
    # Batch normalization momentum for stabilizing tabular mini-batch distribution drift
    batch_norm_momentum: float = 0.1


@dataclass
class TrainingConfig:
    """Hyperparameters and runtime settings for the training pipeline."""

    # Mini-batch size chosen for stable gradients on 5k-50k tabular rows without GPU memory waste
    batch_size: int = 64
    # Learning rate for AdamW
    learning_rate: float = 1e-3
    # Weight decay (L2 penalty) decoupled from gradient updates to regularize dense weights
    weight_decay: float = 1e-4
    # Maximum training epochs
    epochs: int = 50
    # Early stopping patience: halt if validation combined loss doesn't improve for 15 epochs
    early_stopping_patience: int = 15
    # Cosine annealing warm restarts cycle parameters (T_0=10 initial cycle, T_mult=2 doubling)
    scheduler_t0: int = 10
    scheduler_tmult: int = 2
    scheduler_eta_min: float = 1e-5
    # Focal loss parameters for class imbalance (gamma=2.0 focuses on hard cases, alpha=0.25 scales minority)
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    # Huber loss delta/beta threshold where regression transitions from quadratic to linear
    huber_beta: float = 1.0
    # Multi-task loss weights: lambda1 balances classification, lambda2 balances duration regression
    lambda_focal: float = 1.0
    lambda_huber: float = 1.0
    # Train / validation / test split ratios
    val_size: float = 0.15
    test_size: float = 0.15
    # Deterministic seed for reproducible evaluation across runs
    seed: int = 42
    # Output file paths
    data_path: str = "data/land_acquisition_cases.csv"
    model_save_path: str = "model.pt"
    processor_save_path: str = "processor.joblib"
    training_log_path: str = "training_log.csv"


@dataclass
class ContinualLearningConfig:
    """Configuration for Human-In-The-Loop (HITL) retraining, replay buffer, and governance gating."""

    # Experience replay ratio: 1 historical sample for every 4 new incoming samples (1:4 ratio)
    replay_old_ratio: float = 0.2  # 1 / (1 + 4) = 0.20
    # Elastic Weight Consolidation quadratic penalty weight (higher = stronger memory retention)
    lambda_ewc: float = 400.0
    # Retraining epochs on incremental batches (fewer epochs to prevent overfitting small sets)
    retrain_epochs: int = 15
    retrain_learning_rate: float = 5e-4
    # Maximum allowable regression on any protected slice before automated rejection (2%)
    max_slice_regression_pct: float = 0.02
    # Minimum required overall AUROC improvement (or non-degradation) over production
    min_auroc_tolerance: float = 0.00
    # Protected slices monitored during governance staging to prevent localized degradation
    protected_slices: List[str] = field(
        default_factory=lambda: ["land_type", "state"]
    )
    # Staging and model versioning paths
    registry_path: str = "model_registry.json"
    models_dir: str = "models"
    staging_dir: str = "models/staging"
    production_dir: str = "models/production"
    pending_approval_dir: str = "models/pending_human_approval"


# Root application directories
BASE_DIR = Path(__file__).resolve().parent
