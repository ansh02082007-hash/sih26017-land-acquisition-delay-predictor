"""Inference wrapper for Land Acquisition Delay Prediction.

This module provides the DelayPredictor class, designed for plug-and-play integration
into FastAPI backends or batch inference pipelines. It exposes clean predict() and explain()
APIs returning structured, JSON-serializable payloads for frontend dashboards and e-governance audits.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch

from preprocessing import FeatureProcessor, ProcessedBatch
from model import DelayPredictionNet, ModelOutput

logger = logging.getLogger("predictor")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class DelayPredictor:
    """Production inference engine for serving land acquisition delay predictions.

    Loads the saved FeatureProcessor and DelayPredictionNet weights, executes forward
    inference in eval mode without gradient tracking, computes confidence intervals,
    and interfaces with SHAP explainers on demand.
    """

    def __init__(
        self,
        model_path: str = "model.pt",
        processor_path: str = "processor.joblib",
        device: Optional[str] = None,
    ) -> None:
        """Initialize the predictor by loading processor and neural network checkpoints.

        Args:
            model_path: Path to serialized PyTorch state_dict (.pt).
            processor_path: Path to serialized FeatureProcessor (.joblib).
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Loading DelayPredictor on compute device: {self.device}")

        # 1. Restore Feature Processor
        self.processor = FeatureProcessor.load(processor_path)

        # 2. Instantiate and Restore Model Architecture
        self.model = DelayPredictionNet(
            vocab_sizes=self.processor.vocab_sizes_,
            embedding_dims=self.processor.embedding_dims_,
            num_one_hot_features=self.processor.num_one_hot_features_,
            num_numerical_features=self.processor.num_numerical_features_,
        ).to(self.device)

        model_weights = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(model_weights)
        self.model.eval()

        self._explainer = None  # Lazy-loaded on first explain() invocation

    def _get_risk_level(self, probability: float) -> str:
        """Assign standardized administrative risk tier to a case."""
        if probability >= 0.75:
            return "CRITICAL"
        elif probability >= 0.50:
            return "HIGH"
        elif probability >= 0.30:
            return "MEDIUM"
        return "LOW"

    @torch.no_grad()
    def predict(self, case_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """Generate delay probability, delay duration, and risk tier for a given project case.

        Args:
            case_data: Dictionary of case attributes or a 1-row pandas DataFrame.

        Returns:
            Structured dictionary suitable for JSON serialization in FastAPI:
            {
                "case_id": "...",
                "delay_probability": 0.74,
                "is_delay_predicted": True,
                "predicted_delay_months": 18.2,
                "confidence_band": {"lower_months": 15.5, "upper_months": 20.9},
                "stage": "Draft Declaration (Sec 19)",
                "risk_level": "HIGH"
            }
        """
        if isinstance(case_data, dict):
            df = pd.DataFrame([case_data])
        else:
            df = case_data.copy()

        # Transform inputs via fitted FeatureProcessor
        batch: ProcessedBatch = self.processor.transform(df)
        batch = batch.to(self.device)

        # Run forward inference
        output: ModelOutput = self.model(batch)

        prob = float(output.delay_probability[0, 0].item())
        months = float(output.delay_months[0, 0].item())

        case_id = str(df["case_id"].iloc[0]) if "case_id" in df.columns else "CASE_001"
        stage = str(df["rfctlarr_stage"].iloc[0]) if "rfctlarr_stage" in df.columns else "Unknown"

        # Regression confidence band heuristic (± 15% estimated standard deviation)
        # Reflects natural variance in statutory settlement timeframes
        lower_bound = max(0.0, round(months * 0.85, 1))
        upper_bound = round(months * 1.15, 1)

        return {
            "case_id": case_id,
            "delay_probability": round(prob, 4),
            "is_delay_predicted": bool(prob >= 0.50),
            "predicted_delay_months": round(months, 1) if prob >= 0.35 else 0.0,
            "raw_duration_estimate_months": round(months, 1),
            "confidence_band": {
                "lower_months": lower_bound if prob >= 0.35 else 0.0,
                "upper_months": upper_bound if prob >= 0.35 else 0.0,
            },
            "stage": stage,
            "risk_level": self._get_risk_level(prob),
        }

    def explain(
        self,
        case_data: Union[Dict[str, Any], pd.DataFrame],
        background_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Generate SHAP per-feature attribution drivers for Head A and Head B.

        Args:
            case_data: Single case dictionary or DataFrame.
            background_df: Optional reference background distribution.

        Returns:
            JSON-serializable explanation payload with top positive and negative drivers.
        """
        if isinstance(case_data, dict):
            df = pd.DataFrame([case_data])
        else:
            df = case_data.copy()

        # Lazy initialize SHAP explainer
        if self._explainer is None:
            from explain import DualHeadExplainer
            if background_df is None:
                # Load baseline reference dataset if not passed
                from config import TrainingConfig
                bg_path = Path(TrainingConfig.data_path)
                if bg_path.exists():
                    background_df = pd.read_csv(bg_path).head(100)
                else:
                    from data.synthetic_generator import generate_synthetic_dataset
                    background_df = generate_synthetic_dataset(n_samples=100)
            self._explainer = DualHeadExplainer(self.model, self.processor, background_df)

        return self._explainer.explain_case(df)
