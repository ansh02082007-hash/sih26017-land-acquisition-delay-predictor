"""Explainability module for Land Acquisition Delay Prediction using SHAP.

This module provides model-agnostic and gradient-based SHAP (SHapley Additive exPlanations)
for both the Delay Probability (Head A) and Delay Duration (Head B) heads separately.
It enables e-governance auditability: government officials can inspect exactly which factors
(e.g., environmental clearance, compensation disputes, litigation) caused a delay prediction.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for server and CLI execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

from preprocessing import FeatureProcessor, ProcessedBatch
from model import DelayPredictionNet, ModelOutput

logger = logging.getLogger("explain")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ModelHeadContinuousWrapper(nn.Module):
    """Wrapper exposing continuous input tensors for SHAP DeepExplainer and GradientExplainer.

    Why this wrapper is required:
    Discrete integer lookup tensors (PIN/district embedding IDs) do not have defined continuous
    input gradients in PyTorch. This wrapper takes continuous concatenated inputs:
    [dropped_embeddings, one_hot_features, numerical_features] and routes them through the wide,
    deep, and specified head (head_idx=0 for Probability, head_idx=1 for Duration).
    """

    def __init__(self, net: DelayPredictionNet, head_idx: int = 0) -> None:
        """Initialize wrapper.

        Args:
            net: Underlying DelayPredictionNet.
            head_idx: 0 for delay probability (Head A), 1 for delay months (Head B).
        """
        super().__init__()
        self.net = net
        self.head_idx = head_idx
        self.total_emb_dim = sum(net.embedding_dims.values())
        self.num_one_hot = net.num_one_hot
        self.num_num = net.num_numerical

    def forward(self, continuous_inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass over continuous input features.

        continuous_inputs: (B, total_emb_dim + num_one_hot + num_num)
        """
        emb = continuous_inputs[:, :self.total_emb_dim]
        one_hot = continuous_inputs[:, self.total_emb_dim: self.total_emb_dim + self.num_one_hot]
        numericals = continuous_inputs[:, self.total_emb_dim + self.num_one_hot:]

        # Wide path
        wide_in = torch.cat([one_hot, numericals], dim=1)
        wide_out = self.net.wide_linear(wide_in)

        # Deep path
        deep_in = continuous_inputs
        x = self.net.drop1(self.net.act1(self.net.bn1(self.net.deep_fc1(deep_in))))
        x = self.net.drop2(self.net.act2(self.net.bn2(self.net.deep_fc2(x))))
        x = self.net.drop3(self.net.act3(self.net.bn3(self.net.deep_fc3(x))))

        fused = torch.cat([wide_out, x], dim=1)

        if self.head_idx == 0:
            return self.net.sigmoid(self.net.head_classifier(fused))
        else:
            return self.net.softplus(self.net.head_regressor(fused))


class DualHeadExplainer:
    """Manages separate SHAP explanations for Delay Probability and Delay Months."""

    def __init__(
        self,
        model: DelayPredictionNet,
        processor: FeatureProcessor,
        background_df: pd.DataFrame,
        background_size: int = 30,
    ) -> None:
        """Initialize explainers with background baseline distribution."""
        self.model = model
        self.processor = processor
        self.model.eval()

        # Extract background sample
        bg_sample = background_df.head(min(background_size, len(background_df)))
        self.bg_batch = self.processor.transform(bg_sample)

        # Build feature names
        self.feature_names = self._build_feature_names()

        # Build continuous background tensor
        self.bg_continuous = self._extract_continuous_tensor(self.bg_batch)

        # Instantiate SHAP explainers for both heads
        # Try DeepExplainer; if PyTorch version or op graph conflicts, fall back gracefully to GradientExplainer
        self.explainer_prob = self._init_explainer(head_idx=0)
        self.explainer_months = self._init_explainer(head_idx=1)

    def _build_feature_names(self) -> List[str]:
        """Construct descriptive column names for all continuous input dimensions."""
        names = []
        # Embedding dimensions
        for col, dim in self.processor.embedding_dims_.items():
            for d in range(dim):
                names.append(f"{col}_emb_{d}")

        # One-hot encoded feature names
        if hasattr(self.processor.one_hot_encoder_, "get_feature_names_out"):
            ohe_names = list(self.processor.one_hot_encoder_.get_feature_names_out(self.processor.low_cardinality_cols))
            names.extend(ohe_names)
        else:
            names.extend([f"one_hot_{i}" for i in range(self.processor.num_one_hot_features_)])

        # Numerical features
        names.extend(self.processor.skewed_numerical_cols)
        names.extend(self.processor.plain_numerical_cols)
        return names

    def _extract_continuous_tensor(self, batch: ProcessedBatch) -> torch.Tensor:
        """Extract continuous feature tensor incorporating evaluated embedding vectors."""
        with torch.no_grad():
            emb_vectors = []
            for col, emb_layer in self.model.embeddings.items():
                indices = batch.embedding_indices[col]
                emb_vectors.append(emb_layer(indices))
            concat_emb = torch.cat(emb_vectors, dim=1)
            return torch.cat([concat_emb, batch.one_hot_features, batch.numerical_features], dim=1)

    def _init_explainer(self, head_idx: int) -> Any:
        """Initialize SHAP explainer with DeepExplainer and GradientExplainer fallback."""
        wrapped = ModelHeadContinuousWrapper(self.model, head_idx=head_idx)
        wrapped.eval()

        try:
            # Attempt DeepExplainer
            explainer = shap.DeepExplainer(wrapped, self.bg_continuous)
            logger.info(f"Initialized shap.DeepExplainer for Head {head_idx}.")
            return explainer
        except Exception as e:
            logger.warning(
                f"shap.DeepExplainer encountered compatibility limitation ({e}). "
                f"Falling back gracefully to shap.GradientExplainer. "
                f"GradientExplainer computes exact path gradients across the continuous embedding space."
            )
            return shap.GradientExplainer(wrapped, self.bg_continuous)

    def explain_case(
        self,
        case_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute separate per-feature attributions for Head A and Head B.

        Returns a JSON-serializable dictionary formatted for the FastAPI /explain endpoint.
        """
        batch = self.processor.transform(case_df)
        x_continuous = self._extract_continuous_tensor(batch)

        # 1. Explain Probability Head
        shap_prob = self.explainer_prob.shap_values(x_continuous)
        if isinstance(shap_prob, list):
            shap_prob_arr = shap_prob[0]
        else:
            shap_prob_arr = shap_prob
        if isinstance(shap_prob_arr, torch.Tensor):
            shap_prob_arr = shap_prob_arr.detach().cpu().numpy()
        shap_prob_arr = np.asarray(shap_prob_arr)
        if shap_prob_arr.ndim > 2:
            shap_prob_arr = shap_prob_arr.reshape(shap_prob_arr.shape[0], -1)

        # 2. Explain Duration Head
        shap_months = self.explainer_months.shap_values(x_continuous)
        if isinstance(shap_months, list):
            shap_months_arr = shap_months[0]
        else:
            shap_months_arr = shap_months
        if isinstance(shap_months_arr, torch.Tensor):
            shap_months_arr = shap_months_arr.detach().cpu().numpy()
        shap_months_arr = np.asarray(shap_months_arr)
        if shap_months_arr.ndim > 2:
            shap_months_arr = shap_months_arr.reshape(shap_months_arr.shape[0], -1)

        case_id = case_df["case_id"].iloc[0] if "case_id" in case_df.columns else "CASE_001"

        # Group embedding dimensions into single aggregate feature impact
        prob_contributions = self._aggregate_feature_attributions(shap_prob_arr[0].flatten(), case_df.iloc[0])
        months_contributions = self._aggregate_feature_attributions(shap_months_arr[0].flatten(), case_df.iloc[0])

        return {
            "case_id": case_id,
            "delay_probability_attributions": prob_contributions,
            "delay_months_attributions": months_contributions,
            "top_risk_drivers": sorted(
                prob_contributions,
                key=lambda x: abs(x["attribution"]),
                reverse=True,
            )[:5],
            "top_duration_drivers": sorted(
                months_contributions,
                key=lambda x: abs(x["attribution"]),
                reverse=True,
            )[:5],
        }

    def _aggregate_feature_attributions(
        self,
        raw_attributions: np.ndarray,
        raw_row: pd.Series,
    ) -> List[Dict[str, Any]]:
        """Aggregate high-dimensional one-hot and embedding dimensions into case-level features."""
        raw_attributions = np.asarray(raw_attributions).flatten()
        results = []
        idx = 0

        # High-cardinality embeddings: sum attribution over embedding vector
        for col, dim in self.processor.embedding_dims_.items():
            emb_attr = float(np.sum(raw_attributions[idx: idx + dim]))
            raw_val = str(raw_row.get(col, "Unknown"))
            results.append({
                "feature": col,
                "category": "High-Cardinality Entity",
                "value": raw_val,
                "attribution": round(emb_attr, 4),
                "direction": "INCREASES_RISK" if emb_attr > 0 else "DECREASES_RISK",
            })
            idx += dim

        # Low-cardinality one-hot: attribute by column
        if hasattr(self.processor.one_hot_encoder_, "categories_"):
            for col_i, col in enumerate(self.processor.low_cardinality_cols):
                cats = self.processor.one_hot_encoder_.categories_[col_i]
                n_cats = len(cats)
                col_attr = float(np.sum(raw_attributions[idx: idx + n_cats]))
                raw_val = str(raw_row.get(col, "Unknown"))
                results.append({
                    "feature": col,
                    "category": "Statutory / Classification",
                    "value": raw_val,
                    "attribution": round(col_attr, 4),
                    "direction": "INCREASES_RISK" if col_attr > 0 else "DECREASES_RISK",
                })
                idx += n_cats

        # Numerical features
        all_numericals = self.processor.skewed_numerical_cols + self.processor.plain_numerical_cols
        for col in all_numericals:
            if idx < len(raw_attributions):
                attr = float(raw_attributions[idx])
                raw_val = raw_row.get(col, 0.0)
                results.append({
                    "feature": col,
                    "category": "Numerical / Administrative",
                    "value": round(float(raw_val), 2) if isinstance(raw_val, (int, float)) else str(raw_val),
                    "attribution": round(attr, 4),
                    "direction": "INCREASES_RISK" if attr > 0 else "DECREASES_RISK",
                })
                idx += 1

        return results

    def save_waterfall_plot(self, case_df: pd.DataFrame, output_path: str = "shap_waterfall.png") -> str:
        """Render and save a dual-panel SHAP attribution plot comparing Probability vs Duration."""
        explanation = self.explain_case(case_df)
        prob_attrs = explanation["top_risk_drivers"]
        month_attrs = explanation["top_duration_drivers"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Probability drivers
        p_names = [f"{item['feature']}={item['value']}" for item in prob_attrs][::-1]
        p_vals = [item["attribution"] for item in prob_attrs][::-1]
        p_colors = ["#d9534f" if v > 0 else "#5cb85c" for v in p_vals]

        ax1.barh(p_names, p_vals, color=p_colors)
        ax1.axvline(0, color="black", linestyle="--", alpha=0.6)
        ax1.set_title("Head A: Delay Probability Drivers", fontsize=12, fontweight="bold")
        ax1.set_xlabel("SHAP Attribution (Impact on Probability)")

        # Panel 2: Duration drivers
        m_names = [f"{item['feature']}={item['value']}" for item in month_attrs][::-1]
        m_vals = [item["attribution"] for item in month_attrs][::-1]
        m_colors = ["#f0ad4e" if v > 0 else "#0275d8" for v in m_vals]

        ax2.barh(m_names, m_vals, color=m_colors)
        ax2.axvline(0, color="black", linestyle="--", alpha=0.6)
        ax2.set_title("Head B: Delay Duration Drivers (Months)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("SHAP Attribution (Impact on Months)")

        plt.suptitle(
            f"E-Governance Land Acquisition Audit — Case: {explanation['case_id']}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        logger.info(f"SHAP attribution waterfall plot saved to: {out_file}")
        return str(out_file)
