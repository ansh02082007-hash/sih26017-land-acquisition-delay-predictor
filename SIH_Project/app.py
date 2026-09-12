"""FastAPI Backend Server for Land Acquisition Delay Prediction.

Run with:
    uvicorn app:app --reload --port 8000
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from predictor import DelayPredictor
import history_store

app = FastAPI(
    title="Land Acquisition Delay Prediction API (SIH26017)",
    description="Dual-Head Neural Network predicting delay probability and duration under RFCTLARR Act 2013.",
    version="1.0.0",
)

# Global predictor instance
predictor: Optional[DelayPredictor] = None


@app.on_event("startup")
def startup_event():
    """Load model and feature processor on server startup."""
    global predictor
    try:
        predictor = DelayPredictor(model_path="model.pt", processor_path="processor.joblib")
    except Exception as e:
        print(f"Warning: Model weights not found. Please train model first: {e}")


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the interactive e-governance prediction and SHAP explainability dashboard."""
    template_path = Path(__file__).resolve().parent / "templates" / "index.html"
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h2>Dashboard template not found</h2>", status_code=404)


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "system": "SIH26017 Land Delay Prediction",
        "model_loaded": predictor is not None,
    }


class CaseInput(BaseModel):
    """Input payload representing a rural infrastructure land acquisition case."""

    case_id: str = Field(default="LA-2024-001", description="Unique identifier for the project case.")
    pin_code: str = Field(default="500001", description="6-digit Indian postal PIN code.")
    district: str = Field(default="Hyderabad", description="District name.")
    land_type: str = Field(default="Agricultural", description="Land classification: Agricultural, Forest, Commercial, etc.")
    rfctlarr_stage: str = Field(
        default="Preliminary Notification (Sec 11)",
        description="Statutory stage under RFCTLARR Act 2013.",
    )
    land_parcel_area_acres: float = Field(default=15.0, description="Acquisition parcel area in acres.")
    compensation_amount_lakhs: float = Field(default=50.0, description="Total compensation budget in INR Lakhs.")
    days_since_notification: float = Field(default=120.0, description="Days elapsed since Section 11 notice.")
    number_of_displaced_families: float = Field(default=5.0, description="Number of project affected families.")
    litigation_cases_count: float = Field(default=0.0, description="Active court cases or stay orders.")
    social_vulnerability_index: float = Field(default=0.45, description="Socio-economic vulnerability score (0 to 1).")
    official_processing_speed_score: float = Field(default=6.0, description="Administrative clearance velocity (1 to 10).")


@app.post("/predict")
def predict_case(case: CaseInput) -> Dict[str, Any]:
    """Predict delay probability and expected duration in months. Auto-saves to history."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model predictor is not initialized. Run training first.")
    case_data = case.dict()
    result = predictor.predict(case_data)
    # Persist to history so officials can later record the true outcome
    record_id = history_store.save_prediction(case_data, result)
    result["record_id"] = record_id
    return result


@app.post("/explain")
def explain_case(case: CaseInput) -> Dict[str, Any]:
    """Explain case predictions using SHAP attributions for both heads."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model predictor is not initialized. Run training first.")
    return predictor.explain(case.dict())


@app.get("/network/architecture")
def get_architecture() -> Dict[str, Any]:
    """Return structural topology and layer status of the Wide & Deep network."""
    return {
        "model_name": "DelayPredictionNet (Wide & Deep Dual-Head)",
        "inputs": {
            "entity_embeddings": [
                {"name": "PIN Code", "type": "nn.Embedding", "vocab": 30, "dim": 12, "frozen_on_retrain": True},
                {"name": "District", "type": "nn.Embedding", "vocab": 30, "dim": 12, "frozen_on_retrain": True},
            ],
            "embedding_dropout": 0.1,
            "one_hot_features": {"count": 14, "categories": ["Land Type (6)", "RFCTLARR Statutory Stages (8)"]},
            "numerical_features": {"count": 7, "skewed": 3, "standard": 4},
        },
        "wide_branch": {
            "name": "Linear Memorization Path",
            "in_features": 21,
            "out_features": 16,
            "purpose": "Direct linear memorization without non-linear distortion",
        },
        "deep_branch": {
            "name": "Tapering Non-Linear Generalization Tower",
            "layers": [
                {"layer": "Deep Layer 1", "in": 45, "out": 64, "act": "GELU", "bn": True, "dropout": 0.3, "frozen_on_retrain": True},
                {"layer": "Deep Layer 2", "in": 64, "out": 32, "act": "GELU", "bn": True, "dropout": 0.2, "frozen_on_retrain": False},
                {"layer": "Deep Layer 3", "in": 32, "out": 16, "act": "GELU", "bn": True, "dropout": 0.1, "frozen_on_retrain": False},
            ],
            "phase2_hook": "TabNet-style dynamic sparse attention",
        },
        "fused_bottleneck": {
            "dimensions": 32,
            "composition": "Wide Output (16) + Deep Layer 3 (16)",
            "purpose": "Shared multi-task representation co-regularizing classification and duration regression",
        },
        "dual_heads": [
            {
                "head": "Head A: Delay Occurrence Classifier",
                "loss": "Focal Loss (alpha=0.25, gamma=2.0)",
                "activation": "Sigmoid",
                "output_range": "[0.0, 1.0]",
                "target": "Probability of Statutory Schedule Overrun",
            },
            {
                "head": "Head B: Delay Duration Regressor",
                "loss": "Masked Huber Loss (beta=1.0)",
                "activation": "Softplus",
                "output_range": "[0.0, +inf) Months",
                "target": "Overrun Duration in Months (Strictly Non-Negative)",
            }
        ]
    }


@app.post("/retrain/simulate")
def trigger_continual_retraining() -> Dict[str, Any]:
    """Execute live simulation of Human-in-the-Loop retraining and governance staging."""
    from retrain import simulate_continual_learning
    return simulate_continual_learning(epochs=5, lambda_ewc=400.0)


# ---------------------------------------------------------------------------
# History & Feedback Endpoints
# ---------------------------------------------------------------------------

@app.get("/history")
def list_history() -> List[Dict[str, Any]]:
    """Return all prediction records (newest first)."""
    return history_store.get_all_records()


@app.get("/history/stats")
def history_stats() -> Dict[str, Any]:
    """Return aggregate statistics over all prediction history."""
    return history_store.stats()


class OutcomeInput(BaseModel):
    """Payload for recording a real-world land acquisition outcome."""
    actual_delay_months: float = Field(
        description="Actual total delay in months. Use 0 for on-time acquisition."
    )
    acquisition_completed: bool = Field(
        default=True,
        description="Set True when the acquisition process has fully concluded.",
    )
    notes: str = Field(
        default="",
        description="Optional free-text notes from the field/revenue officer.",
    )


@app.post("/history/{record_id}/outcome")
def record_outcome(record_id: str, outcome: OutcomeInput) -> Dict[str, Any]:
    """Attach the real-world outcome to a previous prediction record.

    This creates a labelled training sample for future HITL continual retraining.
    """
    try:
        updated = history_store.add_outcome(
            record_id=record_id,
            actual_delay_months=outcome.actual_delay_months,
            acquisition_completed=outcome.acquisition_completed,
            notes=outcome.notes,
        )
        return {"status": "outcome_recorded", "record": updated}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found in history.")


@app.get("/history/feedback-pairs")
def list_feedback_pairs() -> List[Dict[str, Any]]:
    """Return only records that have ground-truth outcomes — ready for retraining."""
    return history_store.get_feedback_training_pairs()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

