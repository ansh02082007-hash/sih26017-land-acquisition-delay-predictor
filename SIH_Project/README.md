# Wide & Deep Dual-Head Neural Network for Land Acquisition Delay Prediction

### Smart India Hackathon (SIH26017) — Ministry of Rural Development

A production-grade, PyTorch-based **Wide & Deep Dual-Head Neural Network** designed to provide early warnings of land acquisition delays for rural infrastructure projects under the **RFCTLARR Act, 2013** (*Right to Fair Compensation and Transparency in Land Acquisition, Rehabilitation and Resettlement Act*).

The system simultaneously predicts:
1. **Delay Probability** (Binary classification head via custom Focal Loss)
2. **Delay Duration in Months** (Continuous regression head via Huber Loss, strictly masked to delayed cases)

---

## Architecture Overview

```
                          ┌───────────────────────────┐
                          │ Structured Case Attributes│
                          └─────────────┬─────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           │ High-Cardinality           │ Low-Cardinality Categoricals│ Continuous Features
           │ (PIN Code, District)       │ (Land Type, RFCTLARR Stage) │ (Compensation, Area, Days, etc.)
           ▼                            ▼                             ▼
   ┌───────────────┐            ┌───────────────┐             ┌───────────────┐
   │Entity Embeddings│          │One-Hot Encoder│             │Log1p + Scaler │
   │ (dim ≤ 12)    │            └───────┬───────┘             └───────┬───────┘
   └───────┬───────┘                    │                             │
           │ Dropout (0.1)              │                             │
           ▼                            ▼                             ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                          Wide & Deep Paths                               │
   │                                                                          │
   │   [ Wide Path (Memorization) ]          [ Deep Path (Generalization) ]   │
   │   Linear(OneHot + Numericals, 16)       Concat[Embeddings, OneHot, Num]  │
   │                                                       │                  │
   │                                            FC1 (64) + BN + GELU + Drop0.3│
   │                                                       │                  │
   │                                            FC2 (32) + BN + GELU + Drop0.2│
   │                                                       │                  │
   │                                            FC3 (16) + BN + GELU + Drop0.1│
   └────────────────────────────┬──────────────────────────┬──────────────────┘
                                │                          │
                                └────────────┬─────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │   Fused Latent Vector     │
                               │        (32 dims)          │
                               └─────────────┬─────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
              [ Head A: Binary ]                            [ Head B: Regression ]
         Linear(32, 1) -> Sigmoid                      Linear(32, 1) -> Softplus
                      │                                             │
                      ▼                                             ▼
             Delay Probability                              Delay Duration (Months)
                 p ∈ [0, 1]                                       m ≥ 0.0
```

---

## Why Neural Network Over Gradient-Boosted Trees (XGBoost / LightGBM)

A frequent question from technical evaluators and hackathon judges is: *"Why use a PyTorch neural network instead of XGBoost or LightGBM for tabular data?"*

This architecture visibly earns its neural network complexity through four critical design capabilities that tree-based ensembles cannot cleanly or natively support:

### 1. Multi-Task Dual-Head Learning with Shared Representations
- **The Problem in GBTs**: Predicting delay probability and delay duration with XGBoost requires training two completely independent tree ensembles ($M_1$ and $M_2$). $M_1$ learns split thresholds oblivious to duration severity, while $M_2$ cannot leverage the latent boundary features discovered by $M_1$.
- **The Neural Advantage**: The dual-head neural network branches from a single **fused representation bottleneck** (32 dimensions). The shared hidden layers are co-regularized by gradients from both the Focal Loss and Huber Loss, forcing the latent representations to isolate factors that drive both delay likelihood and delay overrun (such as environmental forest clearances or active court stays).

### 2. Dense Entity Embeddings for High-Cardinality Geographic Coordinates
- **The Problem in GBTs**: Indian postal PIN codes and districts represent hundreds of discrete geographic regions. One-hot encoding creates extreme sparsity ($>1000$ columns) that dilutes split candidate histograms. Frequency/target encoding suffers from severe target leakage and cannot generalize to unseen districts.
- **The Neural Advantage**: Learned `nn.Embedding` layers project PIN codes and districts into dense continuous geometries ($\text{dim} \le 12$). Spatial and administrative proximity cluster together in embedding space. Index 0 is reserved for `<UNK>`, providing out-of-the-box robustness to unseen postal codes without re-indexing.

### 3. Human-In-The-Loop Continual Learning with Elastic Weight Consolidation (EWC)
- **The Problem in GBTs**: When new quarterly acquisition data arrives, gradient-boosted trees cannot be incrementally fine-tuned without either suffering catastrophic forgetting or requiring retraining on the entire multi-year historical dataset from scratch.
- **The Neural Advantage**: Neural networks support exact second-order regularization. Using **Elastic Weight Consolidation (Kirkpatrick et al., 2017)** and **Early-Layer Freezing**, our system Retrains only Layers 2–3 and the heads while preserving the Fisher Information Matrix of historical parameters. Historical memory is retained without storing millions of historical records in RAM.

### 4. Direct Phase-2 Hook for TabNet Dynamic Sparse Attention
- GBT splits are static and greedy across the whole tree. Our neural architecture includes an explicit `# PHASE 2 HOOK: TabNet-style attention` allowing instance-wise dynamic feature masks (sparsemax/entmax) to be plugged directly before the deep layers in Phase 2.

---

## Mathematical Formulations

### 1. Custom Focal Loss (Classification Head)
Standard Cross-Entropy is overwhelmed by common non-delayed or easily classified projects. We implement numerically stable Focal Loss from scratch:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where:
- $p_t = p$ if $y=1$ else $(1 - p)$
- $\alpha_t = \alpha = 0.25$ for delayed class, $1 - \alpha = 0.75$ for non-delayed
- $\gamma = 2.0$ dynamically downweights well-classified instances

### 2. Masked Huber Loss (Regression Head)
Delay duration in months is only valid when statutory delay actually occurs ($y_{\text{target}} = 1$). Computing loss on non-delayed instances would falsely force the duration head to learn an artificial zero-spike.

$$\mathcal{L}_{\text{Huber}}(m, y) = \begin{cases} 
0.5 (m - y)^2 & \text{if } |m - y| \le 1.0 \\
|m - y| - 0.5 & \text{otherwise}
\end{cases}$$

$$\mathcal{L}_{\text{Huber, masked}} = \frac{1}{\sum_{i} \mathbb{I}(y_i = 1)} \sum_{i: y_i = 1} \mathcal{L}_{\text{Huber}}(m_i, y_{m, i})$$

### 3. Multi-Task Combined Loss
$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{Focal}} + \lambda_2 \mathcal{L}_{\text{Huber, masked}}$$

---

## Project Structure

```
land_delay_nn/
├── config.py                 # Dataclasses: ModelConfig, TrainingConfig, ContinualLearningConfig
├── preprocessing.py          # FeatureProcessor & ProcessedBatch (sklearn Pipeline compatible)
├── model.py                  # DelayPredictionNet (Wide & Deep + Dual Heads)
├── losses.py                 # Custom FocalLoss and masked MultiTaskDelayLoss
├── train.py                  # Stratified training loop, Cosine Annealing, CSV logging
├── retrain.py                # Experience Replay (1:4), EWC, Governance Staging
├── explain.py                # DualHeadExplainer with SHAP (DeepExplainer + Gradient fallback)
├── predictor.py              # FastAPI-ready inference engine
├── model_registry.json       # Versioned audit ledger for models and approvals
├── pytest.ini                # Pytest configuration restricting tests to tests/
├── requirements.txt          # Python package requirements
├── data/
│   └── synthetic_generator.py # CAG/RTI-anchored realistic data synthesizer
└── tests/
    ├── test_preprocessing.py # Tests encoding, imputation, >15% warning, joblib serialization
    ├── test_model_shapes.py  # Tests tensor dimensions, ranges [0, 1] & m>=0, layer freezing
    ├── test_losses.py        # Tests Focal properties, Huber masking, weight scaling
    └── test_retrain_gating.py# Tests replay ratio, EWC penalty, governance gating & promotion
```

---

## Installation & Quickstart

### 1. Environment Setup
```bash
# Clone and enter directory
cd d:/SIH_Project

# Create Python 3.12 virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset (CAG & RTI Anchored)
Generates 5,000 realistic cases with calibrated delay rates (~45%) and realistic duration tails:
```bash
python data/synthetic_generator.py --samples 5000
```

### 3. Train the Model
Trains with stratified 70/15/15 split, Cosine Annealing with Warm Restarts, early stopping, and logs epoch curves to `training_log.csv`:
```bash
python train.py --config config.py --epochs 25
```

### 4. Run Unit Test Suite
Executes 16 tests verifying shapes, losses, preprocessing, EWC, and governance gating:
```bash
pytest tests/ -v
```

---

## Human-in-the-Loop (HITL) Continual Learning

Retraining never directly overwrites production weights in automated cron jobs. All retraining passes through an auditable governance pipeline:

```
[ New Quarterly Data ]
         │
         ▼
[ Experience Replay Buffer ] ─── Samples 1 Old : 4 New records
         │
         ▼
[ Candidate Retraining ]     ─── Embeddings & Layer 1 Frozen + EWC Penalty
         │
         ▼
[ Automated Validation Gate ]
   ├─ Overall AUROC & MAE Non-Degradation
   └─ Protected Sub-Slice Check: Maximum 2% allowable regression on any State/Land Type
         │
         ├─── Fails Gate ───► Status: REJECTED (Logged in model_registry.json, Prod untouched)
         │
         └─── Passes Gate ──► Status: PENDING_HUMAN_APPROVAL
                                  │
                                  ▼
                         [ Markdown Diff Report Generated ]
                                  │
                                  ▼
                         [ Explicit Administrative Sign-off ]
                         `python retrain.py --approve <ver> --approver "Name"`
                                  │
                                  ▼
                         Status: PRODUCTION (Activated in models/production/)
```

### Running a Retraining Cycle & Promotion:
```bash
# Execute candidate retraining cycle
python retrain.py --test-cycle

# Review audit report in models/pending_human_approval/diff_report_v1.md
# Approve candidate and promote to production
python retrain.py --approve 1 --approver "Dr. R. Sharma (Director, MoRD)"
```

---

## SHAP Explainability & FastAPI Serving

### FastAPI Integration Example:
```python
from fastapi import FastAPI
from predictor import DelayPredictor

app = FastAPI(title="MoRD Land Acquisition Delay Prediction API")
predictor = DelayPredictor("model.pt", "processor.joblib")

@app.post("/predict")
def predict_delay(case: dict):
    """Returns delay probability, duration, confidence band, and risk level."""
    return predictor.predict(case)

@app.post("/explain")
def explain_delay(case: dict):
    """Returns separate SHAP attributions for Probability and Duration."""
    return predictor.explain(case)
```

### Sample Output from `/predict`:
```json
{
  "case_id": "LA-2024-00142",
  "delay_probability": 0.6434,
  "is_delay_predicted": true,
  "predicted_delay_months": 30.9,
  "confidence_band": {
    "lower_months": 26.3,
    "upper_months": 35.5
  },
  "stage": "Award Enquiry (Sec 21-23)",
  "risk_level": "HIGH"
}
```

### Sample Output from `/explain`:
```json
{
  "top_risk_drivers": [
    {"feature": "litigation_cases_count", "value": "2", "attribution": 0.0677, "direction": "INCREASES_RISK"},
    {"feature": "number_of_displaced_families", "value": "25", "attribution": 0.0647, "direction": "INCREASES_RISK"},
    {"feature": "land_type", "value": "Forest", "attribution": 0.0370, "direction": "INCREASES_RISK"}
  ],
  "top_duration_drivers": [
    {"feature": "litigation_cases_count", "value": "2", "attribution": 10.5295, "direction": "INCREASES_RISK"},
    {"feature": "land_type", "value": "Forest", "attribution": 3.3252, "direction": "INCREASES_RISK"}
  ]
}
```

---

## Canonical RFCTLARR Act Statutory Stages

The 8 statutory stages defined in Section 11 through 23 of the RFCTLARR Act, 2013:
1. `Preliminary Notification (Sec 11)`
2. `Survey & SIA (Social Impact Assessment)`
3. `Draft Declaration (Sec 19)`
4. `Final Declaration`
5. `Award Enquiry (Sec 21-23)`
6. `Award Declaration`
7. `Compensation Disbursement`
8. `Possession`
