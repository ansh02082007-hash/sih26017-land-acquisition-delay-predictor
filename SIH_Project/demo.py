"""Quickstart interactive demo for Land Acquisition Delay Prediction System (SIH26017).

Run this script to test all capabilities end-to-end:
    python demo.py
"""

import json
from pathlib import Path
import pandas as pd

from predictor import DelayPredictor
from explain import DualHeadExplainer


def run_demo() -> None:
    print("=" * 70)
    print("  Smart India Hackathon (SIH26017) - Ministry of Rural Development")
    print("  Land Acquisition Delay Prediction: Wide & Deep Dual-Head Network")
    print("=" * 70)

    model_path = Path("model.pt")
    proc_path = Path("processor.joblib")

    if not model_path.exists() or not proc_path.exists():
        print("\n[!] Pre-trained weights not found. Please run training first:")
        print("    python train.py --config config.py --epochs 15\n")
        return

    # 1. Initialize Predictor
    print("\n[1/3] Loading Predictor & Restoring Pretrained Checkpoint...")
    predictor = DelayPredictor(model_path="model.pt", processor_path="processor.joblib")
    print("      [OK] Model and FeatureProcessor loaded successfully.")

    # 2. Test Real-World Acquisition Case
    print("\n[2/3] Simulating a New Land Acquisition Case...")
    test_case = {
        "case_id": "LA-2024-DEMO-01",
        "pin_code": "500001",
        "district": "Hyderabad",
        "land_type": "Forest",  # High environmental clearance friction
        "rfctlarr_stage": "Award Enquiry (Sec 21-23)",  # Highly disputed stage
        "land_parcel_area_acres": 65.5,
        "compensation_amount_lakhs": 420.0,
        "days_since_notification": 380,
        "number_of_displaced_families": 32,
        "litigation_cases_count": 2,  # Active court stays
        "social_vulnerability_index": 0.72,
        "official_processing_speed_score": 3.8,  # Administrative bottleneck
    }

    print("      Case Attributes:")
    for k, v in test_case.items():
        print(f"        * {k:32s}: {v}")

    prediction = predictor.predict(test_case)
    print("\n      --- PREDICTION RESULT ---")
    print(f"      * Delay Predicted?         : {'YES (WARNING)' if prediction['is_delay_predicted'] else 'NO (ON TRACK)'}")
    print(f"      * Delay Probability        : {prediction['delay_probability'] * 100:.1f}%")
    print(f"      * Risk Tier                : {prediction['risk_level']}")
    print(f"      * Expected Delay Duration  : {prediction['predicted_delay_months']:.1f} months")
    print(f"      * Uncertainty Range        : {prediction['confidence_band']['lower_months']:.1f} - {prediction['confidence_band']['upper_months']:.1f} months")

    # 3. Explain with SHAP
    print("\n[3/3] Generating SHAP E-Governance Explainability Drivers...")
    explanation = predictor.explain(test_case)

    print("\n      Top Factors Driving DELAY PROBABILITY (Head A):")
    for item in explanation["top_risk_drivers"]:
        impact = f"+{item['attribution']:.4f}" if item['attribution'] > 0 else f"{item['attribution']:.4f}"
        print(f"        > {item['feature']:30s} = {str(item['value']):12s} ({impact} prob | {item['direction']})")

    print("\n      Top Factors Driving DELAY DURATION (Head B - Months):")
    for item in explanation["top_duration_drivers"]:
        impact = f"+{item['attribution']:.2f} mos" if item['attribution'] > 0 else f"{item['attribution']:.2f} mos"
        print(f"        > {item['feature']:30s} = {str(item['value']):12s} ({impact} | {item['direction']})")

    # 4. Save visualization
    bg_df = pd.read_csv("data/land_acquisition_cases.csv").head(25)
    explainer = DualHeadExplainer(predictor.model, predictor.processor, bg_df)
    plot_file = explainer.save_waterfall_plot(pd.DataFrame([test_case]), "shap_waterfall.png")
    print(f"\n      [OK] Audit-ready SHAP explanation chart saved to: {plot_file}")

    print("\n" + "=" * 70)
    print("  Demo Complete! Ready for Hackathon Presentation & API Serving.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
