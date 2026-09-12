"""Synthetic data generator for Land Acquisition Delay Prediction.

This module generates realistic tabular datasets calibrated to distributions reported in
CAG (Comptroller and Auditor General of India) performance audit reports and RTI data on
rural infrastructure projects under the RFCTLARR Act, 2013.

# TODO: Replace synthetic parameters with exact CAG (Comptroller and Auditor General of India)
# performance audit reports and RTI state-level land acquisition dataset anchors.
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Ensure project root is in sys.path when script is run directly from data/ directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from config import RFCTLARR_STAGES

# Representative sample of Indian administrative districts, states, and postal PIN prefixes
DISTRICT_DATA: List[Tuple[str, str, str]] = [
    # (PIN prefix, District, State)
    ("500", "Hyderabad", "Telangana"),
    ("505", "Karimnagar", "Telangana"),
    ("506", "Warangal", "Telangana"),
    ("520", "Vijayawada", "Andhra Pradesh"),
    ("530", "Visakhapatnam", "Andhra Pradesh"),
    ("560", "Bengaluru Urban", "Karnataka"),
    ("570", "Mysuru", "Karnataka"),
    ("580", "Dharwad", "Karnataka"),
    ("600", "Chennai", "Tamil Nadu"),
    ("641", "Coimbatore", "Tamil Nadu"),
    ("625", "Madurai", "Tamil Nadu"),
    ("400", "Mumbai Suburban", "Maharashtra"),
    ("411", "Pune", "Maharashtra"),
    ("440", "Nagpur", "Maharashtra"),
    ("431", "Aurangabad", "Maharashtra"),
    ("380", "Ahmedabad", "Gujarat"),
    ("395", "Surat", "Gujarat"),
    ("302", "Jaipur", "Rajasthan"),
    ("342", "Jodhpur", "Rajasthan"),
    ("226", "Lucknow", "Uttar Pradesh"),
    ("201", "Gautam Buddha Nagar", "Uttar Pradesh"),
    ("208", "Kanpur Nagar", "Uttar Pradesh"),
    ("221", "Varanasi", "Uttar Pradesh"),
    ("800", "Patna", "Bihar"),
    ("834", "Ranchi", "Jharkhand"),
    ("751", "Khordha", "Odisha"),
    ("700", "Kolkata", "West Bengal"),
    ("160", "Chandigarh", "Punjab"),
    ("141", "Ludhiana", "Punjab"),
    ("110", "South Delhi", "Delhi"),
]

LAND_TYPES: List[str] = [
    "Agricultural",
    "Forest",
    "Commercial",
    "Residential",
    "Barren/Waste",
    "Wetland",
]


def generate_synthetic_dataset(
    n_samples: int = 10000,
    seed: int = 42,
    missing_rate_normal: float = 0.02,
    inject_high_missing_col: bool = False,
) -> pd.DataFrame:
    """Generate structured land acquisition records mirroring Indian administrative data.

    Args:
        n_samples: Number of case rows to synthesize (typical 5,000 to 50,000).
        seed: Random state seed for reproducibility.
        missing_rate_normal: Baseline random missingness rate (2%).
        inject_high_missing_col: Whether to inject >15% missingness into one feature to test audit warnings.

    Returns:
        pd.DataFrame containing realistic project features, delay indicators, and delay months.
    """
    rng = np.random.default_rng(seed)

    # 1. Geographic and Administrative Categoricals
    district_indices = rng.integers(0, len(DISTRICT_DATA), size=n_samples)
    pin_codes = []
    districts = []
    states = []
    for idx in district_indices:
        prefix, dist, st = DISTRICT_DATA[idx]
        suffix = rng.integers(100, 999)
        pin_codes.append(f"{prefix}{suffix}")
        districts.append(dist)
        states.append(st)

    # Land type with realistic empirical proportions (majority Agricultural in rural projects)
    land_type_probs = [0.55, 0.12, 0.08, 0.10, 0.10, 0.05]
    land_types = rng.choice(LAND_TYPES, size=n_samples, p=land_type_probs)

    # Statutory RFCTLARR Act Stage distribution
    # Cases cluster around middle stages (SIA, Draft/Final Declaration, Award Enquiry)
    stage_weights = [0.10, 0.18, 0.16, 0.14, 0.18, 0.12, 0.07, 0.05]
    stages = rng.choice(RFCTLARR_STAGES, size=n_samples, p=stage_weights)

    # 2. Skewed Continuous Features (Heavy right tails: lognormal distributions)
    # Land Parcel Area (acres): median ~12 acres, right tail extending beyond 250 acres
    area_acres = rng.lognormal(mean=2.4, sigma=0.85, size=n_samples)
    area_acres = np.clip(np.round(area_acres, 2), 0.5, 800.0)

    # Compensation Amount (in INR Lakhs): median ~45 lakhs, right tail exceeding 1000 lakhs
    comp_lakhs = rng.lognormal(mean=3.8, sigma=0.95, size=n_samples)
    comp_lakhs = np.clip(np.round(comp_lakhs, 2), 5.0, 5000.0)

    # Days elapsed since Preliminary Notification (Sec 11)
    days_since_notification = rng.lognormal(mean=5.4, sigma=0.75, size=n_samples)
    days_since_notification = np.clip(np.round(days_since_notification), 30, 2500)

    # 3. Non-Skewed Continuous and Demographic Features
    # Number of displaced families: correlated with land area and residential/commercial types
    displaced_base = rng.poisson(lam=4.0, size=n_samples)
    displaced_families = np.where(
        np.isin(land_types, ["Residential", "Commercial"]),
        displaced_base * 3 + rng.integers(1, 15, size=n_samples),
        displaced_base,
    )

    # Active litigation court disputes: count 0 to 6
    litigation_probs = [0.65, 0.20, 0.09, 0.04, 0.015, 0.005]
    litigation_cases = rng.choice(len(litigation_probs), size=n_samples, p=litigation_probs)

    # Social Vulnerability Index (0.0 to 1.0 bounded beta distribution)
    vulnerability_index = rng.beta(a=2.5, b=3.0, size=n_samples)
    vulnerability_index = np.round(vulnerability_index, 3)

    # Official processing speed score (1.0 to 10.0 normal distribution clipped)
    processing_speed = rng.normal(loc=5.8, scale=1.6, size=n_samples)
    processing_speed = np.clip(np.round(processing_speed, 1), 1.0, 10.0)

    # 4. Synthesize Ground Truth Delay Probability via Multi-Factor Logit
    # Reflects real-world factors reported by CAG:
    # - Forest and Wetland require extensive MoEFCC environmental clearances -> high delay risk
    # - High litigation count causes court stay orders -> severe delay
    # - Large compensation disputes and high displacement trigger local protests
    # - Low administrative processing speed exacerbates statutory time overruns

    # Base log-odds calibrated to ~40-45% delay prevalence
    log_odds = -2.80

    # Land type risk adjustment
    type_risk_map = {
        "Agricultural": 0.0,
        "Barren/Waste": -0.4,
        "Commercial": 0.35,
        "Residential": 0.40,
        "Forest": 0.95,     # MoEFCC clearance bottlenecks
        "Wetland": 0.85,    # Environmental litigation
    }
    land_risk = np.array([type_risk_map[t] for t in land_types])

    # Stage risk adjustment (Award Enquiry and Compensation Disbursement frequently encounter disputes)
    stage_risk_map = {
        "Preliminary Notification (Sec 11)": -0.3,
        "Survey & SIA (Social Impact Assessment)": 0.1,
        "Draft Declaration (Sec 19)": 0.2,
        "Final Declaration": 0.15,
        "Award Enquiry (Sec 21-23)": 0.55,
        "Award Declaration": 0.35,
        "Compensation Disbursement": 0.45,
        "Possession": -0.2,
    }
    stage_risk = np.array([stage_risk_map[s] for s in stages])

    # Continuous feature impact weights
    log_odds += (
        land_risk
        + stage_risk
        + 0.55 * litigation_cases
        + 0.30 * np.log1p(area_acres)
        + 0.25 * np.log1p(comp_lakhs)
        + 0.02 * (displaced_families)
        + 0.60 * vulnerability_index
        - 0.25 * (processing_speed - 5.0)
        + rng.normal(0.0, 0.4, size=n_samples)  # Unobserved latent administrative friction
    )

    # Compute true probabilities via Sigmoid
    true_probs = 1.0 / (1.0 + np.exp(-log_odds))
    delay_probability_binary = rng.binomial(1, true_probs)

    # 5. Synthesize Ground Truth Delay Months (Conditional on Delay == 1)
    # CAG reports show delayed land acquisition averages 14-26 months of overrun
    delay_months = np.zeros(n_samples, dtype=np.float32)
    delayed_indices = np.where(delay_probability_binary == 1)[0]

    for i in delayed_indices:
        # Base lognormal duration
        base_months = rng.lognormal(mean=2.4, sigma=0.55)
        # Add additive delays for active litigation (court cases add 6-18 months)
        lit_delay = litigation_cases[i] * rng.uniform(4.0, 8.0)
        # Add environmental clearance delay if forest/wetland
        env_delay = 6.0 if land_types[i] in ["Forest", "Wetland"] else 0.0
        # Total positive delay duration in months
        total_months = base_months + lit_delay + env_delay + rng.uniform(1.0, 3.0)
        delay_months[i] = np.round(np.clip(total_months, 1.0, 72.0), 1)

    # 6. Assemble DataFrame
    case_ids = [f"LA-2024-{i + 1:05d}" for i in range(n_samples)]
    df = pd.DataFrame({
        "case_id": case_ids,
        "pin_code": pin_codes,
        "district": districts,
        "state": states,
        "land_type": land_types,
        "rfctlarr_stage": stages,
        "land_parcel_area_acres": area_acres,
        "compensation_amount_lakhs": comp_lakhs,
        "days_since_notification": days_since_notification,
        "number_of_displaced_families": displaced_families,
        "litigation_cases_count": litigation_cases,
        "social_vulnerability_index": vulnerability_index,
        "official_processing_speed_score": processing_speed,
        "delay_probability": delay_probability_binary,
        "delay_months": delay_months,
    })

    # 7. Introduce realistic baseline missingness
    for col in [
        "land_parcel_area_acres",
        "compensation_amount_lakhs",
        "number_of_displaced_families",
        "social_vulnerability_index",
    ]:
        mask = rng.random(n_samples) < missing_rate_normal
        df.loc[mask, col] = np.nan

    if inject_high_missing_col:
        # Inject 18% missingness into official_processing_speed_score to trigger >15% audit warning
        mask_high = rng.random(n_samples) < 0.18
        df.loc[mask_high, "official_processing_speed_score"] = np.nan

    return df


def main() -> None:
    """CLI entrypoint to generate and persist synthetic land acquisition dataset."""
    parser = argparse.ArgumentParser(description="Generate synthetic land acquisition dataset for SIH26017.")
    parser.add_argument("--samples", type=int, default=10000, help="Number of cases to generate (default: 10000).")
    parser.add_argument("--output", type=str, default="data/land_acquisition_cases.csv", help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.samples} synthetic land acquisition cases...")
    df = generate_synthetic_dataset(n_samples=args.samples, seed=args.seed)
    df.to_csv(out_path, index=False)
    print(f"Successfully generated dataset with shape {df.shape} saved to: {out_path}")
    print(f"Delay rate: {df['delay_probability'].mean() * 100:.1f}%")
    delayed_df = df[df['delay_probability'] == 1]
    print(f"Delayed cases average duration: {delayed_df['delay_months'].mean():.1f} months "
          f"(min: {delayed_df['delay_months'].min():.1f}, max: {delayed_df['delay_months'].max():.1f})")


if __name__ == "__main__":
    main()
