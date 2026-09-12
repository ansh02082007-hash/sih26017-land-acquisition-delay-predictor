"""Prediction History Store for Land Acquisition Delay Prediction.

Persists every prediction to ``prediction_history.json`` so that:
1. Officials can review the AI's estimated delay timeline.
2. When land acquisition completes, the actual duration can be recorded
   alongside the original prediction, creating labelled ground-truth
   feedback samples for continual retraining.

Storage format (JSON array):
[
  {
    "record_id": "uuid4",
    "timestamp": "ISO-8601",
    "case_id": "LA-2024-001",
    "case_data": { ...raw inputs... },
    "prediction": {
      "delay_probability": 0.74,
      "is_delay_predicted": true,
      "predicted_delay_months": 18.2,
      "confidence_band": {"lower_months": 15.5, "upper_months": 20.9},
      "risk_level": "HIGH"
    },
    "outcome": null  |  {
      "actual_delay_months": 22.0,
      "acquisition_completed": true,
      "notes": "Settled after court order",
      "feedback_timestamp": "ISO-8601"
    }
  },
  ...
]
"""

from __future__ import annotations

import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_HISTORY_FILE = Path(__file__).resolve().parent / "prediction_history.json"


def _load() -> List[Dict[str, Any]]:
    """Load full history from disk, returning an empty list if missing."""
    if not _HISTORY_FILE.exists():
        return []
    try:
        with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read history file (%s). Starting fresh.", exc)
        return []


def _save(records: List[Dict[str, Any]]) -> None:
    """Persist records list to disk atomically via a temp-file swap."""
    tmp = _HISTORY_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    tmp.replace(_HISTORY_FILE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_prediction(
    case_data: Dict[str, Any],
    prediction: Dict[str, Any],
) -> str:
    """Append a new prediction record.

    Args:
        case_data:  Raw case input dict (as submitted to the /predict endpoint).
        prediction: Output dict from ``DelayPredictor.predict()``.

    Returns:
        The new record's ``record_id`` (UUID4 string).
    """
    records = _load()
    record_id = str(uuid.uuid4())
    records.append(
        {
            "record_id": record_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": case_data.get("case_id", "unknown"),
            "case_data": case_data,
            "prediction": prediction,
            "outcome": None,
        }
    )
    _save(records)
    logger.info("History: saved prediction for case '%s' (record %s).", case_data.get("case_id"), record_id)
    return record_id


def get_all_records() -> List[Dict[str, Any]]:
    """Return all records, newest first."""
    records = _load()
    return list(reversed(records))


def get_record(record_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single record by its UUID, or None if not found."""
    for rec in _load():
        if rec["record_id"] == record_id:
            return rec
    return None


def add_outcome(
    record_id: str,
    actual_delay_months: float,
    acquisition_completed: bool,
    notes: str = "",
) -> Dict[str, Any]:
    """Attach real-world outcome data to an existing prediction record.

    Args:
        record_id:              UUID of the prediction record to update.
        actual_delay_months:    True total delay in months (0 means on-time).
        acquisition_completed:  Whether acquisition process has fully concluded.
        notes:                  Optional free-text notes from the field officer.

    Returns:
        The updated record.

    Raises:
        KeyError: If ``record_id`` is not found.
    """
    records = _load()
    for rec in records:
        if rec["record_id"] == record_id:
            rec["outcome"] = {
                "actual_delay_months": actual_delay_months,
                "acquisition_completed": acquisition_completed,
                "notes": notes,
                "feedback_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            _save(records)
            logger.info(
                "History: outcome recorded for record %s (actual=%.1f mos).",
                record_id,
                actual_delay_months,
            )
            return rec
    raise KeyError(f"No record found with id: {record_id!r}")


def get_feedback_training_pairs() -> List[Dict[str, Any]]:
    """Return only records that have ground-truth outcomes — ready for retraining.

    Each item includes ``case_data``, ``prediction``, and ``outcome``.
    """
    return [r for r in _load() if r.get("outcome") is not None]


def stats() -> Dict[str, Any]:
    """Summary statistics for the history dashboard."""
    records = _load()
    with_outcome = [r for r in records if r.get("outcome") is not None]
    delayed = [r for r in records if r.get("prediction", {}).get("is_delay_predicted")]
    return {
        "total_predictions": len(records),
        "with_ground_truth": len(with_outcome),
        "predicted_delayed": len(delayed),
        "predicted_on_time": len(records) - len(delayed),
        "feedback_rate_pct": round(100 * len(with_outcome) / len(records), 1) if records else 0.0,
    }
