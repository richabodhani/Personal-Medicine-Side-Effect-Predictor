"""
data_store.py — shared storage layer
Logs every prediction to a local JSON file so both app.py and admin.py can access it.
"""

import json
import os
from datetime import datetime

LOG_FILE = "predictions_log.json"


def _load() -> list:
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save(data: list):
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)


def log_prediction(
    patient_name: str,
    age: int,
    weight: float,
    drug: str,
    thera: str,
    chem: str,
    use: str,
    habit: str,
    severity_score: int,
    total_effects: int,
    side_effects: list,
    summary: str,
):
    """Append one prediction record to the log file."""
    records = _load()
    records.append({
        "id": len(records) + 1,
        "timestamp": datetime.now().isoformat(),
        "patient": {
            "name": patient_name or "Anonymous",
            "age": age,
            "weight": weight,
        },
        "medicine": {
            "drug": drug,
            "therapeutic_class": thera,
            "chemical_class": chem,
            "primary_use": use,
            "habit_forming": habit,
        },
        "result": {
            "severity_score": severity_score,
            "total_effects": total_effects,
            "effects_found": len(side_effects),
            "side_effects": side_effects,
            "summary": summary,
        },
    })
    _save(records)


def get_all() -> list:
    return _load()


def clear_all():
    _save([])
