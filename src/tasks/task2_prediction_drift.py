from typing import Dict, Any, List
from src.models import MLState


def get_initial_state() -> Dict[str, Any]:
    """
    Scenario: Model accuracy dropped 15% overnight
    Root cause: "data_pipeline_schema_change"
    """
    return {
        "services": {"ml_model": "up", "data_pipeline": "degraded"},
        "metrics": {"accuracy": 0.71, "drift_score": 0.82, "error_rate": 0.08, "latency_p99": 210.0},
        "logs": [
            "WARNING: Feature distribution shift detected",
            "ERROR: Schema mismatch in pipeline",
            "WARNING: Prediction confidence dropping"
        ],
        "model_healthy": False,
        "root_cause": "data_pipeline_schema_change"
    }


def get_correct_actions() -> List[str]:
    """
    Returns the sequence of correct actions to resolve the prediction drift issue.
    """
    return ["analyze_drift", "check_deployment", "rollback_model", "verify_fix"]
