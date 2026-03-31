from typing import Dict, Any, List
from src.models import MLState


def get_initial_state() -> Dict[str, Any]:
    """
    Scenario: Primary model OOM, fallback degrading, autoscaling not triggering
    Root cause: "memory_leak_primary"
    """
    return {
        "services": {"primary_model": "down", "fallback_model": "degraded", "autoscaler": "down"},
        "metrics": {"primary_error_rate": 1.0, "fallback_error_rate": 0.45, "cpu_util": 0.94, "memory_util": 0.99},
        "logs": [
            "CRITICAL: Primary model OOM killed",
            "ERROR: Fallback CPU throttling",
            "ERROR: Autoscaler failed to trigger",
            "WARNING: Queue depth increasing"
        ],
        "model_healthy": False,
        "root_cause": "memory_leak_primary"
    }


def get_correct_actions() -> List[str]:
    """
    Returns the sequence of correct actions to resolve the cascading failure issue.
    Correct sequence: check_metrics -> read_logs -> restart_service(primary_model) -> 
                     scale_service(fallback_model) -> verify_fix
    """
    return ["check_metrics", "read_logs", "restart_service", "scale_service", "verify_fix"]
