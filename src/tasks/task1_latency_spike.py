from typing import Dict, Any, List
from src.models import MLState


def get_initial_state() -> Dict[str, Any]:
    """
    Scenario: ML model inference latency jumped from 200ms to 5000ms
    Root cause: "gpu_memory_exhaustion"
    """
    return {
        "services": {"inference_service": "degraded", "api_gateway": "up"},
        "metrics": {"latency_p99": 5000.0, "error_rate": 0.15, "gpu_util": 0.98, "throughput": 12.0},
        "logs": [
            "ERROR: CUDA out of memory",
            "WARNING: Batch size too large",
            "ERROR: Request timeout after 5000ms"
        ],
        "model_healthy": False,
        "root_cause": "gpu_memory_exhaustion"
    }


def get_correct_actions() -> List[str]:
    """
    Returns the sequence of correct actions to resolve the latency spike issue.
    """
    return ["check_metrics", "read_logs", "optimize_batch", "verify_fix"]
