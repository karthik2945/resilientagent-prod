from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime


class MLAction(BaseModel):
    action_type: Literal[
        "check_metrics", "read_logs", "check_deployment",
        "analyze_drift", "scale_service", "rollback_model", "optimize_batch",
        "restart_service", "verify_fix", "notify_team"
    ]
    target: str
    parameters: Optional[Dict[str, Any]] = None


class MLObservation(BaseModel):
    metrics: Dict[str, float]
    recent_logs: List[str]
    alert_status: str
    time_elapsed: float
    last_action_result: str
    root_cause_hint: Optional[str] = None


class MLReward(BaseModel):
    value: float
    reason: str
    partial_progress: float


class MLState(BaseModel):
    task_id: str
    services: Dict[str, str]
    metrics: Dict[str, float]
    logs: List[str]
    incident_start: float
    time_to_resolution: Optional[float] = None
    model_healthy: bool
    actions_taken: List[str]
    wasted_actions: int
    root_cause_identified: Optional[str] = None
    fix_applied: Optional[str] = None
    step_count: int

