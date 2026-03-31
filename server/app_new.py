"""FastAPI application for ResilientAgent-Prod environment."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from server.resilientagent_prod_environment import ResilientAgentEnvironment
from models import ResilientAgentAction, ResilientAgentObservation

app = FastAPI(title="ResilientAgent-Prod Environment")

# Global environment instance
_env: Optional[ResilientAgentEnvironment] = None


class ResetRequest(BaseModel):
    task_id: str


class StepRequest(BaseModel):
    action_type: str
    target: str
    parameters: Optional[Dict[str, Any]] = None


def get_env() -> ResilientAgentEnvironment:
    global _env
    if _env is None:
        _env = ResilientAgentEnvironment()
    return _env


@app.post("/reset")
def reset(request: ResetRequest):
    """Reset environment for a new task."""
    env = get_env()
    env.reset(task_id=request.task_id)
    obs = env._make_observation()
    return {
        "observation": {
            "metrics": obs.metrics,
            "recent_logs": obs.recent_logs,
            "alert_status": obs.alert_status,
            "time_elapsed": obs.time_elapsed,
            "last_action_result": obs.last_action_result,
            "root_cause_hint": obs.root_cause_hint,
            "done": obs.done,
            "reward": obs.reward
        }
    }


@app.post("/step")
def step(request: StepRequest):
    """Execute an action in the environment."""
    env = get_env()
    action = ResilientAgentAction(
        action_type=request.action_type,
        target=request.target,
        parameters=request.parameters
    )
    obs = env.step(action)
    return {
        "observation": {
            "metrics": obs.metrics,
            "recent_logs": obs.recent_logs,
            "alert_status": obs.alert_status,
            "time_elapsed": obs.time_elapsed,
            "last_action_result": obs.last_action_result,
            "root_cause_hint": obs.root_cause_hint,
            "done": obs.done,
            "reward": obs.reward
        },
        "reward": obs.reward,
        "done": obs.done
    }


@app.get("/state")
def state():
    """Get current environment state."""
    env = get_env()
    return {
        "episode_id": env.state.episode_id,
        "step_count": env.state.step_count
    }


@app.post("/grader")
def grader():
    """Get final grade for the episode."""
    env = get_env()
    score = env.grade()
    return {"score": score}


@app.get("/tasks")
def tasks():
    """List available tasks."""
    return {
        "tasks": [
            {"id": "task1_latency_spike", "difficulty": "easy", "description": "Diagnose and fix ML model latency spike"},
            {"id": "task2_prediction_drift", "difficulty": "medium", "description": "Detect and remediate model prediction drift"},
            {"id": "task3_cascading_failure", "difficulty": "hard", "description": "Resolve cascading ML service failure"}
        ]
    }


@app.get("/baseline")
def baseline():
    """Run baseline rule-based agent."""
    env = get_env()
    
    results = {}
    for task_id in ["task1_latency_spike", "task2_prediction_drift", "task3_cascading_failure"]:
        env.reset(task_id=task_id)
        score = env.grade()
        short_name = task_id.split("_", 1)[1]
        results[short_name] = {"score": score, "steps": env._state.step_count, "resolved": env._model_healthy}
    
    return {"model": "rule-based", "results": results}


@app.get("/")
def root():
    """Root endpoint with API info."""
    return {
        "name": "ResilientAgent-Prod Environment",
        "endpoints": ["/reset", "/step", "/state", "/grader", "/tasks", "/baseline"]
    }
