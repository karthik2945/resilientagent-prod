from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import logging
import sys
from pathlib import Path

# Add parent directory and server directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from resilientagent_prod_environment import ResilientAgentEnvironment
from models import ResilientAgentAction, ResilientAgentObservation
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="ResilientAgent-Prod Environment")

# Global environment instance
_env: Optional[ResilientAgentEnvironment] = None
logger = logging.getLogger("app")


def get_env() -> ResilientAgentEnvironment:
    """Get or create the global environment instance."""
    global _env
    if _env is None:
        _env = ResilientAgentEnvironment()
    return _env

# Strong system prompt (same as inference.py)
SYSTEM_PROMPT = """\
You are an autonomous SRE agent that diagnoses and resolves ML production incidents.

## Available actions (pick exactly ONE per step)
check_metrics, read_logs, check_deployment, analyze_drift,
scale_service, rollback_model, optimize_batch, restart_service,
verify_fix, notify_team

## Available targets
inference_service, ml_model, primary_model, fallback_model

## Critical rules
1. NEVER repeat the same (action, target) pair you already used.
2. Follow this general pattern: diagnose first → apply a fix → verify_fix.
3. Task-specific guidance:
   • latency_spike  → check_metrics → read_logs → optimize_batch → verify_fix  (target: inference_service)
   • prediction_drift → analyze_drift → check_deployment → rollback_model → verify_fix  (target: ml_model)
   • cascading_failure → check_metrics(primary_model) → read_logs(primary_model) → restart_service(primary_model) → scale_service(fallback_model) → verify_fix(primary_model)
4. Reply ONLY with a JSON object:  {"action_type": "...", "target": "..."}
   No markdown fences, no extra text.
"""





class StepRequest(BaseModel):
    action_type: str
    target: str
    parameters: Optional[Dict[str, Any]] = None


def build_user_prompt(task_id: str, obs, history: list) -> str:
    """Build a rich user prompt with observation + history (same as inference.py)."""
    obs_summary = {
        "task_id": task_id,
        "alert_status": obs.alert_status,
        "metrics": obs.metrics,
        "recent_logs": obs.recent_logs[:3],
    }

    history_str = ""
    if history:
        history_str = "\n\nActions already taken (DO NOT repeat these):\n"
        for i, h in enumerate(history, 1):
            history_str += f"  {i}. {h['action_type']} -> {h['target']}  (reward={h['reward']:.3f})\n"

    return (
        f"Current observation:\n{json.dumps(obs_summary, indent=2)}"
        f"{history_str}"
        f"\n\nWhat is your next action?"
    )


@app.post("/reset")
async def reset(request: Request):
    """Reset environment for a new task. Accepts empty body or JSON with task_id."""
    env = get_env()
    task_id = "task1_latency_spike"
    
    # Gracefully handle empty body (hackathon validator sends no body)
    try:
        body = await request.body()
        if body and body.strip():
            payload = await request.json()
            if isinstance(payload, dict):
                task_id = payload.get("task_id", task_id)
    except Exception:
        pass  # No body or invalid JSON — use default task_id
        
    obs = env.reset(task_id=task_id)
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
        parameters=request.parameters or {}
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
    return {"state": env.get_state()}


@app.post("/grader")
def grader():
    """Grade current task performance."""
    env = get_env()
    score = env.grade()
    return {"score": score}


@app.get("/tasks")
def tasks():
    """List available tasks."""
    return {
        "tasks": [
            {"id": "task1_latency_spike", "name": "Latency Spike", "description": "Fix ML model latency spike"},
            {"id": "task2_prediction_drift", "name": "Prediction Drift", "description": "Remediate model prediction drift"},
            {"id": "task3_cascading_failure", "name": "Cascading Failure", "description": "Resolve cascading ML service failure"}
        ]
    }


@app.get("/baseline")
def baseline():
    """Run baseline agent on all tasks."""
    env = get_env()
    tasks = [
        ("task1_latency_spike", [
            ("check_metrics", "inference_service"),
            ("read_logs", "inference_service"),
            ("optimize_batch", "inference_service"),
            ("verify_fix", "inference_service"),
        ]),
        ("task2_prediction_drift", [
            ("analyze_drift", "ml_model"),
            ("check_deployment", "ml_model"),
            ("rollback_model", "ml_model"),
            ("verify_fix", "ml_model"),
        ]),
        ("task3_cascading_failure", [
            ("check_metrics", "primary_model"),
            ("read_logs", "primary_model"),
            ("restart_service", "primary_model"),
            ("scale_service", "fallback_model"),
            ("verify_fix", "primary_model"),
        ]),
    ]
    
    results = {}
    all_details = {}
    
    for task_id, action_sequence in tasks:
        obs = env.reset(task_id=task_id)
        steps_data = []
        
        for i, (action_type, target) in enumerate(action_sequence):
            action = ResilientAgentAction(action_type=action_type, target=target)
            obs = env.step(action)
            
            step_info = {
                "step": i + 1,
                "action_type": action_type,
                "target": target,
                "reward": round(obs.reward, 4),
                "done": obs.done,
                "logs": obs.recent_logs[-1:] if obs.recent_logs else []
            }
            steps_data.append(step_info)
            
            if obs.done:
                break
        
        score = env.grade()
        short_name = task_id.split("_", 1)[1]
        results[short_name] = {
            "score": round(score, 4),
            "steps": len(steps_data),
            "resolved": env._model_healthy
        }
        all_details[short_name] = steps_data
    
    return {"results": results, "details": all_details}


def get_llm_action(client, model: str, task_id: str, obs, history: list) -> dict:
    """Ask the LLM for the next action using strong prompt."""
    prompt = build_user_prompt(task_id, obs, history)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.05,
            max_tokens=120,
        )
        reply = response.choices[0].message.content.strip()

        # Strip markdown fences if the model wraps them
        if reply.startswith("```"):
            reply = reply.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        action_dict = json.loads(reply)
        return action_dict

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {"action_type": "notify_team", "target": "inference_service"}


@app.get("/llm-inference")
def llm_inference():
    """Run REAL LLM agent on all tasks using API."""
    # Read evaluator environment variables
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4")
    
    # Check for API key (Handles HF_TOKEN, OPENAI_API_KEY or GROQ_API_KEY)
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No API key found. Set HF_TOKEN or OPENAI_API_KEY environment variable."
        )
    
    # Initialize OpenAI-compatible client
    client = OpenAI(
        base_url=api_base_url,
        api_key=api_key
    )
    
    env = get_env()
    
    tasks = [
        ("task1_latency_spike", "Diagnose and fix ML model latency spike"),
        ("task2_prediction_drift", "Detect and remediate model prediction drift"),
        ("task3_cascading_failure", "Resolve cascading ML service failure")
    ]
    
    results = {}
    all_details = {}
    
    for task_id, task_desc in tasks:
        obs = env.reset(task_id=task_id)
        steps_data = []
        history = []
        max_steps = 10
        
        for step_num in range(max_steps):
            # Use STRONG prompt from inference.py
            action_dict = get_llm_action(client, model_name, task_id, obs, history)
            
            action_type = action_dict.get("action_type", "check_metrics")
            target = action_dict.get("target", "inference_service")
            
            action = ResilientAgentAction(action_type=action_type, target=target)
            obs = env.step(action)
            
            step_info = {
                "step": step_num + 1,
                "action_type": action_type,
                "target": target,
                "reward": round(obs.reward, 4),
                "done": obs.done,
                "logs": obs.recent_logs[-1:] if obs.recent_logs else []
            }
            steps_data.append(step_info)
            history.append({
                "action_type": action_type,
                "target": target,
                "reward": obs.reward
            })
            
            if obs.done:
                break
        
        score = env.grade()
        short_name = task_id.split("_", 1)[1]
        results[short_name] = {
            "score": round(score, 4),
            "steps": len(steps_data),
            "resolved": env._model_healthy
        }
        all_details[short_name] = steps_data
    
    return {
        "model": "llama-3.3-70b-versatile (Groq)",
        "results": results,
        "details": all_details
    }


@app.get("/")
def root():
    """Serve the interactive dashboard UI."""
    return FileResponse("resilientagent_dashboard.html")


@app.get("/health")
def health():
    """Health check endpoint for Docker/Hugging Face Spaces."""
    return {"status": "ok"}
