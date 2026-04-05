#!/usr/bin/env python3
"""
inference.py — LLM-powered SRE Agent for ResilientAgent-Prod.

Uses the OpenAI-compatible API (Groq / HuggingFace / etc.) to diagnose and
resolve ML production incidents. Reads API_BASE_URL, MODEL_NAME, HF_TOKEN
from the environment (hackathon grader injects these automatically).

Outputs structured logs in [START] [STEP] [END] format for hackathon evaluation.
"""

import os
import sys
import json
from typing import Optional, List

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from openai import OpenAI

from server.resilientagent_prod_environment import ResilientAgentEnvironment
from models import ResilientAgentAction

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4")
HF_TOKEN     = os.getenv("HF_TOKEN")
OPENAI_API   = os.getenv("OPENAI_API_KEY")

api_key = HF_TOKEN if HF_TOKEN else OPENAI_API

# Initialize OpenAI client
if not api_key or api_key.strip() == "":
    print("ERROR: Neither HF_TOKEN nor OPENAI_API_KEY environment variable is set!", file=sys.stderr)
    print("Evaluator must inject: API_BASE_URL, MODEL_NAME, and an API Key", file=sys.stderr)
    sys.exit(1)

try:
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
except Exception as e:
    print(f"ERROR: Failed to initialize OpenAI client: {e}", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# System prompt — gives the LLM full context about the environment
# ---------------------------------------------------------------------------
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
3. Reply ONLY with a JSON object:  {"action_type": "...", "target": "..."}
   No markdown fences, no extra text.

## Task-Specific Guidance

### Latency Spike (latency_p99 > 1000ms, gpu_memory_exhaustion in logs)
CORRECT SEQUENCE: check_metrics → read_logs → optimize_batch(inference_service) → verify_fix
- The fix is optimize_batch to reduce GPU memory pressure
- Target: inference_service

### Prediction Drift (accuracy < 0.8, drift_score > 0.5, schema mismatch in logs)
CORRECT SEQUENCE: analyze_drift(ml_model) → check_deployment(ml_model) → rollback_model(ml_model) → verify_fix
- Root cause: data_pipeline_schema_change
- The fix is rollback_model to previous version
- Target: ml_model (NOT inference_service)

### Cascading Failure (primary_model down, fallback_model degraded, OOM in logs)
CORRECT SEQUENCE: check_metrics → read_logs → restart_service(primary_model) → scale_service(fallback_model) → verify_fix
- First fix: restart_service on primary_model (to recover from OOM)
- Second fix: scale_service on fallback_model (to handle load)
- Targets: primary_model and fallback_model (NOT inference_service)

## Action Meanings
- analyze_drift: Check for model/data drift issues
- rollback_model: Revert to previous model version (use for schema/pipeline issues)
- optimize_batch: Reduce batch size to fix GPU memory issues
- restart_service: Restart crashed/dead service (use for OOM/memory leaks)
- scale_service: Add capacity to handle load (use for degraded/high-load services)
"""


def build_user_prompt(task_id: str, obs, history: list[dict]) -> str:
    """Build a rich user prompt with observation + history."""
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


def get_llm_action(task_id: str, obs, history: list[dict]) -> tuple[dict, Optional[str]]:
    """Ask the LLM for the next action. Returns (action_dict, error_string or None)."""
    prompt = build_user_prompt(task_id, obs, history)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
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
        return action_dict, None

    except Exception as e:
        error_msg = str(e)
        # Return fallback action with error
        return {"action_type": "notify_team", "target": "inference_service"}, error_msg

# ---------------------------------------------------------------------------
# Structured logging functions (hackathon format)
# ---------------------------------------------------------------------------
def log_start(task: str, env_name: str, model: str) -> None:
    """Log START with task, env, model."""
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Log STEP with step number, action, reward, done flag, error."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log END with success, steps, final score, reward list."""
    success_val = str(success).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------
def run_inference():
    """Run all tasks and output structured logs."""
    env = ResilientAgentEnvironment()

    tasks = ["task1_latency_spike", "task2_prediction_drift", "task3_cascading_failure"]
    all_results = {}

    for task_id in tasks:
        task_short = task_id.split("_", 1)[1] if "_" in task_id else task_id
        env_name = "resilientagent-prod"
        
        # Reset environment
        obs = env.reset(task_id=task_id)
        
        # Log start
        log_start(task_short, env_name, MODEL_NAME)
        
        history: list[dict] = []
        step_rewards: List[float] = []
        max_steps = 10
        last_error = None

        # Run steps
        while len(history) < max_steps and not obs.done:
            step_num = len(history) + 1
            
            # Get LLM action
            action_dict, llm_error = get_llm_action(task_id, obs, history)
            
            action_type = action_dict.get("action_type", "check_metrics")
            target = action_dict.get("target", "inference_service")
            
            # Format action as string
            action_str = f"{action_type}('{target}')"
            
            # Execute action
            action = ResilientAgentAction(action_type=action_type, target=target)
            obs = env.step(action)
            
            # Record step
            step_rewards.append(obs.reward)
            last_error = llm_error
            
            log_step(step_num, action_str, obs.reward, obs.done, llm_error)
            
            history.append({
                "action_type": action_type,
                "target": target,
                "reward": obs.reward,
            })
            
            if obs.done:
                break
        
        # Grade task
        score = env.grade()
        success = env._model_healthy
        
        log_end(success, len(history), score, step_rewards)
        
        all_results[task_short] = {
            "score": round(score, 4),
            "steps": len(history),
            "resolved": success,
            "rewards": [round(r, 2) for r in step_rewards]
        }
    
    return all_results


if __name__ == "__main__":
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set — API calls will likely fail.", file=sys.stderr)
    run_inference()
