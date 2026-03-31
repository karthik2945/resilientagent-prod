#!/usr/bin/env python3
"""
Baseline inference script for resilientagent-prod.
Uses NVIDIA LLM if API key available, otherwise uses rule-based fallback.
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional

# Configuration
BASE_URL = "http://localhost:8003"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"
MAX_STEPS = 20

# Rule-based action sequences for each task (fallback when no API key)
RULE_BASED_ACTIONS = {
    "task1_latency_spike": [
        {"action_type": "check_metrics", "target": "inference_service", "parameters": None},
        {"action_type": "read_logs", "target": "inference_service", "parameters": None},
        {"action_type": "optimize_batch", "target": "inference_service", "parameters": None},
        {"action_type": "verify_fix", "target": "inference_service", "parameters": None}
    ],
    "task2_prediction_drift": [
        {"action_type": "analyze_drift", "target": "ml_model", "parameters": None},
        {"action_type": "check_deployment", "target": "ml_model", "parameters": None},
        {"action_type": "rollback_model", "target": "ml_model", "parameters": None},
        {"action_type": "verify_fix", "target": "ml_model", "parameters": None}
    ],
    "task3_cascading_failure": [
        {"action_type": "check_metrics", "target": "primary_model", "parameters": None},
        {"action_type": "read_logs", "target": "primary_model", "parameters": None},
        {"action_type": "restart_service", "target": "primary_model", "parameters": None},
        {"action_type": "scale_service", "target": "fallback_model", "parameters": None},
        {"action_type": "verify_fix", "target": "primary_model", "parameters": None}
    ]
}


def get_openai_client() -> Optional[Any]:
    """Initialize OpenAI client with NVIDIA API if key available."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Note: OPENAI_API_KEY not set, using rule-based fallback")
        return None
    
    try:
        from openai import OpenAI
        return OpenAI(
            api_key=api_key,
            base_url=NVIDIA_BASE_URL
        )
    except Exception as e:
        print(f"Warning: Failed to initialize OpenAI client: {e}")
        print("Falling back to rule-based approach")
        return None


def get_rule_based_action(task_id: str, step: int) -> Dict[str, Any]:
    """Get action from predefined rule sequence."""
    actions = RULE_BASED_ACTIONS.get(task_id, [])
    if step < len(actions):
        return actions[step]
    return {"action_type": "check_metrics", "target": "inference_service", "parameters": None}


def get_llm_action(client: Any, observation: Dict[str, Any]) -> Dict[str, Any]:
    """Query LLM for next action."""
    if client is None:
        return {"action_type": "check_metrics", "target": "inference_service", "parameters": None}
    
    obs_text = json.dumps(observation, indent=2)
    system_prompt = """You are an ML ops agent. Available actions: check_metrics, read_logs, check_deployment, analyze_drift, scale_service, rollback_model, optimize_batch, restart_service, verify_fix, notify_team. Respond with JSON: {"action_type": "name", "target": "service"}"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Observation:\n{obs_text}\n\nWhat action? JSON only."}
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=150
        )
        
        content = response.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        action = json.loads(content)
        return {
            "action_type": action.get("action_type", "check_metrics"),
            "target": action.get("target", "inference_service"),
            "parameters": action.get("parameters")
        }
    except Exception as e:
        print(f"  Warning: LLM call failed: {e}, using fallback")
        return {"action_type": "check_metrics", "target": "inference_service", "parameters": None}


def reset_task(task_id: str) -> Dict[str, Any]:
    """Reset environment for a task."""
    response = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    response.raise_for_status()
    return response.json()


def step(action: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an action."""
    response = requests.post(f"{BASE_URL}/step", json=action)
    response.raise_for_status()
    return response.json()


def get_grade() -> Dict[str, Any]:
    """Get final grade."""
    response = requests.post(f"{BASE_URL}/grader")
    response.raise_for_status()
    return response.json()


def run_task(client: Optional[Any], task_id: str) -> Dict[str, Any]:
    """Run a single task."""
    print(f"\n{'='*60}")
    print(f"Running task: {task_id}")
    print(f"Mode: {'Rule-based' if client is None else 'LLM'}")
    print(f"{'='*60}")
    
    result = reset_task(task_id)
    observation = result["observation"]
    
    steps = 0
    terminated = False
    truncated = False
    
    while steps < MAX_STEPS and not terminated and not truncated:
        if client is None:
            action = get_rule_based_action(task_id, steps)
        else:
            action = get_llm_action(client, observation)
        
        print(f"Step {steps + 1}: {action['action_type']} -> {action['target']}")
        
        result = step(action)
        observation = result["observation"]
        reward = result["reward"]
        terminated = result.get("done", False)
        truncated = result.get("done", False)
        
        print(f"  Reward: {reward:.3f}, Done: {terminated}")
        steps += 1
        
        if terminated:
            print(f"  ✓ Task resolved!")
            break
    
    grade_result = get_grade()
    score = grade_result["score"]
    print(f"  Final score: {score:.3f}")
    
    return {"task_id": task_id, "score": score, "steps": steps, "resolved": terminated}


def main():
    """Main entry point."""
    print("Resilient Agent Baseline Inference")
    print(f"Server: {BASE_URL}")
    
    client = get_openai_client()
    
    tasks = ["task1_latency_spike", "task2_prediction_drift", "task3_cascading_failure"]
    results = {}
    
    for task_id in tasks:
        result = run_task(client, task_id)
        short_name = task_id.split("_", 1)[1]
        results[short_name] = result
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    final_output = {
        "model": "rule-based" if client is None else MODEL,
        "results": {name: {"score": r["score"], "steps": r["steps"], "resolved": r["resolved"]} 
                    for name, r in results.items()}
    }
    
    for short_name, result in results.items():
        assert 0.0 <= result["score"] <= 1.0, f"Score {result['score']} out of range"
    
    print(json.dumps(final_output, indent=2))
    
    avg_score = sum(r["score"] for r in results.values()) / len(results)
    resolved = sum(1 for r in results.values() if r["resolved"])
    
    print(f"\nAverage score: {avg_score:.3f}")
    print(f"Tasks resolved: {resolved}/{len(tasks)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
