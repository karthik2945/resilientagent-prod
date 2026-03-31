#!/usr/bin/env python3
"""Direct test of baseline rule-based agent (no HTTP server needed)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.resilientagent_prod_environment import ResilientAgentEnvironment
from models import ResilientAgentAction

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


def run_task(task_id):
    """Run a single task with rule-based agent."""
    print(f"\n{'='*60}")
    print(f"Running task: {task_id}")
    print(f"Mode: Rule-based")
    print(f"{'='*60}")
    
    env = ResilientAgentEnvironment()
    env.reset(task_id=task_id)
    
    actions = RULE_BASED_ACTIONS.get(task_id, [])
    steps = 0
    terminated = False
    
    for action_data in actions:
        if terminated:
            break
            
        action = ResilientAgentAction(**action_data)
        print(f"Step {steps + 1}: {action.action_type} -> {action.target}")
        
        obs = env.step(action)
        print(f"  Reward: {obs.reward:.3f}, Done: {obs.done}")
        
        terminated = obs.done
        steps += 1
        
        if terminated:
            print(f"  ✓ Task resolved!")
            break
    
    score = env.grade()
    print(f"  Final score: {score:.3f}")
    
    return {"task_id": task_id, "score": score, "steps": steps, "resolved": terminated}


def main():
    print("Resilient Agent Baseline Inference (Direct)")
    print("Mode: Rule-based (no HTTP server)")
    
    tasks = ["task1_latency_spike", "task2_prediction_drift", "task3_cascading_failure"]
    results = {}
    
    for task_id in tasks:
        result = run_task(task_id)
        short_name = task_id.split("_", 1)[1]
        results[short_name] = result
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    final_output = {
        "model": "rule-based",
        "results": {name: {"score": r["score"], "steps": r["steps"], "resolved": r["resolved"]} 
                    for name, r in results.items()}
    }
    
    import json
    print(json.dumps(final_output, indent=2))
    
    avg_score = sum(r["score"] for r in results.values()) / len(results)
    resolved = sum(1 for r in results.values() if r["resolved"])
    
    print(f"\nAverage score: {avg_score:.3f}")
    print(f"Tasks resolved: {resolved}/{len(tasks)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
