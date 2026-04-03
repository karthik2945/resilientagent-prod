#!/usr/bin/env python3
"""Standalone evaluation script for ResilientAgent-Prod environment."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.resilientagent_prod_environment import ResilientAgentEnvironment
from models import ResilientAgentAction


def run_task(env, task_id):
    """Run a single task and return results."""
    print(f"\n{'='*60}")
    print(f"Running task: {task_id}")
    print(f"{'='*60}")

    # Reset environment
    obs = env.reset(task_id=task_id)
    print(f"Initial metrics: {obs.metrics}")
    print(f"Initial alert: {obs.alert_status}")
    print(f"Logs: {obs.recent_logs[:3]}")

    # Get correct actions for this task
    correct_actions = env._get_correct_actions_for_task()
    print(f"Correct action sequence: {correct_actions}")

    steps = 0
    max_steps = 20

    # Execute correct actions
    for action_type in correct_actions:
        # Determine target based on action type and task
        target = "inference_service"
        if task_id == "task1_latency_spike":
            target = "inference_service"
        elif task_id == "task2_prediction_drift":
            target = "ml_model"
        elif task_id == "task3_cascading_failure":
            if action_type == "restart_service":
                target = "primary_model"
            elif action_type == "scale_service":
                target = "fallback_model"
            else:
                target = "primary_model"

        action = ResilientAgentAction(
            action_type=action_type,
            target=target,
            parameters=None
        )

        print(f"\nStep {steps + 1}: {action_type} -> {target}")
        obs = env.step(action)
        print(f"  Reward: {obs.reward:.3f}")
        print(f"  Done: {obs.done}")
        print(f"  Metrics: {obs.metrics}")

        steps += 1

        if obs.done:
            print(f"  [OK] Task resolved!")
            break

    # Get final grade
    score = env.grade()
    print(f"\n  Final score: {score:.3f}")

    return {
        "task_id": task_id,
        "score": score,
        "steps": steps,
        "resolved": env._model_healthy
    }


def main():
    """Main evaluation."""
    print("="*60)
    print("ResilientAgent-Prod Environment Evaluation")
    print("="*60)

    env = ResilientAgentEnvironment()

    tasks = [
        "task1_latency_spike",
        "task2_prediction_drift",
        "task3_cascading_failure"
    ]

    results = {}

    for task_id in tasks:
        result = run_task(env, task_id)
        short_name = task_id.split("_", 1)[1]
        results[short_name] = result

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    for short_name, result in results.items():
        print(f"\n{short_name}:")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Steps: {result['steps']}")
        print(f"  Resolved: {result['resolved']}")

    avg_score = sum(r["score"] for r in results.values()) / len(results)
    resolved = sum(1 for r in results.values() if r["resolved"])

    print(f"\n{'='*60}")
    print(f"Average Score: {avg_score:.3f}")
    print(f"Tasks Resolved: {resolved}/{len(tasks)}")
    print(f"{'='*60}")

    # Compare to baseline
    baseline_scores = {
        "latency_spike": 0.80,
        "prediction_drift": 1.00,
        "cascading_failure": 0.80
    }

    print(f"\nComparison to Baseline:")
    for task, result in results.items():
        baseline = baseline_scores.get(task, 0)
        diff = result['score'] - baseline
        status = "UP" if diff > 0 else "DOWN" if diff < 0 else "SAME"
        print(f"  {task}: {result['score']:.3f} vs {baseline:.3f} baseline ({status} {abs(diff):.3f})")


if __name__ == "__main__":
    main()
