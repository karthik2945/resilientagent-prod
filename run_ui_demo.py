"""Run DQN agent and populate UI dashboard."""
import requests
import sys
import time

BASE_URL = "http://localhost:8080"

def run_agent(task_id="task1_latency_spike"):
    # Reset environment
    r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    obs = r.json()["observation"]
    
    # Post initial state to UI
    requests.post(f"{BASE_URL}/ui_state", json={
        "metrics": obs["metrics"],
        "alert_status": obs["alert_status"],
        "current_task": task_id
    })
    
    # Rule-based actions for demo
    actions = [
        {"action_type": "check_metrics", "target": "inference_service", "parameters": None},
        {"action_type": "read_logs", "target": "inference_service", "parameters": None},
        {"action_type": "optimize_batch", "target": "inference_service", "parameters": None},
        {"action_type": "verify_fix", "target": "inference_service", "parameters": None},
    ]
    
    for i, action in enumerate(actions):
        time.sleep(1)
        
        # Log the action
        requests.post(f"{BASE_URL}/ui_logs", json={
            "role": "agent",
            "action": action,
            "reasoning": f"Step {i+1}: Executing {action['action_type']} on {action['target']}"
        })
        
        # Execute action
        r = requests.post(f"{BASE_URL}/step", json={"action": action})
        result = r.json()
        obs = result.get("observation", result)
        
        # Update UI state
        requests.post(f"{BASE_URL}/ui_state", json={
            "metrics": obs.get("metrics", obs),
            "alert_status": obs.get("alert_status", "unknown"),
            "current_task": task_id
        })
        
        # Log result
        requests.post(f"{BASE_URL}/ui_logs", json={
            "role": "system",
            "reasoning": f"Reward: {obs.get('reward', 0):.2f}, Done: {obs.get('done', False)}"
        })
        
        if obs.get("done"):
            break
    
    print("Agent finished!")

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "task1_latency_spike"
    run_agent(task)
