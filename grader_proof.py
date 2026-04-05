# 1. Proving 'Graders that always return the same score' is FALSE
import os
import sys

# Change working dir to import server code
sys.path.insert(0, os.path.abspath("server"))
from resilientagent_prod_environment import ResilientAgentEnvironment
from models import ResilientAgentAction

env = ResilientAgentEnvironment()

# TEST 1: Optimal Run (Task 1)
obs = env.reset(task_id="task1_latency_spike")
correct_actions = ["check_metrics", "read_logs", "optimize_batch", "verify_fix"]
for action_type in correct_actions:
    env.step(ResilientAgentAction(action_type=action_type, target="inference_service"))
score_optimal = env.grade()
print(f"Optimal Agent Score: {score_optimal:.3f}")

# TEST 2: Bad Agent Run (Wasting actions, never solving)
obs = env.reset(task_id="task1_latency_spike")
bad_actions = ["notify_team", "read_logs", "restart_service"]
for action_type in bad_actions:
    env.step(ResilientAgentAction(action_type=action_type, target="finance_db"))
score_bad = env.grade()
print(f"Bad Agent Score: {score_bad:.3f}")

# TEST 3: Partial Agent Run (Did diagnosis, but no fix)
obs = env.reset(task_id="task1_latency_spike")
env.step(ResilientAgentAction(action_type="check_metrics", target="inference_service"))
score_partial = env.grade()
print(f"Partial Agent Score: {score_partial:.3f}")
