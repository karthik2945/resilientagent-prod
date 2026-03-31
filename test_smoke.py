"""Quick smoke test for the upgraded environment and DQN agent."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.resilientagent_prod_environment import ResilientAgentEnvironment
from models import ResilientAgentAction
from baseline.agent import DQNAgent

def test_environment():
    print("=" * 60)
    print("TESTING ENVIRONMENT SIMULATION")
    print("=" * 60)
    env = ResilientAgentEnvironment()

    for task_id in ["task1_latency_spike", "task2_prediction_drift", "task3_cascading_failure"]:
        print(f"\n--- {task_id} ---")
        obs = env.reset(task_id=task_id)
        print(f"  Alert: {obs.alert_status}")
        print(f"  Metrics keys: {list(obs.metrics.keys())}")
        print(f"  Latency: {obs.metrics.get('latency_p99', 'N/A')}")
        print(f"  Logs: {obs.recent_logs[0][:60]}...")

        # Take a few actions
        actions_map = {
            "task1_latency_spike": ["check_metrics", "read_logs", "optimize_batch", "verify_fix"],
            "task2_prediction_drift": ["analyze_drift", "check_deployment", "rollback_model", "verify_fix"],
            "task3_cascading_failure": ["check_metrics", "read_logs", "restart_service", "scale_service", "verify_fix"],
        }
        targets_map = {
            "task1_latency_spike": "inference_service",
            "task2_prediction_drift": "ml_model",
            "task3_cascading_failure": "primary_model",
        }

        for i, act_type in enumerate(actions_map[task_id]):
            target = targets_map[task_id]
            if act_type == "scale_service":
                target = "fallback_model"
            action = ResilientAgentAction(action_type=act_type, target=target)
            obs = env.step(action)
            lat = obs.metrics.get("latency_p99", obs.metrics.get("primary_model_latency_p99", "N/A"))
            print(f"  Step {i+1} ({act_type}): reward={obs.reward:+.4f}, done={obs.done}, latency={lat}")
            if obs.done:
                break

        grade = env.grade()
        print(f"  GRADE: {grade:.4f}")
        assert 0.0 <= grade <= 1.0, f"Grade out of range: {grade}"

    print("\n✓ Environment tests PASSED")


def test_dqn_agent():
    print("\n" + "=" * 60)
    print("TESTING DQN AGENT (PyTorch)")
    print("=" * 60)
    env = ResilientAgentEnvironment()
    agent = DQNAgent(epsilon_start=0.5, epsilon_decay=50)

    task_id = "task1_latency_spike"
    obs = env.reset(task_id=task_id)
    state = agent.observation_to_state(obs.model_dump())
    print(f"  State shape: {state.shape}")
    print(f"  State values: {state[0][:5].tolist()}")

    total_reward = 0
    for step in range(5):
        action_idx = agent.select_action(state)
        action_dict = agent.action_to_dict(action_idx, task_id)
        action = ResilientAgentAction(**action_dict)
        obs = env.step(action)
        reward = obs.reward
        done = obs.done

        next_state = agent.observation_to_state(obs.model_dump())
        agent.memory.push(state, action_idx, reward, next_state, float(done))

        loss = agent.learn()
        state = next_state
        total_reward += reward

        print(f"  Step {step+1}: action={action_dict['action_type']}, reward={reward:+.4f}, loss={loss}, eps={agent.epsilon:.3f}")
        if done:
            break

    grade = env.grade()
    print(f"  Total reward: {total_reward:+.4f}")
    print(f"  Grade: {grade:.4f}")
    print(f"  Model params: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
    print("\n✓ DQN Agent tests PASSED")


if __name__ == "__main__":
    test_environment()
    test_dqn_agent()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
