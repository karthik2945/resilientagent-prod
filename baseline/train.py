#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Training script for the DQN agent on the ResilientAgent environment.

Trains a PyTorch Deep Q-Network to solve ML production incidents by
interacting directly with the environment simulation. Produces loss
curves and evaluation metrics to demonstrate learning.

Usage:
    python -m baseline.train --episodes 500 --eval-every 50
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.agent import DQNAgent, ACTION_SPACE
from server.resilientagent_prod_environment import ResilientAgentEnvironment
from models import ResilientAgentAction


def train(
    num_episodes: int = 2000,
    eval_every: int = 50,
    save_path: str = "baseline/checkpoints/dqn_agent.pt",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train the DQN agent across all three task types.

    Args:
        num_episodes: Total training episodes.
        eval_every: Evaluate greedy policy every N episodes.
        save_path: Path to save best model checkpoint.
        verbose: Print progress.

    Returns:
        Dictionary with training statistics.
    """
    agent = DQNAgent(
        lr=3e-4,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=num_episodes * 10,
        batch_size=64,
        target_update=10,
    )

    env = ResilientAgentEnvironment()
    tasks = ["task1_latency_spike", "task2_prediction_drift", "task3_cascading_failure"]

    # Training stats
    episode_rewards: List[float] = []
    episode_losses: List[float] = []
    eval_scores: List[Dict[str, float]] = []
    best_avg_score = 0.0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    start_time = time.time()

    for episode in range(num_episodes):
        # Cycle through tasks
        task_id = tasks[episode % len(tasks)]

        # Reset environment
        obs = env.reset(task_id=task_id)
        last_action = None
        state = agent.observation_to_state(obs.model_dump(), last_action)

        total_reward = 0.0
        losses = []

        for step in range(env.MAX_STEPS):
            # Select action
            action_idx = agent.select_action(state)
            action_dict = agent.action_to_dict(action_idx, task_id)

            # Step environment
            action = ResilientAgentAction(**action_dict)
            obs = env.step(action)
            reward = obs.reward
            done = obs.done

            next_state = agent.observation_to_state(obs.model_dump(), action_idx)

            # Store transition
            agent.memory.push(state, action_idx, reward, next_state, float(done))

            # Learn
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            state = next_state
            last_action = action_idx
            total_reward += reward

            if done:
                break

        # Update target network periodically
        agent.episodes_done += 1
        if agent.episodes_done % agent.target_update == 0:
            agent.update_target()

        episode_rewards.append(total_reward)
        avg_loss = np.mean(losses) if losses else 0.0
        episode_losses.append(avg_loss)

        if verbose and (episode + 1) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Task: {task_id.split('_', 1)[1]:<20s} | "
                f"Reward: {total_reward:+.3f} | "
                f"Avg(10): {avg_reward:+.3f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        # Evaluation
        if (episode + 1) % eval_every == 0:
            eval_result = evaluate(agent, env, tasks, verbose=verbose)
            eval_scores.append(eval_result)

            avg_score = np.mean(list(eval_result.values()))
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save(save_path)
                if verbose:
                    print(f"  ★ New best model saved (avg score: {avg_score:.4f})")

    elapsed = time.time() - start_time

    # Final evaluation
    final_eval = evaluate(agent, env, tasks, verbose=verbose)

    results = {
        "total_episodes": num_episodes,
        "training_time_seconds": round(elapsed, 2),
        "best_avg_score": round(best_avg_score, 4),
        "final_eval": final_eval,
        "final_epsilon": round(agent.epsilon, 4),
        "total_steps": agent.steps_done,
        "reward_history": [round(r, 4) for r in episode_rewards],
        "loss_history": [round(l, 6) for l in episode_losses],
    }

    # Save training results
    results_path = os.path.join(os.path.dirname(save_path), "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"TRAINING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Episodes: {num_episodes}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Best avg score: {best_avg_score:.4f}")
        print(f"Final scores: {json.dumps(final_eval, indent=2)}")
        print(f"Model saved to: {save_path}")
        print(f"Results saved to: {results_path}")

    return results


def evaluate(
    agent: DQNAgent,
    env: ResilientAgentEnvironment,
    tasks: List[str],
    num_runs: int = 3,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evaluate the agent greedily (no exploration) on all tasks.

    Returns:
        Dict mapping task name to average grade score.
    """
    scores: Dict[str, List[float]] = {t: [] for t in tasks}

    for task_id in tasks:
        for _ in range(num_runs):
            obs = env.reset(task_id=task_id)
            last_action = None
            state = agent.observation_to_state(obs.model_dump(), last_action)

            for step in range(env.MAX_STEPS):
                action_idx = agent.select_action(state, greedy=True)
                action_dict = agent.action_to_dict(action_idx, task_id)
                action = ResilientAgentAction(**action_dict)
                obs = env.step(action)
                state = agent.observation_to_state(obs.model_dump(), action_idx)
                if obs.done:
                    break

            grade = env.grade()
            scores[task_id].append(grade)

    result = {}
    for task_id in tasks:
        name = task_id.split("_", 1)[1]
        avg = np.mean(scores[task_id])
        result[name] = round(float(avg), 4)
        if verbose:
            print(f"  Eval {name}: {avg:.4f} (runs: {[round(s, 3) for s in scores[task_id]]})")

    return result


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent for ResilientAgent environment")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--eval-every", type=int, default=50, help="Evaluate every N episodes")
    parser.add_argument("--save-path", type=str, default="baseline/checkpoints/dqn_agent.pt",
                        help="Path to save model checkpoint")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        eval_every=args.eval_every,
        save_path=args.save_path,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
