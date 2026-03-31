#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Deep Q-Network (DQN) Agent for ResilientAgent Production Environment.

A PyTorch-based reinforcement learning agent that learns to diagnose and
resolve ML production incidents by interacting with the environment and
maximising cumulative reward through experience replay and target networks.
"""

import math
import random
from collections import deque, namedtuple
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_SPACE: List[str] = [
    "check_metrics", "read_logs", "check_deployment",
    "analyze_drift", "scale_service", "rollback_model",
    "optimize_batch", "restart_service", "verify_fix", "notify_team",
]
NUM_ACTIONS = len(ACTION_SPACE)

# Default target names per action (used when the agent picks an action)
DEFAULT_TARGETS: Dict[str, str] = {
    "check_metrics": "inference_service",
    "read_logs": "inference_service",
    "check_deployment": "inference_service",
    "analyze_drift": "ml_model",
    "scale_service": "fallback_model",
    "rollback_model": "ml_model",
    "optimize_batch": "inference_service",
    "restart_service": "primary_model",
    "verify_fix": "inference_service",
    "notify_team": "ops_team",
}

# Observation vector keys – the agent reads these from the metrics dict
STATE_KEYS: List[str] = [
    "latency_p99", "error_rate", "cpu_util", "gpu_util",
    "memory_util", "throughput", "queue_depth",
    "accuracy", "drift_score",
    # Multi-service prefixed keys
    "primary_model_latency_p99", "primary_model_error_rate",
    "primary_model_memory_util", "fallback_model_latency_p99",
    "fallback_model_error_rate", "fallback_model_cpu_util",
]
STATE_DIM = len(STATE_KEYS) + NUM_ACTIONS

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------

class DQN(nn.Module):
    """
    Dueling Deep Q-Network architecture with layer normalisation.

    Uses a shared feature extractor followed by separate value and
    advantage streams, producing more stable Q-value estimates.
    """

    def __init__(self, state_dim: int = STATE_DIM, num_actions: int = NUM_ACTIONS, hidden: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Combine: Q(s,a) = V(s) + A(s,a) - mean(A)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Experience replay buffer with uniform sampling."""

    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    DQN Agent for ML Ops incident response.

    Implements:
      - Epsilon-greedy exploration with decay
      - Double DQN target updates
      - Dueling network architecture
      - Experience replay
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        num_actions: int = NUM_ACTIONS,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 1000,
        batch_size: int = 64,
        target_update: int = 10,
        buffer_size: int = 50_000,
        device: Optional[str] = None,
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.policy_net = DQN(state_dim, num_actions).to(self.device)
        self.target_net = DQN(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)

        self.steps_done: int = 0
        self.episodes_done: int = 0

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-self.steps_done / self.epsilon_decay)

    def observation_to_state(self, obs: Dict[str, Any], last_action: Optional[int] = None) -> torch.Tensor:
        """Convert observation metrics dict to a normalised state tensor."""
        metrics = obs.get("metrics", obs)  # handle both raw dict and obs dict
        vec = []
        for key in STATE_KEYS:
            val = metrics.get(key, 0.0)
            vec.append(float(val))

        # Normalise
        arr = np.array(vec, dtype=np.float32)
        # Clamp extreme values
        arr = np.clip(arr, -1e4, 1e4)
        # Log-scale large values (latency, throughput)
        for i, key in enumerate(STATE_KEYS):
            if "latency" in key or "throughput" in key or "queue" in key:
                arr[i] = np.log1p(max(0, arr[i]))

        # Add one-hot encoded last_action
        action_vec = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if last_action is not None and 0 <= last_action < NUM_ACTIONS:
            action_vec[last_action] = 1.0
            
        arr = np.concatenate([arr, action_vec])

        return torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)

    def select_action(self, state: torch.Tensor, greedy: bool = False) -> int:
        """Select an action using epsilon-greedy policy."""
        self.steps_done += 1
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.argmax(dim=1).item()

    def action_to_dict(self, action_idx: int, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Convert action index to an environment action dict."""
        action_type = ACTION_SPACE[action_idx]

        # Smart target selection based on task context
        target = DEFAULT_TARGETS.get(action_type, "inference_service")
        if task_id:
            if "cascading" in task_id:
                if action_type == "restart_service":
                    target = "primary_model"
                elif action_type == "scale_service":
                    target = "fallback_model"
                elif action_type in ("check_metrics", "read_logs"):
                    target = "primary_model"
            elif "drift" in task_id or "prediction" in task_id:
                target = "ml_model"
            elif "latency" in task_id:
                target = "inference_service"

        return {
            "action_type": action_type,
            "target": target,
            "parameters": None,
        }

    def learn(self) -> Optional[float]:
        """Perform one gradient update from replay buffer. Returns loss."""
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        # Current Q values
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        # Double DQN: use policy net to select actions, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # Huber loss for stability
        loss = F.smooth_l1_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self) -> None:
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "episodes_done": self.episodes_done,
        }, path)

    def load(self, path: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]
        self.episodes_done = checkpoint["episodes_done"]
