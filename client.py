# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ResilientAgent Production Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ResilientAgentAction, ResilientAgentObservation


class ResilientAgentEnv(
    EnvClient[ResilientAgentAction, ResilientAgentObservation, State]
):
    """
    Client for the ResilientAgent Production Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with ResilientAgentEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task_id="task1_latency_spike")
        ...     print(result.observation.alert_status)
        ...
        ...     result = client.step(ResilientAgentAction(action_type="check_metrics", target="inference_service"))
        ...     print(result.observation.metrics["latency_p99"])

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ResilientAgentEnv.from_docker_image("resilientagent_prod-env:latest")
        >>> try:
        ...     result = client.reset(task_id="task1_latency_spike")
        ...     result = client.step(ResilientAgentAction(action_type="check_metrics", target="inference_service"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ResilientAgentAction) -> Dict:
        """
        Convert ResilientAgentAction to JSON payload for step message.

        Args:
            action: ResilientAgentAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type,
            "target": action.target,
            "parameters": action.parameters,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ResilientAgentObservation]:
        """
        Parse server response into StepResult[ResilientAgentObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with ResilientAgentObservation
        """
        obs_data = payload.get("observation", {})
        observation = ResilientAgentObservation(
            metrics=obs_data.get("metrics", {}),
            recent_logs=obs_data.get("recent_logs", []),
            alert_status=obs_data.get("alert_status", "critical"),
            time_elapsed=obs_data.get("time_elapsed", 0.0),
            last_action_result=obs_data.get("last_action_result", "none"),
            root_cause_hint=obs_data.get("root_cause_hint"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
