# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the ResilientAgent Production Environment.

The resilientagent-prod environment simulates ML model production incidents
including latency spikes, prediction drift, and cascading failures.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional, Dict, Any, List, Literal


class ResilientAgentAction(Action):
    """Action for the ResilientAgent environment - ML ops remediation actions."""

    action_type: Literal[
        "check_metrics", "read_logs", "check_deployment",
        "analyze_drift", "scale_service", "rollback_model", "optimize_batch",
        "restart_service", "verify_fix", "notify_team"
    ] = Field(..., description="Type of remediation action to execute")
    target: str = Field(..., description="Target service for the action")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Optional action parameters")


class ResilientAgentObservation(Observation):
    """Observation from the ResilientAgent environment - system metrics and logs."""

    metrics: Dict[str, float] = Field(default_factory=dict, description="Current system metrics")
    recent_logs: List[str] = Field(default_factory=list, description="Recent log entries")
    alert_status: str = Field(default="critical", description="Current alert status: healthy or critical")
    time_elapsed: float = Field(default=0.0, description="Seconds since incident started")
    last_action_result: str = Field(default="none", description="Result of last action taken")
    root_cause_hint: Optional[str] = Field(default=None, description="Hint if root cause identified")
    done: bool = Field(default=False, description="Whether the episode is complete")
    reward: float = Field(default=0.0, description="Reward for the current step")
