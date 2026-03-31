import os
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
import importlib

from src.models import MLAction, MLObservation, MLReward, MLState


class ResilientAgentEnv:
    MAX_STEPS = 20
    TARGET_TIMES = {
        "latency_spike": 300,
        "prediction_drift": 600,
        "cascading_failure": 900
    }

    def __init__(self):
        self.task_id: Optional[str] = None
        self.step_count: int = 0
        self._state: Optional[MLState] = None
        self.incident_start: Optional[float] = None
        self.target_time: float = 300.0
        self._primary_restarted: bool = False
        self._fallback_scaled: bool = False

    def reset(self, task_id: str) -> Tuple[MLObservation, Dict]:
        self.task_id = task_id
        self.step_count = 0
        self.incident_start = datetime.now().timestamp()
        self._primary_restarted = False
        self._fallback_scaled = False

        # Extract task type from task_id (e.g., "task1_latency_spike" -> "latency_spike")
        task_type = task_id.split("_", 1)[1] if "_" in task_id else task_id
        self.target_time = self.TARGET_TIMES.get(task_type, 300.0)

        # Load initial scenario from tasks folder
        try:
            task_module = importlib.import_module(f"src.tasks.{task_id}")
            initial_state = task_module.get_initial_state()
        except (ImportError, AttributeError):
            initial_state = self._default_initial_state(task_id)

        self._state = MLState(
            task_id=task_id,
            services=initial_state.get("services", {}),
            metrics=initial_state.get("metrics", {}),
            logs=initial_state.get("logs", []),
            incident_start=self.incident_start,
            time_to_resolution=None,
            model_healthy=initial_state.get("model_healthy", False),
            actions_taken=[],
            wasted_actions=0,
            root_cause_identified=None,
            fix_applied=None,
            step_count=0
        )

        observation = self._state_to_observation(self._state)
        return observation, {}

    def step(self, action: MLAction) -> Tuple[MLObservation, float, bool, bool, Dict]:
        self.step_count += 1

        # Process action and update state
        new_state = self._process_action(action)
        self._state = new_state

        # Calculate reward based on action correctness
        reward = self._calculate_reward(action)

        # Check termination conditions
        terminated = self._state.model_healthy
        truncated = self.step_count >= self.MAX_STEPS

        observation = self._state_to_observation(self._state)
        info = {"step": self.step_count, "action": action.action_type}

        return observation, reward, terminated, truncated, info

    def state(self) -> MLState:
        return self._state

    def grade(self) -> float:
        if not self._state:
            return 0.0

        # Health score (25%)
        health_score = 0.25 if self._state.model_healthy else 0.0

        # Gate: if not resolved, cap score severely
        if not self._state.model_healthy:
            diagnostic_credit = 0.0
            if self._state.root_cause_identified is not None:
                diagnostic_credit += 0.03
            waste_penalty = min(0.03, self._state.wasted_actions * 0.005)
            return max(0.0, diagnostic_credit - waste_penalty)

        # Time score (25%) — step-based efficiency
        time_score = 0.0
        if self._state.model_healthy:
            steps_taken = self._state.step_count
            correct_actions = self._get_correct_actions_for_task()
            optimal_steps = len(correct_actions) if correct_actions else 4
            if steps_taken <= optimal_steps:
                time_score = 0.25
            elif steps_taken <= optimal_steps + 2:
                time_score = 0.15
            elif steps_taken <= optimal_steps + 5:
                time_score = 0.08

        # Root cause score (15%)
        root_cause_score = 0.15 if self._state.root_cause_identified is not None else 0.0

        # Efficiency score (15%)
        if self._state.wasted_actions == 0:
            efficiency_score = 0.15
        elif self._state.wasted_actions <= 2:
            efficiency_score = 0.08
        else:
            efficiency_score = 0.0

        # Metric-based score (10%)
        metric_score = 0.0
        metrics = self._state.metrics

        latency_ok = metrics.get("latency_p99", 5000) < 1500.0
        error_ok = metrics.get("error_rate", 0.5) < 0.1
        accuracy_ok = metrics.get("accuracy", 0.0) > 0.85 or "accuracy" not in metrics
        cpu_ok = metrics.get("cpu_util", 0.5) < 0.9

        if latency_ok:
            metric_score += 0.04
        if error_ok:
            metric_score += 0.03
        if accuracy_ok:
            metric_score += 0.02
        if cpu_ok:
            metric_score += 0.01

        # Sequence correctness bonus (up to 10% extra)
        sequence_bonus = 0.0
        correct_actions = self._get_correct_actions_for_task()
        if correct_actions and len(self._state.actions_taken) > 0:
            matches = sum(1 for i, action in enumerate(self._state.actions_taken)
                         if i < len(correct_actions) and action == correct_actions[i])
            sequence_bonus = 0.1 * (matches / len(correct_actions))

        total = health_score + time_score + root_cause_score + efficiency_score + metric_score + sequence_bonus
        return min(1.0, max(0.0, total))

    def _process_action(self, action: MLAction) -> MLState:
        actions_taken = self._state.actions_taken + [action.action_type]
        wasted_actions = self._state.wasted_actions

        # Check if action is wasted (simplified logic)
        if not self._is_useful_action(action):
            wasted_actions += 1

        # Update services and metrics based on action
        services = self._state.services.copy()
        metrics = self._state.metrics.copy()
        logs = self._state.logs.copy()

        model_healthy = self._state.model_healthy
        root_cause_identified = self._state.root_cause_identified
        fix_applied = self._state.fix_applied
        time_to_resolution = self._state.time_to_resolution

        # Apply action effects
        if action.action_type == "check_metrics":
            pass
        elif action.action_type == "read_logs":
            logs.append(f"[{self.step_count}] Log entry from {action.target}")
        elif action.action_type == "check_deployment":
            pass
        elif action.action_type == "analyze_drift":
            if "drift" in self.task_id or "prediction" in self.task_id:
                root_cause_identified = "model_drift"
        elif action.action_type == "scale_service":
            if action.target in services:
                services[action.target] = "up"
        elif action.action_type == "rollback_model":
            # Only resolves drift/prediction tasks — NOT latency or cascading
            if self.task_id and ("drift" in self.task_id or "prediction" in self.task_id):
                # Fix applied but verify_fix should mark as healthy
                time_to_resolution = datetime.now().timestamp()
                fix_applied = "rollback_model"
            else:
                # Wrong fix for this task
                wasted_actions += 1
                logs.append(f"[{self.step_count}] rollback_model has no effect on this incident type")
        elif action.action_type == "optimize_batch":
            if "latency" in self.task_id:
                model_healthy = True
                time_to_resolution = datetime.now().timestamp()
                fix_applied = "optimize_batch"
        elif action.action_type == "restart_service":
            if action.target in services:
                services[action.target] = "up"
                # Track health improvement for cascading failure
                if "cascading" in self.task_id and action.target == "primary_model":
                    self._primary_restarted = True
                    logs.append(f"[{self.step_count}] Primary model restarted successfully")
        elif action.action_type == "scale_service":
            if action.target in services:
                services[action.target] = "up"
                # Track health improvement for cascading failure
                if "cascading" in self.task_id and action.target == "fallback_model":
                    self._fallback_scaled = True
                    logs.append(f"[{self.step_count}] Fallback model scaled up successfully")
        elif action.action_type == "verify_fix":
            # For cascading failure, verify_fix sets model_healthy if both services are up
            if "cascading" in self.task_id:
                primary_up = services.get("primary_model") == "up"
                fallback_up = services.get("fallback_model") == "up"
                if primary_up and fallback_up:
                    model_healthy = True
                    time_to_resolution = datetime.now().timestamp()
                    fix_applied = "verify_fix"
                    logs.append(f"[{self.step_count}] Fix verified - all services healthy")
        elif action.action_type == "notify_team":
            logs.append(f"[{self.step_count}] Team notified about {action.target}")

        return MLState(
            task_id=self._state.task_id,
            services=services,
            metrics=metrics,
            logs=logs,
            incident_start=self._state.incident_start,
            time_to_resolution=time_to_resolution,
            model_healthy=model_healthy,
            actions_taken=actions_taken,
            wasted_actions=wasted_actions,
            root_cause_identified=root_cause_identified,
            fix_applied=fix_applied,
            step_count=self.step_count
        )

    def _is_useful_action(self, action: MLAction) -> bool:
        # Simplified logic - determine if action is useful based on current state
        useful_actions = set()
        if "latency" in self.task_id:
            useful_actions = {"check_metrics", "optimize_batch", "read_logs", "check_deployment"}
        elif "drift" in self.task_id or "prediction" in self.task_id:
            useful_actions = {"check_metrics", "analyze_drift", "rollback_model", "check_deployment", "verify_fix"}
        elif "cascading" in self.task_id:
            useful_actions = {"check_metrics", "read_logs", "restart_service", "scale_service", "verify_fix", "check_deployment", "notify_team"}

        return action.action_type in useful_actions

    def _get_correct_actions_for_task(self) -> List[str]:
        """Get the correct action sequence for the current task from ground truth."""
        if self.task_id is None:
            return []
        try:
            task_module = importlib.import_module(f"src.tasks.{self.task_id}")
            return task_module.get_correct_actions()
        except (ImportError, AttributeError):
            # Fallback correct sequences
            if "latency" in self.task_id:
                return ["check_metrics", "read_logs", "optimize_batch", "verify_fix"]
            elif "drift" in self.task_id or "prediction" in self.task_id:
                return ["analyze_drift", "check_deployment", "rollback_model", "verify_fix"]
            elif "cascading" in self.task_id:
                return ["check_metrics", "read_logs", "restart_service", "scale_service", "verify_fix"]
            return []

    def _calculate_reward(self, action: MLAction) -> float:
        if self._state.model_healthy:
            return 1.0

        if self.task_id is None:
            return -0.3

        correct_actions = self._get_correct_actions_for_task()
        current_step = len(self._state.actions_taken) - 1

        if current_step < len(correct_actions):
            expected = correct_actions[current_step]
            if action.action_type == expected:
                return 0.15
            else:
                return -0.3

        if current_step >= len(correct_actions):
            return -0.2

        return -0.3

    def _state_to_observation(self, state: MLState) -> MLObservation:
        time_elapsed = datetime.now().timestamp() - state.incident_start

        return MLObservation(
            metrics=state.metrics,
            recent_logs=state.logs[-10:],
            alert_status="healthy" if state.model_healthy else "critical",
            time_elapsed=time_elapsed,
            last_action_result=state.actions_taken[-1] if state.actions_taken else "none",
            root_cause_hint=state.root_cause_identified
        )

    def _default_initial_state(self, task_id: str) -> Dict[str, Any]:
        if "latency" in task_id:
            return {
                "services": {"api": "up", "inference": "up"},
                "metrics": {"latency_p99": 500.0, "error_rate": 0.05, "gpu_util": 0.8, "throughput": 100.0},
                "logs": ["High latency detected on inference service"],
                "model_healthy": False
            }
        elif "drift" in task_id or "prediction" in task_id:
            return {
                "services": {"api": "up", "inference": "up"},
                "metrics": {"latency_p99": 50.0, "error_rate": 0.15, "gpu_util": 0.6, "throughput": 200.0},
                "logs": ["Prediction accuracy degraded - drift detected"],
                "model_healthy": False
            }
        else:
            return {
                "services": {"api": "up", "inference": "down", "cache": "degraded"},
                "metrics": {"latency_p99": 200.0, "error_rate": 0.25, "gpu_util": 0.3, "throughput": 50.0},
                "logs": ["Cascading failure detected - multiple services affected"],
                "model_healthy": False
            }

