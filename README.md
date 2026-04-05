---
title: ResilientAgent Prod
emoji: 🚀
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
# ResilientAgent-Prod

## Overview
OpenEnv environment for training AI agents to diagnose and resolve ML model production incidents. The agent observes system metrics, logs, and alerts, then takes targeted remediation actions to restore service health.

Built for the **Meta PyTorch OpenEnv Hackathon 2026**.

## Real-World Utility
ML model failures cost companies millions in downtime. This environment trains agents to reduce Mean Time To Resolution (MTTR) from 30 minutes to under 5 minutes, teaching them the correct diagnostic and remediation sequence for each incident class.

## Action Space

| Action | Description |
|--------|-------------|
| `check_metrics` | Query current system metrics (latency, error rate, GPU utilization, throughput) |
| `read_logs` | Read recent log entries from specified service |
| `check_deployment` | Verify deployment status and configuration |
| `analyze_drift` | Analyze model prediction drift and feature distribution |
| `scale_service` | Scale up/down specified service resources |
| `rollback_model` | Rollback to previous model version (only effective for drift incidents) |
| `optimize_batch` | Optimize batch size for inference throughput (only effective for latency incidents) |
| `restart_service` | Restart a degraded or failed service |
| `verify_fix` | Verify that the incident has been resolved |
| `notify_team` | Send alert notification to operations team |

## Observation Space

| Field | Description |
|-------|-------------|
| `metrics` | Dict with latency_p99, error_rate, gpu_util, throughput (and task-specific metrics) |
| `recent_logs` | List of last 10 log entries from monitored services |
| `alert_status` | Current alert level: `"healthy"` or `"critical"` |
| `time_elapsed` | Seconds since incident started |
| `last_action_result` | Description of previous action outcome |
| `root_cause_hint` | Optional hint if root cause has been identified |

## Tasks

### task1_latency_spike (easy)
GPU memory exhaustion causing ML model inference latency to spike from 200ms to 5000ms.

**Correct sequence:** `check_metrics` → `read_logs` → `optimize_batch` → `verify_fix`

### task2_prediction_drift (medium)
Data pipeline schema change causing model accuracy to drop 15% overnight.

**Correct sequence:** `analyze_drift` → `check_deployment` → `rollback_model` → `verify_fix`

### task3_cascading_failure (hard)
Primary model OOM killed, fallback service degrading, autoscaler not triggering due to memory leak.

**Correct sequence:** `check_metrics` → `read_logs` → `restart_service(primary_model)` → `scale_service(fallback_model)` → `verify_fix`

## Grading Breakdown

The environment uses a strict, multi-factor grading system designed to prevent reward hacking:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Health** | 25% | Incident must be fully resolved (`model_healthy = True`) |
| **Step Efficiency** | 25% | Fewer steps = higher score (optimal: matching correct sequence length) |
| **Root Cause** | 15% | Bonus for correctly identifying the root cause |
| **Action Efficiency** | 15% | Penalty for wasted/irrelevant actions |
| **Metric Quality** | 10% | Final system metrics must be within healthy thresholds |
| **Sequence Correctness** | 10% | Bonus for following the exact correct action order |

> **Important:** If the incident is NOT resolved, the maximum possible score is **0.03** (diagnostic credit only). There is no partial credit for failed episodes.

## Reward Design

- **Correct action at correct step:** `+0.15` (with up to `+0.05` target bonus)
- **Wrong action:** `-0.3` (harsh penalty to prevent random exploration from succeeding)
- **Extra actions beyond sequence:** `-0.2`
- **Episode resolved:** `+1.0`

Random agents average score: **~0.0**  
Optimal agents average score: **~0.95**

## Verified LLM Performance

Tested with `llama-3.3-70b-versatile` (Groq):

| Task | Score | Steps | Status |
|------|-------|-------|--------|
| latency_spike | **0.905** | 3 | ✅ Resolved |
| prediction_drift | **1.000** | 4 | ✅ Resolved |
| cascading_failure | **0.960** | 5 | ✅ Resolved |
| **Average** | **0.955** | - | **3/3 Resolved** |

## Setup

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Docker (Hugging Face Spaces Ready)

```bash
docker build -t resilientagent-prod .
docker run -p 7860:7860 resilientagent-prod
```

## API Usage

### Reset environment for a task
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_latency_spike"}'
```

### Execute an action
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "check_metrics",
    "target": "inference_service",
    "parameters": null
  }'
```

### Get current state
```bash
curl http://localhost:7860/state
```

### Get final grade
```bash
curl -X POST http://localhost:7860/grader
```

### List all tasks
```bash
curl http://localhost:7860/tasks
```

### Run baseline agent
```bash
curl http://localhost:7860/baseline
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | **Yes** | The API endpoint for the LLM. |
| `MODEL_NAME` | **Yes** | The model identifier to use for inference. |
| `HF_TOKEN` | **Yes** | Your Hugging Face / API key for the LLM. |

## Requirements

- Python 3.11+
- PyTorch 2.0+
- FastAPI 0.104+
- Uvicorn 0.24+
- Pydantic v2
- OpenAI Python client (`openai`)
- `python-dotenv`

