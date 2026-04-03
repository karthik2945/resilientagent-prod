ResilientAgent — ML Production Incident Simulator

An OpenEnv environment for training autonomous SRE agents to diagnose and resolve real-world ML production incidents.

## Features
- **3 Incident Scenarios**: Latency spikes, prediction drift, cascading failures
- **Dynamic Grading**: 6-factor scoring (health, efficiency, root-cause, quality, metrics, sequence)
- **Interactive Dashboard**: Real-time visualization of environment state and scores
- **LLM-Ready**: Works with OpenAI, Groq, HuggingFace models via Environment Variables

## Quick Start

1. **Access the Dashboard**: Visit this Space's URL
2. **Select a Task**: Choose latency_spike, prediction_drift, or cascading_failure
3. **View Scoring**: Watch real-time scores and metrics
4. **Run Baseline**: Click "Baseline" to see reference performance (0.955 avg)

## API Endpoints

```
POST /reset              → Start new episode
POST /step               → Execute action
GET  /state              → Get current state
GET  /baseline           → Run full evaluation
POST /grader             → Get task score
GET  /tasks              → List available tasks
```

## Scoring (Out of 1.0)

- **Health**: System recovery status (0-0.25)
- **Efficiency**: Steps to resolution (0-0.25)
- **Root Cause**: Correct diagnosis (0-0.15)
- **Quality**: Action relevance (0-0.15)
- **Metrics**: Improvement rate (0-0.20)
- **Sequence**: Optimal ordering (0-0.10)

## Environment Variables (Optional)

```
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4
HF_TOKEN=your_api_key
```

## Baseline Scores

| Task | Score | Steps |
|------|-------|-------|
| Latency Spike | 0.905 | 3 |
| Prediction Drift | 1.000 | 4 |
| Cascading Failure | 0.960 | 5 |
| **Average** | **0.955** | - |

## References

- [OpenEnv Framework](https://github.com/facebookresearch/open_env)
- [ResilientAgent GitHub](https://github.com/caffeinated-coders/resilientagent-prod)
- [Meta PyTorch Hackathon](https://scaler.com/school-of-technology/meta-pytorch-hackathon)
