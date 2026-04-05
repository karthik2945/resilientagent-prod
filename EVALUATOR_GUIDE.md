# EVALUATOR SETUP GUIDE

## For Hackathon Evaluators/Graders

This document explains how to run ResilientAgent-Prod with your own API credentials.

---

## 📋 PRE-REQUISITES

1. **Python 3.11+** installed
2. **One of these API providers with valid credits:**
   - Groq (Free tier: 100k tokens/day)
   - OpenAI (Paid API)
   - Hugging Face (Free with credits)
3. **API credentials ready** (we'll inject them)

---

## 🔑 REQUIRED ENVIRONMENT VARIABLES

Before running `inference.py`, inject these THREE variables:

```bash
export API_BASE_URL="<API_ENDPOINT>"
export MODEL_NAME="<MODEL_IDENTIFIER>"
export HF_TOKEN="<API_KEY>"
```

### Option 1: Groq (Recommended for Hackathon)

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="gsk_YOUR_GROQ_API_KEY_HERE"
```

Get free API key: https://console.groq.com/keys

### Option 2: OpenAI

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4"
export HF_TOKEN="sk_YOUR_OPENAI_API_KEY_HERE"
```

### Option 3: Hugging Face

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_YOUR_HUGGINGFACE_TOKEN_HERE"
```

---

## ⚙️ SETUP STEPS

### 1. Clone/Download Repo
```bash
cd resilientagent-prod
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Inject API Credentials (CRITICAL)

**Option A: Via Command Line (Recommended)**
```bash
# Unix/Linux/Mac
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="gsk_YOUR_KEY"

# Windows PowerShell
$env:API_BASE_URL="https://api.groq.com/openai/v1"
$env:MODEL_NAME="llama-3.3-70b-versatile"
$env:HF_TOKEN="gsk_YOUR_KEY"
```

**Option B: Via .env File**
```bash
# Edit .env with your values
API_BASE_URL="https://api.groq.com/openai/v1"
MODEL_NAME="llama-3.3-70b-versatile"
HF_TOKEN="gsk_YOUR_KEY"
```

---

## 🚀 RUNNING INFERENCE

### Run All 3 Tasks with LLM
```bash
python inference.py
```

**Expected Output (Structured Logs):**
```
[START] task=latency_spike env=resilientagent-prod model=llama-3.3-70b-versatile
[STEP] step=1 action=check_metrics('inference_service') reward=0.15 done=false error=null
[STEP] step=2 action=read_logs('inference_service') reward=0.15 done=false error=null
[STEP] step=3 action=optimize_batch('inference_service') reward=1.00 done=true error=null
[END] success=true steps=3 score=0.905 rewards=0.15,0.15,1.00

[START] task=prediction_drift env=resilientagent-prod model=llama-3.3-70b-versatile
[STEP] step=1 action=analyze_drift('ml_model') reward=0.15 done=false error=null
[STEP] step=2 action=check_deployment('ml_model') reward=0.15 done=false error=null
[STEP] step=3 action=rollback_model('ml_model') reward=0.18 done=false error=null
[STEP] step=4 action=verify_fix('ml_model') reward=1.00 done=true error=null
[END] success=true steps=4 score=1.000 rewards=0.15,0.15,0.18,1.00

[START] task=cascading_failure env=resilientagent-prod model=llama-3.3-70b-versatile
[STEP] step=1 action=check_metrics('primary_model') reward=0.15 done=false error=null
[STEP] step=2 action=read_logs('primary_model') reward=0.15 done=false error=null
[STEP] step=3 action=restart_service('primary_model') reward=0.20 done=false error=null
[STEP] step=4 action=scale_service('fallback_model') reward=0.20 done=false error=null
[STEP] step=5 action=verify_fix('primary_model') reward=1.00 done=true error=null
[END] success=true steps=5 score=0.960 rewards=0.15,0.15,0.20,0.20,1.00
```

### Run Baseline (No LLM)
```bash
python evaluate.py
```

**Expected Output:**
```
Average Score: 0.955
Tasks Resolved: 3/3
latency_spike: 0.905 vs 0.800 baseline (UP 0.105)
prediction_drift: 1.000 vs 1.000 baseline (SAME 0.000)
cascading_failure: 0.960 vs 0.800 baseline (UP 0.160)
```

---

## 🐳 DOCKER DEPLOYMENT

### Build Image
```bash
docker build -t resilientagent-prod:latest .
```

### Run Container
```bash
docker run \
  -e API_BASE_URL="https://api.groq.com/openai/v1" \
  -e MODEL_NAME="llama-3.3-70b-versatile" \
  -e HF_TOKEN="gsk_YOUR_KEY" \
  -p 7860:7860 \
  resilientagent-prod:latest
```

### Health Check
```bash
curl http://localhost:7860/health
# Response: {"status": "ok"}
```

---

## ⏱️ RUNTIME REQUIREMENTS

| Requirement | Actual | Status |
|---|---|---|
| Max Runtime | **<20 min** | ~45 sec per task ✓ |
| CPU | vcpu=2 required | Uses <1 vCPU ✓ |
| Memory | 8GB required | Uses <500MB ✓ |
| Disk | 100MB required | Uses ~50MB ✓ |
| GPU | Optional | Not required ✓ |

**Total 3 Tasks: ~2-3 minutes (well within 20min limit)**

---

## 🧪 TESTING

### Validation Script
```bash
python validate.py
```

### Baseline Test
```bash
python evaluate.py
```

---

## ⚠️ TROUBLESHOOTING

### Error: "HF_TOKEN not set"
**Solution:** Inject API token via environment variable
```bash
export HF_TOKEN="your_api_key"
```

### Error: "Rate limit exceeded"
**Problem:** Free Groq tier hit 100k tokens/day limit
**Solution:** 
- Use a different API provider (OpenAI)
- Wait until next day (Groq resets daily)
- Use a paid Groq tier

### Error: "Connection refused"
**Problem:** Can't reach API endpoint
**Solution:** 
- Verify API_BASE_URL is correct
- Check internet connectivity
- Verify API credentials are valid

### Docker build fails
```bash
# Clear docker cache
docker build --no-cache -t resilientagent-prod:latest .
```

---

## 📊 OUTPUT FORMAT SPECIFICATION

All output strictly follows this format (no deviations):

```
[START] task=<task> env=<env> model=<model>
[STEP] step=<int> action=<action_str> reward=<float> done=<bool> error=<null_or_error>
...
[END] success=<bool> steps=<int> score=<float> rewards=<csv_floats>
```

**Parser Requirements:**
- [START] begins a task run
- [STEP] lines are iterable steps
- [END] scores the task
- Each line is complete (no newlines within)
- Error is "null" when no error, or error message string

---

## 🎯 EXPECTED RESULTS

With a proper LLM API (no rate limits), you should see:

| Task | Score | Resolution |
|---|---|---|
| latency_spike | 0.905 | ✓ RESOLVED |
| prediction_drift | 1.000 | ✓ RESOLVED |
| cascading_failure | 0.960 | ✓ RESOLVED |
| **Average** | **0.955** | **3/3 resolved** |

**Verified Performance:**
- latency_spike: 3 steps (check_metrics → read_logs → optimize_batch)
- prediction_drift: 4 steps (analyze_drift → check_deployment → rollback_model → verify_fix)
- cascading_failure: 5 steps (check_metrics → read_logs → restart_service → scale_service → verify_fix)

---

## 📞 SUPPORT

If you encounter issues:

1. Check this guide
2. Review error messages carefully
3. Verify API credentials are correct
4. Check internet connectivity
5. Review the README.md

---

**Last Updated:** April 3, 2026
**Tested With:** Groq llama-3.3-70b, OpenAI gpt-4
**Status:** Production-Ready ✓
