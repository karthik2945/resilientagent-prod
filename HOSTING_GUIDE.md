# HUGGING FACE SPACES DEPLOYMENT GUIDE

## Hosting ResilientAgent-Prod on HF Spaces

---

## 📦 PRE-DEPLOYMENT CHECKLIST

Before uploading to HF Spaces, ensure:

- [x] `.env` has NO hardcoded secrets ✓
- [x] Dockerfile builds successfully ✓
- [x] inference.py at root directory ✓
- [x] requirements.txt has all dependencies ✓
- [x] .spaceignore filters test files ✓
- [x] README.md is complete ✓
- [x] validate.py passes all checks ✓

---

## 🚀 DEPLOYMENT STEPS

### Step 1: Create Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in:
   - **Space name:** resilientagent-prod (or your team name)
   - **License:** MIT (or your choice)
   - **Space SDK:** Docker
   - **Space hardware:** CPU basic (sufficient for evaluation)
4. Click **"Create Space"**

### Step 2: Clone Space Repository

```bash
git clone https://huggingface.co/spaces/<your-username>/resilientagent-prod
cd resilientagent-prod
```

### Step 3: Add Your Project Files

Copy all files from your local repository:

```bash
# Copy all project files
cp -r /path/to/local/resilientagent-prod/* .

# Verify key files exist
ls -la Dockerfile inference.py requirements.txt openenv.yaml
```

### Step 4: Push to HF Spaces

```bash
git add .
git commit -m "Initial commit: ResilientAgent-Prod ready for evaluation"
git push
```

**HF Spaces will automatically:**
1. Build the Docker image
2. Start the container on port 7860
3. Run healthcheck
4. Make it available at: `https://<username>-resilientagent-prod.hf.space`

### Step 5: Verify Deployment

```bash
# Test health endpoint (wait 2-3 min after push)
curl https://<username>-resilientagent-prod.hf.space/health

# Expected response:
# {"status": "ok"}
```

---

## 🔐 SECRETS MANAGEMENT FOR EVALUATORS

### For Submission Preview (Before Evaluation):

**Do NOT hardcode API keys in .env or code.**

### For Evaluation Day (When Grader Runs):

The evaluator will inject credentials via:

```bash
# HF Spaces will run inference.py with these injected:
docker run \
  -e API_BASE_URL="<grader's API endpoint>" \
  -e MODEL_NAME="<grader's model>" \
  -e HF_TOKEN="<grader's API key>" \
  -p 7860:7860 \
  resilientagent-prod
```

**Your code will read from env vars automatically.**

---

## 📋 ENVIRONMENT VARIABLE INJECTION

When HF Spaces runs your Docker image during evaluation:

```bash
# Grader will set these before executing inference.py
docker run \
  -e API_BASE_URL="https://api.groq.com/openai/v1" \
  -e MODEL_NAME="llama-3.3-70b-versatile" \
  -e HF_TOKEN="gsk_<grader_token>" \
  resilientagent-prod
```

Your `inference.py` will:
```python
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
HF_TOKEN = os.getenv("HF_TOKEN")  # From grader's injection
```

---

## ⏱️ RUNTIME PERFORMANCE

| Metric | Requirement | Your System | Status |
|---|---|---|---|
| Total Runtime | < 20 minutes | ~2-3 minutes | ✅ PASS |
| Task 1 (Latency) | - | ~30 seconds | ✅ OK |
| Task 2 (Drift) | - | ~30 seconds | ✅ OK |
| Task 3 (Cascading) | - | ~30 seconds | ✅ OK |
| Disk Space | 100MB | ~50MB used | ✅ PASS |
| Memory Usage | 8GB | ~200-500MB | ✅ PASS |
| CPU Required | vcpu=2 | <1 vCPU | ✅ PASS |

**HF Spaces CPU basic tier is sufficient** ✓

---

## 🧪 TESTING ON HF SPACES

After deployment goes live:

### 1. Check Health
```bash
curl https://<username>-resilientagent-prod.hf.space/health
```

### 2. Check Endpoints
```bash
# GET tasks endpoint
curl https://<username>-resilientagent-prod.hf.space/tasks

# POST baseline endpoint
curl -X POST https://<username>-resilientagent-prod.hf.space/baseline
```

### 3. Verify Inference Script Works

On your local machine with API credentials:
```bash
# Simulate grader environment
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="gsk_YOUR_KEY"

python inference.py
```

Expected output (structured logs):
```
[START] task=latency_spike env=resilientagent-prod model=llama-3.3-70b-versatile
[STEP] step=1 action=check_metrics('inference_service') reward=0.15 done=false error=null
...
[END] success=true steps=3 score=0.905 rewards=0.15,0.15,1.00
```

---

## 📝 DOCKERFILE VERIFICATION

Your Dockerfile should:

- ✅ Use `python:3.11-slim` base image
- ✅ Install requirements.txt
- ✅ Expose port 7860
- ✅ Include healthcheck
- ✅ Run FastAPI app: `uvicorn server.app:app --host 0.0.0.0 --port 7860`

**Current Dockerfile is correct** ✓

---

## 🗂️ FILE STRUCTURE FOR HF SPACES

```
resilientagent-prod/
├── Dockerfile                    ← HF Spaces builds this
├── requirements.txt              ← Dependencies
├── .spaceignore                  ← Filters test files
├── .env                          ← NO hardcoded secrets
├── inference.py                  ← Grader runs this (root level!)
├── server/
│   ├── app.py                   ← FastAPI endpoints
│   └── resilientagent_prod_environment.py
├── src/
│   ├── tasks/
│   │   ├── task1_latency_spike.py
│   │   ├── task2_prediction_drift.py
│   │   └── task3_cascading_failure.py
│   └── models.py
├── models.py
├── openenv.yaml
└── README.md
```

---

## ❌ COMMON MISTAKES TO AVOID

| Mistake | Impact | Prevention |
|---------|--------|-----------|
| Hardcoded API key in .env | **DISQUALIFICATION** | Use empty vars, let grader inject |
| inference.py not at root | **GRADER CAN'T FIND IT** | Verify path before upload |
| Dockerfile build fails | **CAN'T DEPLOY** | Test locally: `docker build` |
| Missing requirements | **RUNTIME ERROR** | All deps in requirements.txt |
| inference.py >20min runtime | **TIMEOUT** | Verify runs in <5min |
| Wrong output format | **PARSING FAILS** | Follow [START]/[STEP]/[END] exactly |

---

## 🔄 UPDATING AFTER SUBMISSION

If bug fixes needed after initial submission:

```bash
cd resilientagent-prod
git add .
git commit -m "Fix: description of fix"
git push
```

HF Spaces will automatically rebuild and redeploy.

---

## 📊 HF SPACES RESOURCE LIMITS

| Resource | CPU Basic | Sufficient? |
|----------|-----------|-----------|
| CPU | 2 vCPU | Yes ✅ |
| RAM | 4GB | Yes ✅ |
| Storage | 50GB | Yes ✅ |
| Runtime | 20 min | Yes ✅ |

---

## 🎯 FINAL SUBMISSION PROCESS

### Step 1: Verify All Pre-Checks
```bash
python validate.py
# Should show: ✅ ALL CHECKS PASSED
```

### Step 2: Push to HF Spaces
```bash
git add .
git commit -m "Final submission: ResilientAgent-Prod"
git push
```

### Step 3: Wait for Build
- HF Spaces will build Docker image (~2-5 min)
- Container will start
- Healthcheck will verify

### Step 4: Copy Space URL
- Your Space: `https://<username>-resilientagent-prod.hf.space`
- Paste this in the submission form

### Step 5: Submit on Dashboard
- Go to: https://scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard
- Click: "Submit your Assessment"
- Paste Space URL
- Submit!

**Deadline: 8 April 11:59 PM** ⏰

---

## ✅ FINAL CHECKLIST BEFORE SUBMISSION

- [ ] .env has NO hardcoded API keys
- [ ] inference.py is at root directory
- [ ] Dockerfile builds successfully locally
- [ ] validate.py shows 6/6 checks passed
- [ ] evaluate.py shows baseline ~0.955 score
- [ ] README.md is complete and clear
- [ ] EVALUATOR_GUIDE.md created
- [ ] .spaceignore filters test files
- [ ] requirements.txt has all deps
- [ ] openenv.yaml is valid
- [ ] No hardcoded secrets anywhere
- [ ] HF Space created and linked
- [ ] Docker image builds on HF Spaces
- [ ] /health endpoint responds 200

**Once all green ✅ → READY TO SUBMIT**

---

**Last Updated:** April 3, 2026  
**Status:** Production-Ready for HF Spaces ✓
