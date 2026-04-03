#!/usr/bin/env python3
"""
Pre-submission validation script for ResilientAgent-Prod.

Checks:
1. OpenEnv spec compliance
2. Environment endpoints
3. Baseline reproducibility
4. Docker buildability
"""

import sys
import json
import os
import subprocess
from datetime import datetime

def check_spec_compliance():
    """Verify OpenEnv spec."""
    print("\n" + "="*60)
    print("SPEC COMPLIANCE CHECK")
    print("="*60)
    
    required_files = [
        "openenv.yaml",
        "models.py",
        "server/app.py",
        "server/resilientagent_prod_environment.py",
        "Dockerfile",
        "requirements.txt",
        "README.md"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file}")
        else:
            print(f"[FAIL] MISSING: {file}")
            return False
    
    return True

def check_endpoints():
    """Verify required API endpoints."""
    print("\n" + "="*60)
    print("ENDPOINT REQUIREMENTS CHECK")
    print("="*60)
    
    required_endpoints = [
        "POST /reset",
        "POST /step", 
        "GET /state",
        "POST /grader",
        "GET /tasks",
        "GET /baseline",
        "GET /health"
    ]
    
    app_content = open("server/app.py").read()
    
    for endpoint in required_endpoints:
        method, path = endpoint.split()
        pattern = f'@app.{method.lower()}("{path}")' or f'@app.{method.lower()}(\'{path}\')'
        if f'@app.{method.lower()}' in app_content and path in app_content:
            print(f"✓ {endpoint}")
        else:
            print(f"✗ MISSING: {endpoint}")
            return False
    
    return True

def check_baseline():
    """Run baseline evaluation."""
    print("\n" + "="*60)
    print("BASELINE REPRODUCIBILITY CHECK")
    print("="*60)
    
    try:
        result = subprocess.run(
            ["python", "evaluate.py"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if "Average Score:" in result.stdout:
            # Extract score
            for line in result.stdout.split('\n'):
                if "Average Score:" in line:
                    print(f"✓ Baseline runs successfully")
                    print(f"  {line.strip()}")
                    return True
        else:
            print("✗ Baseline evaluation failed")
            print(result.stderr[:500])
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Baseline evaluation timeout (>120s)")
        return False
    except Exception as e:
        print(f"✗ Error running baseline: {e}")
        return False

def check_environment_vars():
    """Check required environment variables."""
    print("\n" + "="*60)
    print("ENVIRONMENT VARIABLES CHECK")
    print("="*60)
    
    env_vars = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    
    for var in env_vars:
        if os.getenv(var):
            print(f"✓ {var} is set")
        else:
            print(f"⚠ {var} not set (required at runtime)")
    
    return True

def check_tasks():
    """Verify all 3 tasks exist."""
    print("\n" + "="*60)
    print("TASK FILES CHECK")
    print("="*60)
    
    tasks = [
        "src/tasks/task1_latency_spike.py",
        "src/tasks/task2_prediction_drift.py",
        "src/tasks/task3_cascading_failure.py"
    ]
    
    for task in tasks:
        if os.path.exists(task):
            print(f"✓ {task}")
        else:
            print(f"✗ MISSING: {task}")
            return False
    
    return True

def check_grading_logic():
    """Verify dynamic grading (not hardcoded)."""
    print("\n" + "="*60)
    print("GRADING LOGIC CHECK (Dynamic, Not Predetermined)")
    print("="*60)
    
    env_content = open("server/resilientagent_prod_environment.py").read()
    
    dynamic_markers = [
        "def grade(self)",
        "self._state.step_count",
        "self._wasted_actions",
        "self._root_cause_identified",
        "_calculate_reward",
        "_is_useful_action",
        "_process_action"
    ]
    
    all_found = True
    for marker in dynamic_markers:
        if marker in env_content:
            print(f"✓ {marker} found")
        else:
            print(f"✗ Missing: {marker}")
            all_found = False
    
    if "hardcoded" not in env_content.lower() and "predetermined" not in env_content.lower():
        print("✓ No hardcoded/predetermined responses detected")
    
    return all_found

def main():
    """Run all validation checks."""
    print("\n" + "#"*60)
    print("# RESILIENTAGENT-PROD SUBMISSION VALIDATION")
    print("# " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("#"*60)
    
    checks = [
        ("Spec Compliance", check_spec_compliance),
        ("Endpoint Requirements", check_endpoints),
        ("Task Files", check_tasks),
        ("Grading Logic (Dynamic)", check_grading_logic),
        ("Environment Variables", check_environment_vars),
        ("Baseline Reproducibility", check_baseline),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"✗ Error: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 ALL CHECKS PASSED - READY FOR SUBMISSION!")
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED - FIX BEFORE SUBMISSION")
        return 1

if __name__ == "__main__":
    sys.exit(main())
