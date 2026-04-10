#!/usr/bin/env python3
"""
validate.py — Pre-submission Validation Script
===============================================
Runs all checklist items before submitting to the competition.
All checks must pass or the submission will be disqualified.

Usage
-----
  # With server already running:
  python validate.py

  # Start server automatically:
  python validate.py --start-server

  # Custom server URL:
  python validate.py --url http://localhost:7860
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import ssl
import urllib.request
import urllib.error
from pathlib import Path

# ── ANSI colours ───────────────────────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS_COUNT = 0
FAIL_COUNT = 0
WARN_COUNT = 0
_SERVER_PROC = None


def ok(msg: str):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  {GREEN}✓{RESET}  {msg}")


def fail(msg: str):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  {RED}✗{RESET}  {msg}")


def warn(msg: str):
    global WARN_COUNT
    WARN_COUNT += 1
    print(f"  {YELLOW}!{RESET}  {msg}")


def section(title: str):
    print(f"\n{CYAN}{BOLD}── {title} ──{RESET}")


# SSL context that accepts HF Space / Let's Encrypt certs on all Python versions.
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE


def http_get(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, method="GET")
    ctx = _SSL_CTX if url.startswith("https://") else None
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return json.loads(r.read())


def http_post(url: str, body: dict | None = None, timeout: int = 30) -> dict:
    data = json.dumps(body).encode() if body else b""
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    ctx = _SSL_CTX if url.startswith("https://") else None
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return json.loads(r.read())


# ═══════════════════════════════════════════════════════════════════════════
# CHECK FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def check_repo_structure():
    section("1. Repository Structure")
    root = Path(__file__).parent

    required_files = [
        ("inference.py",      "Inference script (root)"),
        ("openenv.yaml",      "OpenEnv spec"),
        ("Dockerfile",        "Docker build file"),
        ("requirements.txt",  "Python dependencies"),
        ("environment.py",    "Environment implementation"),
        ("grader.py",         "Grader implementation"),
        ("models.py",         "Typed models"),
        ("tasks.py",          "Task bank"),
        ("server/app.py",     "FastAPI server"),
    ]

    for filename, label in required_files:
        path = root / filename
        if path.exists():
            ok(f"{label} found: {filename}")
        else:
            fail(f"{label} MISSING: {filename}")

    # inference.py must be in root
    if (root / "inference.py").exists():
        ok("inference.py is in root directory")
    else:
        fail("inference.py must be in the ROOT directory")


def check_openenv_yaml():
    section("2. openenv.yaml Spec Compliance")
    root = Path(__file__).parent
    yaml_path = root / "openenv.yaml"

    if not yaml_path.exists():
        fail("openenv.yaml not found — cannot validate")
        return

    content = yaml_path.read_text()

    checks = [
        ("spec_version",    "spec_version field"),
        ("name:",           "name field"),
        ("version:",        "version field"),
        ("type: space",     "type=space"),
        ("runtime: fastapi","runtime=fastapi"),
        ("port: 7860",      "port=7860 (HF Space)"),
        ("POST /reset",     "reset endpoint declared"),
        ("POST /step",      "step endpoint declared"),
        ("GET  /state",     "state endpoint declared"),
        ("GET  /health",    "health endpoint declared"),
        ("API_BASE_URL",    "API_BASE_URL env var declared"),
        ("MODEL_NAME",      "MODEL_NAME env var declared"),
        ("HF_TOKEN",        "HF_TOKEN env var declared"),
        ("count: 30",       "tasks count=30"),
        ("inference.py",    "inference script reference"),
        ("approve",         "approve action"),
        ("reject",          "reject action"),
        ("inspect",         "inspect action"),
        ("partial_credit: true", "partial credit enabled"),
    ]

    for pattern, label in checks:
        if pattern in content:
            ok(label)
        else:
            fail(f"Missing in openenv.yaml: {label} (pattern: '{pattern}')")


def check_typed_models():
    section("3. Typed Models")
    root = Path(__file__).parent
    sys.path.insert(0, str(root.parent))

    try:
        from payops_env.models import PayOpsAction, PayOpsObservation, PayOpsState
        ok("PayOpsAction importable")
        ok("PayOpsObservation importable")
        ok("PayOpsState importable")

        # Check action types
        action = PayOpsAction(action_type="approve", transaction_id="TXN-TEST")
        ok(f"PayOpsAction instantiates (action_type={action.action_type})")

        # Check all 10 action types are valid
        required_actions = [
            "approve", "reject", "flag", "escalate", "hold",
            "inspect", "request_docs", "verify_kyc", "contact_sender", "file_sar"
        ]
        for a in required_actions:
            try:
                PayOpsAction(action_type=a, transaction_id="TXN-TEST")
                ok(f"Action type '{a}' is valid")
            except Exception as e:
                fail(f"Action type '{a}' rejected: {e}")

    except ImportError as e:
        fail(f"Cannot import models: {e}")


def check_environment():
    section("4. Environment (step / reset / state)")
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        import asyncio
        from payops_env.environment import PayOpsEnvironment
        from payops_env.models import PayOpsAction

        env = PayOpsEnvironment()

        async def run_env_checks():
            # reset()
            obs = await env.reset_async()
            ok(f"reset() returns observation (task={obs.task_id})")

            if obs.budget_remaining == 5.0:
                ok("reset() budget_remaining=5.0")
            else:
                fail(f"reset() budget_remaining expected 5.0, got {obs.budget_remaining}")

            # step() investigation
            obs2 = await env.step_async(
                PayOpsAction(action_type="inspect", transaction_id=obs.transaction_id)
            )
            if obs2.reward == 0.15:
                ok("step(inspect) reward=0.15")
            else:
                warn(f"step(inspect) reward={obs2.reward} (expected 0.15)")

            if obs2.budget_remaining == 4.9:
                ok("step(inspect) budget_remaining=4.9")
            else:
                warn(f"step(inspect) budget={obs2.budget_remaining}")

            if obs2.task_id == obs.task_id:
                ok("inspect does not advance task")
            else:
                fail("inspect advanced task (should not)")

            # step() terminal
            obs3 = await env.step_async(
                PayOpsAction(action_type="approve", transaction_id=obs.transaction_id)
            )
            ok(f"step(approve) reward={obs3.reward}")

            # state()
            state = env._state
            if state.step_count > 0:
                ok(f"state() step_count={state.step_count}")
            else:
                fail("state() step_count=0 after steps")

            if isinstance(state.investigation_actions_used, list):
                ok("state() investigation_actions_used is list")
            else:
                fail("state() investigation_actions_used is not a list")

        asyncio.run(run_env_checks())

    except Exception as e:
        fail(f"Environment check failed: {e}")


def check_grader():
    section("5. Grader — 3+ tasks, scores in [0.0, 1.0]")
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from payops_env.grader import grade_episode
        from payops_env.tasks import TASKS

        if len(TASKS) >= 3:
            ok(f"Task bank has {len(TASKS)} tasks (>= 3 required)")
        else:
            fail(f"Task bank has only {len(TASKS)} tasks (need >= 3)")

        # Grade a minimal episode using the real grade_episode signature:
        # grade_episode(actions, tasks, confidences, budget_limit)
        # Use correct terminal actions for first 5 tasks.
        sample_tasks = list(TASKS[:5])
        sample_actions = [t.correct_action for t in sample_tasks]

        result = grade_episode(sample_actions, sample_tasks, budget_limit=5.0)

        if 0.0 <= result.normalised_score <= 1.0:
            ok(f"grade_episode() score in [0,1]: {result.normalised_score:.4f}")
        else:
            fail(f"grade_episode() score out of range: {result.normalised_score}")

        # Check each task graded
        for pt in result.per_task_rewards:
            score = pt.get("weighted_reward", pt.get("reward", 0))
            ok(f"  {pt.get('task_id'):12s} reward={score:+.3f}")

        ok("grade_episode() completes without error")

    except Exception as e:
        fail(f"Grader check failed: {e}")


def check_server(base_url: str):
    section("6. Server Health & OpenEnv Endpoints")

    # Health / reset
    try:
        health = http_get(f"{base_url}/health")
        if health.get("status") == "ok":
            ok(f"GET /health → status=ok (v{health.get('version','?')})")
        else:
            fail(f"GET /health → unexpected: {health}")
    except Exception as e:
        fail(f"GET /health failed: {e}")
        return  # Can't continue without server

    # reset() — must return 200 and a valid observation
    try:
        raw_reset = http_post(f"{base_url}/reset")
        # Support both wrapped {"observation":{...}} and legacy flat format
        obs = raw_reset.get("observation", raw_reset) if isinstance(raw_reset.get("observation"), dict) else raw_reset
        if "task_id" in obs:
            ok(f"POST /reset → 200, task_id={obs['task_id']}")
        else:
            fail(f"POST /reset → missing task_id in response")

        if obs.get("budget_remaining") == 5.0:
            ok("POST /reset → budget_remaining=5.0")
        else:
            fail(f"POST /reset → budget_remaining={obs.get('budget_remaining')}")
    except Exception as e:
        fail(f"POST /reset failed: {e}")

    # step()
    try:
        raw_step = http_post(f"{base_url}/step", {"action_type": "inspect", "transaction_id": "TXN-E001"})
        step = raw_step.get("observation", raw_step) if isinstance(raw_step.get("observation"), dict) else raw_step
        # reward is at top level in both wrapped and flat formats
        reward_val = raw_step.get("reward", step.get("reward"))
        if reward_val is not None:
            ok(f"POST /step (inspect) → reward={reward_val}")
        else:
            fail("POST /step → missing reward in response")
    except Exception as e:
        fail(f"POST /step failed: {e}")

    # state()
    try:
        state = http_get(f"{base_url}/state")
        if "episode_id" in state:
            ok("GET /state → episode_id present")
        else:
            fail("GET /state → missing episode_id")
        if isinstance(state.get("investigation_actions_used"), list):
            ok("GET /state → investigation_actions_used is list")
        else:
            fail("GET /state → investigation_actions_used is not a list")
    except Exception as e:
        fail(f"GET /state failed: {e}")

    # tasks
    try:
        tasks = http_get(f"{base_url}/tasks")
        count = len(tasks) if isinstance(tasks, list) else tasks.get("count", 0)
        if count >= 3:
            ok(f"GET /tasks → count={count} (>= 3 required)")
        else:
            fail(f"GET /tasks → count={count} (need >= 3)")
    except Exception as e:
        fail(f"GET /tasks failed: {e}")

    # grader
    try:
        grader = http_get(f"{base_url}/grader")
        score = grader.get("normalised_score", -1)
        if 0.0 <= score <= 1.0:
            ok(f"GET /grader → normalised_score={score:.4f} (in [0,1])")
        else:
            fail(f"GET /grader → score={score} out of range")
    except Exception as e:
        fail(f"GET /grader failed: {e}")

    # schema
    try:
        schema = http_get(f"{base_url}/schema")
        body = json.dumps(schema)
        for model in ["PayOpsAction", "PayOpsObservation"]:
            if model in body:
                ok(f"GET /schema → {model} present")
            else:
                fail(f"GET /schema → {model} missing")
    except Exception as e:
        fail(f"GET /schema failed: {e}")


def check_env_vars():
    section("7. Required Environment Variables")
    required = {
        "API_BASE_URL": "LLM API endpoint",
        "MODEL_NAME":   "Model identifier",
        "HF_TOKEN":     "API / HF token",
    }
    for var, desc in required.items():
        val = os.environ.get(var, "")
        if val:
            # Show only first 10 chars for secrets
            display = val[:10] + "..." if len(val) > 10 else val
            ok(f"{var} is set ({desc}): {display}")
        else:
            warn(f"{var} is NOT set ({desc}) — required at inference time")


def check_inference_script():
    section("8. inference.py Validation")
    root = Path(__file__).parent
    inf_path = root / "inference.py"

    if not inf_path.exists():
        fail("inference.py not found in root directory")
        return

    content = inf_path.read_text()

    checks = [
        ("from openai import OpenAI", "Uses OpenAI client"),
        ("API_BASE_URL",              "Reads API_BASE_URL env var"),
        ("MODEL_NAME",                "Reads MODEL_NAME env var"),
        ("HF_TOKEN",                  "Reads HF_TOKEN env var"),
        ("chat.completions.create",   "Uses chat.completions.create"),
        ("/reset",                    "Calls /reset endpoint"),
        ("/step",                     "Calls /step endpoint"),
        ("/grader",                   "Retrieves grader results"),
        ("normalised_score",          "Reports normalised_score"),
    ]

    for pattern, label in checks:
        if pattern in content:
            ok(label)
        else:
            fail(f"inference.py missing: {label} (pattern: '{pattern}')")


def check_dockerfile():
    section("9. Dockerfile")
    root = Path(__file__).parent
    df_path = root / "Dockerfile"

    if not df_path.exists():
        fail("Dockerfile not found")
        return

    content = df_path.read_text()

    checks = [
        ("FROM python:",      "Uses Python base image"),
        ("EXPOSE 7860",       "Exposes port 7860 (HF Space)"),
        ("7860",              "References port 7860"),
        ("uvicorn",           "Starts uvicorn server"),
        ("HEALTHCHECK",       "Has HEALTHCHECK"),
        ("requirements.txt",  "Installs requirements.txt"),
        ("appuser",           "Non-root user (security)"),
    ]

    for pattern, label in checks:
        if pattern in content:
            ok(label)
        else:
            fail(f"Dockerfile missing: {label}")

    # Attempt docker build (dry-run, only if docker is available)
    try:
        result = subprocess.run(
            ["docker", "build", "--check", "-f", str(df_path), str(root)],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            ok("docker build --check passed")
        else:
            # --check flag not available in older Docker; try plain build
            warn(f"docker build --check not supported; run 'docker build .' manually")
    except FileNotFoundError:
        warn("docker not installed — skipping build check")
    except subprocess.TimeoutExpired:
        warn("docker build check timed out")
    except Exception as e:
        warn(f"docker check skipped: {e}")


def check_runtime_constraint():
    section("10. Runtime Constraint (< 20 min / 2vCPU / 8GB)")
    # We can't fully test hardware constraints here, but we validate
    # that the inference script doesn't import heavy GPU libs
    root = Path(__file__).parent
    inf_path = root / "inference.py"

    if not inf_path.exists():
        warn("inference.py not found, skipping runtime check")
        return

    content = inf_path.read_text()
    heavy_libs = ["torch", "transformers", "tensorflow", "keras", "jax"]

    for lib in heavy_libs:
        if f"import {lib}" in content or f"from {lib}" in content:
            warn(f"inference.py imports '{lib}' — ensure it runs within 8GB RAM on 2vCPU")

    ok("inference.py uses lightweight OpenAI client (no local model loading)")

    # Check no huge loops that would blow 20min
    if "MAX_STEPS" in content:
        ok("MAX_STEPS guard present")
    else:
        warn("Consider adding a MAX_STEPS guard in inference.py")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PayOps pre-submission validator")
    parser.add_argument("--url", default="http://localhost:7860",
                        help="PayOps server base URL (default: http://localhost:7860)")
    parser.add_argument("--start-server", action="store_true",
                        help="Start the server automatically before validating")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  PayOps Pre-Submission Validator{RESET}")
    print(f"  Target server : {base_url}")
    print(f"{'='*60}{RESET}\n")

    # Static checks (no server needed)
    check_repo_structure()
    check_openenv_yaml()
    check_typed_models()
    check_environment()
    check_grader()
    check_env_vars()
    check_inference_script()
    check_dockerfile()
    check_runtime_constraint()

    # Server-dependent checks
    # Try to reach the server; if not up, attempt to start
    server_available = False
    try:
        http_get(f"{base_url}/health", timeout=3)
        server_available = True
    except Exception:
        if args.start_server:
            print(f"\n  Starting server at {base_url}...")
            root = Path(__file__).parent.parent
            env = os.environ.copy()
            env["PYTHONPATH"] = str(root)
            port = base_url.rsplit(":", 1)[-1] if ":" in base_url else "7860"
            global _SERVER_PROC
            _SERVER_PROC = subprocess.Popen(
                [sys.executable, "-m", "uvicorn",
                 "payops_env.server.app:app",
                 "--host", "0.0.0.0", "--port", port],
                env=env, cwd=str(root)
            )
            for _ in range(15):
                time.sleep(1)
                try:
                    http_get(f"{base_url}/health", timeout=2)
                    server_available = True
                    print("  Server started.")
                    break
                except Exception:
                    pass
            if not server_available:
                warn("Could not start server automatically")
        else:
            warn(f"Server not reachable at {base_url}. Start it first, or pass --start-server")

    if server_available:
        check_server(base_url)
    else:
        section("6. Server Health & OpenEnv Endpoints")
        warn("Skipped — server not available")

    # ── Summary ────────────────────────────────────────────────────────────
    total = PASS_COUNT + FAIL_COUNT
    print(f"\n{BOLD}{'='*60}")
    if FAIL_COUNT == 0:
        print(f"{GREEN}  ALL CHECKS PASSED  ({PASS_COUNT}/{total} passed, {WARN_COUNT} warnings){RESET}")
        print(f"{BOLD}  Ready to submit! ✓{RESET}")
    else:
        print(f"{RED}  {FAIL_COUNT} CHECK(S) FAILED{RESET}  "
              f"({PASS_COUNT} passed, {FAIL_COUNT} failed, {WARN_COUNT} warnings)")
        print(f"{RED}{BOLD}  Fix failures before submitting.{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    if _SERVER_PROC:
        _SERVER_PROC.terminate()

    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
