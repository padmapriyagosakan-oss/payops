"""
inference.py — PayOps LLM Inference Script
===========================================
Runs an LLM agent (via OpenAI-compatible client) against the PayOps
OpenEnv environment and prints per-task scores plus a final normalised score.

Required environment variables
--------------------------------
  API_BASE_URL   OpenAI-compatible API endpoint (e.g. https://api.openai.com/v1
                 or a HuggingFace Inference endpoint)
  MODEL_NAME     Model identifier  (e.g. gpt-4o-mini, meta-llama/Llama-3-8B-Instruct)
  HF_TOKEN       API key / HuggingFace token used as the Bearer credential

Optional
--------
  PAYOPS_BASE_URL  Base URL of the running PayOps server (default: http://localhost:7860)

Usage
-----
  export API_BASE_URL="https://api.openai.com/v1"
  export MODEL_NAME="gpt-4o-mini"
  export HF_TOKEN="sk-..."
  python inference.py
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import time
import urllib.request
import urllib.error
from typing import Optional

# SSL context: HF Spaces use Let's Encrypt certs that Python 3.14 on macOS
# cannot verify because the system trust store is not used by default.
# We disable verification only for requests to the PayOps server itself;
# the OpenAI client has its own SSL handling.
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# ── OpenAI client (uses env vars) ─────────────────────────────────────────

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────

API_BASE_URL: str    = (os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1").rstrip("/")
MODEL_NAME: str      = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
# OPENAI_API_KEY is the spec-standard credential; HF_TOKEN is the HF-native alias.
# OPENAI_API_KEY takes precedence when both are set.
OPENAI_API_KEY: str  = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN: str        = os.environ.get("HF_TOKEN", "")
_API_KEY: str        = OPENAI_API_KEY or HF_TOKEN   # resolved credential
PAYOPS_URL: str      = os.environ.get("PAYOPS_BASE_URL", "http://localhost:7860").rstrip("/")
# Fixed seed keeps per-episode amount/risk jitter deterministic across runs.
# Override with INFERENCE_SEED=<int> or INFERENCE_SEED=random for a fresh episode.
_SEED_ENV             = os.environ.get("INFERENCE_SEED", "42")
INFERENCE_SEED: Optional[int] = None if _SEED_ENV.lower() == "random" else int(_SEED_ENV)

VALID_ACTIONS = {
    "approve", "reject", "flag", "escalate", "hold",
    "inspect", "request_docs", "verify_kyc", "contact_sender", "file_sar",
}

# ── Contest-spec structured logging ──────────────────────────────────────────
# Exactly three line types to stdout (machine-parseable by the evaluator):
#   [START]  once per episode
#   [STEP]   once per env.step() call
#   [END]    always, even on exception

TASK_NAME = os.environ.get("PAYOPS_TASK", "payops")
ENV_NAME  = "payops_env"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

SYSTEM_PROMPT = """You are a Payment Operations analyst reviewing financial transactions.

For each transaction you will receive structured data including:
- transaction_id, amount, currency, sender, receiver
- risk_score (0.0=safe, 1.0=highly risky)
- ml_confidence (model confidence in fraud detection)
- flags (list of risk indicators)
- kyc_status (verified/pending/failed/none)
- velocity_1h, velocity_24h (transaction counts in time windows)
- country_risk (sender/receiver country risk levels)
- account_age_days, previous_sars
- inspection_notes, docs_notes, kyc_notes, contact_notes (if investigation was done)

You must respond with EXACTLY one of these action names (lowercase, no punctuation):
  approve      - transaction appears legitimate, release funds
  reject       - transaction is fraudulent, block and reverse
  flag         - suspicious, mark for manual review
  escalate     - complex case, send to senior analyst
  hold         - pause transaction pending further info
  inspect      - examine transaction history and patterns (investigation, costs 0.1 budget)
  request_docs - request supporting documents from sender (costs 0.2 budget)
  verify_kyc   - run full KYC verification (costs 0.2 budget)
  contact_sender - contact transaction sender for confirmation (costs 0.3 budget)
  file_sar     - file Suspicious Activity Report with regulators (costs 0.05 budget)

Investigation actions (inspect/request_docs/verify_kyc/contact_sender/file_sar) do NOT
advance the task — they reveal more information. Terminal actions (approve/reject/flag/
escalate/hold) resolve the task and advance to the next one.

Budget: you start with 5.0 units. Investigation actions cost budget. Stay within budget.

Respond with ONLY the action name. No explanation, no punctuation, no quotes.
"""


def _make_request(url: str, method: str = "GET", body: Optional[dict] = None) -> dict:
    """Simple HTTP helper using stdlib (no httpx dependency)."""
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code} from {url}: {err_body}") from e


def _unwrap_response(resp: dict) -> dict:
    """
    Normalise server responses that may use the official OpenEnv wire format
    ``{"observation": {...}, "reward": ..., "done": ...}`` or the legacy flat
    format where the observation fields are at the top level.  Always returns
    a flat observation dict with ``reward`` and ``done`` populated.
    """
    if "observation" in resp and isinstance(resp["observation"], dict):
        obs = dict(resp["observation"])
        # Promote top-level reward/done into the obs dict for uniform access
        if "reward" in resp:
            obs["reward"] = resp["reward"]
        if "done" in resp:
            obs["done"] = resp["done"]
        return obs
    # Legacy flat format — already the observation
    return resp


def build_observation_text(obs: dict) -> str:
    """Format the observation dict into a readable prompt for the LLM."""
    lines = [
        f"Transaction ID : {obs.get('transaction_id', 'N/A')}",
        f"Task           : {obs.get('task_id', 'N/A')} ({obs.get('task_difficulty', 'N/A')} difficulty)",
        f"Amount         : {obs.get('currency', 'USD')} {obs.get('amount', 0):,.2f}",
        f"Sender         : {obs.get('sender', 'N/A')}",
        f"Receiver       : {obs.get('receiver', 'N/A')}",
        f"Risk Score     : {obs.get('risk_score', 0):.2f}",
        f"ML Confidence  : {obs.get('ml_confidence', 0):.2f}",
        f"Flags          : {', '.join(obs.get('flags', [])) or 'none'}",
        f"KYC Status     : {obs.get('kyc_status', 'N/A')}",
        f"Velocity 1h    : {obs.get('velocity_1h', 0)} txns",
        f"Velocity 24h   : {obs.get('velocity_24h', 0)} txns",
        f"Country Risk   : {obs.get('country_risk', 'N/A')}",
        f"Account Age    : {obs.get('account_age_days', 0)} days",
        f"Previous SARs  : {obs.get('previous_sars', 0)}",
        f"Budget Remaining: {obs.get('budget_remaining', 5.0):.2f}",
        f"Step           : {obs.get('chain_step', 1)}/{obs.get('chain_total', 1)}",
    ]
    # Append revealed intel if present
    for field, label in [
        ("inspection_notes", "Inspection Notes"),
        ("docs_notes",       "Documents Notes"),
        ("kyc_notes",        "KYC Notes"),
        ("contact_notes",    "Contact Notes"),
    ]:
        if obs.get(field):
            lines.append(f"{label:15}: {obs[field]}")

    return "\n".join(lines)


def call_llm(client: OpenAI, observation_text: str) -> str:
    """Call the LLM and return a cleaned action string."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Transaction details:\n\n{observation_text}\n\nYour action:"},
        ],
        max_tokens=16,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip().lower()
    # Strip any punctuation/quotes the model might add
    action = raw.strip("\"'.,;: \n\t")
    return action


def run_inference() -> tuple:
    """
    Run one full episode of PayOps with the LLM agent.
    Returns (grader_result_dict, per_step_rewards, total_steps).
    """
    # ── validate env vars ───────────────────────────────────────────────
    # OPENAI_API_KEY or HF_TOKEN must be set (either is acceptable)
    if not _API_KEY:
        print("ERROR: Set OPENAI_API_KEY (or HF_TOKEN) with your API credential.", file=sys.stderr)
        sys.exit(1)
    # API_BASE_URL and MODEL_NAME always have built-in defaults (via 'or' fallback above).
    # Only HF_TOKEN / OPENAI_API_KEY genuinely requires the caller to supply a value.

    print(f"PayOps Inference", file=sys.stderr)
    print(f"  Model      : {MODEL_NAME}", file=sys.stderr)
    print(f"  API Base   : {API_BASE_URL}", file=sys.stderr)
    print(f"  Server     : {PAYOPS_URL}", file=sys.stderr)
    print(file=sys.stderr)

    # ── init OpenAI client ──────────────────────────────────────────────
    client = OpenAI(
        api_key=_API_KEY,
        base_url=API_BASE_URL,
    )

    # ── health check ────────────────────────────────────────────────────
    try:
        health = _make_request(f"{PAYOPS_URL}/health")
        print(f"Server health : {health.get('status')} (v{health.get('version', '?')})", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Cannot reach PayOps server at {PAYOPS_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    # ── reset environment ────────────────────────────────────────────────
    # Pass a fixed seed so per-episode jitter is deterministic (reproducible scores).
    reset_body: dict = {}
    if INFERENCE_SEED is not None:
        reset_body["seed"] = INFERENCE_SEED
        print(f"Episode seed  : {INFERENCE_SEED} (set INFERENCE_SEED=random for a fresh run)", file=sys.stderr)
    _reset_resp = _make_request(f"{PAYOPS_URL}/reset", method="POST", body=reset_body or None)
    # Support official wrapped format {"observation":{...},"reward":...} and legacy flat format
    obs = _unwrap_response(_reset_resp)
    print(f"Episode start : task={obs['task_id']}, budget={obs['budget_remaining']}", file=sys.stderr)
    print(file=sys.stderr)

    step_count = 0
    rewards_list: list = []
    start_time = time.time()
    # Hard caps to guarantee <20 min runtime on any inference endpoint.
    # Worst case: 55 steps × 15 s/call ≈ 825 s (~14 min); well inside the 20 min limit.
    MAX_STEPS = 55        # absolute hard cap across the whole episode
    MAX_INV_PER_TASK = 2  # max investigation actions per task before forcing terminal decision

    # Per-task investigation counter: task_id → count
    inv_counts: dict = {}

    # ── main loop ────────────────────────────────────────────────────────
    while not obs.get("done", False) and step_count < MAX_STEPS:
        obs_text = build_observation_text(obs)
        task_id  = obs.get("task_id", "?")
        budget   = obs.get("budget_remaining", 5.0)

        # Get action from LLM
        try:
            action = call_llm(client, obs_text)
        except Exception as e:
            print(f"  LLM error on {task_id}: {e} — defaulting to 'flag'", file=sys.stderr)
            action = "flag"

        # Validate action
        if action not in VALID_ACTIONS:
            print(f"  ⚠ LLM returned invalid action '{action}' → defaulting to 'flag'", file=sys.stderr)
            action = "flag"

        # Budget guard: if budget is low, skip expensive investigation actions
        expensive = {"contact_sender": 0.3, "request_docs": 0.2, "verify_kyc": 0.2}
        if action in expensive and budget < expensive[action] + 0.1:
            print(f"  ⚠ Insufficient budget ({budget:.2f}) for '{action}' → using 'inspect'", file=sys.stderr)
            action = "inspect" if budget >= 0.1 else "flag"

        # Loop guard: if model keeps investigating without deciding, force a terminal action
        INVESTIGATION_ACTIONS = {"inspect", "request_docs", "verify_kyc", "contact_sender", "file_sar"}
        TERMINAL_ACTIONS = {"approve", "reject", "flag", "escalate", "hold"}
        if action in INVESTIGATION_ACTIONS:
            inv_counts[task_id] = inv_counts.get(task_id, 0) + 1
            if inv_counts[task_id] > MAX_INV_PER_TASK:
                print(f"  ⚠ Investigation loop on {task_id} ({inv_counts[task_id]} sub-actions) → forcing 'flag'", file=sys.stderr)
                action = "flag"
        else:
            # Reset counter on terminal action (task will advance)
            inv_counts[task_id] = 0

        # Step
        try:
            _step_resp = _make_request(
                f"{PAYOPS_URL}/step",
                method="POST",
                body={"action_type": action, "transaction_id": obs.get("transaction_id")},
            )
        except Exception as e:
            print(f"  Step error: {e}", file=sys.stderr)
            break

        # Unwrap official {"observation":{...},"reward":...} or legacy flat format
        obs = _unwrap_response(_step_resp)
        reward    = obs.get("reward", 0)
        new_task  = obs.get("task_id", "?")
        done_flag = obs.get("done", False)
        step_count += 1
        rewards_list.append(reward)
        log_step(step=step_count, action=action, reward=reward, done=done_flag, error=None)

        marker = "✓" if reward > 0 else ("✗" if reward < 0 else "·")
        print(f"  [{marker}] {task_id:12s} → {action:15s}  reward={reward:+.3f}  "
              f"budget={obs.get('budget_remaining', 0):.2f}", file=sys.stderr)

    elapsed = time.time() - start_time
    print(f"\nEpisode complete: {step_count} steps in {elapsed:.1f}s", file=sys.stderr)

    # ── grade ────────────────────────────────────────────────────────────
    try:
        grader = _make_request(f"{PAYOPS_URL}/grader")
    except Exception as e:
        print(f"ERROR: Could not retrieve grader results: {e}", file=sys.stderr)
        sys.exit(1)

    score   = grader.get("normalised_score", 0)
    passed  = grader.get("passed", False)
    budgets = grader.get("budget_spent", 0)

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"  Normalised Score : {score:.4f}", file=sys.stderr)
    print(f"  Total Reward     : {grader.get('total_reward', 0):.3f}", file=sys.stderr)
    print(f"  Max Possible     : {grader.get('max_possible_reward', 0):.3f}", file=sys.stderr)
    print(f"  Budget Spent     : {budgets:.2f}", file=sys.stderr)
    print(f"  Budget Penalty   : {grader.get('budget_penalty', 0):.3f}", file=sys.stderr)
    print(f"  Passed           : {'YES ✓' if passed else 'NO ✗'}", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)

    # Per-task breakdown
    per_task = grader.get("per_task", [])
    correct_count = sum(1 for t in per_task if t.get("correct", False))
    wrong_count   = len(per_task) - correct_count

    print("\nPer-task breakdown:", file=sys.stderr)
    print(f"  {'Task':12s} {'Difficulty':10s} {'Agent Action':15s} {'Correct Action':15s} {'Reward':>8s}", file=sys.stderr)
    print(f"  {'-'*65}", file=sys.stderr)
    for t in per_task:
        ok  = t.get("correct", False)
        sym = "✓" if ok else "✗"
        print(f"  [{sym}] {t.get('task_id','?'):12s} {t.get('difficulty','?'):10s} "
              f"{t.get('terminal_action','?'):15s} {t.get('correct_action','?'):15s} "
              f"{t.get('weighted_reward', 0):+8.3f}", file=sys.stderr)

    print(f"\n  Tasks correct : {correct_count}/{len(per_task)}  "
          f"({100*correct_count/len(per_task):.0f}%)  "
          f"Wrong: {wrong_count}", file=sys.stderr)

    return grader, rewards_list, step_count


if __name__ == "__main__":
    _rewards: list = []
    _steps   = 0
    _score   = 0.0
    _success = False
    try:
        log_start(task=TASK_NAME, env=ENV_NAME, model=MODEL_NAME)
        result, _rewards, _steps = run_inference()
        _score   = result.get("normalised_score", 0.0)
        _success = result.get("passed", False)
    except Exception as exc:
        print(f"[DEBUG] Fatal: {exc}", file=sys.stderr)
    finally:
        log_end(success=_success, steps=_steps, score=_score, rewards=_rewards)
    sys.exit(0 if _success else 1)
