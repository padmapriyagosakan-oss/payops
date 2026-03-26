"""
Baseline agent for the PayOps OpenEnv environment.

A rule-based agent that decides an action based on observable risk signals.
Use this as a sanity-check to prove the environment works end-to-end and
as a lower-bound performance reference before training a learned agent.

Usage
-----
    # Run directly
    python scripts/baseline_agent.py

    # Or via the /baseline API endpoint
    curl -X POST http://localhost:8000/baseline

Policy
------
1. If any sanctioned / high-risk flag        → reject
2. velocity_1h >= 10 (velocity burst)        → reject
3. kyc_status in (failed, none)              → escalate
4. risk_score >= 0.80                        → reject
5. risk_score >= 0.60                        → escalate
6. risk_score >= 0.35 OR any flag present    → flag
7. Otherwise                                 → approve
"""

from __future__ import annotations

import asyncio
import sys
import os

# Allow running from the project root or the scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from payops_env.environment import PayOpsEnvironment
from payops_env.grader import grade_episode
from payops_env.models import PayOpsAction
from payops_env.tasks import TASKS


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

DANGER_FLAGS = {
    "sanctioned_country",
    "app_scam_indicator",
    "mule_account_pattern",
    "structuring_pattern",
    "ctr_threshold_avoidance",
}


def rule_based_policy(obs) -> str:
    """
    Deterministic rule-based policy.

    Priority order ensures the most dangerous patterns are caught first,
    even when the ML risk score is low (adversarial/poisoned score inputs).
    """
    # Priority 1: known fraud or sanctioned patterns regardless of risk score
    if any(flag in DANGER_FLAGS for flag in obs.flags):
        return "reject"

    # Priority 2: transaction velocity burst (potential account takeover)
    if obs.velocity_1h is not None and obs.velocity_1h >= 10:
        return "reject"

    # Priority 3: identity not confirmed
    if obs.kyc_status in ("failed", "none"):
        return "escalate"

    # Priority 4–6: risk score tiers
    if obs.risk_score >= 0.80:
        return "reject"
    elif obs.risk_score >= 0.60:
        return "escalate"
    elif obs.risk_score >= 0.35 or obs.flags:
        return "flag"
    else:
        return "approve"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run():
    env = PayOpsEnvironment()
    obs = await env.reset_async()
    total_reward = 0.0
    step = 0
    actions_taken = []

    print("=" * 60)
    print("  PayOps Baseline Agent — Rule-Based Policy")
    print("=" * 60)
    print(f"  Tasks in episode : {len(TASKS)}")
    print("=" * 60)

    while not obs.done:
        action_type = rule_based_policy(obs)
        action = PayOpsAction(
            action_type=action_type,
            transaction_id=obs.transaction_id,
        )

        print(f"\nStep {step + 1}  [{obs.task_difficulty.upper()}]  {obs.task_id}")
        print(f"  TXN        : {obs.transaction_id}")
        print(f"  Amount     : {obs.amount:,.2f} {obs.currency}")
        print(f"  Sender     : {obs.sender}")
        print(f"  Receiver   : {obs.receiver}")
        print(f"  Risk score : {obs.risk_score:.2f}")
        print(f"  KYC        : {obs.kyc_status or 'n/a'}  |  "
              f"Country risk: {obs.country_risk or 'n/a'}  |  "
              f"Velocity 1h: {obs.velocity_1h or 'n/a'}")
        print(f"  Flags      : {obs.flags or '[]'}")
        print(f"  → Agent    : {action_type}")

        obs = await env.step_async(action)
        actions_taken.append(action_type)
        total_reward += obs.reward
        step += 1

        print(f"  ✓ Reward   : {obs.reward:+.2f}  "
              f"(correct: {obs.info.get('correct_action', '?')})")

    env.close()

    # Grade the episode
    result = grade_episode(actions_taken, list(TASKS))

    print("\n" + "=" * 60)
    print("  Episode Summary")
    print("=" * 60)
    print(f"  Steps             : {step}")
    print(f"  Total reward      : {result.total_reward:+.2f}")
    print(f"  Max possible      : {result.max_possible_reward:.2f}")
    print(f"  Normalised score  : {result.normalised_score:.4f}")
    print(f"  Passed (≥0.5)     : {'YES ✓' if result.passed else 'NO ✗'}")

    print("\n  Per-task breakdown:")
    for t in result.per_task_rewards:
        mark = "✓" if t["correct"] else "✗"
        print(
            f"    {mark}  {t['task_id']:12s}  [{t['difficulty']:6s}]  "
            f"action={t['action_taken']:10s}  "
            f"correct={t['correct_action']:10s}  "
            f"reward={t['reward']:+.2f}"
        )

    print("=" * 60)
    return result


if __name__ == "__main__":
    asyncio.run(run())

