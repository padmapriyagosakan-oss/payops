"""
Utilities for running the baseline agent programmatically (used by /baseline endpoint).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from payops_env.environment import PayOpsEnvironment
from payops_env.grader import grade_episode
from payops_env.models import PayOpsAction
from payops_env.tasks import TASKS


# ---------------------------------------------------------------------------
# Adaptive rule-based policy
# ---------------------------------------------------------------------------

_DANGER_FLAGS = {
    "sanctioned_country", "app_scam_indicator", "mule_account_pattern",
    "structuring_pattern", "ctr_threshold_avoidance", "fraud_ring_indicator",
    "geo_impossible_login", "account_takeover_indicator",
}

_WATCHLIST_FLAGS = {
    "new_account_7d", "large_first_transfer", "solicitor_mule_pattern",
    "dormant_receiver", "sudden_activity", "insider_threat", "internal_to_personal",
    "invoice_mismatch", "trade_finance",
}


def _should_investigate(obs) -> Optional[str]:
    """
    Decide whether to issue an investigation sub-action first.

    Returns the sub-action name, or None if we should decide directly.

    Priority:
      1. KYC expired / pending → verify_kyc (once)
      2. Watchlist flags AND high amount AND docs not yet requested → request_docs
      3. Low ml_confidence AND medium risk → inspect (once)
      4. contact_sender: APP scam pattern or insider threat
      5. file_sar: if structuring/fraud-ring flags and not yet filed
    """
    # The env sets both "already_used" (bool for this action) and
    # "investigation_used" (list of all inv actions used for this task)
    if isinstance(obs.info, dict):
        inv_used = obs.info.get("investigation_used", [])
        already = set(inv_used) if isinstance(inv_used, (list, set)) else set()
    else:
        already = set()

    # file_sar if structuring / fraud ring and SAR not yet filed
    sar_flags = {"structuring_pattern", "ctr_threshold_avoidance",
                 "fraud_ring_indicator", "coordinated_transfers"}
    if sar_flags & set(obs.flags) and "file_sar" not in already:
        return "file_sar"

    # contact_sender for APP scam or insider
    contact_flags = {"app_scam_indicator", "internal_to_personal", "account_takeover_indicator"}
    if contact_flags & set(obs.flags) and "contact_sender" not in already:
        return "contact_sender"

    # verify_kyc if expired or pending
    if obs.kyc_status in ("expired", "pending") and "verify_kyc" not in already:
        return "verify_kyc"

    # request_docs for first-time payees with high value
    doc_flags = {"first_time_payee", "large_first_transfer", "invoice_mismatch", "trade_finance"}
    if (doc_flags & set(obs.flags)
            and obs.amount >= 50_000
            and "request_docs" not in already):
        return "request_docs"

    # inspect when ml_confidence is low or watchlist flags are present
    ml_conf = getattr(obs, "ml_confidence", 0.9) or 0.9
    watchlist_hit = bool(_WATCHLIST_FLAGS & set(obs.flags))
    if (ml_conf < 0.60 or watchlist_hit) and "inspect" not in already:
        return "inspect"

    return None


def _terminal_decision(obs) -> str:
    """
    Make a terminal decision.  Uses enriched observation fields where available
    (inspect_reveal, docs_notes, kyc_notes, contact_notes already in obs.info).
    """
    # Definitive danger signals
    if any(f in _DANGER_FLAGS for f in obs.flags):
        # Exception: if we confirmed via contact that it is genuine, approve
        contact = getattr(obs, "contact_notes", None) or ""
        if "CEO confirms they did NOT" in contact or "Classic APP scam" in contact:
            return "reject"
        return "reject"

    vel = obs.velocity_1h
    if vel is not None and vel >= 10:
        return "reject"

    if obs.kyc_status in ("failed", "none"):
        return "escalate"

    # KYC expired → hold (wait for renewal)
    if obs.kyc_status == "expired":
        return "hold"

    # Very high risk
    if obs.risk_score >= 0.85:
        return "reject"

    # Watchlist flags with high value
    if _WATCHLIST_FLAGS & set(obs.flags) and obs.amount >= 20_000:
        return "escalate"

    # FX / correspondent banking settlement
    if "fx_settlement" in obs.flags:
        return "approve"

    # After inspection revealed legitimacy (corp-level, FX)
    inspection = getattr(obs, "inspection_notes", None) or ""
    if "correspondent banking" in inspection.lower() or "on file" in inspection.lower():
        if obs.risk_score < 0.75:
            return "approve"

    if obs.risk_score >= 0.65:
        return "escalate"
    elif obs.risk_score >= 0.40 or obs.flags:
        return "flag"
    else:
        return "approve"


def _rule_based_policy(obs) -> str:
    """
    Adaptive policy.  First checks whether investigation is needed; if so
    returns a sub-action.  Otherwise returns a terminal decision.
    """
    sub = _should_investigate(obs)
    if sub is not None:
        return sub
    return _terminal_decision(obs)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_baseline() -> Tuple[List[Dict[str, Any]], float, float, int]:
    """
    Run the adaptive rule-based baseline over the full task set.

    Returns:
        (per_task_rewards, total_reward, normalised_score, steps)
    """
    env = PayOpsEnvironment()
    obs = await env.reset_async()
    actions_taken: List[str] = []
    confs: List[Optional[float]] = []
    step = 0

    while not obs.done:
        action_type = _rule_based_policy(obs)
        action = PayOpsAction(
            action_type=action_type,
            transaction_id=obs.transaction_id,
        )
        obs = await env.step_async(action)
        actions_taken.append(action_type)
        confs.append(None)
        step += 1

    jittered_tasks = list(env._tasks)
    env.close()

    result = grade_episode(actions_taken, jittered_tasks, confs)
    return result.per_task_rewards, result.total_reward, result.normalised_score, step

