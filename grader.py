"""
Grader for the PayOps environment.

Reward design
-------------
The grader provides a *partial-credit* reward signal so agents receive
feedback for cautious-but-not-perfect decisions.

Base rewards
~~~~~~~~~~~~
  Correct action                        → +1.0  (full credit)
  Partial-credit adjacent action        → per-task fraction of +1.0
  approve when should be reject/escalate→ -1.0  (worst mistake — approving fraud)
  approve when should be flag/hold      → -0.5
  reject when should be approve         → -0.5  (over-rejection)
  any other wrong terminal action       → -0.25

  Investigation sub-actions  (non-terminal):
    inspect / request_docs / verify_kyc / contact_sender used BEFORE the
    terminal decision when the task requests them  → +0.15 bonus each
    Same investigation action used twice on same task → +0.0 (no double-dip)

Modifiers
~~~~~~~~~
  Difficulty weight     — multiplies the base reward:
    easy=1.0, medium=1.2, hard=1.5, critical=2.0

  Confidence bonus/penalty  — applied when the agent provides confidence:
    high-confidence (≥0.8) AND correct   → +0.10
    high-confidence (≥0.8) AND wrong     → -0.10

  Cost penalty          — investigation actions have per-action budget costs
    (see tasks.ACTION_COSTS).  Spending beyond budget_limit reduces episode
    score by budget_overspend × 0.1.

  Time penalty          — excessive investigation steps on a single task:
    each investigation sub-action beyond the first on the same task → -0.05

  Regulatory bonus      — for tasks requiring a SAR filing:
    agent called file_sar before terminal action → +0.20 bonus

Episode score (0–1)
~~~~~~~~~~~~~~~~~~~
  Computed by summing difficulty-weighted reward across all tasks and
  normalising against the maximum theoretically achievable reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from payops_env.tasks import ACTION_COSTS, PayOpsTask

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FULL_CREDIT = 1.0

WRONG_APPROVE_FRAUD   = -1.0
WRONG_APPROVE_CAUTION = -0.5
WRONG_REJECT_GOOD     = -0.5
WRONG_DEFAULT         = -0.25

INVESTIGATION_BONUS       = 0.15   # per eligible sub-action used before terminal
TIME_PENALTY_PER_EXTRA_STEP = 0.05 # per duplicate investigation on same task
CONFIDENCE_CORRECT_BONUS  = 0.10
CONFIDENCE_WRONG_PENALTY  = -0.10
REGULATORY_BONUS          = 0.20   # filing SAR when required
BUDGET_OVERSPEND_PENALTY  = 0.10   # per unit of budget exceeded

DIFFICULTY_WEIGHT: Dict[str, float] = {
    "easy":     1.0,
    "medium":   1.2,
    "hard":     1.5,
    "critical": 2.0,
}

INVESTIGATION_ACTIONS: Set[str] = {
    "inspect", "request_docs", "verify_kyc", "contact_sender", "file_sar"
}


# ---------------------------------------------------------------------------
# Per-step helpers
# ---------------------------------------------------------------------------

def _is_investigation(action_type: str) -> bool:
    return action_type in INVESTIGATION_ACTIONS


def _base_terminal_reward(action_type: str, task: PayOpsTask) -> float:
    """Return the base reward for a terminal action against a task."""
    if action_type == task.correct_action:
        return FULL_CREDIT

    if action_type in task.partial_credit_actions:
        return FULL_CREDIT * task.partial_credit_actions[action_type]

    if action_type == "approve" and task.correct_action in ("reject", "escalate"):
        return WRONG_APPROVE_FRAUD

    if action_type == "approve" and task.correct_action in ("flag", "hold"):
        return WRONG_APPROVE_CAUTION

    if action_type == "reject" and task.correct_action == "approve":
        return WRONG_REJECT_GOOD

    return WRONG_DEFAULT


def step_reward(
    action_type: str,
    task: PayOpsTask,
    inspected_already: bool = False,
) -> float:
    """
    Backward-compatible single-step reward used by the real-time environment.

    Investigation actions return 0 (already handled inside environment.step_async
    through the richer grader logic).  Terminal actions use the base reward table.
    """
    if _is_investigation(action_type):
        return 0.0 if inspected_already else INVESTIGATION_BONUS

    return _base_terminal_reward(action_type, task)


# ---------------------------------------------------------------------------
# Extended per-task grader (used by grade_episode)
# ---------------------------------------------------------------------------

@dataclass
class TaskGradeDetail:
    task_id: str
    difficulty: str
    weight: float
    correct_action: str
    terminal_action: str
    investigation_actions_used: List[str]
    base_reward: float
    investigation_bonus: float
    time_penalty: float
    confidence_modifier: float
    regulatory_bonus: float
    total_reward: float
    correct: bool
    reward_breakdown: Dict[str, float] = field(default_factory=dict)


def _grade_single_task(
    terminal_action: str,
    investigation_actions: List[str],   # sub-actions used BEFORE terminal
    task: PayOpsTask,
    agent_confidence: Optional[float] = None,
) -> TaskGradeDetail:
    weight = DIFFICULTY_WEIGHT.get(task.difficulty, 1.0)
    base   = _base_terminal_reward(terminal_action, task)
    correct = terminal_action == task.correct_action

    # ── investigation bonus and time penalty ────────────────────────────────
    inv_bonus  = 0.0
    time_pen   = 0.0
    eligible   = task.requires_investigation  # set of actions agent should use
    seen_counts: Dict[str, int] = {}
    for inv_action in investigation_actions:
        seen_counts[inv_action] = seen_counts.get(inv_action, 0) + 1
        if inv_action in eligible and seen_counts[inv_action] == 1:
            inv_bonus += INVESTIGATION_BONUS
        elif seen_counts[inv_action] > 1:
            time_pen += TIME_PENALTY_PER_EXTRA_STEP

    # ── confidence modifier ─────────────────────────────────────────────────
    conf_mod = 0.0
    if agent_confidence is not None and agent_confidence >= 0.8:
        conf_mod = CONFIDENCE_CORRECT_BONUS if correct else CONFIDENCE_WRONG_PENALTY

    # ── regulatory bonus ────────────────────────────────────────────────────
    reg_bonus = 0.0
    if task.regulatory_action and "file_sar" in investigation_actions:
        reg_bonus = REGULATORY_BONUS

    total = weight * (base + inv_bonus - time_pen + conf_mod + reg_bonus)

    return TaskGradeDetail(
        task_id=task.task_id,
        difficulty=task.difficulty,
        weight=weight,
        correct_action=task.correct_action,
        terminal_action=terminal_action,
        investigation_actions_used=investigation_actions,
        base_reward=round(base, 4),
        investigation_bonus=round(inv_bonus, 4),
        time_penalty=round(time_pen, 4),
        confidence_modifier=round(conf_mod, 4),
        regulatory_bonus=round(reg_bonus, 4),
        total_reward=round(total, 4),
        correct=correct,
        reward_breakdown={
            "base":           round(base, 4),
            "weight":         weight,
            "investigation":  round(inv_bonus, 4),
            "time_penalty":   round(-time_pen, 4),
            "confidence":     round(conf_mod, 4),
            "regulatory":     round(reg_bonus, 4),
            "weighted_total": round(total, 4),
        },
    )


# ---------------------------------------------------------------------------
# Episode grader
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    total_reward: float
    max_possible_reward: float
    normalised_score: float        # 0.0 – 1.0
    per_task_rewards: List[dict]
    budget_spent: float
    budget_overspend: float
    budget_penalty: float
    passed: bool                   # normalised_score >= 0.5


def grade_episode(
    actions: List[str],
    tasks: List[PayOpsTask],
    confidences: Optional[List[Optional[float]]] = None,
    budget_limit: float = 5.0,
) -> EpisodeResult:
    """
    Grade a complete episode.

    ``actions`` is the flat list of all actions taken (including investigation
    sub-actions interspersed between terminal decisions).

    The grader separates them by treating any action in INVESTIGATION_ACTIONS as
    a sub-action that accumulates per-task until a terminal action closes the task.

    Args:
        actions:       Ordered list of action_type strings.
        tasks:         Ordered list of PayOpsTask objects.
        confidences:   Optional parallel list of agent confidence values (same
                       length as ``actions``).  Use None for missing entries.
        budget_limit:  Maximum investigation budget.  Overspend is penalised.

    Returns:
        EpisodeResult with comprehensive score breakdown.
    """
    if confidences is None:
        confidences = [None] * len(actions)

    per_task_details: List[TaskGradeDetail] = []
    budget_spent = 0.0

    task_idx   = 0
    pending_inv: List[str]           = []   # sub-actions for current task
    pending_conf: List[Optional[float]] = []

    for i, (action, conf) in enumerate(zip(actions, confidences)):
        budget_spent += ACTION_COSTS.get(action, 0.0)

        if _is_investigation(action):
            pending_inv.append(action)
            pending_conf.append(conf)
        else:
            # terminal action → grade and advance task pointer
            if task_idx >= len(tasks):
                break
            task = tasks[task_idx]
            # Use the confidence from the terminal action step
            detail = _grade_single_task(action, pending_inv, task, agent_confidence=conf)
            per_task_details.append(detail)
            pending_inv   = []
            pending_conf  = []
            task_idx     += 1

    # Handle any tasks with no actions (agent ran out of steps)
    while task_idx < len(tasks):
        task = tasks[task_idx]
        detail = _grade_single_task("approve", [], task, agent_confidence=None)
        detail.base_reward = WRONG_DEFAULT
        detail.total_reward = DIFFICULTY_WEIGHT.get(task.difficulty, 1.0) * WRONG_DEFAULT
        per_task_details.append(detail)
        task_idx += 1

    # ── budget overspend penalty ─────────────────────────────────────────────
    budget_overspend = max(0.0, budget_spent - budget_limit)
    budget_penalty   = round(budget_overspend * BUDGET_OVERSPEND_PENALTY, 4)

    total = sum(d.total_reward for d in per_task_details) - budget_penalty
    max_possible = sum(
        DIFFICULTY_WEIGHT.get(t.difficulty, 1.0) * FULL_CREDIT for t in tasks
    )
    normalised = max(0.0, min(1.0, total / max_possible)) if max_possible > 0 else 0.0

    return EpisodeResult(
        total_reward=round(total, 4),
        max_possible_reward=round(max_possible, 4),
        normalised_score=round(normalised, 4),
        per_task_rewards=[
            {
                "task_id":                  d.task_id,
                "difficulty":               d.difficulty,
                "weight":                   d.weight,
                "terminal_action":          d.terminal_action,
                "correct_action":           d.correct_action,
                "investigation_used":       d.investigation_actions_used,
                "correct":                  d.correct,
                "reward_breakdown":         d.reward_breakdown,
                "weighted_reward":          d.total_reward,
            }
            for d in per_task_details
        ],
        budget_spent=round(budget_spent, 4),
        budget_overspend=round(budget_overspend, 4),
        budget_penalty=budget_penalty,
        passed=normalised >= 0.5,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper used by the environment
# ---------------------------------------------------------------------------

def grade(action_type: str, task: PayOpsTask, inspected_already: bool = False) -> float:
    """Single-step reward used inside environment.step_async."""
    return step_reward(action_type, task, inspected_already=inspected_already)

