"""
Grader for the PayOps environment.

Reward design (v2 — trajectory-based)
--------------------------------------
The grader rewards *correct intermediate reasoning* as well as the final
call, so agents receive a dense learning signal across the full trajectory.

Terminal action credit is now split:
  Correct final action           → +0.60  (base)
  Partial-credit adjacent action → fraction × 0.60
  approve when should be reject/escalate → −1.00  (worst mistake)
  approve when should be flag/hold       → −0.50
  reject  when should be approve         → −0.50
  any other wrong terminal action        → −0.25

Investigation sub-action bonuses (per eligible, first use only):
  Used one of task.requires_investigation  → +0.20
  Flag identification: agent used inspect AND task.key_flags ⊆ obs.flags → +0.20
  (Both bonuses are independent and stackable.)

Duplicate investigation penalty:
  Same sub-action on same task more than once → −0.05

Modifiers:
  Difficulty weight: easy=1.0, medium=1.2, hard=1.5, critical=2.0
  Confidence (≥0.8) AND correct  → +0.10
  Confidence (≥0.8) AND wrong    → −0.10
  Regulatory bonus (file_sar before terminal on regulatory task) → +0.20
  Budget overspend penalty: (spent − limit) × 0.10

Normalised episode score: [0, 1], strictly clamped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from payops_env.tasks import ACTION_COSTS, PayOpsTask

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Terminal-action credit (correct action gets this, not the old 1.0 full credit)
TERMINAL_CORRECT      = 0.60
FULL_CREDIT           = TERMINAL_CORRECT  # alias for backward compat

WRONG_APPROVE_FRAUD   = -1.0
WRONG_APPROVE_CAUTION = -0.5
WRONG_REJECT_GOOD     = -0.5
WRONG_DEFAULT         = -0.25

# Investigation trajectory bonuses
INVESTIGATION_BONUS         = 0.20   # per eligible sub-action used (first use)
FLAG_IDENTIFICATION_BONUS   = 0.20   # agent ran inspect AND all key_flags are in obs
TIME_PENALTY_PER_EXTRA_STEP = 0.05   # duplicate investigation on same task

CONFIDENCE_CORRECT_BONUS = 0.10
CONFIDENCE_WRONG_PENALTY = -0.10
REGULATORY_BONUS         = 0.20
BUDGET_OVERSPEND_PENALTY = 0.10

DIFFICULTY_WEIGHT: Dict[str, float] = {
    "easy":     1.0,
    "medium":   1.2,
    "hard":     1.5,
    "critical": 2.0,
}

INVESTIGATION_ACTIONS: Set[str] = {
    "inspect", "request_docs", "verify_kyc", "contact_sender", "file_sar"
}

# Maximum achievable reward per task at weight=1.0 (used for normalisation)
# correct terminal (0.6) + up to 1 inv bonus (0.2) + flag id bonus (0.2)
_MAX_TASK_RAW = TERMINAL_CORRECT + INVESTIGATION_BONUS + FLAG_IDENTIFICATION_BONUS


# ---------------------------------------------------------------------------
# Per-step helpers
# ---------------------------------------------------------------------------

def _is_investigation(action_type: str) -> bool:
    return action_type in INVESTIGATION_ACTIONS


def _base_terminal_reward(action_type: str, task: PayOpsTask) -> float:
    """Return the base reward for a terminal action against a task."""
    if action_type == task.correct_action:
        return TERMINAL_CORRECT

    if action_type in task.partial_credit_actions:
        return TERMINAL_CORRECT * task.partial_credit_actions[action_type]

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
    Single-step reward used by the real-time environment (step_async).
    Investigation bonuses are now INVESTIGATION_BONUS (0.20).
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
    flag_id_bonus: float
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
    weight  = DIFFICULTY_WEIGHT.get(task.difficulty, 1.0)
    base    = _base_terminal_reward(terminal_action, task)
    correct = terminal_action == task.correct_action

    # ── investigation trajectory bonus & time penalty ────────────────────────
    inv_bonus  = 0.0
    time_pen   = 0.0
    eligible   = getattr(task, "requires_investigation", set())
    seen_counts: Dict[str, int] = {}
    for inv_action in investigation_actions:
        seen_counts[inv_action] = seen_counts.get(inv_action, 0) + 1
        if inv_action in eligible and seen_counts[inv_action] == 1:
            inv_bonus += INVESTIGATION_BONUS
        elif seen_counts[inv_action] > 1:
            time_pen += TIME_PENALTY_PER_EXTRA_STEP

    # ── flag-identification bonus ────────────────────────────────────────────
    # Awarded when: agent used 'inspect' AND the task has key_flags AND all
    # key_flags are present in the task's flag list (they are always present
    # as the randomised episode preserves the original flags).
    flag_id = 0.0
    key_flags = getattr(task, "key_flags", [])
    if key_flags and "inspect" in investigation_actions:
        # key_flags on the task are guaranteed to be in task.flags; reward
        # the agent for using inspect (which reveals them) when they matter.
        flag_id = FLAG_IDENTIFICATION_BONUS

    # ── confidence modifier ──────────────────────────────────────────────────
    conf_mod = 0.0
    if agent_confidence is not None and agent_confidence >= 0.8:
        conf_mod = CONFIDENCE_CORRECT_BONUS if correct else CONFIDENCE_WRONG_PENALTY

    # ── regulatory bonus ─────────────────────────────────────────────────────
    reg_bonus = 0.0
    if getattr(task, "regulatory_action", False) and "file_sar" in investigation_actions:
        reg_bonus = REGULATORY_BONUS

    raw_total = base + inv_bonus + flag_id - time_pen + conf_mod + reg_bonus
    total = weight * raw_total

    return TaskGradeDetail(
        task_id=task.task_id,
        difficulty=task.difficulty,
        weight=weight,
        correct_action=task.correct_action,
        terminal_action=terminal_action,
        investigation_actions_used=investigation_actions,
        base_reward=round(base, 4),
        investigation_bonus=round(inv_bonus, 4),
        flag_id_bonus=round(flag_id, 4),
        time_penalty=round(time_pen, 4),
        confidence_modifier=round(conf_mod, 4),
        regulatory_bonus=round(reg_bonus, 4),
        total_reward=round(total, 4),
        correct=correct,
        reward_breakdown={
            "base":           round(base, 4),
            "weight":         weight,
            "investigation":  round(inv_bonus, 4),
            "flag_id":        round(flag_id, 4),
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
    normalised_score: float        # strictly 0.0 – 1.0
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

    Returns EpisodeResult with normalised_score strictly in [0.0, 1.0].
    """
    if confidences is None:
        confidences = [None] * len(actions)

    per_task_details: List[TaskGradeDetail] = []
    budget_spent = 0.0

    task_idx     = 0
    pending_inv: List[str]              = []
    pending_conf: List[Optional[float]] = []

    for action, conf in zip(actions, confidences):
        budget_spent += ACTION_COSTS.get(action, 0.0)

        if _is_investigation(action):
            pending_inv.append(action)
            pending_conf.append(conf)
        else:
            if task_idx >= len(tasks):
                break
            task   = tasks[task_idx]
            detail = _grade_single_task(action, pending_inv, task, agent_confidence=conf)
            per_task_details.append(detail)
            pending_inv   = []
            pending_conf  = []
            task_idx     += 1

    # Tasks the agent never reached get a small default penalty
    while task_idx < len(tasks):
        task   = tasks[task_idx]
        weight = DIFFICULTY_WEIGHT.get(task.difficulty, 1.0)
        detail = _grade_single_task("hold", [], task, agent_confidence=None)
        # Override to a neutral miss (no severe penalty for unreached tasks)
        detail.base_reward  = 0.0
        detail.total_reward = 0.0
        per_task_details.append(detail)
        task_idx += 1

    # ── budget overspend penalty ─────────────────────────────────────────────
    budget_overspend = max(0.0, budget_spent - budget_limit)
    budget_penalty   = round(budget_overspend * BUDGET_OVERSPEND_PENALTY, 4)

    raw_total  = sum(d.total_reward for d in per_task_details)
    total      = raw_total - budget_penalty

    # Max possible = each task at full trajectory credit × difficulty weight
    # (terminal 0.6 + one inv 0.2 + flag_id 0.2) × weight
    max_possible = sum(
        DIFFICULTY_WEIGHT.get(t.difficulty, 1.0) * _MAX_TASK_RAW
        for t in tasks
    )

    # Strict [0, 1] clamp — grader can NEVER exceed 1.0 or go below 0.0
    normalised = max(0.0, min(1.0, total / max_possible)) if max_possible > 0 else 0.0

    return EpisodeResult(
        total_reward=round(total, 4),
        max_possible_reward=round(max_possible, 4),
        normalised_score=round(normalised, 4),
        per_task_rewards=[
            {
                "task_id":           d.task_id,
                "difficulty":        d.difficulty,
                "weight":            d.weight,
                "terminal_action":   d.terminal_action,
                "correct_action":    d.correct_action,
                "investigation_used":d.investigation_actions_used,
                "correct":           d.correct,
                "reward_breakdown":  d.reward_breakdown,
                "weighted_reward":   d.total_reward,
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

