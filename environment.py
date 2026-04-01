"""
PayOps Environment — core simulation.

Supports all 10 action types:
  Terminal decisions: approve | reject | flag | escalate | hold
  Investigation sub-actions: inspect | request_docs | verify_kyc |
                             contact_sender | file_sar

Investigation sub-actions do NOT advance the task pointer; they reveal
additional context and consume budget.  A file_sar sub-action is required
for full credit on regulatory tasks.

Multi-step chain tasks (chain_total > 1) are presented as a single task
entry; the grader handles them as one decision unit.
"""

from __future__ import annotations

import copy
import random
import uuid
from collections import deque
from typing import Deque, List, Optional

from payops_env.grader import INVESTIGATION_BONUS, grade
from payops_env.models import PayOpsAction, PayOpsObservation, PayOpsState
from payops_env.tasks import ACTION_COSTS, TASK_VARIANTS, TASKS, PayOpsTask


TERMINAL_ACTIONS   = {"approve", "reject", "flag", "escalate", "hold"}
INVESTIGATION_ACTIONS = {"inspect", "request_docs", "verify_kyc", "contact_sender", "file_sar"}
VALID_ACTIONS      = TERMINAL_ACTIONS | INVESTIGATION_ACTIONS

_RECENT_WINDOW = 4   # how many past (task_id, action) pairs to surface in obs


class PayOpsEnvironment:
    """
    OpenEnv-compatible environment for Payment Operations triage.

    An episode proceeds through every task in TASKS.  For each task the
    agent may issue any number of investigation sub-actions before a
    terminal decision closes the task.

    Budget
    ------
    The agent starts with ``budget_limit`` points.  Each investigation
    sub-action deducts its cost (see tasks.ACTION_COSTS).  Budget overspend
    is penalised in grade_episode; within the environment it is tracked but
    does not terminate the episode.
    """

    BUDGET_LIMIT = 5.0

    def __init__(self):
        self._tasks: List[PayOpsTask] = []
        self._current_task: Optional[PayOpsTask] = None
        self._state: PayOpsState = PayOpsState()
        self._used_inv: dict = {}       # task_id → set of investigation actions used
        self._sar_filed: set = set()    # task_ids where file_sar was issued
        self._recent_decisions: Deque  = deque(maxlen=_RECENT_WINDOW)

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    async def reset_async(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> PayOpsObservation:
        # --- per-episode jitter: prevents agent overfitting to fixed values ---
        # Use caller-supplied seed when provided (enables reproducibility).
        episode_seed = seed if seed is not None else int(uuid.uuid4().int % 2**31)
        rng = random.Random(episode_seed)
        jittered: List[PayOpsTask] = []
        for t in TASKS:
            jt = copy.copy(t)
            jt.amount     = round(t.amount * rng.uniform(0.85, 1.20), 2)
            jt.risk_score = round(min(1.0, max(0.0, t.risk_score + rng.gauss(0, 0.03))), 4)
            if t.velocity_1h  is not None:
                jt.velocity_1h  = max(0, t.velocity_1h  + rng.randint(-3, 3))
            if t.velocity_24h is not None:
                jt.velocity_24h = max(0, t.velocity_24h + rng.randint(-3, 3))
            jittered.append(jt)
        self._tasks = jittered

        # ── Variant pool selection ──────────────────────────────────────────────
        # Each task in TASK_VARIANTS has 2 alternative scenarios (in addition to
        # the base).  The episode seed selects which scenario plays out, making
        # the correct_action unknowable from the task_id alone — the agent MUST
        # investigate to discover the decisive evidence.
        #
        # seed=0 is the canonical episode (all base variants); used by the test
        # suite so that hardcoded expected rewards remain stable.
        if episode_seed != 0:
            for i, jt in enumerate(self._tasks):
                variants = TASK_VARIANTS.get(jt.task_id, [])
                if not variants:
                    continue
                # variant_idx=0 → base (no changes); 1..N → variants[0..N-1]
                variant_idx = (episode_seed * 1009 + i * 17) % (len(variants) + 1)
                if variant_idx == 0:
                    continue
                overrides = variants[variant_idx - 1]
                for key, val in overrides.items():
                    setattr(jt, key, val)
        self._current_task = self._tasks[0]
        self._used_inv = {}
        self._sar_filed = set()
        self._recent_decisions = deque(maxlen=_RECENT_WINDOW)
        self._episode_seed = episode_seed
        self._state = PayOpsState(
            episode_id=episode_id if episode_id is not None else str(uuid.uuid4()),
            episode_seed=episode_seed,
            step_count=0,
            current_task_id=self._current_task.task_id,
            transactions_processed=0,
            total_tasks=len(self._tasks),
            cumulative_reward=0.0,
            actions_taken=[],
            last_action=None,
            done=False,
            budget_spent=0.0,
            budget_limit=self.BUDGET_LIMIT,
            investigation_actions_used=[],
            correct_decisions=0,
            wrong_high_cost=0,
            recent_decisions=[],
        )
        return self._make_observation(reward=0.0, done=False, info={"event": "reset"})

    async def step_async(self, action: PayOpsAction) -> PayOpsObservation:
        if self._current_task is None:
            raise RuntimeError("Environment must be reset before stepping.")

        action_type = action.action_type.lower()
        if action_type not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{action_type}'. "
                f"Valid actions: {sorted(VALID_ACTIONS)}"
            )

        task = self._current_task
        task_id = task.task_id
        cost = ACTION_COSTS.get(action_type, 0.0)

        # ── MULTI-STEP CHAIN GATE ─────────────────────────────────────────────
        # Critical tasks with chain_total > 1 require (chain_total − 1)
        # investigation sub-actions before a terminal decision is accepted.
        # Blocked attempts return a helpful message without advancing the task.
        if action_type in TERMINAL_ACTIONS:
            chain_min = max(0, getattr(task, "chain_total", 1) - 1)
            inv_done  = len(self._used_inv.get(task_id, set()))
            if chain_min > 0 and inv_done < chain_min:
                needed = chain_min - inv_done
                return self._make_observation(
                    reward=-0.05,
                    done=False,
                    info={
                        "event":              "chain_gate_blocked",
                        "chain_status":       "investigation_required",
                        "chain_steps_needed": needed,
                        "message": (
                            f"This {task.difficulty} transaction requires {needed} "
                            f"more investigation step(s) before a terminal decision. "
                            f"Please investigate first."
                        ),
                    },
                )

        # Deduct cost
        self._state.budget_spent = round(self._state.budget_spent + cost, 4)
        self._state.step_count   += 1
        self._state.actions_taken.append(action_type)
        self._state.last_action   = action_type

        # ── INVESTIGATION SUB-ACTIONS ────────────────────────────────────
        if action_type in INVESTIGATION_ACTIONS:
            used = self._used_inv.setdefault(task_id, set())
            already = action_type in used
            reward = 0.0 if already else INVESTIGATION_BONUS
            used.add(action_type)
            self._state.investigation_actions_used.append(action_type)
            self._state.cumulative_reward = round(
                self._state.cumulative_reward + reward, 4
            )

            # Determine reveal text
            reveal_text: Optional[str] = None
            reveal_field: Optional[str] = None
            if action_type == "inspect":
                reveal_text  = task.inspect_reveal  or "No additional information available."
                reveal_field = "inspection_notes"
            elif action_type == "request_docs":
                reveal_text  = task.docs_reveal     or "No documents on record for this transaction."
                reveal_field = "docs_notes"
            elif action_type == "verify_kyc":
                reveal_text  = task.kyc_reveal      or "KYC records could not be retrieved."
                reveal_field = "kyc_notes"
            elif action_type == "contact_sender":
                reveal_text  = task.contact_reveal  or "Sender did not respond to contact attempt."
                reveal_field = "contact_notes"
            elif action_type == "file_sar":
                self._sar_filed.add(task_id)
                reveal_text  = (
                    "SAR filed with FinCEN. Reference number will be generated within 24 h."
                    if task.regulatory_action
                    else "SAR filed. Note: this transaction may not meet SAR-filing threshold."
                )
                reveal_field = "docs_notes"

            used_so_far = list(self._used_inv.get(task_id, set()))
            info = {
                "event":               action_type,
                "already_used":        already,
                "investigation_used":  used_so_far,  # full list for this task
                reveal_field:          reveal_text,
                "budget_remaining":    round(
                    self.BUDGET_LIMIT - self._state.budget_spent, 4
                ),
            }
            return self._make_observation(
                reward=reward, done=False, info=info, reveal_field=reveal_field, reveal_text=reveal_text
            )

        # ── TERMINAL DECISION ────────────────────────────────────────────
        used_inv = list(self._used_inv.get(task_id, set()))
        sar_used = task_id in self._sar_filed
        inspected_already = "inspect" in self._used_inv.get(task_id, set())

        investigation_done = bool(self._used_inv.get(task_id, set()))
        reward = grade(
            action_type, task,
            inspected_already=inspected_already,
            investigation_done=investigation_done,
        )
        self._state.cumulative_reward = round(
            self._state.cumulative_reward + reward, 4
        )

        is_correct = action_type == task.correct_action
        if is_correct:
            self._state.correct_decisions += 1
        elif action_type == "approve" and task.correct_action in ("reject", "escalate"):
            self._state.wrong_high_cost += 1

        self._recent_decisions.append(
            {"task_id": task_id, "action": action_type, "correct": is_correct}
        )
        self._state.recent_decisions = list(self._recent_decisions)
        self._state.transactions_processed += 1

        # Advance task pointer
        task_idx = self._tasks.index(task)
        remaining = self._tasks[task_idx + 1:]
        done = len(remaining) == 0

        if not done:
            self._current_task = remaining[0]
            self._state.current_task_id = self._current_task.task_id
        else:
            self._state.done = True

        return self._make_observation(
            reward=reward,
            done=done,
            info={
                "event":          "step",
                "action_taken":   action_type,
                "correct_action": task.correct_action,
                "task_id":        task_id,
                "difficulty":     task.difficulty,
                "is_correct":     is_correct,
                "investigation_used": used_inv,
                "budget_remaining":   round(
                    self.BUDGET_LIMIT - self._state.budget_spent, 4
                ),
            },
        )

    def state(self) -> PayOpsState:
        return self._state

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(
        self,
        reward: float,
        done: bool,
        info: dict,
        reveal_field: Optional[str] = None,
        reveal_text: Optional[str] = None,
    ) -> PayOpsObservation:
        task = self._current_task
        steps_remaining = self._state.total_tasks - self._state.transactions_processed

        # Progressive disclosure: sensitive forensic fields are only surfaced after
        # the agent has called 'inspect' on this task.  This makes the investigate-
        # before-deciding mechanic load-bearing rather than cosmetic.
        task_id = task.task_id
        inspected = "inspect" in self._used_inv.get(task_id, set())
        contacted = "contact_sender" in self._used_inv.get(task_id, set())

        return PayOpsObservation(
            # ── transaction core ──
            transaction_id=task.transaction_id,
            amount=task.amount,
            currency=task.currency,
            sender=task.sender,
            receiver=task.receiver,
            transaction_type=task.transaction_type,
            status=(
                "inspected"
                if info.get("event") in INVESTIGATION_ACTIONS
                else ("done" if done else "pending")
            ),
            # ── risk signals ──
            risk_score=task.risk_score,
            ml_confidence=getattr(task, "ml_confidence", 0.90),
            flags=list(task.flags),
            velocity_1h=task.velocity_1h,
            velocity_24h=getattr(task, "velocity_24h", None),
            avg_transaction_amount=getattr(task, "avg_transaction_amount", None),
            account_age_days=getattr(task, "account_age_days", None),
            country_risk=task.country_risk,
            kyc_status=task.kyc_status,
            kyc_expiry_days=getattr(task, "kyc_expiry_days", None),
            previous_violations=task.previous_violations if inspected else None,
            previous_sars=getattr(task, "previous_sars", None) if inspected else None,
            counterparty_risk=getattr(task, "counterparty_risk", None) if inspected else None,
            # ── chain context ──
            chain_total=getattr(task, "chain_total", 1),
            chain_step=self._state.step_count,
            chain_context=task.description,
            # ── investigation reveals ──
            inspection_notes=(
                reveal_text if reveal_field == "inspection_notes" else info.get("inspection_notes")
            ),
            docs_notes=(
                reveal_text if reveal_field == "docs_notes" else None
            ),
            kyc_notes=(
                reveal_text if reveal_field == "kyc_notes" else None
            ),
            contact_notes=(
                reveal_text if reveal_field == "contact_notes"
                else (info.get("contact_notes") if contacted else None)
            ),
            # ── episode meta ──
            task_id=task.task_id,
            task_difficulty=task.difficulty,
            step_in_episode=self._state.step_count,
            steps_remaining=steps_remaining,
            action_cost=ACTION_COSTS.get(info.get("event", ""), 0.0),
            budget_remaining=round(self.BUDGET_LIMIT - self._state.budget_spent, 4),
            investigation_hints=[],  # not surfaced upfront; agent must explore
            recent_decisions=list(self._recent_decisions),
            reward=reward,
            cumulative_reward=self._state.cumulative_reward,
            done=done,
            network_graph=getattr(task, "network_graph", None) if inspected else None,
            info=info,
        )


