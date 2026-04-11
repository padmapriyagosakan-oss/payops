"""
Grader callables for PayOps OpenEnv tasks.

Each class is referenced from openenv.yaml as:
    payops_env.server.graders:EASY001Grader

The platform reads the grader field from openenv.yaml, imports the class,
and uses it to grade agent actions.  Each grader is a callable that accepts
an action string and returns a float reward in [0.0, 1.0].
"""
from __future__ import annotations


class _BaseGrader:
    """Base grader: correct action → 1.0, partial credit if configured, else 0.0."""

    task_id: str = ""

    def grade(self, action: str, **kwargs) -> float:
        from payops_env.tasks import TASKS_BY_ID  # lazy import avoids circular deps

        t = TASKS_BY_ID.get(self.task_id)
        if t is None:
            return 0.0
        if action == t.correct_action:
            return 1.0
        return float(
            dict(getattr(t, "partial_credit_actions", {})).get(action, 0.0)
        )

    def __call__(self, action: str, **kwargs) -> float:
        return self.grade(action, **kwargs)


def _make(task_id: str) -> type:
    """Factory: create a named grader class for the given task_id."""
    return type(
        task_id.replace("-", "") + "Grader",
        (_BaseGrader,),
        {"task_id": task_id},
    )


# ── Easy ─────────────────────────────────────────────────────────────────────
EASY001Grader = _make("EASY-001")  # approve
EASY002Grader = _make("EASY-002")  # reject
EASY003Grader = _make("EASY-003")  # approve
EASY004Grader = _make("EASY-004")  # flag
EASY005Grader = _make("EASY-005")  # approve
EASY006Grader = _make("EASY-006")  # flag

# ── Medium ────────────────────────────────────────────────────────────────────
MED001Grader = _make("MED-001")   # escalate
MED002Grader = _make("MED-002")   # hold
MED003Grader = _make("MED-003")   # flag
MED004Grader = _make("MED-004")   # flag
MED005Grader = _make("MED-005")   # hold
MED006Grader = _make("MED-006")   # escalate
MED007Grader = _make("MED-007")   # hold
MED008Grader = _make("MED-008")   # flag

# ── Hard ──────────────────────────────────────────────────────────────────────
HARD001Grader = _make("HARD-001")  # escalate
HARD002Grader = _make("HARD-002")  # reject
HARD003Grader = _make("HARD-003")  # reject
HARD004Grader = _make("HARD-004")  # approve
HARD005Grader = _make("HARD-005")  # escalate
HARD006Grader = _make("HARD-006")  # flag
HARD007Grader = _make("HARD-007")  # reject
HARD008Grader = _make("HARD-008")  # reject
HARD009Grader = _make("HARD-009")  # escalate
HARD010Grader = _make("HARD-010")  # reject

# ── Critical ──────────────────────────────────────────────────────────────────
CRIT001Grader = _make("CRIT-001")  # approve
CRIT002Grader = _make("CRIT-002")  # reject
CRIT003Grader = _make("CRIT-003")  # escalate
CRIT004Grader = _make("CRIT-004")  # reject
CRIT005Grader = _make("CRIT-005")  # reject
CRIT006Grader = _make("CRIT-006")  # escalate
