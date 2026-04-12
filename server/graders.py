"""
Grader callables for PayOps OpenEnv tasks.

Referenced from openenv.yaml as:  server.graders:EASY001Grader

Each grader is fully self-contained (zero external imports) so the platform
validator can import and call them in any isolated environment.

Returns {"score": float, "feedback": str} as required by the platform.
Scores are always strictly in (0, 1) — never exactly 0.0 or 1.0.
"""
from __future__ import annotations

# Platform requires scores strictly between 0 and 1 (exclusive).
_SCORE_MIN = 0.001
_SCORE_MAX = 0.999


def _clamp(score: float) -> float:
    """Clamp score to the open interval (0, 1)."""
    if score <= 0.0:
        return _SCORE_MIN
    if score >= 1.0:
        return _SCORE_MAX
    return score


class _BaseGrader:
    """
    Self-contained base grader.

    Subclasses set:
      correct_action : str
      partial_credit : dict[str, float]   (action -> score, all in (0, 1))
    """

    correct_action: str = ""
    partial_credit: dict = {}

    def grade(self, action: str, **kwargs):
        if action == self.correct_action:
            return {"score": _SCORE_MAX, "feedback": "Correct action"}
        raw = float(self.partial_credit.get(action, 0.0))
        score = _clamp(raw)
        return {
            "score": score,
            "feedback": "Partial credit" if raw > 0 else "Incorrect action",
        }

    def __call__(self, action: str, **kwargs):
        return self.grade(action, **kwargs)


# ── Easy ─────────────────────────────────────────────────────────────────────
class EASY001Grader(_BaseGrader):
    correct_action = "approve"
    partial_credit = {}

class EASY002Grader(_BaseGrader):
    correct_action = "reject"
    partial_credit = {"escalate": 0.4}

class EASY003Grader(_BaseGrader):
    correct_action = "approve"
    partial_credit = {}

class EASY004Grader(_BaseGrader):
    correct_action = "flag"
    partial_credit = {"escalate": 0.6, "hold": 0.5}

class EASY005Grader(_BaseGrader):
    correct_action = "approve"
    partial_credit = {}

class EASY006Grader(_BaseGrader):
    correct_action = "flag"
    partial_credit = {"hold": 0.6}

# ── Medium ────────────────────────────────────────────────────────────────────
class MED001Grader(_BaseGrader):
    correct_action = "escalate"
    partial_credit = {"flag": 0.5, "hold": 0.4}

class MED002Grader(_BaseGrader):
    correct_action = "hold"
    partial_credit = {"escalate": 0.6, "flag": 0.4}

class MED003Grader(_BaseGrader):
    correct_action = "flag"
    partial_credit = {"hold": 0.5, "escalate": 0.3}

class MED004Grader(_BaseGrader):
    correct_action = "flag"
    partial_credit = {"escalate": 0.5, "hold": 0.4}

class MED005Grader(_BaseGrader):
    correct_action = "hold"
    partial_credit = {"flag": 0.5, "escalate": 0.4}

class MED006Grader(_BaseGrader):
    correct_action = "escalate"
    partial_credit = {"flag": 0.4, "hold": 0.5}

class MED007Grader(_BaseGrader):
    correct_action = "hold"
    partial_credit = {"escalate": 0.6, "flag": 0.4}

class MED008Grader(_BaseGrader):
    correct_action = "flag"
    partial_credit = {"hold": 0.6, "escalate": 0.4}

# ── Hard ──────────────────────────────────────────────────────────────────────
class HARD001Grader(_BaseGrader):
    correct_action = "escalate"
    partial_credit = {"flag": 0.6, "hold": 0.5, "reject": 0.3}

class HARD002Grader(_BaseGrader):
    correct_action = "reject"
    partial_credit = {"escalate": 0.6, "hold": 0.5, "flag": 0.3}

class HARD003Grader(_BaseGrader):
    correct_action = "reject"
    partial_credit = {"escalate": 0.5, "flag": 0.2}

class HARD004Grader(_BaseGrader):
    correct_action = "approve"
    partial_credit = {"escalate": 0.5, "hold": 0.4, "flag": 0.3}

class HARD005Grader(_BaseGrader):
    correct_action = "escalate"
    partial_credit = {"flag": 0.5, "hold": 0.4, "reject": 0.3}

class HARD006Grader(_BaseGrader):
    correct_action = "flag"
    partial_credit = {"hold": 0.6, "escalate": 0.5}

class HARD007Grader(_BaseGrader):
    correct_action = "reject"
    partial_credit = {"hold": 0.5, "escalate": 0.4, "flag": 0.3}

class HARD008Grader(_BaseGrader):
    correct_action = "reject"
    partial_credit = {"hold": 0.5, "escalate": 0.4, "flag": 0.3}

class HARD009Grader(_BaseGrader):
    correct_action = "escalate"
    partial_credit = {"hold": 0.5, "flag": 0.4, "reject": 0.3}

class HARD010Grader(_BaseGrader):
    correct_action = "reject"
    partial_credit = {"hold": 0.5, "escalate": 0.4, "flag": 0.3}

# ── Critical ──────────────────────────────────────────────────────────────────
class CRIT001Grader(_BaseGrader):
    correct_action = "approve"
    partial_credit = {"escalate": 0.5, "hold": 0.4, "flag": 0.3}

class CRIT002Grader(_BaseGrader):
    correct_action = "reject"
    partial_credit = {"escalate": 0.4, "flag": 0.2}

class CRIT003Grader(_BaseGrader):
    correct_action = "escalate"
    partial_credit = {"flag": 0.4, "hold": 0.35, "reject": 0.3}

class CRIT004Grader(_BaseGrader):
    correct_action = "reject"
    partial_credit = {"escalate": 0.5, "hold": 0.4}

class CRIT005Grader(_BaseGrader):
    correct_action = "reject"
    partial_credit = {"escalate": 0.5, "hold": 0.4}

class CRIT006Grader(_BaseGrader):
    correct_action = "escalate"
    partial_credit = {"hold": 0.5, "reject": 0.4}
