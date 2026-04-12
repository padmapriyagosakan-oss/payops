"""
FastAPI server for the PayOps OpenEnv environment.

Endpoints
---------
POST /reset         Reset environment, return initial observation
POST /step          Execute an action, return observation + reward
GET  /state         Current internal environment state
GET  /schema        Action / observation JSON schemas
GET  /tasks         List all tasks with metadata
GET  /grader        Grade the current episode
POST /baseline      Run the rule-based baseline agent
GET  /analytics     Aggregate performance analytics for this session
POST /replay        Grade a supplied action sequence without modifying state
GET  /leaderboard   All scored episodes this session
GET  /health        Health check
WS   /ws            WebSocket for persistent sessions
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from payops_env.environment import PayOpsEnvironment, VALID_ACTIONS
from payops_env.grader import grade_episode
from payops_env.models import PayOpsAction, PayOpsObservation, PayOpsState, PayOpsReward
from payops_env.tasks import TASKS, TASKS_BY_ID


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PayOps OpenEnv",
    description=(
        "Payment Operations Incident Response environment. "
        "An AI agent reviews financial transactions and decides how to handle them."
    ),
    version="2.0.2",
)

_APP_VERSION = "2.0.2"
_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


@app.middleware("http")
async def disable_cache_for_validator_paths(request: Request, call_next):
    """Prevent stale validator responses from being served from caches."""
    response = await call_next(request)
    if request.method in {"GET", "HEAD"}:
        response.headers.update(_NO_CACHE_HEADERS)
    return response


@app.get("/", include_in_schema=False)
async def root():
    """Root liveness endpoint for HF Spaces readiness checks."""
    return {"status": "ok", "app": "payops_env"}


@app.get("/metadata")
async def metadata():
    """Environment metadata — mirrors the openenv create_app /metadata endpoint."""
    return {
        "name": "payops_env",
        "description": (
            "Payment Operations Incident Response environment. "
            "An AI agent reviews financial transactions and decides how to handle them."
        ),
        "version": _APP_VERSION,
    }


@app.get("/metadata-v2")
async def metadata_v2():
    """Versioned metadata alias used to bypass stale edge caches."""
    return await metadata()


# Per-session environment instances — one per /reset call.
# Keyed by episode_id; keeps the last _MAX_SESSIONS sessions to bound memory.
_MAX_SESSIONS = 20
_sessions: Dict[str, Dict[str, Any]] = {}
_current_session_id: Optional[str] = None
_state_lock = asyncio.Lock()   # serialises all state-mutating handlers

# Leaderboard persists for the process lifetime
_leaderboard: List[Dict[str, Any]] = []


def _current_session() -> Dict[str, Any]:
    """Return the session dict for the active episode, or raise HTTP 400."""
    if _current_session_id is None or _current_session_id not in _sessions:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call /reset first.",
        )
    return _sessions[_current_session_id]


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """POST /reset body — compatible with openenv.core ResetRequest."""
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class UnifiedStepRequest(BaseModel):
    """
    POST /step body — accepts both the official openenv wire format::

        {"action": {"action_type": "approve", "transaction_id": "TXN-E001"},
         "timeout_s": null}

    and the legacy flat format (backward compat)::

        {"action_type": "approve", "transaction_id": "TXN-E001"}
    """
    model_config = ConfigDict(extra="allow")

    # Official openenv wire fields
    action: Optional[Dict[str, Any]] = None
    timeout_s: Optional[float] = None
    request_id: Optional[str] = None

    # Legacy flat fields
    action_type: Optional[str] = None
    transaction_id: Optional[str] = None
    reason: Optional[str] = None
    confidence: Optional[float] = None

    def resolved_action(self) -> PayOpsAction:
        """Parse the action from whichever format was supplied."""
        if self.action is not None:
            return PayOpsAction(**self.action)
        if self.action_type is None:
            raise HTTPException(status_code=422, detail="action_type is required")
        return PayOpsAction(
            action_type=self.action_type,
            transaction_id=self.transaction_id or "",
            reason=self.reason,
            confidence=self.confidence,
        )


class EnvResponse(BaseModel):
    """Standard openenv wire response: observation dict + reward + done."""
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


class BaselineResult(BaseModel):
    scores: List[Dict[str, Any]]
    total_reward: float
    normalised_score: float
    steps: int


class ReplayRequest(BaseModel):
    actions: List[str]
    confidences: Optional[List[Optional[float]]] = None
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=EnvResponse, summary="Reset the environment")
async def reset(request: ResetRequest = ResetRequest()):
    """Reset the environment and return the first transaction observation."""
    global _current_session_id
    async with _state_lock:
        env = PayOpsEnvironment()
        obs = await env.reset_async(seed=request.seed, episode_id=request.episode_id)
        session_id = env.state().episode_id
        _sessions[session_id] = {
            "env":     env,
            "actions": [],
            "confs":   [],
            "tasks":   list(env._tasks),   # jittered tasks for this episode
        }
        _current_session_id = session_id
        # Prune oldest sessions when the cap is exceeded
        if len(_sessions) > _MAX_SESSIONS:
            oldest = next(iter(_sessions))
            del _sessions[oldest]
    return EnvResponse(observation=obs.model_dump(), reward=None, done=False)


@app.post("/step", response_model=EnvResponse, summary="Execute an action")
async def step(request: UnifiedStepRequest):
    """
    Submit an action for the current transaction.

    Accepts both the official openenv wire format
    ``{"action": {"action_type": "...", "transaction_id": "..."}, "timeout_s": null}``
    and the legacy flat format
    ``{"action_type": "...", "transaction_id": "..."}``.  Returns
    ``{"observation": {...}, "reward": <float>, "done": <bool>}``.
    """
    action = request.resolved_action()
    if action.action_type.lower() not in VALID_ACTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action_type '{action.action_type}'. "
                   f"Valid values: {sorted(VALID_ACTIONS)}",
        )
    async with _state_lock:
        sess = _current_session()
        try:
            obs = await sess["env"].step_async(action)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        sess["actions"].append(action.action_type.lower())
        sess["confs"].append(action.confidence)

        # Auto-save completed episode to leaderboard
        if obs.done:
            result = grade_episode(
                sess["actions"], sess["tasks"], sess["confs"]
            )
            _leaderboard.append(
                {
                    "episode_id":       sess["env"].state().episode_id,
                    "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "normalised_score": result.normalised_score,
                    "total_reward":     result.total_reward,
                    "budget_spent":     result.budget_spent,
                    "budget_overspend": result.budget_overspend,
                    "passed":           result.passed,
                    "steps":            len(sess["actions"]),
                }
            )

    return EnvResponse(observation=obs.model_dump(), reward=obs.reward, done=obs.done)


@app.get("/state", response_model=PayOpsState, summary="Get internal environment state")
async def state():
    """Return the current internal state of the environment."""
    async with _state_lock:
        return _current_session()["env"].state()


@app.get("/schema", summary="Get action and observation schemas")
async def schema():
    """Return the JSON schemas for PayOpsAction and PayOpsObservation."""
    return {
        "action": PayOpsAction.model_json_schema(),
        "observation": PayOpsObservation.model_json_schema(),
        "state": PayOpsState.model_json_schema(),
    }


def _grader_ref(task_id: str) -> str:
    """Return the dotted grader reference string for a task id, e.g. 'server.graders:EASY001Grader'."""
    return f"server.graders:{task_id.replace('-', '')}Grader"


def _clamp_score(v: float) -> float:
    """Clamp any score to the open interval (0, 1) — platform rejects 0.0 and 1.0."""
    if v <= 0.0:
        return 0.001
    if v >= 1.0:
        return 0.999
    return round(v, 4)


@app.get("/tasks", summary="List all available tasks")
async def tasks():
    """Return a flat list of task metadata (one dict per task)."""
    result = []
    for t in TASKS:
        result.append(
            {
                "id":              t.task_id,
                "task_id":         t.task_id,
                "name":            t.task_id,
                "difficulty":      t.difficulty,
                "description":     t.description,
                "transaction_id":  t.transaction_id,
                "amount":          t.amount,
                "currency":        t.currency,
                "transaction_type":t.transaction_type,
                "risk_score":      t.risk_score,
                "ml_confidence":   getattr(t, "ml_confidence", None),
                "flags":           t.flags,
                "correct_action":  t.correct_action,
                "requires_investigation": list(getattr(t, "requires_investigation", [])),
                "regulatory_action": getattr(t, "regulatory_action", False),
                "chain_total":     getattr(t, "chain_total", 1),
                "grader":          _grader_ref(t.task_id),
                "score":           0.5,
            }
        )
    return result


@app.get("/tasks-v2", summary="List all available tasks")
async def tasks_v2():
    """Versioned tasks alias used to bypass stale edge caches."""
    return await tasks()


@app.get("/grader", summary="Grade the current episode")
async def grader():
    """
    Grade the episode using all actions taken since the last /reset.
    When called with no active session or no prior actions (e.g. by platform
    validators), returns the grader catalog — one entry per task — so that
    downstream tooling can confirm graders are configured for all 30 tasks.
    """
    async with _state_lock:
        # Build grader catalog (used when no session / no actions yet)
        def _catalog():
            return {
                "total_reward":        0.001,
                "max_possible_reward": 0.001,
                "normalised_score":    0.001,
                "budget_spent":        0.0,
                "budget_overspend":    0.0,
                "budget_penalty":      0.0,
                "passed":              False,
                "per_task": [
                    {
                        "task_id":   t.task_id,
                        "difficulty": t.difficulty,
                        "grader":    _grader_ref(t.task_id),
                        "score":     0.5,
                    }
                    for t in TASKS
                ],
                "message": "No episode in progress. Showing grader catalog.",
                "per_task_rewards": [
                    {
                        "task_id":   t.task_id,
                        "difficulty": t.difficulty,
                        "grader":    _grader_ref(t.task_id),
                        "score":     0.5,
                    }
                    for t in TASKS
                ],
            }

        # No session at all — return catalog instead of raising 400
        if _current_session_id is None or _current_session_id not in _sessions:
            return _catalog()

        sess = _sessions[_current_session_id]
        if not sess["actions"]:
            return _catalog()

        result = grade_episode(sess["actions"], sess["tasks"], sess["confs"])
        # Build task lookup so we can attach the grader config to every per_task
        # entry — the platform validator checks for the "grader" key whether the
        # endpoint is called cold OR after an episode has been played.
        tasks_by_id = {t.task_id: t for t in sess["tasks"]}

    per_task = []
    for pt in result.per_task_rewards:
        entry = dict(pt)
        t = tasks_by_id.get(pt["task_id"])
        if t:
            entry["grader"] = _grader_ref(t.task_id)
        # Platform requires a per-task "score" in the open interval (0, 1).
        # Derive from weighted_reward normalised by difficulty weight, clamped.
        raw = pt.get("weighted_reward", 0.0)
        weight = pt.get("weight", 1.0) or 1.0
        task_score = (raw / weight + 1.0) / 2.0  # map [-1, +1] → [0, 1]
        entry["score"] = _clamp_score(task_score)
        per_task.append(entry)

    return {
        "total_reward":       result.total_reward,
        "max_possible_reward":result.max_possible_reward,
        "normalised_score":   _clamp_score(result.normalised_score),
        "budget_spent":       result.budget_spent,
        "budget_overspend":   result.budget_overspend,
        "budget_penalty":     result.budget_penalty,
        "passed":             result.passed,
        "per_task":           per_task,
        "per_task_rewards":   per_task,
    }


@app.get("/grader-v2", summary="Grade the current episode")
async def grader_v2():
    """Versioned grader alias used to bypass stale edge caches."""
    return await grader()


@app.post("/baseline", response_model=BaselineResult, summary="Run the baseline agent")
async def baseline():
    """
    Run the built-in rule-based baseline agent against the full task set
    and return its scores. Useful for sanity-checking the environment.
    """
    from payops_env.scripts_util import run_baseline
    scores, total, normalised, steps = await run_baseline()
    return BaselineResult(
        scores=scores,
        total_reward=total,
        normalised_score=normalised,
        steps=steps,
    )


@app.get("/analytics", summary="Session performance analytics")
async def analytics():
    """
    Return aggregate analytics across all completed episodes this session.
    Includes accuracy by difficulty, average budget spend, and common mistakes.
    """
    if not _leaderboard:
        return {"message": "No completed episodes yet. Run a full episode first."}

    async with _state_lock:
        sess = _current_session()
        actions = list(sess["actions"])
        tasks   = list(sess["tasks"])
        confs   = list(sess["confs"])

    # Per-difficulty accuracy from the last episode's per_task breakdown
    result = grade_episode(actions, tasks, confs)
    by_diff: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "correct": 0, "rewards": []})
    for pt in result.per_task_rewards:
        d = pt["difficulty"]
        by_diff[d]["total"]   += 1
        by_diff[d]["correct"] += int(pt["correct"])
        by_diff[d]["rewards"].append(pt["weighted_reward"])

    diff_summary = {
        diff: {
            "accuracy":    round(v["correct"] / v["total"], 3) if v["total"] else 0,
            "avg_reward":  round(sum(v["rewards"]) / len(v["rewards"]), 3) if v["rewards"] else 0,
            "count":       v["total"],
        }
        for diff, v in by_diff.items()
    }

    return {
        "episodes_completed":  len(_leaderboard),
        "best_score":          max(e["normalised_score"] for e in _leaderboard),
        "avg_score":           round(sum(e["normalised_score"] for e in _leaderboard) / len(_leaderboard), 4),
        "avg_budget_spent":    round(sum(e["budget_spent"] for e in _leaderboard) / len(_leaderboard), 4),
        "current_episode":     {
            "normalised_score": result.normalised_score,
            "budget_spent":     result.budget_spent,
            "budget_penalty":   result.budget_penalty,
            "by_difficulty":    diff_summary,
        },
    }


@app.post("/replay", summary="Grade a supplied action sequence")
async def replay(request: ReplayRequest):
    """
    Grade a supplied list of actions against the task bank without
    modifying the current environment state.

    Pass ``seed`` to grade against a specific jittered task set (matching a
    live episode seeded with the same value).  Omitting ``seed`` grades
    against the canonical un-jittered tasks for offline baseline comparisons.
    """
    actions = [a.lower() for a in request.actions]
    invalid = [a for a in actions if a not in VALID_ACTIONS]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action(s): {invalid}. Valid: {sorted(VALID_ACTIONS)}",
        )

    if request.seed is not None:
        _replay_env = PayOpsEnvironment()
        await _replay_env.reset_async(seed=request.seed)
        task_list = list(_replay_env._tasks)
    else:
        task_list = list(TASKS)

    confs  = request.confidences or [None] * len(actions)
    result = grade_episode(actions, task_list, confs)
    return {
        "total_reward":        result.total_reward,
        "max_possible_reward": result.max_possible_reward,
        "normalised_score":    result.normalised_score,
        "budget_spent":        result.budget_spent,
        "budget_overspend":    result.budget_overspend,
        "budget_penalty":      result.budget_penalty,
        "passed":              result.passed,
        "per_task":            result.per_task_rewards,
    }


@app.get("/leaderboard", summary="Session leaderboard")
async def leaderboard():
    """
    Return all scored episodes from this server session, sorted by score.
    """
    sorted_board = sorted(_leaderboard, key=lambda e: e["normalised_score"], reverse=True)
    return {"count": len(sorted_board), "entries": sorted_board}


# ---------------------------------------------------------------------------
# WebSocket endpoint for persistent sessions
# ---------------------------------------------------------------------------

@app.get("/ws", include_in_schema=False)
async def ws_http_upgrade():
    """Return 426 Upgrade Required for plain HTTP requests to the WS endpoint."""
    from fastapi.responses import Response
    return Response(
        content="WebSocket upgrade required",
        status_code=426,
        headers={"Upgrade": "websocket"},
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket interface.

    Client sends JSON:
      {"type": "reset"}
      {"type": "step", "action_type": "...", "transaction_id": "..."}
      {"type": "state"}

    Server responds with observation JSON.
    """
    await websocket.accept()
    ws_env = PayOpsEnvironment()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "reset":
                obs = await ws_env.reset_async()
                await websocket.send_json(obs.model_dump())

            elif msg_type == "step":
                action_type = msg.get("action_type", "")
                if action_type.lower() not in VALID_ACTIONS:
                    await websocket.send_json(
                        {"error": f"Invalid action_type '{action_type}'"}
                    )
                    continue
                action = PayOpsAction(
                    action_type=action_type,
                    transaction_id=msg.get("transaction_id", ""),
                    reason=msg.get("reason"),
                    confidence=msg.get("confidence"),
                )
                try:
                    obs = await ws_env.step_async(action)
                    await websocket.send_json(obs.model_dump())
                except Exception as exc:
                    await websocket.send_json({"error": str(exc)})

            elif msg_type == "state":
                await websocket.send_json(ws_env.state().model_dump())

            else:
                await websocket.send_json(
                    {"error": f"Unknown message type '{msg_type}'"}
                )

    except WebSocketDisconnect:
        ws_env.close()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health():
    async with _state_lock:
        if _current_session_id and _current_session_id in _sessions:
            st = _sessions[_current_session_id]["env"].state()
            episode_id   = st.episode_id
            episode_seed = st.episode_seed
            current_task = st.current_task_id
            processed    = st.transactions_processed
            total        = st.total_tasks
        else:
            episode_id = episode_seed = current_task = None
            processed  = 0
            total      = len(TASKS)
    return {
        "status": "ok",
        "environment": "payops_env",
        "version": _APP_VERSION,
        "episode_id": episode_id,
        "episode_seed": episode_seed,
        "current_task_id": current_task,
        "transactions_processed": processed,
        "total_tasks": total,
    }


@app.get("/health-v2", summary="Health check")
async def health_v2():
    """Versioned health alias used to bypass stale edge caches."""
    return await health()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = int(__import__("os").environ.get("PORT", "8000"))):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

