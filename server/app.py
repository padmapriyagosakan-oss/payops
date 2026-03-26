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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from payops_env.environment import PayOpsEnvironment, VALID_ACTIONS
from payops_env.grader import grade_episode
from payops_env.models import PayOpsAction, PayOpsObservation, PayOpsState
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
    version="2.0.0",
)

# Single shared environment instance (suitable for single-user / HF Space use)
_env = PayOpsEnvironment()
_episode_actions: List[str] = []          # ALL actions including investigation
_episode_confs: List[Optional[float]] = []
_episode_tasks: List[Any] = []

# Leaderboard persists for the process lifetime
_leaderboard: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------

class StepRequest(BaseModel):
    action_type: str
    transaction_id: str
    reason: Optional[str] = None
    confidence: Optional[float] = None


class BaselineResult(BaseModel):
    scores: List[Dict[str, Any]]
    total_reward: float
    normalised_score: float
    steps: int


class ReplayRequest(BaseModel):
    actions: List[str]
    confidences: Optional[List[Optional[float]]] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=PayOpsObservation, summary="Reset the environment")
async def reset():
    """Reset the environment and return the first transaction observation."""
    global _episode_actions, _episode_tasks, _episode_confs
    _episode_actions = []
    _episode_confs   = []
    _episode_tasks   = list(TASKS)
    obs = await _env.reset_async()
    return obs


@app.post("/step", response_model=PayOpsObservation, summary="Execute an action")
async def step(request: StepRequest):
    """
    Submit an action for the current transaction.

    Valid action_type values:
      Terminal:      approve | reject | flag | escalate | hold
      Investigation: inspect | request_docs | verify_kyc | contact_sender | file_sar
    """
    if request.action_type.lower() not in VALID_ACTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action_type '{request.action_type}'. "
                   f"Valid values: {sorted(VALID_ACTIONS)}",
        )
    action = PayOpsAction(
        action_type=request.action_type,
        transaction_id=request.transaction_id,
        reason=request.reason,
        confidence=request.confidence,
    )
    try:
        obs = await _env.step_async(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _episode_actions.append(request.action_type.lower())
    _episode_confs.append(request.confidence)

    # Auto-save completed episode to leaderboard
    if obs.done:
        result = grade_episode(
            _episode_actions, _episode_tasks, _episode_confs
        )
        _leaderboard.append(
            {
                "episode_id":       _env.state().episode_id,
                "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "normalised_score": result.normalised_score,
                "total_reward":     result.total_reward,
                "budget_spent":     result.budget_spent,
                "budget_overspend": result.budget_overspend,
                "passed":           result.passed,
                "steps":            len(_episode_actions),
            }
        )

    return obs


@app.get("/state", response_model=PayOpsState, summary="Get internal environment state")
async def state():
    """Return the current internal state of the environment."""
    return _env.state()


@app.get("/schema", summary="Get action and observation schemas")
async def schema():
    """Return the JSON schemas for PayOpsAction and PayOpsObservation."""
    return {
        "action": PayOpsAction.model_json_schema(),
        "observation": PayOpsObservation.model_json_schema(),
        "state": PayOpsState.model_json_schema(),
    }


@app.get("/tasks", summary="List all available tasks")
async def tasks():
    """Return metadata for all tasks grouped by difficulty."""
    result = []
    for t in TASKS:
        result.append(
            {
                "task_id":         t.task_id,
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
            }
        )
    return {"count": len(result), "tasks": result}


@app.get("/grader", summary="Grade the current episode")
async def grader():
    """
    Grade the episode using all actions taken since the last /reset.
    """
    if not _episode_actions:
        return JSONResponse(
            status_code=400,
            content={"error": "No actions recorded. Run /reset then /step first."},
        )
    result = grade_episode(_episode_actions, _episode_tasks, _episode_confs)
    return {
        "total_reward":       result.total_reward,
        "max_possible_reward":result.max_possible_reward,
        "normalised_score":   result.normalised_score,
        "budget_spent":       result.budget_spent,
        "budget_overspend":   result.budget_overspend,
        "budget_penalty":     result.budget_penalty,
        "passed":             result.passed,
        "per_task":           result.per_task_rewards,
    }


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

    # Per-difficulty accuracy from the last episode's per_task breakdown
    result = grade_episode(_episode_actions, _episode_tasks, _episode_confs)
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
    Grade a supplied list of actions against the full task bank without
    modifying the current environment state.

    Useful for offline evaluation and leaderboard submissions.
    """
    actions = [a.lower() for a in request.actions]
    invalid = [a for a in actions if a not in VALID_ACTIONS]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action(s): {invalid}. Valid: {sorted(VALID_ACTIONS)}",
        )

    confs  = request.confidences or [None] * len(actions)
    result = grade_episode(actions, list(TASKS), confs)
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
    return {"status": "ok", "environment": "payops_env", "version": "2.0.0"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)

