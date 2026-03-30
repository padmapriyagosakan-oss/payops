#!/usr/bin/env bash
# run_tests.sh — PayOps v2 full test suite
# Usage: bash payops_env/run_tests.sh [BASE_URL]
# Requires the server to be running. Default: http://localhost:8000

BASE="${1:-http://localhost:8000}"
PASS=0
FAIL=0
SKIP=0

# ── helpers ──────────────────────────────────────────────────────────────────

check() {
  local name="$1"
  local got="$2"
  local want="$3"
  if echo "$got" | grep -qE "$want"; then
    printf "  \033[32m✓\033[0m  %s\n" "$name"
    PASS=$((PASS+1))
  else
    printf "  \033[31m✗\033[0m  %s\n" "$name"
    printf "     expected pattern : %s\n" "$want"
    printf "     got              : %s\n" "$(echo "$got" | head -c 200)"
    FAIL=$((FAIL+1))
  fi
}

check_absent() {
  local name="$1"
  local got="$2"
  local absent="$3"
  if echo "$got" | grep -qE "$absent"; then
    printf "  \033[31m✗\033[0m  %s  (unexpected: %s)\n" "$name" "$absent"
    FAIL=$((FAIL+1))
  else
    printf "  \033[32m✓\033[0m  %s\n" "$name"
    PASS=$((PASS+1))
  fi
}

step() {
  curl -s -X POST "$BASE/step" \
    -H "Content-Type: application/json" \
    -d "$1"
}

reset() {
  curl -s -X POST "$BASE/reset" > /dev/null
}

section() {
  echo ""
  echo "── $1 ──"
}

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║    PayOps v2 — Full Post-Main Test Suite             ║"
echo "║    Target: $BASE"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════
# GROUP A — Server / Infrastructure
# ═══════════════════════════════════════════════════════════
section "GROUP A — Server / Infrastructure"

# A-01  Health check
check "A-01  GET /health → status ok" \
  "$(curl -s $BASE/health)" \
  '"status":"ok"'

# A-02  Version is v2
check "A-02  GET /health → version 2.0.0" \
  "$(curl -s $BASE/health)" \
  '"version":"2.0.0"'

# A-03  Schema action model present
check "A-03  GET /schema → PayOpsAction schema" \
  "$(curl -s $BASE/schema)" \
  '"PayOpsAction"'

# A-04  Schema observation model present
check "A-04  GET /schema → PayOpsObservation schema" \
  "$(curl -s $BASE/schema)" \
  '"PayOpsObservation"'

# A-05  Schema state model present
check "A-05  GET /schema → PayOpsState schema" \
  "$(curl -s $BASE/schema)" \
  '"PayOpsState"'

# ═══════════════════════════════════════════════════════════
# GROUP B — Tasks endpoint
# ═══════════════════════════════════════════════════════════
section "GROUP B — Tasks endpoint"

TASKS_RESP="$(curl -s $BASE/tasks)"

# B-01  20 tasks
check "B-01  GET /tasks → count=20" \
  "$TASKS_RESP" '"count":20'

# B-02  All 4 difficulty tiers present
check "B-02  Tasks include 'easy' tier" "$TASKS_RESP" '"difficulty":"easy"'
check "B-03  Tasks include 'medium' tier" "$TASKS_RESP" '"difficulty":"medium"'
check "B-04  Tasks include 'hard' tier" "$TASKS_RESP" '"difficulty":"hard"'
check "B-05  Tasks include 'critical' tier" "$TASKS_RESP" '"difficulty":"critical"'

# B-06  Regulatory tasks present
check "B-06  Tasks include regulatory_action flag" \
  "$TASKS_RESP" '"regulatory_action":true'

# B-07  Multi-step chain tasks present
check "B-07  Tasks include chain_total > 1" \
  "$TASKS_RESP" '"chain_total":3'

# B-08  requires_investigation populated
check "B-08  Tasks include requires_investigation" \
  "$TASKS_RESP" '"requires_investigation":\['

# ═══════════════════════════════════════════════════════════
# GROUP C — Reset
# ═══════════════════════════════════════════════════════════
section "GROUP C — Reset"

RESET_RESP="$(curl -s -X POST $BASE/reset)"
check "C-01  POST /reset → returns EASY-001" "$RESET_RESP" '"task_id":"EASY-001"'
check "C-02  POST /reset → done=false" "$RESET_RESP" '"done":false'
check "C-03  POST /reset → budget_remaining=5.0" "$RESET_RESP" '"budget_remaining":5.0'
check "C-04  POST /reset → risk_score present" "$RESET_RESP" '"risk_score":'
check "C-05  POST /reset → ml_confidence present" "$RESET_RESP" '"ml_confidence":'
check "C-06  POST /reset → chain_total present" "$RESET_RESP" '"chain_total":'

# ═══════════════════════════════════════════════════════════
# GROUP D — Terminal Actions (correct decisions)
# ═══════════════════════════════════════════════════════════
section "GROUP D — Terminal Actions (correct decisions)"

reset

# D-01  approve correct → reward=1.0
check "D-01  EASY-001 approve → reward=1.0" \
  "$(step '{"action_type":"approve","transaction_id":"TXN-E001"}')" \
  '"reward":1.0'

# D-02  reject correct → reward=1.0
check "D-02  EASY-002 reject → reward=1.0" \
  "$(step '{"action_type":"reject","transaction_id":"TXN-E002"}')" \
  '"reward":1.0'

# D-03  approve correct (refund) → reward=1.0
check "D-03  EASY-003 approve → reward=1.0" \
  "$(step '{"action_type":"approve","transaction_id":"TXN-E003"}')" \
  '"reward":1.0'

# D-04  flag correct → reward=1.0
check "D-04  EASY-004 flag → reward=1.0" \
  "$(step '{"action_type":"flag","transaction_id":"TXN-E004"}')" \
  '"reward":1.0'

# D-05  escalate correct → reward=1.0
check "D-05  MED-001 escalate → reward=1.0" \
  "$(step '{"action_type":"escalate","transaction_id":"TXN-M001"}')" \
  '"reward":1.0'

# D-06  hold correct → reward=1.0
check "D-06  MED-002 hold → reward=1.0" \
  "$(step '{"action_type":"hold","transaction_id":"TXN-M002"}')" \
  '"reward":1.0'

# D-07  task advances after terminal
R=$(step '{"action_type":"flag","transaction_id":"TXN-M003"}')
check "D-07  After flag on MED-003, next task is MED-004" "$R" '"task_id":"MED-004"'

# ═══════════════════════════════════════════════════════════
# GROUP E — Terminal Actions (wrong decisions / partial credit)
# ═══════════════════════════════════════════════════════════
section "GROUP E — Wrong actions & partial credit"

reset

# E-01  approve when should reject → -1.0 (fraud approval)
step '{"action_type":"approve","transaction_id":"TXN-E001"}' > /dev/null  # advance past E001
check "E-01  Approve fraud (EASY-002) → reward=-1.0" \
  "$(step '{"action_type":"approve","transaction_id":"TXN-E002"}')" \
  '"reward":-1.0'

reset
step '{"action_type":"approve","transaction_id":"TXN-E001"}' > /dev/null
step '{"action_type":"reject","transaction_id":"TXN-E002"}' > /dev/null

# E-02  reject correct → approve → -0.5
check "E-02  Reject legit (EASY-003) → reward=-0.5" \
  "$(step '{"action_type":"reject","transaction_id":"TXN-E003"}')" \
  '"reward":-0.5'

reset
# E-03  partial credit — escalate instead of escalate on MED-001 is correct
#        flag instead of escalate earns partial
step '{"action_type":"approve","transaction_id":"TXN-E001"}' > /dev/null
step '{"action_type":"reject","transaction_id":"TXN-E002"}' > /dev/null
step '{"action_type":"approve","transaction_id":"TXN-E003"}' > /dev/null
step '{"action_type":"flag","transaction_id":"TXN-E004"}' > /dev/null
R=$(step '{"action_type":"flag","transaction_id":"TXN-M001"}')
check "E-03  Partial credit: flag on MED-001 (correct=escalate) → reward > 0" \
  "$R" '"reward":0\.[0-9]'

# ═══════════════════════════════════════════════════════════
# GROUP F — Investigation Sub-Actions
# ═══════════════════════════════════════════════════════════
section "GROUP F — Investigation sub-actions"

reset

# F-01  inspect → reward=0.15, stays on same task
R=$(step '{"action_type":"inspect","transaction_id":"TXN-E001"}')
check "F-01  inspect → reward=0.15" "$R" '"reward":0.15'
check "F-02  inspect → does NOT advance task (still EASY-001)" "$R" '"task_id":"EASY-001"'
check "F-03  inspect → inspection_notes populated" "$R" '"inspection_notes":"'
check "F-04  inspect → budget_remaining=4.9" "$R" '"budget_remaining":4.9'

# F-05  inspect again on same task → reward=0.0 (no double-dip)
check "F-05  second inspect → reward=0.0" \
  "$(step '{"action_type":"inspect","transaction_id":"TXN-E001"}')" \
  '"reward":0.0'

# Advance to EASY-002
step '{"action_type":"approve","transaction_id":"TXN-E001"}' > /dev/null

# F-06  request_docs → reward=0.15, docs_notes populated
R=$(step '{"action_type":"request_docs","transaction_id":"TXN-E002"}')
check "F-06  request_docs → reward=0.15" "$R" '"reward":0.15'
check "F-07  request_docs → docs_notes populated" "$R" '"docs_notes":"'
check "F-08  request_docs → budget_remaining=4.6 (cost=0.2)" "$R" '"budget_remaining":4.6'

# F-09  request_docs again → reward=0.0
check "F-09  second request_docs → reward=0.0" \
  "$(step '{"action_type":"request_docs","transaction_id":"TXN-E002"}')" \
  '"reward":0.0'

step '{"action_type":"reject","transaction_id":"TXN-E002"}' > /dev/null  # advance

# F-10  verify_kyc → reward=0.15, kyc_notes populated
R=$(step '{"action_type":"verify_kyc","transaction_id":"TXN-E003"}')
check "F-10  verify_kyc → reward=0.15" "$R" '"reward":0.15'
check "F-11  verify_kyc → kyc_notes populated" "$R" '"kyc_notes":"'
check "F-12  verify_kyc → budget cost=0.2 deducted" "$R" '"budget_remaining":4.[0-9]'

step '{"action_type":"approve","transaction_id":"TXN-E003"}' > /dev/null

# F-13  contact_sender → reward=0.15, contact_notes populated
R=$(step '{"action_type":"contact_sender","transaction_id":"TXN-E004"}')
check "F-13  contact_sender → reward=0.15" "$R" '"reward":0.15'
check "F-14  contact_sender → contact_notes populated" "$R" '"contact_notes":"'
check "F-15  contact_sender → budget cost=0.3 deducted" "$R" '"budget_remaining":3\.[0-9]'

step '{"action_type":"flag","transaction_id":"TXN-E004"}' > /dev/null

# F-16  file_sar → reward=0.15, docs_notes mentions SAR
R=$(step '{"action_type":"file_sar","transaction_id":"TXN-M001"}')
check "F-16  file_sar → reward=0.15" "$R" '"reward":0.15'
check "F-17  file_sar → docs_notes mentions SAR" "$R" '"docs_notes":"SAR'
check "F-18  file_sar → budget cost=0.05 deducted" "$R" '"budget_remaining":3\.[0-9]'

# ═══════════════════════════════════════════════════════════
# GROUP G — /state endpoint
# ═══════════════════════════════════════════════════════════
section "GROUP G — State endpoint"

STATE="$(curl -s $BASE/state)"
check "G-01  GET /state → episode_id set" "$STATE" '"episode_id":"'
check "G-02  GET /state → step_count > 0" "$STATE" '"step_count":[1-9]'
check "G-03  GET /state → budget_spent > 0" "$STATE" '"budget_spent":[0-9]'
check "G-04  GET /state → investigation_actions_used list" "$STATE" '"investigation_actions_used":\['
check "G-05  GET /state → recent_decisions list" "$STATE" '"recent_decisions":\['
check "G-06  GET /state → correct_decisions >= 0" "$STATE" '"correct_decisions":[0-9]'

# ═══════════════════════════════════════════════════════════
# GROUP H — /grader endpoint
# ═══════════════════════════════════════════════════════════
section "GROUP H — Grader endpoint"

GRADER="$(curl -s $BASE/grader)"
check "H-01  GET /grader → normalised_score present" "$GRADER" '"normalised_score":'
check "H-02  GET /grader → total_reward present" "$GRADER" '"total_reward":'
check "H-03  GET /grader → budget_spent present" "$GRADER" '"budget_spent":'
check "H-04  GET /grader → budget_penalty present" "$GRADER" '"budget_penalty":'
check "H-05  GET /grader → per_task array" "$GRADER" '"per_task":\['
check "H-06  GET /grader → reward_breakdown in per_task" "$GRADER" '"reward_breakdown":'
check "H-07  GET /grader → passed field present" "$GRADER" '"passed":'

# ═══════════════════════════════════════════════════════════
# GROUP I — /replay endpoint
# ═══════════════════════════════════════════════════════════
section "GROUP I — /replay endpoint (offline eval)"

# I-01  Replay with all correct actions → score=1.0
REPLAY_PERFECT='{"actions":["approve","reject","approve","flag",
  "escalate","hold","flag","flag","hold","escalate",
  "escalate","reject","reject","approve","escalate","flag",
  "approve","reject","escalate","reject"]}'
R=$(curl -s -X POST $BASE/replay \
    -H "Content-Type: application/json" \
    -d "$REPLAY_PERFECT")
check "I-01  Replay 20 correct actions → score=1.0" "$R" '"normalised_score":1.0'
check "I-02  Replay → passed=true" "$R" '"passed":true'
check "I-03  Replay → budget_spent=0.0 (no inv actions)" "$R" '"budget_spent":0.0'

# I-04  Replay with investigation actions included
REPLAY_WITH_INV='{"actions":["inspect","approve","reject","approve","flag",
  "escalate","hold","flag","flag","hold","escalate",
  "escalate","reject","reject","approve","escalate","flag",
  "approve","reject","escalate","reject"]}'
R=$(curl -s -X POST $BASE/replay \
    -H "Content-Type: application/json" \
    -d "$REPLAY_WITH_INV")
check "I-04  Replay with inspect → budget_spent=0.1" "$R" '"budget_spent":0.1'

# I-05  Replay with invalid action → 422
check "I-05  Replay invalid action → 422 error" \
  "$(curl -s -X POST $BASE/replay \
      -H 'Content-Type: application/json' \
      -d '{"actions":["delete","approve"]}')" \
  "Invalid action"

# I-06  Replay result has per_task breakdown
check "I-06  Replay → per_task array present" "$R" '"per_task":\['
check "I-07  Replay → reward_breakdown in each task" "$R" '"reward_breakdown":'

# I-08  Replay with confidences
REPLAY_CONF='{"actions":["approve","reject"],"confidences":[0.95,0.90]}'
check "I-08  Replay with confidences → ok" \
  "$(curl -s -X POST $BASE/replay \
      -H 'Content-Type: application/json' \
      -d "$REPLAY_CONF")" \
  '"normalised_score":'

# ═══════════════════════════════════════════════════════════
# GROUP J — /baseline endpoint
# ═══════════════════════════════════════════════════════════
section "GROUP J — Baseline agent"

BASELINE="$(curl -s -X POST $BASE/baseline)"
check "J-01  POST /baseline → normalised_score present" "$BASELINE" '"normalised_score":'
check "J-02  POST /baseline → total_reward present" "$BASELINE" '"total_reward":'
check "J-03  POST /baseline → steps > 0" "$BASELINE" '"steps":[1-9]'
check "J-04  POST /baseline → per_task scores present" "$BASELINE" '"scores":\['
check "J-05  POST /baseline → score >= 0.5 (passes)" "$BASELINE" '"normalised_score":0\.[5-9]'

# ═══════════════════════════════════════════════════════════
# GROUP K — /analytics endpoint
# ═══════════════════════════════════════════════════════════
section "GROUP K — Analytics endpoint"

# Run a full episode first so analytics has data
reset
BASE=$BASE python3 - <<'PYEOF'
import os, urllib.request, json, sys
BASE = os.environ.get("BASE", "http://localhost:8000")
def post(path, body=None):
    req = urllib.request.Request(f"{BASE}{path}",
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r: return json.loads(r.read())
def get(path):
    with urllib.request.urlopen(f"{BASE}{path}") as r: return json.loads(r.read())

post("/reset")
actions = [
  ("approve","TXN-E001"),("reject","TXN-E002"),("approve","TXN-E003"),("flag","TXN-E004"),
  ("escalate","TXN-M001"),("hold","TXN-M002"),("flag","TXN-M003"),("flag","TXN-M004"),
  ("hold","TXN-M005"),("escalate","TXN-M006"),
  ("escalate","TXN-H001"),("reject","TXN-H002"),("reject","TXN-H003"),("approve","TXN-H004"),
  ("escalate","TXN-H005"),("flag","TXN-H006"),
  ("approve","TXN-C001"),("reject","TXN-C002"),("escalate","TXN-C003"),("reject","TXN-C004"),
]
for action, txn in actions:
    post("/step", {"action_type": action, "transaction_id": txn})
PYEOF

ANA="$(curl -s $BASE/analytics)"
check "K-01  GET /analytics → episodes_completed >= 1" "$ANA" '"episodes_completed":[1-9]'
check "K-02  GET /analytics → best_score present" "$ANA" '"best_score":'
check "K-03  GET /analytics → avg_score present" "$ANA" '"avg_score":'
check "K-04  GET /analytics → avg_budget_spent present" "$ANA" '"avg_budget_spent":'
check "K-05  GET /analytics → current_episode present" "$ANA" '"current_episode":'
check "K-06  GET /analytics → by_difficulty present" "$ANA" '"by_difficulty":'
check "K-07  GET /analytics → easy accuracy" "$ANA" '"easy":'
check "K-08  GET /analytics → critical accuracy" "$ANA" '"critical":'

# ═══════════════════════════════════════════════════════════
# GROUP L — /leaderboard endpoint
# ═══════════════════════════════════════════════════════════
section "GROUP L — Leaderboard endpoint"

LB="$(curl -s $BASE/leaderboard)"
check "L-01  GET /leaderboard → count >= 1 (episode recorded)" "$LB" '"count":[1-9]'
check "L-02  GET /leaderboard → entries array present" "$LB" '"entries":\['
check "L-03  GET /leaderboard → episode_id in entry" "$LB" '"episode_id":"'
check "L-04  GET /leaderboard → normalised_score in entry" "$LB" '"normalised_score":'
check "L-05  GET /leaderboard → timestamp in entry" "$LB" '"timestamp":"'
check "L-06  GET /leaderboard → budget_spent in entry" "$LB" '"budget_spent":'

# ═══════════════════════════════════════════════════════════
# GROUP M — Perfect episode score
# ═══════════════════════════════════════════════════════════
section "GROUP M — Perfect episode (all 20 correct, no investigation cost)"

reset
BASE=$BASE python3 - <<'PYEOF'
import os, urllib.request, json
BASE = os.environ.get("BASE", "http://localhost:8000")
def post(path, body=None):
    req = urllib.request.Request(f"{BASE}{path}",
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r: return json.loads(r.read())

post("/reset")
actions = [
  ("approve","TXN-E001"),("reject","TXN-E002"),("approve","TXN-E003"),("flag","TXN-E004"),
  ("escalate","TXN-M001"),("hold","TXN-M002"),("flag","TXN-M003"),("flag","TXN-M004"),
  ("hold","TXN-M005"),("escalate","TXN-M006"),
  ("escalate","TXN-H001"),("reject","TXN-H002"),("reject","TXN-H003"),("approve","TXN-H004"),
  ("escalate","TXN-H005"),("flag","TXN-H006"),
  ("approve","TXN-C001"),("reject","TXN-C002"),("escalate","TXN-C003"),("reject","TXN-C004"),
]
for action, txn in actions:
    post("/step", {"action_type": action, "transaction_id": txn})
PYEOF

GRADER="$(curl -s $BASE/grader)"
check "M-01  Perfect episode → normalised_score=1.0" "$GRADER" '"normalised_score":1.0'
check "M-02  Perfect episode → passed=true" "$GRADER" '"passed":true'
check "M-03  Perfect episode → budget_penalty=0.0" "$GRADER" '"budget_penalty":0.0'

# ═══════════════════════════════════════════════════════════
# GROUP N — Difficulty weighting
# ═══════════════════════════════════════════════════════════
section "GROUP N — Difficulty weighting in grader"

# A critical task correct should contribute more weight than easy
REPLAY_CRIT='{"actions":["approve","reject","approve","flag",
  "escalate","hold","flag","flag","hold","escalate",
  "escalate","reject","reject","approve","escalate","flag",
  "approve","reject","escalate","reject"]}'

FULL=$(curl -s -X POST $BASE/replay \
    -H "Content-Type: application/json" \
    -d "$REPLAY_CRIT")
check "N-01  Full correct replay → max_possible_reward includes weights" \
  "$FULL" '"max_possible_reward":2[0-9]\.'
check "N-02  Full correct → total_reward equals max_possible" \
  "$(echo "$FULL" | python3 -c "
import sys,json,re
d=json.load(sys.stdin)
print('EQUAL' if abs(d['total_reward']-d['max_possible_reward'])<0.01 else 'DIFF')
")" "EQUAL"

# ═══════════════════════════════════════════════════════════
# GROUP O — Budget mechanics
# ═══════════════════════════════════════════════════════════
section "GROUP O — Budget mechanics"

reset

# Burn through budget with contact_sender (cost=0.3) × 5 = 1.5
# then request_docs × 5 = 1.0, then verify_kyc × 5 = 1.0  (total = 3.5 so far ≤5)
# Total exceeding: won't exceed in these few calls, test budget tracking
step '{"action_type":"contact_sender","transaction_id":"TXN-E001"}' > /dev/null  # -0.30
step '{"action_type":"contact_sender","transaction_id":"TXN-E001"}' > /dev/null  # dup, but cost still deducted
step '{"action_type":"request_docs","transaction_id":"TXN-E001"}' > /dev/null    # -0.20
step '{"action_type":"verify_kyc","transaction_id":"TXN-E001"}' > /dev/null      # -0.20
step '{"action_type":"file_sar","transaction_id":"TXN-E001"}' > /dev/null        # -0.05
R=$(step '{"action_type":"approve","transaction_id":"TXN-E001"}')
# budget_remaining = 5.0 - 0.30 - 0.30 - 0.20 - 0.20 - 0.05 = 3.95
check "O-01  Budget correctly deducted after multiple inv actions" \
  "$R" '"budget_remaining":3\.[0-9]'

# grader shows budget_spent
GRADER="$(curl -s $BASE/grader)"
check "O-02  Grader → budget_spent > 0" "$GRADER" '"budget_spent":[1-9]'

# replay that overshoots budget → budget_penalty > 0
HEAVY='{"actions":["contact_sender","contact_sender","contact_sender","contact_sender","contact_sender","contact_sender","contact_sender","contact_sender","contact_sender","contact_sender","contact_sender","contact_sender","contact_sender","contact_sender","contact_sender","contact_sender","contact_sender","contact_sender","approve","reject","approve","flag","escalate","hold","flag","flag","hold","escalate","escalate","reject","reject","approve","escalate","flag","approve","reject","escalate","reject"]}'
R=$(curl -s -X POST $BASE/replay \
    -H "Content-Type: application/json" \
    -d "$HEAVY")
check "O-03  Over-budget replay → budget_penalty > 0" "$R" '"budget_penalty":0\.[0-9]*[1-9]'
check "O-04  Over-budget replay → budget_overspend > 0" "$R" '"budget_overspend":0\.[1-9]'

# ═══════════════════════════════════════════════════════════
# GROUP P — Edge cases & errors
# ═══════════════════════════════════════════════════════════
section "GROUP P — Edge cases & error handling"

# P-01  Step before reset → 400
check "P-01  Step on fresh env (reset then done) → handled" \
  "$(curl -s -X POST $BASE/reset > /dev/null && curl -s -X POST $BASE/step \
      -H 'Content-Type: application/json' \
      -d '{"action_type":"approve","transaction_id":"TXN-NONE"}')" \
  '"reward"'

# P-02  Unknown action_type → 422
check "P-02  Unknown action_type → 422" \
  "$(curl -s -X POST $BASE/step \
      -H 'Content-Type: application/json' \
      -d '{"action_type":"nuke","transaction_id":"TXN-E001"}')" \
  "Invalid action_type"

# P-03  Missing action_type field → 422
check "P-03  Missing action_type field → error" \
  "$(curl -s -X POST $BASE/step \
      -H 'Content-Type: application/json' \
      -d '{"transaction_id":"TXN-E001"}')" \
  '"detail"'

# P-04  Empty replay actions list
check "P-04  Replay empty actions → returns score" \
  "$(curl -s -X POST $BASE/replay \
      -H 'Content-Type: application/json' \
      -d '{"actions":[]}')" \
  '"normalised_score"'

# P-05  Replay with confidences shorter than actions → still works
check "P-05  Replay with partial confidences → score present" \
  "$(curl -s -X POST $BASE/replay \
      -H 'Content-Type: application/json' \
      -d '{"actions":["approve","reject"],"confidences":[0.9]}')" \
  '"normalised_score"'

# P-06  Grader before any reset → 400 or returns score
check "P-06  Grader before actions → handled gracefully" \
  "$(curl -s $BASE/grader)" \
  '"normalised_score"|"error"'

# P-07  Analytics before any completed episode → handled
reset
check "P-07  Analytics after fresh reset → message or data" \
  "$(curl -s $BASE/analytics)" \
  '"message"|"episodes_completed"'

# ═══════════════════════════════════════════════════════════
# GROUP Q — WebSocket
# ═══════════════════════════════════════════════════════════
section "GROUP Q — WebSocket (requires wscat or python)"

if command -v python3 &>/dev/null; then
  WS_RESULT=$(BASE=$BASE python3 - <<'PYEOF'
import os, urllib.request, json
try:
    BASE = os.environ.get("BASE", "http://localhost:8000")
    # ws upgrade check via HTTP (will get 426 Upgrade Required — proves endpoint exists)
    req = urllib.request.Request(f"{BASE}/ws")
    try:
        urllib.request.urlopen(req)
    except Exception as e:
        msg = str(e)
        if "426" in msg or "101" in msg or "Switching" in msg or "upgrade" in msg.lower():
            print("WS_OK")
        else:
            print(f"WS_UNKNOWN:{msg[:60]}")
except Exception as e:
    print(f"WS_ERR:{e}")
PYEOF
)
  check "Q-01  WS /ws endpoint exists (426 upgrade = correct)" \
    "$WS_RESULT" "WS_OK"
else
  printf "  \033[33m-\033[0m  Q-01  WebSocket check skipped (no python3)\n"
  SKIP=$((SKIP+1))
fi

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════╗"
TOTAL=$((PASS+FAIL))
if [ "$FAIL" -eq 0 ]; then
  printf "║  ✓  All %d tests passed" "$TOTAL"
else
  printf "║  Results: %d/%d passed, %d failed" "$PASS" "$TOTAL" "$FAIL"
fi
[ "$SKIP" -gt 0 ] && printf "  (%d skipped)" "$SKIP"
echo ""
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Exit with failure code if any tests failed
[ "$FAIL" -eq 0 ]
