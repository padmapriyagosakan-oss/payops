# PayOps Environment — Test Cases & Testing Guide

This document covers every testable behaviour of the PayOps OpenEnv, organised
by endpoint and scenario. Each test shows the exact command to run, the
expected response, and what a failure looks like.

---

## Prerequisites

```bash
# 1. Start the server (run from /Users/padmapriya)
PYTHONPATH=/Users/padmapriya uvicorn payops_env.server.app:app --host 0.0.0.0 --port 8000

# 2. Confirm it is up (should return {"status":"ok",...})
curl -s http://localhost:8000/health
```

All `curl` commands below assume the server is running on `localhost:8000`.

---

## T-01  Health Check

**Goal:** Confirm the server is alive and returns version metadata.

```bash
curl -s http://localhost:8000/health
```

**Expected output**
```json
{"status": "ok", "environment": "payops_env", "version": "2.0.0"}
```

**Failure indicator:** Connection refused, or any field missing / wrong value.

---

## T-02  Schema Endpoint

**Goal:** Verify that action, observation, and state JSON schemas are served correctly.

```bash
curl -s http://localhost:8000/schema | python3 -m json.tool
```

**Expected output (condensed)**
```json
{
  "action":      { "title": "PayOpsAction", "type": "object", ... },
  "observation": { "title": "PayOpsObservation", "type": "object", ... },
  "state":       { "title": "PayOpsState", "type": "object", ... }
}
```

**Checks to verify manually:**
- `action.properties` includes `action_type`, `transaction_id`, `reason`, `confidence`
- `observation.properties` includes `risk_score`, `flags`, `kyc_status`, `velocity_1h`
- HTTP status code is `200`

---

## T-03  Tasks Endpoint

**Goal:** Confirm all 20 tasks are returned with the correct difficulty distribution.

```bash
curl -s http://localhost:8000/tasks | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Total tasks:', d['count'])
from collections import Counter
c = Counter(t['difficulty'] for t in d['tasks'])
print('By difficulty:', dict(c))
print()
for t in d['tasks']:
    print(f\"  {t['task_id']:12} [{t['difficulty']:8}] correct={t['correct_action']}\")
"
```

**Expected output**
```
Total tasks: 20
By difficulty: {'easy': 4, 'medium': 6, 'hard': 6, 'critical': 4}

  EASY-001     [easy    ] correct=approve
  EASY-002     [easy    ] correct=reject
  EASY-003     [easy    ] correct=approve
  EASY-004     [easy    ] correct=flag
  MED-001      [medium  ] correct=escalate
  MED-002      [medium  ] correct=hold
  MED-003      [medium  ] correct=flag
  MED-004      [medium  ] correct=flag
  MED-005      [medium  ] correct=hold
  MED-006      [medium  ] correct=escalate
  HARD-001     [hard    ] correct=escalate
  HARD-002     [hard    ] correct=reject
  HARD-003     [hard    ] correct=reject
  HARD-004     [hard    ] correct=approve
  HARD-005     [hard    ] correct=escalate
  HARD-006     [hard    ] correct=flag
  CRIT-001     [critical] correct=approve
  CRIT-002     [critical] correct=reject
  CRIT-003     [critical] correct=escalate
  CRIT-004     [critical] correct=reject
```

> Note: correct_action values for jitter-variant tasks (EASY-004, MED-001/003/004/006,
> HARD-001/006, CRIT-001/003/004) may differ per episode seed — the above shows default values.

**Failure indicator:** count != 20, missing difficulty tier, wrong correct_action.

---

## T-04  Reset

**Goal:** Reset the environment and confirm the first task is EASY-001.

```bash
curl -s -X POST http://localhost:8000/reset | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('task_id          :', d['task_id'])
print('transaction_id   :', d['transaction_id'])
print('difficulty        :', d['task_difficulty'])
print('status            :', d['status'])
print('done              :', d['done'])
print('reward            :', d['reward'])
print('cumulative_reward :', d['cumulative_reward'])
print('risk_score        :', d['risk_score'])
"
```

**Expected output**
```
task_id          : EASY-001
transaction_id   : TXN-E001
difficulty        : easy
status            : pending
done              : false
reward            : 0.0
cumulative_reward : 0.0
risk_score        : 0.05
```

**Failure indicator:** `done=true`, `reward != 0`, wrong `task_id`.

---

## T-05  Correct Action — Full Credit (+1.0)

**Goal:** Submit the correct action for EASY-001 (`approve`) and receive reward +1.0.

```bash
curl -s -X POST http://localhost:8000/reset > /dev/null   # fresh start

curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"approve","transaction_id":"TXN-E001"}' \
| python3 -c "
import sys, json
d = json.load(sys.stdin)
print('reward           :', d['reward'])
print('correct info     :', d['info'].get('correct_action'))
print('action taken     :', d['info'].get('action_taken'))
"
```

**Expected output**
```
reward           : 1.0
correct info     : approve
action taken     : approve
```

---

## T-06  Wrong Action — Penalty (approve on fraud = -1.0)

**Goal:** Skip to EASY-002 (textbook fraud) and approve it. Expect -1.0 penalty.

```bash
curl -s -X POST http://localhost:8000/reset > /dev/null
# Step past EASY-001
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"approve","transaction_id":"TXN-E001"}' > /dev/null

# Now on EASY-002 (correct=reject). Try approving it.
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"approve","transaction_id":"TXN-E002"}' \
| python3 -c "
import sys, json
d = json.load(sys.stdin)
print('reward       :', d['reward'])
print('correct was  :', d['info'].get('correct_action'))
"
```

**Expected output**
```
reward       : -1.0
correct was  : reject
```

---

## T-07  Partial Credit Action

**Goal:** On MED-001 (correct=`escalate`), submit `flag` — should earn +0.5 partial credit.

```bash
curl -s -X POST http://localhost:8000/reset > /dev/null
# Step through EASY tasks 1-4 with any actions
for ACTION in approve reject approve reject; do
  curl -s -X POST http://localhost:8000/step \
    -H "Content-Type: application/json" \
    -d "{\"action_type\":\"$ACTION\",\"transaction_id\":\"dummy\"}" > /dev/null
done

# Now on MED-001 (correct=escalate). Submit flag.
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"flag","transaction_id":"TXN-M001"}' \
| python3 -c "
import sys, json
d = json.load(sys.stdin)
print('reward       :', d['reward'])
print('correct was  :', d['info'].get('correct_action'))
print('partial?     :', 0 < d['reward'] < 1.0)
"
```

**Expected output**
```
reward       : 0.5
correct was  : escalate
partial?     : True
```

---

## T-08  Inspect Action — Information Reveal

**Goal:** Use `inspect` on EASY-001 to receive investigation notes and a small reward (+0.15). The episode should NOT advance (still on same transaction).

```bash
curl -s -X POST http://localhost:8000/reset > /dev/null

curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"inspect","transaction_id":"TXN-E001"}' \
| python3 -c "
import sys, json
d = json.load(sys.stdin)
print('reward            :', d['reward'])
print('status            :', d['status'])
print('task_id           :', d['task_id'])    # should still be EASY-001
print('inspection_notes  :', d['inspection_notes'])
"
```

**Expected output**
```
reward            : 0.15
status            : inspected
task_id           : EASY-001
inspection_notes  : Sender account opened 3 years ago. Consistent transaction history. KYC fully verified.
```

---

## T-09  Double Inspect — No Double-Dipping

**Goal:** Inspect the same transaction twice. Second inspect should return reward 0.0 (already inspected).

```bash
curl -s -X POST http://localhost:8000/reset > /dev/null

# First inspect — reward 0.15
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"inspect","transaction_id":"TXN-E001"}' \
| python3 -c "import sys,json; d=json.load(sys.stdin); print('First inspect reward:', d['reward'])"

# Second inspect — reward 0.0
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"inspect","transaction_id":"TXN-E001"}' \
| python3 -c "import sys,json; d=json.load(sys.stdin); print('Second inspect reward:', d['reward'])"
```

**Expected output**
```
First inspect reward: 0.15
Second inspect reward: 0.0
```

---

## T-10  Invalid Action Type

**Goal:** Send an unsupported action type and receive a 422 validation error.

```bash
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"delete","transaction_id":"TXN-E001"}' \
| python3 -c "import sys,json; d=json.load(sys.stdin); print('status_code:', d.get('detail','')[:60])"
```

**Expected output**
```
status_code: Invalid action_type 'delete'. Valid values: ['approve', 'escal
```

HTTP status code should be `422`.

---

## T-11  Step Without Reset

**Goal:** Call `/step` without calling `/reset` first. Should return a `400` error.

```bash
# Kill and restart server to guarantee clean state
# Then immediately step without reset:
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"approve","transaction_id":"TXN-E001"}'
```

**Expected output**
```
400
```

---

## T-12  State Endpoint Tracking

**Goal:** Confirm `/state` reflects the episode progress correctly.

```bash
curl -s -X POST http://localhost:8000/reset > /dev/null

curl -s http://localhost:8000/state | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('step_count            :', d['step_count'])
print('transactions_processed:', d['transactions_processed'])
print('total_tasks           :', d['total_tasks'])
print('done                  :', d['done'])
"

# Take one step
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"approve","transaction_id":"TXN-E001"}' > /dev/null

curl -s http://localhost:8000/state | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('step_count            :', d['step_count'])
print('transactions_processed:', d['transactions_processed'])
print('last_action           :', d['last_action'])
print('cumulative_reward     :', d['cumulative_reward'])
"
```

**Expected output (before step)**
```
step_count            : 0
transactions_processed: 0
total_tasks           : 12
done                  : false
```

**Expected output (after step)**
```
step_count            : 1
transactions_processed: 1
last_action           : approve
cumulative_reward     : 1.0
```

---

## T-13  Complete Episode — Done Flag

**Goal:** Step through all 12 tasks and confirm `done=true` on the last step.

```bash
curl -s -X POST http://localhost:8000/reset > /dev/null

python3 - <<'EOF'
import httpx, asyncio

BASE = "http://localhost:8000"
ACTIONS = [
    "approve","reject","approve","flag",   # easy
    "escalate","hold","flag","flag",       # medium
    "escalate","reject","reject","approve" # hard (perfect sequence)
]

client = httpx.Client()
txn_ids = [t["transaction_id"] for t in client.get(f"{BASE}/tasks").json()["tasks"]]

for i, (action, txn) in enumerate(zip(ACTIONS, txn_ids)):
    resp = client.post(f"{BASE}/step", json={"action_type": action, "transaction_id": txn}).json()
    print(f"Step {i+1:2d}  {txn:12}  action={action:10}  reward={resp['reward']:+.2f}  done={resp['done']}")

client.close()
EOF
```

**Expected output (last line)**
```
Step 12  TXN-H004     action=approve      reward=+1.00  done=True
```

All other steps should show `done=False`.

---

## T-14  Grader Endpoint

**Goal:** Grade and score the episode immediately after completing all steps.

```bash
# Run the perfect sequence first (T-13 above), then:
curl -s http://localhost:8000/grader | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('total_reward      :', d['total_reward'])
print('max_possible      :', d['max_possible_reward'])
print('normalised_score  :', d['normalised_score'])
print('passed            :', d['passed'])
print()
for t in d['per_task']:
    mark = '✓' if t['correct'] else '✗'
    print(f\"  {mark} {t['task_id']:12} action={t['action_taken']:10} correct={t['correct_action']:10} reward={t['reward']:+.2f}\")
"
```

**Expected output (perfect run)**
```
total_reward      : 12.0
max_possible      : 12.0
normalised_score  : 1.0
passed            : True

  ✓ EASY-001     action=approve     correct=approve     reward=+1.00
  ✓ EASY-002     action=reject      correct=reject      reward=+1.00
  ...
  ✓ HARD-004     action=approve     correct=approve     reward=+1.00
```

---

## T-15  Grader Without Episode

**Goal:** Call `/grader` before any steps — should return a 400 error.

```bash
curl -s -X POST http://localhost:8000/reset > /dev/null
curl -s http://localhost:8000/grader
```

**Expected output**
```json
{"error": "No actions recorded. Run /reset then /step first."}
```

---

## T-16  Baseline Endpoint

**Goal:** Confirm `/baseline` runs the rule-based agent and returns a normalised score ≥ 0.5.

```bash
curl -s -X POST http://localhost:8000/baseline | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('normalised_score :', d['normalised_score'])
print('total_reward     :', d['total_reward'])
print('steps            :', d['steps'])
print('passed (>=0.5)   :', d['normalised_score'] >= 0.5)
print()
for t in d['scores']:
    mark = '✓' if t['correct'] else '✗'
    print(f\"  {mark} {t['task_id']:12} [{t['difficulty']:6}] action={t['action_taken']:10} reward={t['reward']:+.2f}\")
"
```

**Expected output**
```
normalised_score : 0.7292
total_reward     : 8.75
steps            : 12
passed (>=0.5)   : True

  ✓ EASY-001     [easy  ] action=approve     reward=+1.00
  ✓ EASY-002     [easy  ] action=reject      reward=+1.00
  ...
```

---

## T-17  Optional Reason and Confidence Fields

**Goal:** Submit an action with optional `reason` and `confidence`. Should be accepted without error.

```bash
curl -s -X POST http://localhost:8000/reset > /dev/null

curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "flag",
    "transaction_id": "TXN-E001",
    "reason": "Low risk but flagging for audit trail verification",
    "confidence": 0.82
  }' \
| python3 -c "import sys,json; d=json.load(sys.stdin); print('HTTP OK, reward:', d['reward'])"
```

**Expected output**
```
HTTP OK, reward: 0.3
```

(0.3 = partial credit for flagging when correct action is approve)

---

## T-18  Hard Task — Model Poisoning (HARD-001)

**Goal:** Demonstrate that the low risk_score (0.18) is a trap. The correct action is `escalate`.

```bash
curl -s -X POST http://localhost:8000/reset > /dev/null
# Advance to HARD-001 (task 9) — step through easy + medium with dummy actions
python3 - <<'EOF'
import httpx
BASE = "http://localhost:8000"
c = httpx.Client()
tasks = c.get(f"{BASE}/tasks").json()["tasks"]

# Steps 1-8: easy + medium
for t in tasks[:8]:
    c.post(f"{BASE}/step", json={"action_type": "approve", "transaction_id": t["transaction_id"]})

# Step 9: HARD-001 — agent trusts the low risk_score and approves (wrong!)
resp_wrong = c.post(f"{BASE}/step", json={
    "action_type": "approve", "transaction_id": "TXN-H001"
}).json()
print("Trusted ML score → approve")
print("  reward   :", resp_wrong["reward"])   # expect -0.5
print("  correct  :", resp_wrong["info"]["correct_action"])

# Reset and do it correctly
c.post(f"{BASE}/reset")
for t in tasks[:8]:
    c.post(f"{BASE}/step", json={"action_type": "approve", "transaction_id": t["transaction_id"]})

resp_correct = c.post(f"{BASE}/step", json={
    "action_type": "escalate", "transaction_id": "TXN-H001"
}).json()
print("\nOverrode ML score → escalate")
print("  reward   :", resp_correct["reward"])  # expect +1.0
c.close()
EOF
```

**Expected output**
```
Trusted ML score → approve
  reward   : -0.5
  correct  : escalate

Overrode ML score → escalate
  reward   : 1.0
```

---

## T-19  Inspect Reveals Hidden Context (HARD-001)

**Goal:** Inspect HARD-001 to reveal the mule-account intelligence note before deciding.

```bash
curl -s -X POST http://localhost:8000/reset > /dev/null
# Advance to HARD-001
python3 - <<'EOF'
import httpx
BASE = "http://localhost:8000"
c = httpx.Client()
tasks = c.get(f"{BASE}/tasks").json()["tasks"]
for t in tasks[:8]:
    c.post(f"{BASE}/step", json={"action_type": "approve", "transaction_id": t["transaction_id"]})

# Inspect HARD-001
resp = c.post(f"{BASE}/step", json={
    "action_type": "inspect", "transaction_id": "TXN-H001"
}).json()
print("Inspect reward :", resp["reward"])
print("Notes          :", resp["inspection_notes"])
c.close()
EOF
```

**Expected output**
```
Inspect reward : 0.15
Notes          : Account created 7 days ago. This is the first outbound transfer. Receiver matches a pattern of solicitor-impersonation mule accounts flagged in last month's intelligence bulletin. Risk model underscored due to clean transaction history (new account).
```

---

## T-20  WebSocket Session

**Goal:** Run a full reset → step sequence over the WebSocket endpoint.

```bash
pip install websockets -q   # if not already installed

python3 - <<'EOF'
import asyncio, json, websockets

async def test_ws():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as ws:
        # Reset
        await ws.send(json.dumps({"type": "reset"}))
        obs = json.loads(await ws.recv())
        print("Reset  →", obs["transaction_id"], "risk:", obs["risk_score"])

        # Step – approve
        await ws.send(json.dumps({
            "type": "step",
            "action_type": "approve",
            "transaction_id": obs["transaction_id"]
        }))
        obs2 = json.loads(await ws.recv())
        print("Step   →", "reward:", obs2["reward"], "next:", obs2["transaction_id"])

        # State
        await ws.send(json.dumps({"type": "state"}))
        state = json.loads(await ws.recv())
        print("State  →", "steps:", state["step_count"], "txns:", state["transactions_processed"])

asyncio.run(test_ws())
EOF
```

**Expected output**
```
Reset  → TXN-E001 risk: 0.05
Step   → reward: 1.0 next: TXN-E002
State  → steps: 1 txns: 1
```

---

## T-21  Baseline Agent Script (Standalone)

**Goal:** Run the standalone Python baseline script independently of the server.

```bash
cd /Users/padmapriya
PYTHONPATH=/Users/padmapriya python3 payops_env/scripts/baseline_agent.py
```

**Expected output (last few lines)**
```
============================================================
  Episode Summary
============================================================
  Steps             : 12
  Total reward      : +8.75
  Max possible      : 12.00
  Normalised score  : 0.7292
  Passed (≥0.5)     : YES ✓
============================================================
```

---

## T-22  All Actions Are Valid on Each Task

**Goal:** Confirm every action type is accepted without error (even if penalised).

```bash
python3 - <<'EOF'
import httpx

BASE = "http://localhost:8000"
ACTIONS = ["approve", "reject", "flag", "escalate", "inspect", "hold"]

c = httpx.Client()
for action in ACTIONS:
    c.post(f"{BASE}/reset")
    resp = c.post(f"{BASE}/step", json={
        "action_type": action,
        "transaction_id": "TXN-E001"
    })
    print(f"action={action:10}  HTTP={resp.status_code}  reward={resp.json()['reward']:+.2f}")
c.close()
EOF
```

**Expected output**
```
action=approve     HTTP=200  reward=+1.00
action=reject      HTTP=200  reward=-0.50
action=flag        HTTP=200  reward=+0.30
action=escalate    HTTP=200  reward=-0.25
action=inspect     HTTP=200  reward=+0.15
action=hold        HTTP=200  reward=-0.25
```

---

## T-23  Full Perfect Episode (Score = 1.0)

**Goal:** Submit all 12 correct actions and confirm normalised_score = 1.0.

```bash
python3 - <<'EOF'
import httpx

BASE = "http://localhost:8000"
c = httpx.Client()

tasks = c.get(f"{BASE}/tasks").json()["tasks"]
c.post(f"{BASE}/reset")

for t in tasks:
    resp = c.post(f"{BASE}/step", json={
        "action_type": t["correct_action"],
        "transaction_id": t["transaction_id"]
    }).json()
    mark = "✓" if resp["reward"] == 1.0 else "✗"
    print(f"{mark} {t['task_id']:12} action={t['correct_action']:10} reward={resp['reward']:+.2f}")

score = c.get(f"{BASE}/grader").json()
print()
print("Normalised score:", score["normalised_score"])
print("Passed          :", score["passed"])
c.close()
EOF
```

**Expected output**
```
✓ EASY-001     action=approve     reward=+1.00
✓ EASY-002     action=reject      reward=+1.00
✓ EASY-003     action=approve     reward=+1.00
✓ EASY-004     action=flag        reward=+1.00
✓ MED-001      action=escalate    reward=+1.00
✓ MED-002      action=hold        reward=+1.00
✓ MED-003      action=flag        reward=+1.00
✓ MED-004      action=flag        reward=+1.00
✓ HARD-001     action=escalate    reward=+1.00
✓ HARD-002     action=reject      reward=+1.00
✓ HARD-003     action=reject      reward=+1.00
✓ HARD-004     action=approve     reward=+1.00

Normalised score: 1.0
Passed          : True
```

---

## T-24  Worst-Case Episode (Approve Everything)

**Goal:** Approve all 12 transactions (maximally wrong) and confirm very low score.

```bash
python3 - <<'EOF'
import httpx

BASE = "http://localhost:8000"
c = httpx.Client()

tasks = c.get(f"{BASE}/tasks").json()["tasks"]
c.post(f"{BASE}/reset")

total = 0
for t in tasks:
    resp = c.post(f"{BASE}/step", json={
        "action_type": "approve",
        "transaction_id": t["transaction_id"]
    }).json()
    total += resp["reward"]
    print(f"{t['task_id']:12} correct={t['correct_action']:10} reward={resp['reward']:+.2f}")

score = c.get(f"{BASE}/grader").json()
print()
print(f"Total reward     : {total:+.2f}")
print(f"Normalised score : {score['normalised_score']}")
print(f"Passed           : {score['passed']}")
c.close()
EOF
```

**Expected outcome:** Several `-1.0` and `-0.5` penalties. Normalised score near or equal to `0.0`. `passed=False`.

---

## Quick Reference — Expected Rewards Per Action

| Scenario | Action | Reward |
|----------|--------|--------|
| Correct decision | any | `+1.0` |
| Inspect (first time) | `inspect` | `+0.15` |
| Inspect (already inspected) | `inspect` | `0.0` |
| Partial credit (task-specific) | adjacent | `+0.2` – `+0.6` |
| Approve fraud/escalation | `approve` | `-1.0` |
| Approve flagged/held | `approve` | `-0.5` |
| Reject legitimate tx | `reject` | `-0.5` |
| Any other wrong action | any | `-0.25` |

---

## Quick Reference — Correct Actions per Task

| Task ID | Difficulty | Correct Action | Key Signal |
|---------|-----------|----------------|------------|
| EASY-001 | easy | `approve` | risk=0.05, no flags, verified KYC |
| EASY-002 | easy | `reject` | sanctioned country, unknown sender, risk=0.97 |
| EASY-003 | easy | `approve` | small refund, risk=0.03, verified |
| EASY-004 | easy | `flag` | velocity_1h=15 (ATM burst) |
| MED-001 | medium | `escalate` | large B2B, cross-border, medium-risk country |
| MED-002 | medium | `hold` | KYC pending, large internal transfer |
| MED-003 | medium | `flag` | amount 3× historical average |
| MED-004 | medium | `flag` | crypto exchange, moderate risk |
| HARD-001 | hard | `escalate` | risk_score=0.18 is poisoned — manual flags say escalate |
| HARD-002 | hard | `reject` | APP scam, mule account pattern |
| HARD-003 | hard | `reject` | structuring/smurfing, KYC failed |
| HARD-004 | hard | `approve` | legitimate FX settlement — looks scary, is fine |

---

## Running All Tests in One Script

Save the following as `run_tests.sh` and execute from `/Users/padmapriya`:

```bash
#!/usr/bin/env bash
# run_tests.sh — smoke-test all PayOps endpoints

set -e
BASE="http://localhost:8000"
PASS=0
FAIL=0

check() {
  local name="$1"
  local got="$2"
  local want="$3"
  if echo "$got" | grep -q "$want"; then
    echo "  ✓  $name"
    ((PASS++))
  else
    echo "  ✗  $name  (expected '$want', got '$got')"
    ((FAIL++))
  fi
}

echo "=== PayOps Test Suite ==="

check "T-01 health"        "$(curl -s $BASE/health)"              '"status":"ok"'
check "T-02 schema"        "$(curl -s $BASE/schema)"              '"PayOpsAction"'
check "T-03 tasks count"   "$(curl -s $BASE/tasks)"               '"count":12'
check "T-04 reset"         "$(curl -s -X POST $BASE/reset)"       '"task_id":"EASY-001"'
check "T-05 correct step"  "$(curl -s -X POST $BASE/step -H 'Content-Type: application/json' -d '{"action_type":"approve","transaction_id":"TXN-E001"}')"  '"reward":1.0'
check "T-10 invalid action" "$(curl -s -X POST $BASE/step -H 'Content-Type: application/json' -d '{"action_type":"delete","transaction_id":"TXN-E001"}')" "Invalid action_type"
check "T-16 baseline"      "$(curl -s -X POST $BASE/baseline)"    '"normalised_score"'

echo ""
echo "Results: $PASS passed, $FAIL failed"
```

```bash
cd /Users/padmapriya
bash payops_env/run_tests.sh
```

**Expected output**
```
=== PayOps Test Suite ===
  ✓  T-01 health
  ✓  T-02 schema
  ✓  T-03 tasks count
  ✓  T-04 reset
  ✓  T-05 correct step
  ✓  T-10 invalid action
  ✓  T-16 baseline

Results: 7 passed, 0 failed
```

---

## Interactive API Explorer

FastAPI serves auto-generated interactive docs. Open in a browser while the server is running:

```
http://localhost:8000/docs      ← Swagger UI (try endpoints in-browser)
http://localhost:8000/redoc     ← ReDoc documentation
```
