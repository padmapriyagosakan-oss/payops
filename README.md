---
title: PayOps — Payment Operations Incident Response
emoji: 💳
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - finance
  - fraud-detection
  - compliance
  - reinforcement-learning
pinned: false
---

# PayOps — Payment Operations Incident Response

An **OpenEnv-compatible** reinforcement-learning environment where an AI agent
acts as a Payment Operations analyst.  The agent reviews financial transactions
one by one and must decide the correct compliance action for each.

---

## Motivation

Payment operations teams process thousands of transactions every day.  A
skilled analyst uses dozens of signals — risk scores, velocity, KYC status,
flag patterns — to make fast, accurate decisions.  This environment lets an AI
agent learn and be evaluated on exactly this task, spanning clear-cut cases all
the way to subtle adversarial patterns like model-score poisoning and
Authorised Push Payment (APP) scams.

---

## Environment Description

Each **episode** steps through all **20 transactions** (4 easy, 6 medium, 6 hard, 4 critical).
For each transaction the agent observes a rich set of signals and chooses one
of **10 possible actions** — 5 terminal decisions and 5 investigation sub-actions.
A reward is returned immediately, and the next transaction is presented until
the episode is complete.

---

## Action Space

| Action       | Description |
|-------------|-------------|
| `approve`   | Mark transaction as legitimate; allow it through |
| `reject`    | Block the transaction outright |
| `flag`      | Soft hold; mark for manual review |
| `escalate`  | Route to senior compliance officer / fraud team |
| `inspect`   | Request additional signals (logs, KYC, velocity) — yields reveal notes and a small reward; agent then acts again on the same transaction |
| `hold`      | Temporary hold pending more information |

---

## Observation Space

| Field                 | Type           | Description |
|----------------------|----------------|-------------|
| `transaction_id`     | `str`          | Unique transaction identifier |
| `amount`             | `float`        | Transaction amount |
| `currency`           | `str`          | ISO-4217 currency code |
| `sender`             | `str`          | Sender identifier |
| `receiver`           | `str`          | Receiver identifier |
| `transaction_type`   | `str`          | transfer \| payment \| withdrawal \| refund \| internal |
| `status`             | `str`          | pending \| approved \| rejected \| flagged \| escalated \| held \| inspected |
| `risk_score`         | `float [0,1]`  | Composite ML risk score |
| `flags`              | `List[str]`    | Active risk flags |
| `velocity_1h`        | `int?`         | Transactions from sender in the past hour |
| `country_risk`       | `str?`         | low \| medium \| high \| sanctioned |
| `kyc_status`         | `str?`         | verified \| pending \| failed \| none |
| `previous_violations`| `int?`         | Prior compliance violations |
| `inspection_notes`   | `str?`         | Extra details revealed after an `inspect` action |
| `task_id`            | `str`          | Identifier of the active task |
| `task_difficulty`    | `str`          | easy \| medium \| hard |
| `step_in_episode`    | `int`          | Steps elapsed in this episode |
| `reward`             | `float`        | Reward from the last action |
| `cumulative_reward`  | `float`        | Total reward so far this episode |
| `done`               | `bool`         | Whether the episode has ended |
| `info`               | `dict`         | Diagnostic info (event, correct action, etc.) |

---

## Task Descriptions

### Easy (4 tasks — clear signals)

| ID        | Description | Correct Action |
|----------|-------------|----------------|
| EASY-001 | Low-value domestic transfer between verified users; no flags | `approve` |
| EASY-002 | Textbook fraud: unknown sender, offshore, sanctioned country, risk=0.97 | `reject` |
| EASY-003 | Standard refund to verified customer; tiny amount, no flags | `approve` |
| EASY-004 | ATM withdrawal burst — 15 withdrawals in 58 minutes | `flag` |

### Medium (6 tasks — ambiguous, multi-signal reasoning required)

| ID       | Description | Correct Action |
|---------|-------------|----------------|
| MED-001 | Large B2B wire, verified CFO, cross-border to medium-risk jurisdiction | `escalate` |
| MED-002 | Internal treasury transfer; large amount, KYC pending renewal | `hold` |
| MED-003 | Recurring subscription 3× higher than historical average | `flag` |
| MED-004 | Payment to licensed crypto exchange from verified personal account | `flag` |
| MED-005 | Payroll disbursement to 40 employees from new payroll account | `approve` |
| MED-006 | Cross-border remittance to high-risk corridor; regular migrant-worker pattern | `hold` |

### Hard (6 tasks — adversarial / edge-case)

| ID        | Description | Correct Action |
|----------|-------------|----------------|
| HARD-001 | Fraud model poisoning: risk_score=0.18 but manual signals scream escalate | `escalate` |
| HARD-002 | APP (Authorised Push Payment) scam: victim sending willingly to mule account | `reject` |
| HARD-003 | Structuring / smurfing: just-below-CTR-threshold payments, same UBO | `reject` |
| HARD-004 | Legitimate FX correspondent banking settlement — looks alarming, is not | `approve` |
| HARD-005 | Insider threat: employee initiating transfers to personal family accounts | `escalate` |
| HARD-006 | Zero-day mule account: new account receiving high-velocity micro-deposits | `reject` |

### Critical (4 tasks — regulatory + multi-step investigation chains)

| ID        | Description | Correct Action |
|----------|-------------|----------------|
| CRIT-001 | AML structuring ring: requires inspection + SAR filing before terminal action | `reject` + `file_sar` |
| CRIT-002 | KYC expiry bypass: counterparty exploiting grace period; verify_kyc needed | `hold` |
| CRIT-003 | Sanctions evasion via shell company chain; contact_sender + escalate required | `escalate` |
| CRIT-004 | Politically Exposed Person (PEP) large transfer to unknown shell entity | `escalate` |

---

## Reward Design

| Outcome | Reward |
|---------|--------|
| Correct action | **+1.0** |
| Partial-credit adjacent action (per-task) | **+0.2 – +0.6** |
| `inspect` (information seeking, first time) | **+0.15** |
| `approve` when correct is `reject` / `escalate` | **−1.0** |
| `approve` when correct is `flag` / `hold` | **−0.5** |
| `reject` when correct is `approve` | **−0.5** |
| Any other wrong action | **−0.25** |

The **episode score** (0–1) is: `max(0, total_reward) / max_possible_reward`.
A score ≥ 0.5 is considered a passing episode.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Reset environment, return first observation |
| `POST` | `/step` | Execute an action |
| `GET`  | `/state` | Current internal environment state |
| `GET`  | `/schema` | JSON schemas for action / observation / state |
| `GET`  | `/tasks` | Full task list with metadata |
| `GET`  | `/grader` | Grade the current episode |
| `POST` | `/baseline` | Run rule-based baseline and return scores |
| `GET`  | `/health` | Health check |
| `WS`   | `/ws` | WebSocket persistent session |

Interactive API docs: `http://localhost:8000/docs`

---

## Setup & Running

### Local (Python)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server (from the parent directory of payops_env)
PYTHONPATH=$(pwd) uvicorn payops_env.server.app:app --host 0.0.0.0 --port 8000

# 3. Verify
curl http://localhost:8000/health
```

### Run the baseline agent

```bash
# From the project root
PYTHONPATH=$(pwd) python payops_env/scripts/baseline_agent.py
```

### Docker

```bash
# Build
docker build -t payops-env .

# Run locally on port 8000
docker run -p 8000:7860 -e PORT=7860 payops-env

# Verify
curl http://localhost:8000/health
```

### HuggingFace Space

The `Dockerfile` exposes port **7860** (HF Spaces default).  Push the repo to
a HF Space with Docker runtime — no additional configuration required.

---

## Example Agent Interaction

```python
import httpx

base = "http://localhost:8000"

# Reset
obs = httpx.post(f"{base}/reset").json()
print(obs["transaction_id"], obs["risk_score"], obs["flags"])

# Step
while not obs["done"]:
    # ... agent decides action_type ...
    obs = httpx.post(f"{base}/step", json={
        "action_type": "approve",
        "transaction_id": obs["transaction_id"],
    }).json()
    print(f"reward={obs['reward']:+.2f}  done={obs['done']}")

# Grade
score = httpx.get(f"{base}/grader").json()
print(f"Episode score: {score['normalised_score']:.4f}")
```

---

## Baseline Results

The rule-based baseline agent uses a deterministic priority-ordered policy.

| Metric | Baseline (v2, 20 tasks) |
|--------|-------------------------|
| Normalised score | 0.68–0.76 |
| Passed (≥ 0.5) | Yes |
| Strong at | Easy tasks, clear velocity/flag patterns |
| Weak at | Hard adversarial tasks (HARD-001 model-poisoning, HARD-004 FX settlement) |
| Critical coverage | Partial — misses some SAR filing requirements |

Scores vary slightly per run due to per-episode parameter jitter (see below).

Run `POST /baseline` or `python payops_env/scripts/baseline_agent.py` to reproduce.

---

## Project Structure

```
payops_env/
├── models.py              # PayOpsAction, PayOpsObservation, PayOpsState (Pydantic)
├── environment.py         # PayOpsEnvironment — reset_async / step_async / state
├── tasks.py               # 20 tasks (EASY×4, MED×6, HARD×6, CRIT×4) with ground-truth labels
├── grader.py              # Partial-credit reward function + episode grader
├── scripts_util.py        # Baseline runner helper (used by /baseline endpoint)
├── scripts/
│   └── baseline_agent.py  # Standalone rule-based baseline agent
├── server/
│   └── app.py             # FastAPI server with all required endpoints
├── inference.py           # Competition inference script (OpenAI client, root-level)
├── validate.py            # Pre-submission checklist validator
├── openenv.yaml           # OpenEnv manifest v2.0.0
├── Dockerfile             # Docker / HuggingFace Space container (port 7860)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Evaluation Criteria Alignment

| Criterion | Implementation |
|-----------|---------------|
| Real-world utility | Payment fraud and compliance triage — deployed daily by fintech ops teams worldwide |
| Task & grader quality | 20 tasks across 4 difficulty tiers (easy→critical); partial-credit grader; clear pass/fail |
| Environment design | 30-field observation space; 10-action space (5 terminal + 5 investigation); budget mechanic; episode state tracking |
| Code quality & spec compliance | Pydantic v2 models; async API; all 11 required endpoints; openenv.yaml v2; Dockerfile; validate.py |
| Creativity & novelty | Adversarial model-poisoning task; APP scam; AML structuring with SAR requirement; PEP detection |


---

## Reward Design (v2 — Trajectory-Based)

Rewards are dense across the full trajectory, not just on the final decision:

| Component | Value | Condition |
|-----------|-------|-----------|
| Correct terminal action | **+0.60** | per task |
| Investigation sub-action | **+0.20** | per eligible sub-action, first use only |
| Flag identification | **+0.20** | agent used `inspect` AND task has key diagnostic flags |
| Confidence bonus | +0.10 | confidence ≥ 0.8 AND correct |
| Confidence penalty | −0.10 | confidence ≥ 0.8 AND wrong |
| Regulatory SAR bonus | +0.20 | `file_sar` before terminal on regulatory task |
| Duplicate investigation | −0.05 | same sub-action used twice on same task |
| Approve a fraud/sanctioned | **−1.00** | worst mistake |

Difficulty weights: easy×1.0, medium×1.2, hard×1.5, critical×2.0  
Episode score is **strictly clamped to `[0.0, 1.0]`**.  Passing threshold: **0.5**.

### Per-Episode Parameter Jitter

Each `POST /reset` generates a unique `episode_seed` and applies small random perturbations to prevent agent overfitting:

| Field | Jitter |
|-------|--------|
| `amount` | × Uniform(0.85, 1.20) |
| `risk_score` | + Gauss(0, 0.03), clamped [0,1] |
| `velocity_1h` | + Randint(−3, +3), min 0 |
| `velocity_24h` | + Randint(−3, +3), min 0 |

The `correct_action` and all ground-truth labels are **never changed** — only the observable values the agent uses to make decisions.

The `episode_seed` is returned by `GET /health` and `GET /state` for reproducibility.

### Network Graph

Selected tasks include a `network_graph` field in the observation exposing mule-chain / correspondent-banking relationships (e.g. victim → mule → offshore). This gives agents richer context for complex fraud patterns.
