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

Terminal decisions (no budget cost) commit to a final outcome for the transaction.
Investigation sub-actions (with budget cost) reveal more information and let the agent act again on the same transaction.

| Action           | Type          | Description | Budget Cost |
|-----------------|---------------|-------------|-------------|
| `approve`        | terminal      | Mark transaction as legitimate; allow it through | — |
| `reject`         | terminal      | Block the transaction outright | — |
| `flag`           | terminal      | Soft hold; mark for manual review | — |
| `escalate`       | terminal      | Route to senior compliance officer / fraud team | — |
| `hold`           | terminal      | Temporary hold pending more information | — |
| `inspect`        | investigation | Pull additional signals (logs, KYC, velocity) — yields `inspection_notes` | 0.10 |
| `request_docs`   | investigation | Ask sender for supporting documents (invoice, contract) — yields `docs_notes` | 0.20 |
| `verify_kyc`     | investigation | Trigger an active KYC re-verification check — yields `kyc_notes` | 0.20 |
| `contact_sender` | investigation | Contact the sender directly to confirm intent — yields `contact_notes` | 0.30 |
| `file_sar`       | investigation | File a Suspicious Activity Report to the regulator (required on AML/structuring tasks) | 0.10 |

---

## Observation Space

| Field                   | Type              | Description |
|------------------------|-------------------|-------------|
| `transaction_id`        | `str`             | Unique transaction identifier |
| `amount`                | `float`           | Transaction amount in the stated currency |
| `currency`              | `str`             | ISO-4217 currency code |
| `sender`                | `str`             | Sender identifier (email / account / alias) |
| `receiver`              | `str`             | Receiver identifier |
| `transaction_type`      | `str`             | transfer \| payment \| withdrawal \| refund \| internal \| loan_repayment \| payroll |
| `status`                | `str`             | pending \| approved \| rejected \| flagged \| escalated \| held \| inspected \| docs_requested \| kyc_triggered \| sender_contacted \| sar_filed |
| `risk_score`            | `float [0,1]`     | Composite ML risk score |
| `ml_confidence`         | `float [0,1]`     | Model's self-reported confidence in `risk_score` — low value signals possible model poisoning |
| `flags`                 | `List[str]`       | Active risk flags (e.g. `high_value`, `unknown_sender`, `velocity_breach`) |
| `velocity_1h`           | `int?`            | Transactions from sender in the past hour |
| `velocity_24h`          | `int?`            | Transactions from sender in the past 24 hours |
| `avg_transaction_amount`| `float?`          | Sender's historical average transaction amount |
| `account_age_days`      | `int?`            | Age of the sender account in days |
| `country_risk`          | `str?`            | low \| medium \| high \| sanctioned |
| `kyc_status`            | `str?`            | verified \| pending \| failed \| none \| expired |
| `kyc_expiry_days`       | `int?`            | Days until KYC expires (negative = already expired) |
| `previous_violations`   | `int?`            | Prior compliance violations for this sender |
| `previous_sars`         | `int?`            | Suspicious Activity Reports previously filed for this sender |
| `counterparty_risk`     | `str?`            | clean \| unknown \| watchlist \| blacklist |
| `chain_step`            | `int`             | Current step in a multi-hop investigation chain (1 = initial presentation) |
| `chain_total`           | `int`             | Total investigation steps for this task (1 = single-step) |
| `chain_context`         | `str?`            | Accumulated summary of findings from earlier chain steps |
| `steps_remaining`       | `int?`            | Investigation sub-steps remaining before a terminal decision is required |
| `action_cost`           | `float`           | Budget cost incurred by the last action |
| `budget_remaining`      | `float [0,1]`     | Fraction of investigation budget remaining (1.0 = full, 0.0 = exhausted) |
| `inspection_notes`      | `str?`            | Additional details revealed after an `inspect` action |
| `docs_notes`            | `str?`            | Document review findings after a `request_docs` action |
| `kyc_notes`             | `str?`            | KYC re-verification outcome after a `verify_kyc` action |
| `contact_notes`         | `str?`            | Outcome after a `contact_sender` action |
| `recent_decisions`      | `List[dict]`      | Last ≤3 completed decisions in this episode (for pattern context) |
| `network_graph`         | `dict?`           | Mule-chain / correspondent-bank relationship graph where present |
| `task_id`               | `str`             | Identifier of the active task |
| `task_difficulty`       | `str`             | easy \| medium \| hard \| critical |
| `step_in_episode`       | `int`             | Steps elapsed in this episode |
| `reward`                | `float`           | Reward from the last action |
| `reward_breakdown`      | `dict`            | Itemised reward components: base, confidence_bonus, cost_penalty, etc. |
| `cumulative_reward`     | `float`           | Total reward accumulated so far in this episode |
| `done`                  | `bool`            | Whether the episode has ended |
| `info`                  | `dict`            | Diagnostic info (event, correct action, etc.) |

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
| MED-005 | Expired KYC on high-frequency corporate payroll account; KYC lapsed 12 days ago | `hold` |
| MED-006 | Real estate advance payment; large first-time transfer to new receiver but signed contract exists | `escalate` |

### Hard (6 tasks — adversarial / edge-case)

| ID        | Description | Correct Action |
|----------|-------------|----------------|
| HARD-001 | Fraud model poisoning: risk_score=0.18 but manual signals scream escalate | `escalate` |
| HARD-002 | APP (Authorised Push Payment) scam: victim sending willingly to mule account | `reject` |
| HARD-003 | Structuring / smurfing: just-below-CTR-threshold payments, same UBO | `reject` |
| HARD-004 | Legitimate FX correspondent banking settlement — looks alarming, is not | `approve` |
| HARD-005 | Insider threat: employee initiating transfers to personal family accounts | `escalate` |
| HARD-006 | Ghost account: dormant 5 years, suddenly received 20 inbound transfers this week — possible account takeover | `flag` |

### Critical (4 tasks — regulatory + multi-step investigation chains)

| ID        | Description | Correct Action |
|----------|-------------|----------------|
| CRIT-001 | Multi-step chain: large wire to new counterparty; agent must inspect then request docs before deciding (chain of 3) | `approve` |
| CRIT-002 | Fraud ring: coordinated small payments from 3 related accounts aggregating above reporting threshold; SAR required | `reject` |
| CRIT-003 | Trade-based money laundering: over-invoiced international trade payment (4× market price); regulatory escalation required | `escalate` |
| CRIT-004 | Compromised corporate account: geo-impossible login (NY → Lagos in 8 min); confirmed account takeover | `reject` |

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
# Via the API endpoint (no extra script needed)
curl -s -X POST http://localhost:8000/baseline | python3 -m json.tool
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

### Rule-based baseline (`POST /baseline`)

The rule-based baseline uses a deterministic priority-ordered policy in `scripts_util.py`.

| Metric | Rule-based baseline (v2, 20 tasks) |
|--------|------------------------------------|
| Normalised score | 0.68–0.76 |
| Passed (≥ 0.5) | Yes |
| Strong at | Easy tasks, clear velocity/flag patterns |
| Weak at | Hard adversarial tasks (HARD-001 model-poisoning, HARD-004 FX settlement) |
| Critical coverage | Partial — misses some SAR filing requirements |

Scores vary slightly per run due to per-episode parameter jitter.

Run `POST /baseline` to reproduce.

### LLM baseline (`inference.py` — `Qwen/Qwen2.5-7B-Instruct` via HF Inference)

Run on 30 March 2026 against the live HF Space (`https://padmapriyagosakan-payops-env.hf.space`).

| Metric | Qwen/Qwen2.5-7B-Instruct |
|--------|--------------------------|
| Normalised score | 0.3612 |
| Total reward | 10.185 / 28.200 max |
| Budget spent | 3.10 / 5.00 |
| Budget penalty | 0.00 |
| Passed (≥ 0.5) | No |
| LLM calls completed | ~10 of 20 tasks (HF free-tier credits exhausted mid-run; remainder used `flag` fallback) |

**Per-task LLM decisions (first ~10 tasks, before credits exhausted):**

| Task | LLM Action | Correct | Result |
|------|-----------|---------|--------|
| EASY-001 | inspect → approve | approve | ✓ correct (+investigation bonus) |
| EASY-002 | escalate | reject | partial credit |
| EASY-003 | inspect → approve | approve | ✓ correct (+investigation bonus) |
| EASY-004 | reject | flag | ✗ wrong (−0.25) |
| MED-001 | inspect → approve | escalate | ✗ wrong approve (−1.0) |
| MED-002 | inspect → hold | hold | ✓ correct (+investigation bonus) |
| MED-003 | flag | flag | ✓ correct |
| MED-004 | flag | flag | ✓ correct |

To reproduce with your own API key:

```bash
export HF_TOKEN="hf_..."                  # or OPENAI_API_KEY for OpenAI
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export PAYOPS_BASE_URL="https://padmapriyagosakan-payops-env.hf.space"
python inference.py
```

---

## Project Structure

```
payops_env/
├── models.py              # PayOpsAction, PayOpsObservation, PayOpsState (Pydantic)
├── environment.py         # PayOpsEnvironment — reset_async / step_async / state
├── tasks.py               # 20 tasks (EASY×4, MED×6, HARD×6, CRIT×4) with ground-truth labels
├── grader.py              # Partial-credit reward function + episode grader
├── scripts_util.py        # Baseline runner helper (used by /baseline endpoint)
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
| Correct terminal action | **+1.0** | per task (difficulty-weighted in episode score) |
| Investigation sub-action | **+0.15** | per eligible sub-action, first use only |
| Flag identification | **+0.20** | agent used `inspect` AND key diagnostic flags present |
| Confidence bonus | +0.10 | confidence ≥ 0.8 AND correct |
| Confidence penalty | −0.10 | confidence ≥ 0.8 AND wrong |
| Regulatory SAR bonus | +0.20 | `file_sar` before terminal on a regulatory task |
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
