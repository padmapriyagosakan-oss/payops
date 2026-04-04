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

Each **episode** steps through all **30 transactions** (6 easy, 8 medium, 10 hard, 6 critical).
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
| `budget_remaining`      | `float`           | Remaining investigation budget (starts at 5.0; decreases with each investigation action) |
| `inspection_notes`      | `str?`            | Additional details revealed after an `inspect` action |
| `docs_notes`            | `str?`            | Document review findings after a `request_docs` action |
| `kyc_notes`             | `str?`            | KYC re-verification outcome after a `verify_kyc` action |
| `contact_notes`         | `str?`            | Outcome after a `contact_sender` action |
| `investigation_hints`   | `List[str]`       | Sub-actions recommended for this task (e.g. `inspect`, `verify_kyc`). Using them before the terminal decision earns bonus reward. Empty = no specific investigation required. |
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
| EASY-005 | Scheduled monthly mortgage repayment; regular amount, verified borrower | `approve` |
| EASY-006 | Suspected duplicate payment: same sender/receiver/amount submitted twice in 4 minutes | `flag` |

### Medium (8 tasks — ambiguous, multi-signal reasoning required)

| ID       | Description | Correct Action |
|---------|-------------|----------------|
| MED-001 | Large B2B wire, verified CFO, cross-border to medium-risk jurisdiction | `escalate` |
| MED-002 | Internal treasury transfer; large amount, KYC pending renewal | `hold` |
| MED-003 | Recurring subscription 3× higher than historical average | `flag` |
| MED-004 | Payment to licensed crypto exchange from verified personal account | `flag` |
| MED-005 | Expired KYC on high-frequency corporate payroll account; KYC lapsed 12 days ago | `hold` |
| MED-006 | Real estate advance payment; large first-time transfer to new receiver but signed contract exists | `escalate` |
| MED-007 | Supplier emails to say bank details have changed; first payment to new account matches large invoice (BEC indicator) | `hold` |
| MED-008 | Buy Now Pay Later high-value purchase; new account, thin credit file, elevated risk signals | `flag` |

### Hard (10 tasks — adversarial / edge-case)

| ID        | Description | Correct Action |
|----------|-------------|----------------|
| HARD-001 | Fraud model poisoning: risk_score=0.18 but manual signals scream escalate | `escalate` |
| HARD-002 | APP (Authorised Push Payment) scam: victim sending willingly to mule account | `reject` |
| HARD-003 | Structuring / smurfing: just-below-CTR-threshold payments, same UBO | `reject` |
| HARD-004 | Legitimate FX correspondent banking settlement — looks alarming, is not | `approve` |
| HARD-005 | Insider threat: employee initiating transfers to personal family accounts | `escalate` |
| HARD-006 | Ghost account: dormant 5 years, suddenly received 20 inbound transfers this week | `flag` |
| HARD-007 | SIM-swap attack: phone ported 6 hours ago; account now requesting large crypto withdrawal to new address | `reject` |
| HARD-008 | Romance scam / pig butchering: 4th escalating transfer to overseas 'romantic partner' met online | `reject` |
| HARD-009 | Synthetic identity fraud: new business account with AI-generated-looking perfect profile | `escalate` |
| HARD-010 | Payroll diversion: HR system breach rerouted employee salary to newly added account | `reject` |

### Critical (6 tasks — regulatory + multi-step investigation chains)

| ID        | Description | Correct Action |
|----------|-------------|----------------|
| CRIT-001 | Multi-step chain: large PE wire to new counterparty; inspect then request docs before deciding (chain of 3) | `approve` |
| CRIT-002 | Fraud ring: coordinated small payments from 3 related accounts aggregating above reporting threshold; SAR required | `reject` |
| CRIT-003 | Trade-based money laundering: over-invoiced international trade payment (4× market price) | `escalate` |
| CRIT-004 | Compromised corporate account: geo-impossible login (NY → Lagos in 8 min); confirmed account takeover | `reject` |
| CRIT-005 | OFAC sanctions evasion: large USD payment routed through UAE shell chain; UBO is on SDN list (chain of 3) | `reject` |
| CRIT-006 | Correspondent banking: partner bank added to FinCEN 311 Special Measures list; in-flight payments must be escalated | `escalate` |

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

| Metric | Rule-based baseline (v2, 30 tasks) |
|--------|------------------------------------|
| Normalised score | 0.68–0.76 |
| Passed (≥ 0.5) | Yes |
| Strong at | Easy tasks, clear velocity/flag patterns |
| Weak at | Hard adversarial tasks (HARD-001 model-poisoning, HARD-004 FX settlement) |
| Critical coverage | Partial — misses some SAR filing requirements |

Scores vary slightly per run due to per-episode parameter jitter.

Run `POST /baseline` to reproduce.

### LLM baseline (`inference.py` — `llama-3.1-8b-instant` via Groq)

Run locally against seed 42 (reproducible) with investigation sub-actions enabled.

| Metric | llama-3.1-8b-instant (Groq) |
|--------|-----------------------------|
| Normalised score | **0.6028** |
| Total reward | 17.000 / 28.200 max |
| Tasks correct | 6 / 20 (30%) |
| Budget spent | 5.50 / 5.00 |
| Budget penalty | 0.05 |
| Episode steps | 57 (incl. investigation sub-actions) |
| Duration | ~290 s |
| Passed (≥ 0.5) | **YES ✓** |
| Seed | 42 (fixed — deterministic across re-runs) |

**Per-task decisions:**

| Task | LLM Action | Correct Action | Weighted Reward |
|------|-----------|----------------|----------------|
| EASY-001 | `approve` | `approve` | +1.000 ✓ |
| EASY-002 | `flag` | `reject` | −0.250 ✗ (flag no longer partial credit) |
| EASY-003 | `approve` | `approve` | +1.000 ✓ |
| EASY-004 | `flag` | `flag` | +1.000 ✓ |
| MED-001 | `flag` | `escalate` | +0.900 (partial + investigation bonus) |
| MED-002 | `flag` | `hold` | +0.540 (partial + investigation bonus) |
| MED-003 | `flag` | `flag` | +1.200 ✓ |
| MED-004 | `flag` | `flag` | +1.200 ✓ |
| MED-005 | `flag` | `hold` | +0.660 (partial + investigation bonus) |
| MED-006 | `flag` | `escalate` | +0.600 (partial + investigation bonus) |
| HARD-001 | `flag` | `escalate` | +1.275 (partial + investigation bonus) |
| HARD-002 | `flag` | `reject` | +0.525 (partial + investigation bonus) |
| HARD-003 | `flag` | `reject` | +0.675 (partial + investigation bonus) |
| HARD-004 | `flag` | `approve` | +0.825 (partial + investigation bonus) |
| HARD-005 | `flag` | `escalate` | +0.825 (partial + investigation bonus) |
| HARD-006 | `flag` | `flag` | +2.025 ✓ (+ investigation bonus) |
| CRIT-001 | `flag` | `approve` | +1.100 (partial + investigation bonus) |
| CRIT-002 | `flag` | `reject` | +0.900 (partial + investigation bonus) |
| CRIT-003 | `flag` | `escalate` | +1.300 (partial + investigation bonus) |
| CRIT-004 | `flag` | `reject` | −0.250 ✗ |

**Observations:** The model used investigation sub-actions (`inspect`, `verify_kyc`, `contact_sender`) before terminal decisions, earning investigation bonuses that raised the score from a naive always-flag baseline. Easy cases with clear evidence now penalise lazy `flag` decisions (e.g. EASY-002). Agents that correctly identify terminal actions on top of proper investigation can exceed 0.90.

To reproduce exactly (seed=42 is the default):

```bash
export OPENAI_API_KEY="gsk_..."           # your Groq API key
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export PAYOPS_BASE_URL="https://padmapriyagosakan-payops-env.hf.space"
# INFERENCE_SEED=42  # default; set to "random" for a fresh episode
PYTHONPATH=$(pwd) python payops_env/inference.py
```

For Groq setup instructions see the **Running inference with Groq** section below.

---

## Running inference with Groq (recommended — free)

[Groq](https://console.groq.com) provides a completely free API with no monthly credit cap and no installation required. It uses the same OpenAI-compatible interface that `inference.py` already targets.

### Prerequisites

1. **Create a free Groq account** — go to [console.groq.com](https://console.groq.com) and sign up (Google / GitHub login available)

2. **Generate an API key** — click **API Keys → Create API Key**, copy the key (starts with `gsk_`)

3. **Install the Python dependency** (already in `requirements.txt`):
   ```bash
   pip install openai
   ```

### Run inference

```bash
cd /path/to/payops_env      # project root (parent of payops_env/)

export OPENAI_API_KEY="gsk_..."          # your Groq API key
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export PAYOPS_BASE_URL="https://padmapriyagosakan-payops-env.hf.space"

PYTHONPATH=$(pwd) python payops_env/inference.py
```

> **Why Groq?**
> - Free tier: 14,400 requests/day, 500,000 tokens/minute — a 20-task episode uses ~30 calls
> - No monthly credit pool that runs out mid-run (unlike the HF free tier)
> - No installation or model download (unlike Ollama)
> - `temperature=0.0` is already set in `inference.py` so results are reproducible
> - Inference speed: ~750 tok/s → full episode completes in under 30 seconds

### Alternative free models on Groq

| Model | Notes |
|-------|-------|
| `llama-3.1-8b-instant` | Fastest, good reasoning |
| `llama-3.3-70b-versatile` | Best quality on hard tasks; same free tier |
| `mixtral-8x7b-32768` | Large context window |
| `gemma2-9b-it` | Google Gemma 2 |

### Alternative: Ollama (fully local, no internet required for LLM calls)

If you prefer to run the model entirely on your machine:

```bash
# 1. Install
brew install ollama

# 2. Pull a model (choose based on available RAM)
ollama pull qwen2.5:3b      # ~2 GB  –  8 GB RAM
ollama pull qwen2.5:7b      # ~4.7 GB – 16 GB RAM

# 3. Start the server (keep running in a separate terminal)
ollama serve

# 4. Run inference
export OPENAI_API_KEY=ollama
export API_BASE_URL="http://localhost:11434/v1"
export MODEL_NAME="qwen2.5:3b"
export PAYOPS_BASE_URL="https://padmapriyagosakan-payops-env.hf.space"
PYTHONPATH=$(pwd) python payops_env/inference.py
```

---

## Project Structure

```
payops_env/
├── models.py              # PayOpsAction, PayOpsObservation, PayOpsState (Pydantic)
├── environment.py         # PayOpsEnvironment — reset_async / step_async / state
├── tasks.py               # 30 tasks (EASY×6, MED×8, HARD×10, CRIT×6) with ground-truth labels
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
| Task & grader quality | 30 tasks across 4 difficulty tiers (easy→critical); partial-credit grader; clear pass/fail |
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
