from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

class PayOpsAction(BaseModel):
    """
    Action submitted by the agent for a single transaction.

    action_type choices
    -------------------
    approve        – mark transaction as legitimate and allow it through
    reject         – block the transaction outright
    flag           – mark for manual review with a soft hold
    escalate       – route to senior compliance officer / fraud team
    inspect        – pull additional signals (logs, KYC data, velocity)
    hold           – temporary hold pending more information
    request_docs   – ask sender for supporting documents (e.g. invoice, contract)
    verify_kyc     – trigger an active KYC re-verification check
    contact_sender – contact the sender directly to confirm intent
    file_sar       – file a Suspicious Activity Report to regulator
    """

    action_type: str = Field(
        ...,
        description=(
            "One of: approve | reject | flag | escalate | inspect | hold "
            "| request_docs | verify_kyc | contact_sender | file_sar"
        ),
    )
    transaction_id: str = Field(..., description="ID of the transaction being acted on")
    reason: Optional[str] = Field(
        default=None, description="Free-text rationale from the agent"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Agent self-reported confidence [0, 1]. Used in reward shaping.",
    )


# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------

class PayOpsObservation(BaseModel):
    """
    Structured observation returned after each step (and on reset).
    Describes the current transaction visible to the agent.
    """

    # --- transaction identity ---
    transaction_id: str
    amount: float = Field(..., description="Transaction amount in the stated currency")
    currency: str = Field(..., description="ISO-4217 currency code, e.g. USD, EUR")
    sender: str = Field(..., description="Sender identifier (email / account / alias)")
    receiver: str = Field(..., description="Receiver identifier")
    transaction_type: str = Field(
        default="transfer",
        description="Type: transfer | payment | withdrawal | refund | internal | loan_repayment | payroll",
    )

    # --- risk signals ---
    status: str = Field(
        default="pending",
        description=(
            "Current status: pending | approved | rejected | flagged | escalated "
            "| held | inspected | docs_requested | kyc_triggered | sender_contacted | sar_filed"
        ),
    )
    risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Composite ML risk score [0=low, 1=high]"
    )
    ml_confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Model's self-reported confidence in its own risk_score. Low = possibly poisoned.",
    )
    flags: List[str] = Field(
        default_factory=list,
        description="Active risk flags e.g. high_value, unknown_sender, velocity_breach",
    )

    # --- sender behaviour signals ---
    velocity_1h: Optional[int] = Field(
        default=None,
        description="Number of transactions from this sender in the past hour",
    )
    velocity_24h: Optional[int] = Field(
        default=None,
        description="Number of transactions from this sender in the past 24 hours",
    )
    avg_transaction_amount: Optional[float] = Field(
        default=None,
        description="Sender's historical average transaction amount",
    )
    account_age_days: Optional[int] = Field(
        default=None,
        description="Age of the sender account in days",
    )

    # --- counterparty / geography ---
    country_risk: Optional[str] = Field(
        default=None,
        description="Receiver country risk tier: low | medium | high | sanctioned",
    )
    kyc_status: Optional[str] = Field(
        default=None,
        description="KYC verification status: verified | pending | failed | none | expired",
    )
    kyc_expiry_days: Optional[int] = Field(
        default=None,
        description="Days until KYC expires (negative = already expired)",
    )
    previous_violations: Optional[int] = Field(
        default=None,
        description="Number of prior compliance violations for this sender",
    )
    previous_sars: Optional[int] = Field(
        default=None,
        description="Number of Suspicious Activity Reports previously filed for this sender",
    )
    counterparty_risk: Optional[str] = Field(
        default=None,
        description="Known risk profile of the receiver: clean | unknown | watchlist | blacklist",
    )

    # --- chain context (multi-hop investigation) ---
    chain_step: int = Field(
        default=1,
        description="Which step within a multi-hop investigation chain (1=initial presentation)",
    )
    chain_total: int = Field(
        default=1,
        description="Total number of chained investigation steps for this task",
    )
    chain_context: Optional[str] = Field(
        default=None,
        description="Summary of findings from earlier chain steps",
    )

    # --- resource tracking ---
    steps_remaining: Optional[int] = Field(
        default=None,
        description="How many investigation sub-steps remain before a terminal decision is required",
    )
    action_cost: float = Field(
        default=0.0,
        description="Operational cost penalty incurred by the last action",
    )
    budget_remaining: float = Field(
        default=1.0,
        description="Fraction of investigation budget remaining (1.0=full, 0.0=exhausted)",
    )

    # --- context from prior investigation actions ---
    inspection_notes: Optional[str] = Field(
        default=None,
        description="Additional details revealed after an 'inspect' action",
    )
    docs_notes: Optional[str] = Field(
        default=None,
        description="Document review findings after a 'request_docs' action",
    )
    kyc_notes: Optional[str] = Field(
        default=None,
        description="KYC re-verification outcome after a 'verify_kyc' action",
    )
    contact_notes: Optional[str] = Field(
        default=None,
        description="Outcome of contacting the sender via 'contact_sender' action",
    )

    # --- recent decision context (last 3 decisions in this episode) ---
    recent_decisions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Last up to 3 completed decisions in this episode for pattern context",
    )

    # --- episode bookkeeping ---
    task_id: str = Field(default="", description="Identifier of the active task")
    task_difficulty: str = Field(
        default="easy", description="Difficulty tier: easy | medium | hard | critical"
    )
    step_in_episode: int = Field(
        default=0, description="How many steps have elapsed in this episode"
    )
    reward: float = Field(default=0.0, description="Reward from the last action")
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Itemised reward components: base, time_penalty, confidence_bonus, cost_penalty",
    )
    cumulative_reward: float = Field(
        default=0.0, description="Total reward accumulated so far in this episode"
    )
    done: bool = Field(default=False, description="Whether the episode has ended")
    network_graph: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Mule-chain / correspondent-bank relationship graph for tasks where present",
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra diagnostic information (action taken, correct action, etc.)",
    )


# ---------------------------------------------------------------------------
# Internal state (used by the server's /state endpoint)
# ---------------------------------------------------------------------------

class PayOpsState(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0
    current_task_id: str = ""
    transactions_processed: int = 0
    total_tasks: int = 0
    cumulative_reward: float = 0.0
    budget_spent: float = Field(default=0.0, description="Total action costs accumulated")
    budget_limit: float = Field(default=5.0, description="Max investigation budget per episode")
    actions_taken: List[str] = Field(default_factory=list)
    last_action: Optional[str] = None
    investigation_actions_used: List[str] = Field(
        default_factory=list,
        description="All investigation sub-actions used this episode (inspect, request_docs, etc.)",
    )
    correct_decisions: int = Field(default=0, description="Terminal decisions that matched ground truth")
    wrong_high_cost: int = Field(
        default=0, description="Count of approve-on-fraud type mistakes"
    )
    recent_decisions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent completed task outcomes for analytics",
    )
    done: bool = False
    episode_seed: Optional[int] = Field(
        default=None,
        description="Random seed used to jitter task parameters this episode (for reproducibility)",
    )

