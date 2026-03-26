"""
Task bank for the PayOps environment — 20 tasks across 4 difficulty tiers.

Difficulty tiers
----------------
  easy      – single clear signal; one-hop decision
  medium    – ambiguous or competing signals; reasoning required
  hard      – adversarial, conflicting, or edge-case patterns
  critical  – multi-step investigation chains; regulatory compliance stakes

Multi-step chains
-----------------
Tasks with chain_total > 1 require the agent to issue investigation sub-actions
(inspect / request_docs / verify_kyc / contact_sender) before a terminal decision.
Each chain step reveals progressively more context via the appropriate _reveal field.
The terminal decision is only scored on the final chain step.

Action costs
------------
Each investigation sub-action incurs a budget cost.  If the agent exhausts the
budget (spend > budget_limit), a cumulative cost penalty is applied to the reward.

  inspect        → cost 0.1
  request_docs   → cost 0.2
  verify_kyc     → cost 0.2
  contact_sender → cost 0.3
  file_sar       → cost 0.1  (required for structuring/AML — free if correct)

Reward structure per task
-------------------------
Each task defines:
  correct_action          – ground-truth terminal decision
  partial_credit_actions  – {action: fraction_of_full_credit}
  requires_investigation  – set of sub-actions ideally used before deciding
  regulatory_action       – if True, file_sar is a required prerequisite for full credit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class PayOpsTask:
    # --- identity ---
    task_id: str
    difficulty: str          # easy | medium | hard | critical
    description: str

    # --- transaction fields ---
    transaction_id: str
    amount: float
    currency: str
    sender: str
    receiver: str
    transaction_type: str   # transfer | payment | withdrawal | refund | internal | loan_repayment | payroll

    # --- primary risk signals ---
    risk_score: float                    # 0.0–1.0
    ml_confidence: float = 0.90         # model's confidence in its own risk_score
    flags: List[str] = field(default_factory=list)

    # --- sender behaviour ---
    velocity_1h: Optional[int] = None
    velocity_24h: Optional[int] = None
    avg_transaction_amount: Optional[float] = None
    account_age_days: Optional[int] = None

    # --- counterparty / geo ---
    country_risk: Optional[str] = None    # low | medium | high | sanctioned
    kyc_status: Optional[str] = None      # verified | pending | failed | none | expired
    kyc_expiry_days: Optional[int] = None
    previous_violations: Optional[int] = None
    previous_sars: Optional[int] = None
    counterparty_risk: Optional[str] = None  # clean | unknown | watchlist | blacklist

    # --- grading ---
    correct_action: str = "approve"
    partial_credit_actions: Dict[str, float] = field(default_factory=dict)
    requires_investigation: Set[str] = field(default_factory=set)
    # Set of sub-actions the agent SHOULD use for best score.
    # Using them before the terminal decision grants a bonus (applied in grader).
    regulatory_action: bool = False
    # If True, filing a SAR is required to get full credit.

    # --- investigation reveals (each unlocked by the matching sub-action) ---
    inspect_reveal: Optional[str] = None
    docs_reveal: Optional[str] = None
    kyc_reveal: Optional[str] = None
    contact_reveal: Optional[str] = None

    # --- multi-step chain metadata ---
    chain_total: int = 1
    # chain_total > 1 means the task spans multiple investigation steps.
    # The grader only scores the terminal decision on the last chain step.


# =============================================================================
# EASY  (4 tasks — single clear signal)
# =============================================================================

EASY_TASKS: List[PayOpsTask] = [
    PayOpsTask(
        task_id="EASY-001",
        difficulty="easy",
        description="Low-value domestic transfer between two long-standing verified users. No flags.",
        transaction_id="TXN-E001",
        amount=47.50,
        currency="USD",
        sender="alice@example.com",
        receiver="bob@example.com",
        transaction_type="transfer",
        risk_score=0.05,
        ml_confidence=0.97,
        flags=[],
        velocity_1h=1, velocity_24h=3,
        avg_transaction_amount=55.0,
        account_age_days=1_095,
        country_risk="low", kyc_status="verified", kyc_expiry_days=320,
        previous_violations=0, previous_sars=0, counterparty_risk="clean",
        correct_action="approve",
        partial_credit_actions={"flag": 0.3},
        inspect_reveal="Sender account 3 yrs old. Consistent transaction history. KYC fully verified.",
    ),
    PayOpsTask(
        task_id="EASY-002",
        difficulty="easy",
        description="Textbook fraud: unknown sender, offshore, sanctioned country, risk=0.97.",
        transaction_id="TXN-E002",
        amount=9_800.00,
        currency="USD",
        sender="unknown_user_x99",
        receiver="offshore_acct_442",
        transaction_type="transfer",
        risk_score=0.97,
        ml_confidence=0.95,
        flags=["high_value", "unknown_sender", "offshore_receiver", "sanctioned_country"],
        velocity_1h=12, velocity_24h=28,
        avg_transaction_amount=None,
        account_age_days=2,
        country_risk="sanctioned", kyc_status="none", kyc_expiry_days=None,
        previous_violations=3, previous_sars=1, counterparty_risk="blacklist",
        correct_action="reject",
        partial_credit_actions={"escalate": 0.4, "flag": 0.2},
        inspect_reveal="Receiver linked to 3 prior rejected transactions. IP on threat-intel blocklist.",
    ),
    PayOpsTask(
        task_id="EASY-003",
        difficulty="easy",
        description="Standard refund to a verified customer. Tiny amount, no flags.",
        transaction_id="TXN-E003",
        amount=12.99,
        currency="USD",
        sender="shop@retailer.com",
        receiver="carol@customer.com",
        transaction_type="refund",
        risk_score=0.03,
        ml_confidence=0.98,
        flags=[],
        velocity_1h=1, velocity_24h=5,
        avg_transaction_amount=14.0,
        account_age_days=730,
        country_risk="low", kyc_status="verified", kyc_expiry_days=180,
        previous_violations=0, previous_sars=0, counterparty_risk="clean",
        correct_action="approve",
        partial_credit_actions={"flag": 0.3},
        inspect_reveal="Standard return via web checkout. Matching order ID on file. No anomalies.",
    ),
    PayOpsTask(
        task_id="EASY-004",
        difficulty="easy",
        description="ATM withdrawal burst — 15 withdrawals in 58 minutes across 4 ATMs. Velocity violation.",
        transaction_id="TXN-E004",
        amount=200.00,
        currency="USD",
        sender="david@accounts.com",
        receiver="atm_node_77",
        transaction_type="withdrawal",
        risk_score=0.78,
        ml_confidence=0.89,
        flags=["velocity_breach", "atm_burst"],
        velocity_1h=15, velocity_24h=17,
        avg_transaction_amount=80.0,
        account_age_days=540,
        country_risk="medium", kyc_status="verified", kyc_expiry_days=90,
        previous_violations=0, previous_sars=0, counterparty_risk="unknown",
        correct_action="flag",
        partial_credit_actions={"escalate": 0.6, "hold": 0.5},
        inspect_reveal="15 ATM withdrawals in 58 min across 4 ATMs. Pattern consistent with card clone.",
    ),
]


# =============================================================================
# MEDIUM  (6 tasks — ambiguous, multi-signal)
# =============================================================================

MEDIUM_TASKS: List[PayOpsTask] = [
    PayOpsTask(
        task_id="MED-001",
        difficulty="medium",
        description="Large B2B wire. Verified CFO. Cross-border to medium-risk EU jurisdiction.",
        transaction_id="TXN-M001",
        amount=85_000.00,
        currency="EUR",
        sender="cfo@globalcorp.com",
        receiver="vendor@eu-supplier.de",
        transaction_type="transfer",
        risk_score=0.52,
        ml_confidence=0.72,
        flags=["high_value", "cross_border"],
        velocity_1h=2, velocity_24h=4,
        avg_transaction_amount=45_000.0,
        account_age_days=2_190,
        country_risk="medium", kyc_status="verified", kyc_expiry_days=200,
        previous_violations=0, previous_sars=0, counterparty_risk="clean",
        correct_action="escalate",
        partial_credit_actions={"flag": 0.5, "hold": 0.4},
        requires_investigation={"inspect"},
        inspect_reveal="Contract on file for €85k milestone. Receiver is licensed EU entity. Quarterly vendor pattern.",
    ),
    PayOpsTask(
        task_id="MED-002",
        difficulty="medium",
        description="Internal treasury transfer. Large amount. KYC pending renewal — risk is procedural, not fraud.",
        transaction_id="TXN-M002",
        amount=250_000.00,
        currency="USD",
        sender="treasury@holdco.com",
        receiver="subsidiary@holdco-us.com",
        transaction_type="internal",
        risk_score=0.41,
        ml_confidence=0.78,
        flags=["high_value", "kyc_expiry_90d"],
        velocity_1h=1, velocity_24h=2,
        avg_transaction_amount=200_000.0,
        account_age_days=3_650,
        country_risk="low", kyc_status="pending", kyc_expiry_days=5,
        previous_violations=0, previous_sars=0, counterparty_risk="clean",
        correct_action="hold",
        partial_credit_actions={"escalate": 0.6, "flag": 0.4},
        requires_investigation={"verify_kyc"},
        kyc_reveal="KYC renewal submitted 10 days ago. Both accounts share same UBO. Transfer aligns with Q1 plan.",
    ),
    PayOpsTask(
        task_id="MED-003",
        difficulty="medium",
        description="Subscription payment 3× historical average. Possible upgrade billing or card compromise.",
        transaction_id="TXN-M003",
        amount=449.97,
        currency="USD",
        sender="eve@subscriber.com",
        receiver="billing@saas-platform.com",
        transaction_type="payment",
        risk_score=0.44,
        ml_confidence=0.68,
        flags=["amount_spike", "pattern_deviation"],
        velocity_1h=1, velocity_24h=1,
        avg_transaction_amount=149.99,
        account_age_days=820,
        country_risk="low", kyc_status="verified", kyc_expiry_days=150,
        previous_violations=0, previous_sars=0, counterparty_risk="clean",
        correct_action="flag",
        partial_credit_actions={"hold": 0.5, "escalate": 0.3},
        inspect_reveal="Historical monthly charge $149.99. This charge = 3-month annual upgrade. Merchant confirmation pending.",
    ),
    PayOpsTask(
        task_id="MED-004",
        difficulty="medium",
        description="Payment to regulated crypto exchange. Moderate risk. Sender has clean history.",
        transaction_id="TXN-M004",
        amount=5_000.00,
        currency="USD",
        sender="frank@personal.com",
        receiver="exchange@cryptovault.io",
        transaction_type="payment",
        risk_score=0.58,
        ml_confidence=0.74,
        flags=["crypto_exchange", "high_value"],
        velocity_1h=1, velocity_24h=2,
        avg_transaction_amount=2_000.0,
        account_age_days=1_460,
        country_risk="medium", kyc_status="verified", kyc_expiry_days=270,
        previous_violations=0, previous_sars=0, counterparty_risk="unknown",
        correct_action="flag",
        partial_credit_actions={"escalate": 0.5, "hold": 0.4},
        inspect_reveal="Exchange licensed in sender's jurisdiction. Sender made 4 similar payments over 6 months. Within limits.",
    ),
    PayOpsTask(
        task_id="MED-005",
        difficulty="medium",
        description="Expired KYC on a high-frequency corporate account. Routine transactions continue but KYC lapsed 12 days ago.",
        transaction_id="TXN-M005",
        amount=28_000.00,
        currency="GBP",
        sender="payments@logistics-uk.com",
        receiver="driver-pool@payroll.co.uk",
        transaction_type="payroll",
        risk_score=0.38,
        ml_confidence=0.82,
        flags=["kyc_expired", "high_value"],
        velocity_1h=1, velocity_24h=6,
        avg_transaction_amount=26_000.0,
        account_age_days=1_800,
        country_risk="low", kyc_status="expired", kyc_expiry_days=-12,
        previous_violations=0, previous_sars=0, counterparty_risk="clean",
        correct_action="hold",
        partial_credit_actions={"flag": 0.5, "escalate": 0.4},
        requires_investigation={"verify_kyc"},
        kyc_reveal="KYC expired 12 days ago due to administrative oversight. Re-submission in progress. No fraud indicators.",
    ),
    PayOpsTask(
        task_id="MED-006",
        difficulty="medium",
        description="Real estate advance payment. Large amount. First payment to this receiver but contract exists.",
        transaction_id="TXN-M006",
        amount=120_000.00,
        currency="USD",
        sender="buyer@realestate-client.com",
        receiver="escrow@property-agent.co",
        transaction_type="transfer",
        risk_score=0.56,
        ml_confidence=0.65,
        flags=["high_value", "first_time_payee", "large_first_transfer"],
        velocity_1h=1, velocity_24h=1,
        avg_transaction_amount=5_000.0,
        account_age_days=2_555,
        country_risk="medium", kyc_status="verified", kyc_expiry_days=90,
        previous_violations=0, previous_sars=0, counterparty_risk="unknown",
        correct_action="escalate",
        partial_credit_actions={"flag": 0.4, "hold": 0.5},
        requires_investigation={"request_docs"},
        docs_reveal="Signed purchase agreement found. Escrow agent licensed and registered. Standard conveyancing practice.",
    ),
]


# =============================================================================
# HARD  (6 tasks — adversarial, conflicting, edge-cases)
# =============================================================================

HARD_TASKS: List[PayOpsTask] = [
    PayOpsTask(
        task_id="HARD-001",
        difficulty="hard",
        description="ML model poisoning: risk_score=0.18 but manual flags scream fraud. Agent must override the model.",
        transaction_id="TXN-H001",
        amount=47_000.00,
        currency="GBP",
        sender="grace@newaccount.co.uk",
        receiver="payee@solicitor-uk.com",
        transaction_type="transfer",
        risk_score=0.18,     # poisoned — deliberately underscored
        ml_confidence=0.31,  # low confidence signals model uncertainty
        flags=["new_account_7d", "large_first_transfer", "solicitor_mule_pattern"],
        velocity_1h=1, velocity_24h=1,
        avg_transaction_amount=None,
        account_age_days=7,
        country_risk="low", kyc_status="pending", kyc_expiry_days=None,
        previous_violations=0, previous_sars=0, counterparty_risk="watchlist",
        correct_action="escalate",
        partial_credit_actions={"flag": 0.6, "hold": 0.5, "reject": 0.3},
        requires_investigation={"inspect"},
        inspect_reveal=(
            "Account 7 days old. First outbound transfer. Receiver matches solicitor-impersonation "
            "mule pattern from last month's intelligence bulletin. ML underscored — new clean account."
        ),
    ),
    PayOpsTask(
        task_id="HARD-002",
        difficulty="hard",
        description="APP scam: victim willingly sending money but has been social-engineered. Receiver is a mule.",
        transaction_id="TXN-H002",
        amount=12_500.00,
        currency="GBP",
        sender="henry@victim-bank.co.uk",
        receiver="mule_relay_2287@fastpay.io",
        transaction_type="transfer",
        risk_score=0.61,
        ml_confidence=0.77,
        flags=["app_scam_indicator", "mule_account_pattern", "first_time_payee"],
        velocity_1h=1, velocity_24h=1,
        avg_transaction_amount=300.0,
        account_age_days=3_285,
        country_risk="medium", kyc_status="verified", kyc_expiry_days=180,
        previous_violations=0, previous_sars=0, counterparty_risk="blacklist",
        correct_action="reject",
        partial_credit_actions={"escalate": 0.6, "hold": 0.5, "flag": 0.3},
        requires_investigation={"contact_sender"},
        contact_reveal=(
            "Sender says they were called by someone claiming to be their bank. "
            "Instructed to move savings to a 'safe account'. Classic APP scam confirmed."
        ),
    ),
    PayOpsTask(
        task_id="HARD-003",
        difficulty="hard",
        description="Structuring/smurfing: just-below-CTR-threshold payments from same beneficial owner across accounts.",
        transaction_id="TXN-H003",
        amount=9_450.00,    # just below $10k CTR threshold
        currency="USD",
        sender="irene_acct_A@shadow.net",
        receiver="irene_acct_B@shadow.net",
        transaction_type="transfer",
        risk_score=0.71,
        ml_confidence=0.83,
        flags=["structuring_pattern", "same_ubo", "ctr_threshold_avoidance", "high_value"],
        velocity_1h=3, velocity_24h=9,
        avg_transaction_amount=9_200.0,
        account_age_days=120,
        country_risk="high", kyc_status="failed", kyc_expiry_days=None,
        previous_violations=2, previous_sars=1, counterparty_risk="watchlist",
        correct_action="reject",
        partial_credit_actions={"escalate": 0.5, "flag": 0.2},
        requires_investigation={"inspect"},
        regulatory_action=True,
        inspect_reveal=(
            "3 transactions in 24h: $9,450 + $9,200 + $9,100 from related accounts. "
            "Same UBO. KYC failed on inconsistent ID docs. Classic CTR structuring."
        ),
    ),
    PayOpsTask(
        task_id="HARD-004",
        difficulty="hard",
        description="Legitimate FX settlement: huge amount, looks alarming — is a standard correspondent banking transfer.",
        transaction_id="TXN-H004",
        amount=4_200_000.00,
        currency="USD",
        sender="nostro@bank-a-swift.com",
        receiver="vostro@bank-b-swift.com",
        transaction_type="internal",
        risk_score=0.67,
        ml_confidence=0.58,  # model uncertain on legitimate large transfers
        flags=["high_value", "cross_border", "fx_settlement"],
        velocity_1h=8, velocity_24h=24,
        avg_transaction_amount=3_900_000.0,
        account_age_days=7_300,
        country_risk="medium", kyc_status="verified", kyc_expiry_days=400,
        previous_violations=0, previous_sars=0, counterparty_risk="clean",
        correct_action="approve",
        partial_credit_actions={"escalate": 0.5, "hold": 0.4, "flag": 0.3},
        requires_investigation={"inspect"},
        inspect_reveal=(
            "Both SWIFT BIC-verified. Part of daily USD/EUR FX settlement cycle. "
            "8 similar settlements this month, all cleared. Nostro/vostro agreement on file."
        ),
    ),
    PayOpsTask(
        task_id="HARD-005",
        difficulty="hard",
        description=(
            "Insider threat: employee of the bank initiating an unauthorised wire "
            "to their personal account, disguised as a vendor payment."
        ),
        transaction_id="TXN-H005",
        amount=22_000.00,
        currency="USD",
        sender="staff_ops@bank-internal.com",
        receiver="jake.smith.personal@gmail.com",
        transaction_type="payment",
        risk_score=0.54,
        ml_confidence=0.61,
        flags=["internal_to_personal", "unusual_beneficiary", "after_hours"],
        velocity_1h=1, velocity_24h=1,
        avg_transaction_amount=75_000.0,
        account_age_days=2_000,
        country_risk="low", kyc_status="verified", kyc_expiry_days=240,
        previous_violations=0, previous_sars=0, counterparty_risk="unknown",
        correct_action="escalate",
        partial_credit_actions={"flag": 0.5, "hold": 0.4, "reject": 0.3},
        requires_investigation={"inspect", "contact_sender"},
        inspect_reveal=(
            "Initiation time: 11:47 PM. No vendor contract matches this beneficiary. "
            "Staff member placed on PIP last week. Receiver email matches staff home address."
        ),
        contact_reveal=(
            "Staff member claims it is a legitimate vendor. Unable to provide contract reference. "
            "Story changes on follow-up. Escalate to HR and Fraud immediately."
        ),
    ),
    PayOpsTask(
        task_id="HARD-006",
        difficulty="hard",
        description=(
            "Ghost account: receiver account was dormant for 5 years and suddenly "
            "received 20 inbound transfers this week. Possible account takeover."
        ),
        transaction_id="TXN-H006",
        amount=3_200.00,
        currency="EUR",
        sender="layla@verified-sender.eu",
        receiver="old_dormant_acct_889@bank.eu",
        transaction_type="transfer",
        risk_score=0.63,
        ml_confidence=0.69,
        flags=["dormant_receiver", "sudden_activity", "high_inbound_velocity"],
        velocity_1h=1, velocity_24h=3,
        avg_transaction_amount=800.0,
        account_age_days=890,
        country_risk="medium", kyc_status="verified", kyc_expiry_days=60,
        previous_violations=0, previous_sars=0, counterparty_risk="unknown",
        correct_action="flag",
        partial_credit_actions={"hold": 0.6, "escalate": 0.5},
        requires_investigation={"inspect"},
        inspect_reveal=(
            "Receiver dormant 5 years. 20 inbound transfers this week totalling €64k. "
            "All immediately forwarded offshore. Classic money-mule account reactivation."
        ),
    ),
]


# =============================================================================
# CRITICAL  (4 tasks — multi-step chains, regulatory stakes)
# =============================================================================

CRITICAL_TASKS: List[PayOpsTask] = [
    PayOpsTask(
        task_id="CRIT-001",
        difficulty="critical",
        description=(
            "Multi-step investigation: large wire to a new counterparty. "
            "Agent must inspect logs, then request supporting documents, "
            "then make a final decision. Chain of 3 steps."
        ),
        transaction_id="TXN-C001",
        amount=375_000.00,
        currency="USD",
        sender="deal-team@pe-firm.com",
        receiver="newco@acquisition-target.io",
        transaction_type="transfer",
        risk_score=0.59,
        ml_confidence=0.55,
        flags=["high_value", "first_time_payee", "cross_border", "new_counterparty"],
        velocity_1h=1, velocity_24h=2,
        avg_transaction_amount=50_000.0,
        account_age_days=3_000,
        country_risk="medium", kyc_status="verified", kyc_expiry_days=120,
        previous_violations=0, previous_sars=0, counterparty_risk="unknown",
        correct_action="approve",
        partial_credit_actions={"escalate": 0.5, "hold": 0.4, "flag": 0.3},
        requires_investigation={"inspect", "request_docs"},
        chain_total=3,
        inspect_reveal="PE firm confirmed. Series B investment round. Deal announced in press last week.",
        docs_reveal="Signed SPA (Share Purchase Agreement) on file. Notarised. Receiver is the acquisition target.",
    ),
    PayOpsTask(
        task_id="CRIT-002",
        difficulty="critical",
        description=(
            "Fraud ring: three related accounts sending coordinated small payments "
            "that aggregate above the reporting threshold. Requires SAR filing."
        ),
        transaction_id="TXN-C002",
        amount=4_900.00,
        currency="USD",
        sender="ring_node_1@tempmail.com",
        receiver="collector_hub@offshore.net",
        transaction_type="transfer",
        risk_score=0.76,
        ml_confidence=0.80,
        flags=["fraud_ring_indicator", "coordinated_transfers", "threshold_avoidance", "high_value"],
        velocity_1h=5, velocity_24h=18,
        avg_transaction_amount=4_800.0,
        account_age_days=45,
        country_risk="high", kyc_status="failed", kyc_expiry_days=None,
        previous_violations=1, previous_sars=0, counterparty_risk="blacklist",
        correct_action="reject",
        partial_credit_actions={"escalate": 0.4, "flag": 0.2},
        requires_investigation={"inspect"},
        regulatory_action=True,
        chain_total=2,
        inspect_reveal=(
            "3 accounts (ring_node_1/2/3) sending $4,900 / $4,850 / $4,750 simultaneously. "
            "All created same day. Receiver account drained offshore within minutes. "
            "SAR filing required under BSA §5318(g)."
        ),
    ),
    PayOpsTask(
        task_id="CRIT-003",
        difficulty="critical",
        description=(
            "Trade-based money laundering: over- and under-invoiced international trade payments. "
            "Documents don't match transfer amounts. Regulatory escalation required."
        ),
        transaction_id="TXN-C003",
        amount=680_000.00,
        currency="USD",
        sender="importer@trade-co.hk",
        receiver="exporter@goods-supplier.cn",
        transaction_type="payment",
        risk_score=0.72,
        ml_confidence=0.67,
        flags=["trade_finance", "invoice_mismatch", "cross_border", "high_value"],
        velocity_1h=1, velocity_24h=3,
        avg_transaction_amount=120_000.0,
        account_age_days=730,
        country_risk="high", kyc_status="verified", kyc_expiry_days=30,
        previous_violations=1, previous_sars=0, counterparty_risk="watchlist",
        correct_action="escalate",
        partial_credit_actions={"flag": 0.4, "hold": 0.35, "reject": 0.3},
        requires_investigation={"request_docs", "inspect"},
        regulatory_action=True,
        chain_total=3,
        docs_reveal="Invoice declares 500 units @ $1,360 each = $680k. Market price is $320/unit. 4× over-invoiced.",
        inspect_reveal="Shipping records show only 200 units dispatched. Payment/goods ratio 3.4× above market norm.",
    ),
    PayOpsTask(
        task_id="CRIT-004",
        difficulty="critical",
        description=(
            "Compromised corporate account: valid credentials but geo-impossible login. "
            "Someone in Nigeria logged into a US-only account 8 minutes after the CEO logged out in NY."
        ),
        transaction_id="TXN-C004",
        amount=198_000.00,
        currency="USD",
        sender="ceo@target-corp.com",
        receiver="urgent-wire@third-party-finance.com",
        transaction_type="transfer",
        risk_score=0.81,
        ml_confidence=0.85,
        flags=["geo_impossible_login", "account_takeover_indicator", "high_value", "urgency_flag"],
        velocity_1h=1, velocity_24h=1,
        avg_transaction_amount=15_000.0,
        account_age_days=4_380,
        country_risk="high", kyc_status="verified", kyc_expiry_days=365,
        previous_violations=0, previous_sars=0, counterparty_risk="unknown",
        correct_action="reject",
        partial_credit_actions={"escalate": 0.5, "hold": 0.4},
        requires_investigation={"inspect", "contact_sender"},
        chain_total=2,
        inspect_reveal=(
            "Last login: NY (US) at 14:23. This session: Lagos (NG) at 14:31. "
            "Physical travel impossible in 8 minutes. Likely credential compromise."
        ),
        contact_reveal="CEO confirms they did NOT initiate this transfer. Account takeover confirmed.",
    ),
]


# =============================================================================
# Combine all tasks
# =============================================================================

TASKS: List[PayOpsTask] = EASY_TASKS + MEDIUM_TASKS + HARD_TASKS + CRITICAL_TASKS

TASKS_BY_ID: Dict[str, PayOpsTask] = {t.task_id: t for t in TASKS}

# ---------------------------------------------------------------------------
# Action cost table (investigation sub-actions consume budget)
# ---------------------------------------------------------------------------

ACTION_COSTS: Dict[str, float] = {
    "approve": 0.0,
    "reject": 0.0,
    "flag": 0.0,
    "escalate": 0.0,
    "hold": 0.0,
    "inspect": 0.10,
    "request_docs": 0.20,
    "verify_kyc": 0.20,
    "contact_sender": 0.30,
    "file_sar": 0.05,   # intentionally cheap — incentivise regulatory compliance
}


