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

    # --- key diagnostic flags (used for flag-identification reward shaping) ---
    key_flags: List[str] = field(default_factory=list)
    # Subset of flags that are causally decisive for the correct_action.
    # An agent that calls inspect AND those flags are in the observation
    # receives an additional flag_identification bonus in the grader.

    # --- network graph (mule chain / correspondent bank relationships) ---
    network_graph: Optional[Dict[str, str]] = None
    # Optional dict mapping roles to entity IDs, e.g.:
    #   {"sender": "acc_123", "mule": "acc_456", "offshore": "bank_789"}
    # Exposed in observation so agents can reason about chains.

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
        partial_credit_actions={},  # no partial credit — clear approve, wrong actions penalised
        inspect_reveal="Sender account 3 yrs old. Consistent transaction history. KYC fully verified.",
        key_flags=[],
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
        partial_credit_actions={"escalate": 0.4},  # flag removed — textbook fraud needs decisive action
        inspect_reveal="Receiver linked to 3 prior rejected transactions. IP on threat-intel blocklist.",
        key_flags=["sanctioned_country", "unknown_sender"],
        network_graph={"sender": "unknown_user_x99", "offshore": "offshore_acct_442"},
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
        partial_credit_actions={},  # no partial credit — tiny refund, no flags, flagging is wrong
        inspect_reveal="Standard return via web checkout. Matching order ID on file. No anomalies.",
        key_flags=[],
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
        key_flags=["velocity_breach"],
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
        key_flags=["cross_border", "high_value"],
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
        key_flags=["kyc_expiry_90d"],
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
        key_flags=["amount_spike", "pattern_deviation"],
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
        key_flags=["crypto_exchange"],
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
        key_flags=["kyc_expired"],
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
        key_flags=["first_time_payee", "large_first_transfer"],
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
        key_flags=["solicitor_mule_pattern"],
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
        key_flags=["app_scam_indicator", "mule_account_pattern"],
        network_graph={"victim": "henry@victim-bank.co.uk", "mule": "mule_relay_2287@fastpay.io"},
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
        key_flags=["structuring_pattern", "ctr_threshold_avoidance"],
        network_graph={"node_a": "irene_acct_A@shadow.net", "node_b": "irene_acct_B@shadow.net", "ubo": "irene_ubo"},
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
        key_flags=["fx_settlement"],
        network_graph={"correspondent": "nostro@bank-a-swift.com", "settlement": "vostro@bank-b-swift.com"},
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
        key_flags=["internal_to_personal", "unusual_beneficiary"],
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
        key_flags=["dormant_receiver", "sudden_activity"],
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
        key_flags=["first_time_payee", "new_counterparty"],
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
        ),        key_flags=["fraud_ring_indicator", "coordinated_transfers"],
        network_graph={"ring_1": "ring_node_1@tempmail.com", "ring_2": "ring_node_2", "collector": "collector_hub@offshore.net"},    ),
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
        key_flags=["invoice_mismatch", "trade_finance"],
        network_graph={"importer": "importer@trade-co.hk", "exporter": "exporter@goods-supplier.cn"},
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
        key_flags=["geo_impossible_login", "account_takeover_indicator"],
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

# ---------------------------------------------------------------------------
# Task variant pools — anti-memorisation & investigation-gating
# ---------------------------------------------------------------------------
# Each entry is a list of override dicts for alternative scenarios of that task.
# Variant index 0 always means "use base task" (no overrides applied).
# Variants 1..N override the fields listed, so the correct_action is only
# determinable after performing the required investigation sub-actions.
# The variant is selected deterministically from the episode seed in environment.py.
#
# Key design constraint: the BASE observation (pre-investigation) must be
# genuinely ambiguous — investigation reveals ARE the decisive evidence.
# ---------------------------------------------------------------------------
TASK_VARIANTS: Dict[str, List[Dict]] = {

    # ── MEDIUM ────────────────────────────────────────────────────────────

    "MED-003": [
        # v1: unauthorised recurring billing → hold to freeze
        {
            "correct_action": "hold",
            "partial_credit_actions": {"flag": 0.5, "escalate": 0.3},
            "inspect_reveal": (
                "Merchant billing records show 3 prior declined charges this month "
                "from different cards linked to the same subscriber IP. Pattern is "
                "consistent with credential stuffing / unauthorised recurring charge. "
                "Freeze payment and contact customer before releasing."
            ),
        },
        # v2: confirmed annual plan upgrade → approve
        {
            "correct_action": "approve",
            "partial_credit_actions": {"flag": 0.3},
            "inspect_reveal": (
                "Customer service log: subscriber upgraded to annual plan via phone "
                "this morning. $449.97 = 3 × monthly fee (annual discount applied). "
                "Merchant has confirmed the charge. Safe to approve."
            ),
        },
    ],

    "MED-004": [
        # v1: structuring pattern across exchanges → escalate
        {
            "correct_action": "escalate",
            "partial_credit_actions": {"flag": 0.5, "hold": 0.3},
            "inspect_reveal": (
                "Sender has made 7 crypto exchange payments in 5 days totalling $28k "
                "across 3 different platforms. Amounts stay just below exchange reporting "
                "limits. Pattern matches structuring. Escalate for AML review."
            ),
        },
        # v2: clean recurring investment → approve
        {
            "correct_action": "approve",
            "partial_credit_actions": {"flag": 0.3},
            "inspect_reveal": (
                "Sender's 3-year payment history shows consistent quarterly crypto "
                "purchases of similar size. Exchange is fully regulated and KYC-verified. "
                "Amount within personal investment limits. No structuring indicators. Approve."
            ),
        },
    ],

    "MED-005": [
        # v1: KYC discrepancy found → escalate
        {
            "correct_action": "escalate",
            "partial_credit_actions": {"flag": 0.5, "hold": 0.3},
            "kyc_reveal": (
                "KYC expired because the compliance team found discrepancies between "
                "the business registration address and Companies House records. The "
                "case has been flagged for investigation by the KYC team. Escalate — "
                "do not approve or hold without senior sign-off."
            ),
        },
        # v2: KYC renewed, routine payroll → approve
        {
            "correct_action": "approve",
            "partial_credit_actions": {"hold": 0.4},
            "kyc_reveal": (
                "KYC renewal was submitted 10 days ago and completed processing "
                "yesterday. All checks passed. The lapse was an administrative "
                "oversight only. Payroll has an 18-month history with no anomalies. "
                "Approve and process."
            ),
        },
    ],

    "MED-006": [
        # v1: forged purchase agreement → reject
        {
            "correct_action": "reject",
            "partial_credit_actions": {"escalate": 0.4, "flag": 0.3},
            "docs_reveal": (
                "Purchase agreement is a forgery: the notary seal serial number "
                "does not exist in the notary registry; the property title number "
                "returns no match in Land Registry; the 'escrow agent' is not "
                "registered with the Council for Licensed Conveyancers. "
                "Classic conveyancing fraud — reject immediately."
            ),
        },
        # v2: title dispute — hold pending resolution
        {
            "correct_action": "hold",
            "partial_credit_actions": {"escalate": 0.5, "flag": 0.3},
            "docs_reveal": (
                "SPA is authentic and escrow agent is licensed. However, a co-owner "
                "has filed a title dispute on the property. The buyer's solicitor "
                "advises funds should be held in escrow until the dispute is resolved "
                "(expected 3–4 weeks). Hold pending legal clearance."
            ),
        },
    ],

    # ── HARD ──────────────────────────────────────────────────────────────

    "HARD-001": [
        # v1: deeper forensics confirm active fraud ring → reject
        {
            "correct_action": "reject",
            "partial_credit_actions": {"escalate": 0.5, "flag": 0.3},
            "inspect_reveal": (
                "Cross-database query: same receiver account (payee@solicitor-uk.com) "
                "appeared in 4 prior rejected fraud wires this quarter, total £190k. "
                "Funds clear to offshore within 45 min each time. ML score was "
                "deliberately poisoned via clean-account seeding. Reject immediately."
            ),
        },
        # v2: flags are false positives, ML score correct → approve
        {
            "correct_action": "approve",
            "partial_credit_actions": {"escalate": 0.4, "flag": 0.3},
            "inspect_reveal": (
                "Manual flag investigation: 'solicitor_mule_pattern' rule mis-fired "
                "on a legitimate SRA-registered firm (reg. SRA-443210) that rebranded "
                "last month, causing a new account. 'New_account_7d' flag triggered "
                "by the rebrand. ALL flags are false positives. ML risk score 0.18 is "
                "accurate. Safe to approve."
            ),
        },
    ],

    "HARD-002": [
        # v1: new legitimate supplier, flags were incorrect → approve
        {
            "correct_action": "approve",
            "partial_credit_actions": {"flag": 0.4, "hold": 0.3},
            "contact_reveal": (
                "Sender confirmed: mule_relay_2287@fastpay.io is their new supplier "
                "in Singapore. The VPN flagged by geo-risk monitoring is the company's "
                "standard corporate security policy. Purchase order #PO-44821 is on "
                "file matching the amount. Safe to approve."
            ),
        },
        # v2: inconsistent story, AML review needed → escalate
        {
            "correct_action": "escalate",
            "partial_credit_actions": {"reject": 0.4, "flag": 0.3},
            "contact_reveal": (
                "Contacted sender: story is vague and inconsistent across two "
                "follow-up calls. Cannot confirm supplier identity. No purchase order "
                "provided. The receiver's business registration cannot be located. "
                "Cannot approve or reject — escalate to AML for investigation."
            ),
        },
    ],

    "HARD-003": [
        # v1: confirmed structuring but scale needs MLRO → escalate
        {
            "correct_action": "escalate",
            "partial_credit_actions": {"reject": 0.5, "flag": 0.3},
            "inspect_reveal": (
                "Pattern strongly resembles CTR structuring. However, the total "
                "aggregate ($27,750) and the involvement of 3 related entities "
                "means this crosses the threshold for mandatory MLRO referral under "
                "internal policy. Escalate rather than unilaterally reject."
            ),
        },
        # v2: same UBO transfers are authorised corporate restructuring → hold
        {
            "correct_action": "hold",
            "partial_credit_actions": {"reject": 0.4, "escalate": 0.5},
            "inspect_reveal": (
                "Same-UBO transfers are documented in a board resolution dated last "
                "week authorising an internal corporate restructuring. KYC failure "
                "was due to an ID document resubmission still in queue. Structuring "
                "indicators are coincidental. Hold pending KYC re-verification."
            ),
        },
    ],

    "HARD-004": [
        # v1: unrecognised SWIFT BIC, potential impersonation → reject
        {
            "correct_action": "reject",
            "partial_credit_actions": {"escalate": 0.5, "flag": 0.3},
            "inspect_reveal": (
                "SWIFT BIC vostro@bank-b-swift.com resolved to a bank that does NOT "
                "appear in the correspondent banking agreement registry. The BIC is "
                "visually similar to a legitimate bank (one character substituted). "
                "Likely impersonation of a correspondent partner — reject and alert."
            ),
        },
        # v2: ghost employees detected → flag for payroll audit
        {
            "correct_action": "flag",
            "partial_credit_actions": {"approve": 0.5, "hold": 0.4},
            "inspect_reveal": (
                "FX settlement accounts are SWIFT-verified and legitimate. However, "
                "the initiating staff workflow shows this $4.2M wire was split from "
                "a larger batch that included 2 non-existent internal accounts. "
                "Flag for payroll/treasury audit before releasing."
            ),
        },
    ],

    "HARD-005": [
        # v1: confirmed insider fraud, no response from staff → reject
        {
            "correct_action": "reject",
            "partial_credit_actions": {"escalate": 0.5, "flag": 0.3},
            "contact_reveal": (
                "Staff member's phone is disconnected; email bounced. HR confirms "
                "the employee was placed on garden leave 2 days ago pending "
                "investigation. Receiver account drained offshore within hours of "
                "prior similar payments. Insider fraud confirmed — reject and freeze."
            ),
        },
        # v2: authorized vendor payment, staff working late → approve
        {
            "correct_action": "approve",
            "partial_credit_actions": {"escalate": 0.4, "flag": 0.3},
            "contact_reveal": (
                "Staff member confirmed: personal Gmail was added as a forwarding "
                "alias last month when the vendor switched to a personal invoicing "
                "service. Contract reference VC-2024-189 verified by procurement. "
                "After-hours initiation matches remote work schedule. Approve."
            ),
        },
    ],

    "HARD-006": [
        # v1: confirmed active mule network → reject
        {
            "correct_action": "reject",
            "partial_credit_actions": {"escalate": 0.5, "flag": 0.4},
            "inspect_reveal": (
                "Receiver account confirmed to be receiving from 12 other dormant "
                "accounts this week (total €78k). All funds exit to the same offshore "
                "cluster within 2 hours. Active money-mule aggregation network. "
                "Reject and file SAR immediately."
            ),
        },
        # v2: estate probate settlement, legitimate size → escalate
        {
            "correct_action": "escalate",
            "partial_credit_actions": {"flag": 0.6, "hold": 0.4},
            "inspect_reveal": (
                "Account was frozen under a probate order that was lifted 3 days ago. "
                "Estate settlement documentation is on file. Amount (€3,200) is "
                "consistent with a partial distribution from the estate. Legitimate "
                "reactivation — escalate for senior approval given the reactivation flag."
            ),
        },
    ],

    # ── CRITICAL ──────────────────────────────────────────────────────────

    "CRIT-001": [
        # v1: deepfake deal, BEC fraud → reject
        {
            "correct_action": "reject",
            "partial_credit_actions": {"escalate": 0.4, "hold": 0.35, "flag": 0.3},
            "inspect_reveal": (
                "PE firm name verified but domain is a typo-squatting lookalike "
                "(pe-flrm.com vs pe-firm.com). No SEC filing exists for this deal. "
                "The 'press announcement' URL leads to a page created 6 days ago. "
                "Classic BEC / CEO impersonation fraud."
            ),
            "docs_reveal": (
                "'Signed SPA' is a doctored template — notarization certificate "
                "serial number LN-2024-00512 does not exist in the notary registry. "
                "Acquisition is fabricated. Reject and alert security team."
            ),
        },
        # v2: legitimate deal, wrong destination account → hold
        {
            "correct_action": "hold",
            "partial_credit_actions": {"approve": 0.4, "escalate": 0.5},
            "inspect_reveal": (
                "PE firm and deal confirmed as genuine (SEC filing #0001234567). "
                "Press announcement verified with three external sources. Acquisition "
                "is legitimate. However, the destination account does not match the "
                "independent escrow account specified in clause 7.3 of the SPA."
            ),
            "docs_reveal": (
                "SPA authenticated by legal team. Per clause 7.3, all milestone "
                "payments must route to independent escrow account "
                "escrow@trustco-escrow.com, not directly to the acquisition target. "
                "Hold and redirect to the correct escrow account."
            ),
        },
    ],

    "CRIT-002": [
        # v1: same SME, multiple business accounts — coincidental threshold → hold
        {
            "correct_action": "hold",
            "partial_credit_actions": {"reject": 0.5, "escalate": 0.4},
            "inspect_reveal": (
                "The 3 flagged accounts all belong to the same registered SME "
                "(Companies House reg. 09876543) using separate business accounts "
                "for different cost centres. The $4,900 pattern matches their "
                "internal expense approval limit — not structuring intent. "
                "KYC re-verification recommended before releasing. Hold."
            ),
        },
        # v2: ring is larger (9 accounts), MLRO escalation needed → escalate
        {
            "correct_action": "escalate",
            "partial_credit_actions": {"reject": 0.5, "flag": 0.3},
            "inspect_reveal": (
                "Network analysis reveals the ring involves 9 accounts total (not 3), "
                "all created on the same day, all funnelling to the same offshore "
                "collector. Aggregate total is $42k. Volume and complexity require "
                "MLRO involvement and a formal SAR — escalate rather than unilaterally reject."
            ),
        },
    ],

    "CRIT-003": [
        # v1: legitimate specialized goods, market price correct → approve
        {
            "correct_action": "approve",
            "partial_credit_actions": {"escalate": 0.5, "hold": 0.3},
            "docs_reveal": (
                "Independent commodity valuation confirms: these are custom precision "
                "industrial components for semiconductor manufacturing. Verified market "
                "price: $1,310–1,420 per unit. Invoice price of $1,360 is within "
                "normal range. No mismatch. Documentation is clean."
            ),
            "inspect_reveal": (
                "Full shipping manifest verified: 500 units dispatched per bill of "
                "lading BoL-HK-2024-8821. Customs clearance certified in both "
                "Hong Kong and the destination country. No discrepancies."
            ),
        },
        # v2: criminal-scale TBML, reject immediately
        {
            "correct_action": "reject",
            "partial_credit_actions": {"escalate": 0.5, "flag": 0.3},
            "docs_reveal": (
                "Invoice declares 500 units @ $1,360 = $680k. Independent valuation: "
                "market price is $250/unit. Over-invoiced by 5.4×. Matches a "
                "trade-based money laundering typology filed with FATF last year. "
                "Criminal-level TBML — reject and file SAR immediately."
            ),
            "inspect_reveal": (
                "Shipping records show only 100 units dispatched vs 500 invoiced. "
                "Invoice-to-goods ratio 5.4× above market. Direct evidence of false "
                "customs declarations. Reject immediately."
            ),
        },
    ],

    "CRIT-004": [
        # v1: CEO delegated to CFO who was in Lagos — hold for confirmation
        {
            "correct_action": "hold",
            "partial_credit_actions": {"escalate": 0.5, "reject": 0.4},
            "inspect_reveal": (
                "Geo-impossible login flagged. However, a delegation email from the "
                "CEO was sent 2 hours ago via their verified corporate account granting "
                "the CFO emergency signing authority for this exact transfer. "
                "CFO was in Lagos for a board meeting (flight records available). "
            ),
            "contact_reveal": (
                "CEO confirmed via secure callback: CFO had full authority to initiate "
                "this transfer. Delegation is documented in the board minutes. "
                "Hold briefly to verify the papertrail, then release."
            ),
        },
        # v2: CEO unreachable, dual-approval controls met, escalate
        {
            "correct_action": "escalate",
            "partial_credit_actions": {"reject": 0.5, "hold": 0.4},
            "inspect_reveal": (
                "Account takeover strongly suspected. CEO's registered number is "
                "unreachable (OOO message). However, the transfer went through the "
                "standard dual-approval workflow with two internal approvers (both "
                "records look genuine). Outcome ambiguous — escalate to Fraud "
                "Operations for emergency investigation."
            ),
            "contact_reveal": (
                "Cannot reach CEO. Deputy CFO states they approved the transfer but "
                "cannot confirm whether the CEO or an impersonator initiated it. "
                "Escalate immediately to Fraud Ops and place a temporary hold."
            ),
        },
    ],
}


