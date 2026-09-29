"""Sentinel engine — Phase 2.
Adds: real evaluation harness, hard negatives, multi-destination structuring,
all-amounts circular-transfer graph, temporal+weighted correlation, synthesized
narrative, employee-specific off-hours, dynamic entitlements for inject.
"""
import random, json, os, urllib.request, datetime as dt
from collections import defaultdict
import numpy as np, networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

# ── Constants ────────────────────────────────────────────────────────────────
T0 = dt.datetime(2026, 9, 21, 9, 0)
H  = dt.timedelta(hours=1)
f  = lambda t: t.strftime("%d %b %H:%M")

# Configurable detector thresholds
STRUCTURING_THRESHOLD      = 200_000   # reporting threshold (₹)
STRUCTURING_PROXIMITY_RATIO = 0.80     # flag if amt >= 80% of threshold
STRUCTURING_COUNT_MIN      = 3         # minimum transfers in window
STRUCTURING_WINDOW_HOURS   = 24        # rolling window for grouping
CIRCULAR_MIN_HOPS          = 3         # minimum cycle length
CIRCULAR_WINDOW_HOURS      = 24
CIRCULAR_MIN_AMT           = 50_000    # minimum transaction amount to include in cycle graph
INSIDER_AMOUNT_THRESHOLD   = 100_000
INSIDER_WINDOW_MINUTES     = 60
CORRELATION_WINDOW_HOURS   = 48        # signals >48 h apart → separate cases

# ── Signal weight table ───────────────────────────────────────────────────────
# Higher weight = stronger contribution to risk escalation
SIGNAL_WEIGHTS = {
    "UNAUTHORIZED_ACCESS": 3,
    "ACCESS_CHAIN":        3,
    "INSIDER_LINK":        3,
    "CIRCULAR_TRANSFER":   3,
    "STRUCTURING":         2,
    "PROFILE_MISMATCH":    2,
    "OFF_HOURS_ACCESS":    1,
}

# ── Domain model ─────────────────────────────────────────────────────────────
EMPLOYEES = {
    "EMP1": dict(employee_id="EMP1", name="Anil Deshmukh",  role="Branch Manager",       department="Branch Ops",  hours=(8, 18)),
    "EMP2": dict(employee_id="EMP2", name="Priya Menon",    role="Teller",               department="Branch Ops",  hours=(9, 17)),
    "EMP3": dict(employee_id="EMP3", name="Raj Kapoor",     role="Relationship Manager",  department="Wealth Mgmt", hours=(9, 18)),
    "EMP4": dict(employee_id="EMP4", name="Sunita Rao",     role="Operations Analyst",    department="Operations",  hours=(10, 19)),
    "EMP5": dict(employee_id="EMP5", name="Vikram Singh",   role="Audit / Compliance",    department="Compliance",  hours=(8, 20)),
}

CUSTOMERS = {
    "CUST1001": dict(customer_id="CUST1001", name="Meera Patel",       customer_type="INDIVIDUAL"),
    "CUST1002": dict(customer_id="CUST1002", name="Arjun Reddy",       customer_type="INDIVIDUAL"),
    "CUST1005": dict(customer_id="CUST1005", name="Kavitha Nair",      customer_type="INDIVIDUAL"),
    "CUST1010": dict(customer_id="CUST1010", name="Deepak Sharma",     customer_type="INDIVIDUAL"),
    "CUST1040": dict(customer_id="CUST1040", name="Farooque Trading",  customer_type="BUSINESS"),
    "CUST1045": dict(customer_id="CUST1045", name="Zain Exports",      customer_type="BUSINESS"),
    "CUST1050": dict(customer_id="CUST1050", name="Lakshmi Iyer",      customer_type="INDIVIDUAL"),
    "CUST1051": dict(customer_id="CUST1051", name="Nandini Das",       customer_type="INDIVIDUAL"),
    "CUST_CORP": dict(customer_id="CUST_CORP", name="MegaCorp Industries", customer_type="CORPORATE"),
}

ACCOUNTS = {
    "AC1001": dict(account_id="AC1001", owner_customer_id="CUST1001", account_type="SAVINGS"),
    "AC1002": dict(account_id="AC1002", owner_customer_id="CUST1002", account_type="SAVINGS"),
    "AC1005": dict(account_id="AC1005", owner_customer_id="CUST1005", account_type="SAVINGS"),
    "AC1006": dict(account_id="AC1006", owner_customer_id="CUST1005", account_type="CURRENT"),
    "AC1010": dict(account_id="AC1010", owner_customer_id="CUST1010", account_type="SAVINGS"),
    "AC1040": dict(account_id="AC1040", owner_customer_id="CUST1040", account_type="CURRENT"),
    "AC1045": dict(account_id="AC1045", owner_customer_id="CUST1045", account_type="CURRENT"),
    "AC1050": dict(account_id="AC1050", owner_customer_id="CUST1050", account_type="SAVINGS"),
    "AC1051": dict(account_id="AC1051", owner_customer_id="CUST1051", account_type="SAVINGS"),
    "AC1052": dict(account_id="AC1052", owner_customer_id="CUST1050", account_type="CURRENT"),
    "AC1053": dict(account_id="AC1053", owner_customer_id="CUST1051", account_type="CURRENT"),
    "AC1054": dict(account_id="AC1054", owner_customer_id="CUST1051", account_type="SAVINGS"),
    "CORP1":  dict(account_id="CORP1",  owner_customer_id="CUST_CORP", account_type="CORPORATE"),
}

# Entitlements: employee → [{accounts, actions, source}]
ENTITLEMENTS = {
    "EMP1": [dict(accounts=["AC1001","AC1002","AC1005","AC1006","AC1010"],
                  actions=["VIEW","EDIT_PROFILE","EDIT_LIMIT"], source="Branch Manager override")],
    "EMP2": [dict(accounts=["AC1010"],
                  actions=["VIEW","EDIT_PROFILE"], source="Assigned teller")],
    "EMP3": [dict(accounts=["AC1001","AC1002"],
                  actions=["VIEW","EDIT_PROFILE"], source="Relationship Manager portfolio")],
    "EMP4": [dict(accounts=["AC1050"],
                  actions=["VIEW","EDIT_PROFILE"], source="Operations assignment")],
    "EMP5": [dict(accounts="ALL",
                  actions=["VIEW"], source="Compliance read-only audit access")],
}

# Ring accounts — set by gen() and extended by inject()
RING = [f"AC{n}" for n in range(1050, 1055)]
BAD  = set(RING) | {"AC1040", "AC1045"}   # ground truth for legacy evaluate()


# ── Entitlement helpers ───────────────────────────────────────────────────────

def is_authorized(employee_id, account_id, action):
    """Return structured authorization result from ENTITLEMENTS table."""
    emp  = EMPLOYEES.get(employee_id)
    role = emp["role"] if emp else "Unknown"
    rules = ENTITLEMENTS.get(employee_id, [])
    for ent in rules:
        acct_match   = (ent["accounts"] == "ALL") or (account_id in ent["accounts"])
        action_match = action in ent["actions"]
        if acct_match and action_match:
            return dict(allowed=True, employee_id=employee_id, account_id=account_id,
                        action=action, role=role, entitlement_source=ent["source"], reason="Authorized")
    return dict(allowed=False, employee_id=employee_id, account_id=account_id,
                action=action, role=role,
                reason=f"No entitlement for {action} on {account_id}")


def _grant_temp_entitlement(employee_id, account_id, actions, source):
    """Add a temporary entitlement entry (used by inject). Idempotent."""
    rules = ENTITLEMENTS.setdefault(employee_id, [])
    # Avoid duplicating if already present
    for r in rules:
        if r.get("accounts") != "ALL" and account_id in r.get("accounts", []):
            return
    rules.append(dict(accounts=[account_id], actions=actions, source=source))


# ── Synthetic data generation ─────────────────────────────────────────────────

def gen(seed=7):
    r   = random.Random(seed)
    inc = {f"AC{1000+i}": r.choice([25000, 40000, 60000, 90000, 150000]) for i in range(60)}
    for a in RING: inc[a] = 25000
    inc["AC1040"] = 40000; inc["CORP1"] = 500_000
    accts = [a for a in inc if a != "CORP1"]; tx = []
    add = lambda s, d, a, t: tx.append(
        dict(id=f"T{len(tx)+1:04d}", src=s, dst=d, amt=round(a), ts=t))

    # ── Normal retail traffic
    for _ in range(450):
        s, d = r.sample(accts, 2)
        add(s, d, min(inc[s] * r.uniform(.03, .4), 60000),
            T0 + dt.timedelta(minutes=r.randint(0, 7 * 1440)))

    # ── LEGIT hard-negative #1: weekly payroll CORP1 → many accounts
    for a in accts[:12]:
        for k in (0, 6): add("CORP1", a, inc[a], T0 + dt.timedelta(days=k, hours=1))

    # ── LEGIT hard-negative #2: recurring family transfer AC1005→AC1006 (rent)
    for k in range(3):
        add("AC1005", "AC1006", 15000, T0 + dt.timedelta(days=2*k, hours=5))

    # ── LEGIT hard-negative #3: authorized employee + large legitimate outflow
    # EMP1 edits AC1001 (authorized); AC1001 sends large transfer 2 days later
    add("AC1001", "AC1002", 120_000, T0 + dt.timedelta(days=2, hours=10, minutes=30))

    # ── LEGIT hard-negative #4: near-threshold recurring payment around ₹190k
    # AC1002 pays rent to AC1005 three times — similar amount but legitimate recurring
    for k in range(3):
        add("AC1002", "AC1005", 188_000 + k * 500,
            T0 + dt.timedelta(days=7 + k * 30, hours=2))

    # ── LEGIT hard-negative #5: small legitimate loop (family)
    # AC1001 → AC1002 → AC1005 → AC1001, each only ₹8k — below suspicious threshold
    base_fam = T0 + dt.timedelta(days=5, hours=14)
    add("AC1001", "AC1002",  8_000, base_fam)
    add("AC1002", "AC1005",  8_000, base_fam + dt.timedelta(hours=2))
    add("AC1005", "AC1001",  8_000, base_fam + dt.timedelta(hours=4))

    # ── SUSPICIOUS scenario 1: laundering ring + insider
    base = T0 + dt.timedelta(days=3, hours=1)
    for i in range(5):
        add(RING[i], RING[(i+1)%5], 480_000 - i*8_000,
            base + dt.timedelta(minutes=31 + 22*i))

    # ── SUSPICIOUS scenario 2a: classic same-pair structuring
    for i in range(3):
        add("AC1040", "AC1045", 190_000 + i*1_500,
            T0 + dt.timedelta(days=4, hours=3, minutes=50*i))

    # ── SUSPICIOUS scenario 2b: multi-destination structuring from AC1040
    # AC1040 → AC1001, AC1002, AC1010 each just under threshold, same day
    for j, dst in enumerate(["AC1001", "AC1002", "AC1010"]):
        add("AC1040", dst, 191_000 + j*1_000,
            T0 + dt.timedelta(days=4, hours=8, minutes=j*40))

    # ── Access logs
    logs = [
        # SUSPICIOUS: EMP3 VIEW→VIEW→EDIT_LIMIT on RING[0] = AC1050 (unauthorized)
        dict(emp="EMP3", action="VIEW",       acct=RING[0], ts=base + dt.timedelta(minutes=1)),
        dict(emp="EMP3", action="VIEW",       acct=RING[0], ts=base + dt.timedelta(minutes=3)),
        dict(emp="EMP3", action="EDIT_LIMIT", acct=RING[0], ts=base + dt.timedelta(minutes=5)),
        # LEGIT: EMP2 authorized edit on AC1010
        dict(emp="EMP2", action="EDIT_PROFILE", acct="AC1010",
             ts=T0 + dt.timedelta(days=2)),
        # LEGIT: EMP1 authorized edit on AC1001 at 09:00 on day 2 (within EMP1's 08:00-18:00 schedule)
        dict(emp="EMP1", action="EDIT_LIMIT", acct="AC1001",
             ts=T0 + dt.timedelta(days=2, hours=0)),
        # LEGIT: EMP5 heavy audit reads — 300 VIEW events, must not flood alerts
    ]
    for _ in range(300):
        logs.append(dict(emp="EMP5", action="VIEW",
                         acct=r.choice(accts),
                         ts=T0 + dt.timedelta(minutes=r.randint(0, 7*1440))))

    return inc, sorted(tx, key=lambda t: t["ts"]), sorted(logs, key=lambda l: l["ts"])


# ── Phase 2: Evaluation harness with explicit ground truth ───────────────────

GROUND_TRUTH_SCENARIOS = [
    # ── Suspicious ──────────────────────────────────────────────────────────
    dict(scenario_id="SUSP_CIRCULAR_01",
         label="SUSPICIOUS", expected_detection=True, category="circular_transfer",
         description="5-hop circular transfer: RING accounts AC1050–AC1054 pass ₹480k–440k in a loop within 1.5h.",
         detection_criteria=lambda alerts: any(
             "CIRCULAR_TRANSFER" in a["kinds"] and
             len(set(a["accounts"]) & set(RING)) >= 4
             for a in alerts if a["risk"] in ("HIGH","MEDIUM"))),

    dict(scenario_id="SUSP_STRUCTURING_PAIR_02",
         label="SUSPICIOUS", expected_detection=True, category="structuring",
         description="Classic same-pair structuring: AC1040→AC1045, 3 transfers of ₹190k–193k.",
         detection_criteria=lambda alerts: any(
             "STRUCTURING" in a["kinds"] and
             "AC1040" in a["accounts"] and "AC1045" in a["accounts"]
             for a in alerts if a["risk"] in ("HIGH","MEDIUM"))),

    dict(scenario_id="SUSP_STRUCTURING_MULTIDEST_03",
         label="SUSPICIOUS", expected_detection=True, category="structuring",
         description="Multi-destination structuring: AC1040→AC1001/AC1002/AC1010, each ~₹191-193k in same day.",
         detection_criteria=lambda alerts: any(
             "STRUCTURING" in a["kinds"] and "AC1040" in a["accounts"]
             and len(set(a["accounts"]) - {"AC1040"}) >= 2
             for a in alerts if a["risk"] in ("HIGH","MEDIUM"))),

    dict(scenario_id="SUSP_UNAUTHORIZED_EDIT_04",
         label="SUSPICIOUS", expected_detection=True, category="insider",
         description="EMP3 performs EDIT_LIMIT on AC1050 which is outside their entitlement.",
         detection_criteria=lambda alerts: any(
             "UNAUTHORIZED_ACCESS" in a["kinds"] and a.get("employee") == "EMP3"
             for a in alerts)),

    dict(scenario_id="SUSP_ACCESS_CHAIN_05",
         label="SUSPICIOUS", expected_detection=True, category="insider",
         description="EMP3: VIEW→VIEW→EDIT_LIMIT on AC1050 then AC1050 sends ₹480k 26 min later.",
         detection_criteria=lambda alerts: any(
             "ACCESS_CHAIN" in a["kinds"] and a.get("employee") == "EMP3"
             for a in alerts)),

    dict(scenario_id="SUSP_PROFILE_MISMATCH_06",
         label="SUSPICIOUS", expected_detection=True, category="profile",
         description="RING accounts (AC1050–1054) each move 20x declared income; Isolation Forest should flag them.",
         detection_criteria=lambda alerts: any(
             "PROFILE_MISMATCH" in a["kinds"] and
             len(set(a["accounts"]) & set(RING)) >= 2
             for a in alerts)),

    dict(scenario_id="SUSP_MIXED_CORRELATED_07",
         label="SUSPICIOUS", expected_detection=True, category="mixed",
         description="AC1040 is flagged for both structuring and profile mismatch — correlated multi-signal case.",
         detection_criteria=lambda alerts: any(
             "STRUCTURING" in a["kinds"] and "PROFILE_MISMATCH" in a["kinds"] and
             "AC1040" in a["accounts"]
             for a in alerts)),

    # ── Legitimate / Hard negatives ─────────────────────────────────────────
    dict(scenario_id="LEGIT_BUSY_TELLER_08",
         label="LEGITIMATE", expected_detection=False, category="hard_negative",
         description="EMP5 performs 300 VIEW-only audit reads across all accounts. Should NOT trigger insider alerts.",
         detection_criteria=lambda alerts: any(
             a.get("employee") == "EMP5" and
             any(k in a["kinds"] for k in ("INSIDER_LINK","ACCESS_CHAIN","UNAUTHORIZED_ACCESS"))
             and a["risk"] == "HIGH"
             for a in alerts)),

    dict(scenario_id="LEGIT_FAMILY_LOOP_09",
         label="LEGITIMATE", expected_detection=False, category="hard_negative",
         description="AC1001→AC1002→AC1005→AC1001, each ₹8k. Low-value loop must not trigger CIRCULAR_TRANSFER HIGH.",
         detection_criteria=lambda alerts: any(
             "CIRCULAR_TRANSFER" in a["kinds"] and
             {"AC1001","AC1002","AC1005"} <= set(a["accounts"]) and
             a["risk"] == "HIGH"
             for a in alerts)),

    dict(scenario_id="LEGIT_RECURRING_PAYMENT_10",
         label="LEGITIMATE", expected_detection=False, category="hard_negative",
         description="AC1002→AC1005 pays ₹188-189k monthly (3 times over 3 months). Too spread out to be structuring.",
         detection_criteria=lambda alerts: any(
             "STRUCTURING" in a["kinds"] and
             "AC1002" in a["accounts"] and "AC1005" in a["accounts"] and
             a["risk"] in ("HIGH","MEDIUM")
             for a in alerts)),

    dict(scenario_id="LEGIT_AUTHORIZED_EDIT_11",
         label="LEGITIMATE", expected_detection=False, category="hard_negative",
         description="EMP1 authorized EDIT_LIMIT on AC1001 (within entitlement); AC1001 sends ₹120k 1.5h later. Should NOT be flagged as insider.",
         detection_criteria=lambda alerts: any(
             a.get("employee") == "EMP1" and
             any(k in a["kinds"] for k in ("INSIDER_LINK","ACCESS_CHAIN","UNAUTHORIZED_ACCESS"))
             and a["risk"] == "HIGH"
             for a in alerts)),

    dict(scenario_id="LEGIT_PAYROLL_12",
         label="LEGITIMATE", expected_detection=False, category="hard_negative",
         description="CORP1 weekly payroll to 12 accounts: identical repeated payments. Should be auto-triaged or LOW.",
         detection_criteria=lambda alerts: any(
             "CORP1" in a["accounts"] and a["risk"] == "HIGH"
             for a in alerts)),
]


def run_evaluation(alerts):
    """
    Run all GROUND_TRUTH_SCENARIOS against live alert list.
    Returns structured evaluation with TP/FP/FN/TN/Precision/Recall/FPR/F1.
    Ground truth is defined BY SCENARIO LABEL, not by detector output.
    """
    results = []
    for sc in GROUND_TRUTH_SCENARIOS:
        detected = sc["detection_criteria"](alerts)
        label    = sc["label"]   # SUSPICIOUS or LEGITIMATE
        expected = sc["expected_detection"]

        if label == "SUSPICIOUS":
            if detected:   outcome = "TP"
            else:          outcome = "FN"
        else:  # LEGITIMATE
            if detected:   outcome = "FP"
            else:          outcome = "TN"

        results.append(dict(
            scenario_id=sc["scenario_id"],
            label=label,
            category=sc["category"],
            description=sc["description"],
            expected_detection=expected,
            detected=detected,
            outcome=outcome,
        ))

    suspicious = [r for r in results if r["label"] == "SUSPICIOUS"]
    legitimate = [r for r in results if r["label"] == "LEGITIMATE"]
    TP = sum(1 for r in results if r["outcome"] == "TP")
    FP = sum(1 for r in results if r["outcome"] == "FP")
    FN = sum(1 for r in results if r["outcome"] == "FN")
    TN = sum(1 for r in results if r["outcome"] == "TN")

    precision = round(TP / max(TP + FP, 1), 3)
    recall    = round(TP / max(TP + FN, 1), 3)
    fpr       = round(FP / max(FP + TN, 1), 3)
    f1        = round(2 * precision * recall / max(precision + recall, 1e-9), 3)

    return dict(
        scenarios=results,
        counts=dict(
            total=len(results),
            suspicious=len(suspicious),
            legitimate=len(legitimate),
            TP=TP, FP=FP, FN=FN, TN=TN,
        ),
        metrics=dict(
            precision=precision,
            recall=recall,
            fpr=fpr,
            f1=f1,
        ),
        # Legacy evaluate() fields preserved for API backward compat
        alerts=len(alerts),
        actionable=len([a for a in alerts if a["risk"] in ("HIGH","MEDIUM")]),
        ring_found=any("CIRCULAR_TRANSFER" in a["kinds"] and
                       len(set(a["accounts"]) & set(RING)) >= 4 for a in alerts),
        structuring_found=any("STRUCTURING" in a["kinds"] and
                              "AC1040" in a["accounts"] for a in alerts),
        insider_linked=any(a.get("employee") == "EMP3" for a in alerts),
        auto_triaged=sum(a["status"] == "AUTO_TRIAGED" for a in alerts),
        busy_teller_alerts=sum(a.get("employee") == "EMP5" for a in alerts),
        n_scenarios=len(results),
    )


# ── Detectors ─────────────────────────────────────────────────────────────────

def detect(tx, logs, inc):
    sig = []

    # ── 1. Circular transfer — use ALL transactions (not just ≥100k), then filter by min_amt
    G_circ = nx.DiGraph()
    for t in tx:
        if t["amt"] >= CIRCULAR_MIN_AMT:
            # Multi-edge: keep the highest-amount edge between a pair for cycle detection
            if G_circ.has_edge(t["src"], t["dst"]):
                if t["amt"] > G_circ[t["src"]][t["dst"]]["amt"]:
                    nx.set_edge_attributes(G_circ, {(t["src"], t["dst"]): t})
            else:
                G_circ.add_edge(t["src"], t["dst"], **t)

    for c in nx.simple_cycles(G_circ, length_bound=8):
        if len(c) < CIRCULAR_MIN_HOPS: continue
        es      = [G_circ[c[i]][c[(i+1)%len(c)]] for i in range(len(c))]
        ts_list = [e["ts"] for e in es]
        span    = max(ts_list) - min(ts_list)
        total   = sum(e["amt"] for e in es)
        min_amt = min(e["amt"] for e in es)
        avg_amt = total / len(es)

        if span > CIRCULAR_WINDOW_HOURS * H: continue

        # Risk feature: total amount matters, not just presence of cycle
        # Low-value cycles (avg < 20k) are LOW severity
        severity = "HIGH" if avg_amt >= 100_000 else "LOW"

        sig.append(dict(
            kind="CIRCULAR_TRANSFER", rule="CIRCULAR_FLOW", severity=severity,
            accts=sorted(set(c)), tx=[e["id"] for e in es],
            parameters=dict(min_hops=CIRCULAR_MIN_HOPS,
                            max_span_hours=CIRCULAR_WINDOW_HOURS,
                            min_amt_included=CIRCULAR_MIN_AMT),
            observed=dict(hops=len(c),
                          span_hours=round(span.total_seconds()/3600, 1),
                          total=total,
                          min_tx_amt=min_amt,
                          avg_tx_amt=round(avg_amt)),
            text=(f"{len(c)}-hop loop: ₹{total:,} moved in {span.total_seconds()/3600:.1f}h "
                  f"(avg ₹{avg_amt:,.0f}/transfer, min ₹{min_amt:,})")))

    # ── 2. Structuring — rolling window, multi-destination and multi-source
    #    Candidate transactions: between 80% and 100% of threshold
    lo = STRUCTURING_PROXIMITY_RATIO * STRUCTURING_THRESHOLD
    hi = STRUCTURING_THRESHOLD
    candidates = [t for t in tx if lo <= t["amt"] < hi]
    # Group by source — one source fanning out to many destinations
    by_src = defaultdict(list)
    for t in candidates:
        by_src[t["src"]].append(t)

    seen_struct_txids = set()
    for src, txs in by_src.items():
        txs_sorted = sorted(txs, key=lambda t: t["ts"])
        # Rolling window: find windows of >=3 transfers within STRUCTURING_WINDOW_HOURS
        for i, start_tx in enumerate(txs_sorted):
            window = [start_tx]
            for j in range(i+1, len(txs_sorted)):
                span = (txs_sorted[j]["ts"] - start_tx["ts"]).total_seconds()/3600
                if span <= STRUCTURING_WINDOW_HOURS:
                    window.append(txs_sorted[j])
                else:
                    break
            if len(window) < STRUCTURING_COUNT_MIN: continue
            tx_ids = frozenset(t["id"] for t in window)
            if tx_ids <= seen_struct_txids: continue  # already emitted
            seen_struct_txids |= tx_ids
            dsts   = sorted({t["dst"] for t in window})
            total  = sum(t["amt"] for t in window)
            sig.append(dict(
                kind="STRUCTURING", rule="JUST_UNDER_THRESHOLD", severity="MEDIUM",
                accts=sorted({src} | set(dsts)), tx=[t["id"] for t in window],
                parameters=dict(reporting_threshold=STRUCTURING_THRESHOLD,
                                proximity_ratio=STRUCTURING_PROXIMITY_RATIO,
                                count_threshold=STRUCTURING_COUNT_MIN,
                                window_hours=STRUCTURING_WINDOW_HOURS),
                observed=dict(count=len(window),
                              destinations=len(dsts),
                              min_amt=min(t["amt"] for t in window),
                              max_amt=max(t["amt"] for t in window),
                              total=total),
                text=(f"{len(window)} outgoing transfers from {src} across "
                      f"{len(dsts)} recipient(s) totalled ₹{total:,} within "
                      f"{STRUCTURING_WINDOW_HOURS}h — each just under the "
                      f"₹{STRUCTURING_THRESHOLD:,} reporting threshold")))

    # Group by destination — many sources fanning into one destination
    by_dst = defaultdict(list)
    for t in candidates:
        by_dst[t["dst"]].append(t)
    for dst, txs in by_dst.items():
        txs_sorted = sorted(txs, key=lambda t: t["ts"])
        for i, start_tx in enumerate(txs_sorted):
            window = [start_tx]
            for j in range(i+1, len(txs_sorted)):
                span = (txs_sorted[j]["ts"] - start_tx["ts"]).total_seconds()/3600
                if span <= STRUCTURING_WINDOW_HOURS:
                    window.append(txs_sorted[j])
                else:
                    break
            if len(window) < STRUCTURING_COUNT_MIN: continue
            # Only if multiple distinct sources (distinguishes from src-side which is covered above)
            srcs = {t["src"] for t in window}
            if len(srcs) < 2: continue
            tx_ids = frozenset(t["id"] for t in window)
            if tx_ids <= seen_struct_txids: continue
            seen_struct_txids |= tx_ids
            total = sum(t["amt"] for t in window)
            sig.append(dict(
                kind="STRUCTURING", rule="JUST_UNDER_THRESHOLD_INBOUND", severity="MEDIUM",
                accts=sorted(srcs | {dst}), tx=[t["id"] for t in window],
                parameters=dict(reporting_threshold=STRUCTURING_THRESHOLD,
                                proximity_ratio=STRUCTURING_PROXIMITY_RATIO,
                                count_threshold=STRUCTURING_COUNT_MIN,
                                window_hours=STRUCTURING_WINDOW_HOURS),
                observed=dict(count=len(window),
                              sources=len(srcs),
                              min_amt=min(t["amt"] for t in window),
                              max_amt=max(t["amt"] for t in window),
                              total=total),
                text=(f"{len(window)} inbound transfers into {dst} from "
                      f"{len(srcs)} source(s) totalled ₹{total:,} within "
                      f"{STRUCTURING_WINDOW_HOURS}h — each just under the "
                      f"₹{STRUCTURING_THRESHOLD:,} reporting threshold")))

    # ── 3. Profile mismatch: Isolation Forest
    feats = {}
    for a in inc:
        o = [t["amt"] for t in tx if t["src"] == a]
        feats[a] = [sum(o)/inc[a], (max(o) if o else 0)/inc[a], len(o)]
    X = np.array(list(feats.values()))
    m = IsolationForest(n_estimators=200, contamination=0.12, random_state=0).fit(X)
    for a, p, s in zip(feats, m.predict(X), -m.score_samples(X)):
        if p == -1:
            sig.append(dict(
                kind="PROFILE_MISMATCH", rule="ISOLATION_FOREST_ANOMALY", severity="MEDIUM",
                accts=[a], tx=[],
                parameters=dict(contamination=0.12, n_estimators=200),
                observed=dict(outflow_ratio=round(feats[a][0], 2),
                              anomaly_score=round(s, 2),
                              declared_income=inc[a]),
                text=(f"{a} activity is isolated as anomalous vs declared income "
                      f"₹{inc[a]:,}/mo (outflow {feats[a][0]:.1f}x income, "
                      f"anomaly score {s:.2f})")))

    # ── 4. Insider signals
    access_by_emp_acct = defaultdict(list)
    for l in logs:
        access_by_emp_acct[(l["emp"], l["acct"])].append(l)

    # ── Off-hours: use EMPLOYEE-SPECIFIC schedule; anti-flood by count per employee
    emp_offhours = defaultdict(list)
    for l in logs:
        emp_sched = EMPLOYEES.get(l["emp"], {}).get("hours", (8, 20))
        h_val = l["ts"].hour
        if not (emp_sched[0] <= h_val < emp_sched[1]):
            emp_offhours[l["emp"]].append(l)

    for emp, events in emp_offhours.items():
        emp_sched = EMPLOYEES.get(emp, {}).get("hours", (8, 20))
        if len(events) <= 3:   # anti-flood: only if sparse / unusual
            for ev in events:
                auth = is_authorized(emp, ev["acct"], ev["action"])
                sig.append(dict(
                    kind="OFF_HOURS_ACCESS", rule="OFF_HOURS", severity="LOW",
                    accts=[ev["acct"]], tx=[], emp=emp, authorization=auth,
                    parameters=dict(schedule_start=emp_sched[0],
                                    schedule_end=emp_sched[1]),
                    observed=dict(access_hour=ev["ts"].hour),
                    ts=ev["ts"].isoformat(),
                    text=(f"{emp} accessed {ev['acct']} ({ev['action']}) outside "
                          f"scheduled hours ({emp_sched[0]:02d}:00–{emp_sched[1]:02d}:00) "
                          f"at {f(ev['ts'])}")))

    # ── Unauthorized access + access chain + insider link
    for l in logs:
        if not l["action"].startswith("EDIT"): continue
        auth = is_authorized(l["emp"], l["acct"], l["action"])
        ts_l = l["ts"].isoformat()

        if not auth["allowed"]:
            sig.append(dict(
                kind="UNAUTHORIZED_ACCESS", rule="UNAUTHORIZED_EDIT", severity="HIGH",
                accts=[l["acct"]], tx=[], emp=l["emp"], authorization=auth, ts=ts_l,
                text=(f"{l['emp']} performed unauthorized {l['action']} on {l['acct']}. "
                      f"{auth['reason']}")))

        hist        = access_by_emp_acct[(l["emp"], l["acct"])]
        prior_views = [x for x in hist if x["action"]=="VIEW" and x["ts"]<=l["ts"]]

        for t in tx:
            dtm = (t["ts"] - l["ts"]).total_seconds() / 60
            if (t["src"] == l["acct"] and 0 <= dtm <= INSIDER_WINDOW_MINUTES
                    and t["amt"] >= INSIDER_AMOUNT_THRESHOLD):
                has_chain  = len(prior_views) >= 2
                kind_k     = "ACCESS_CHAIN" if has_chain else "INSIDER_LINK"
                rule_k     = "VIEW_EDIT_OUTFLOW_CHAIN" if has_chain else "EDIT_THEN_LARGE_OUTFLOW"
                chain_desc = (" → ".join(x["action"] for x in prior_views[-3:])
                              + f" → {l['action']}") if has_chain else l["action"]
                sig.append(dict(
                    kind=kind_k, rule=rule_k, severity="HIGH",
                    accts=sorted({l["acct"], t["dst"]}), tx=[t["id"]],
                    emp=l["emp"], authorization=auth, ts=ts_l,
                    parameters=dict(amount_threshold=INSIDER_AMOUNT_THRESHOLD,
                                    window_minutes=INSIDER_WINDOW_MINUTES),
                    observed=dict(amount=t["amt"],
                                  delay_minutes=round(dtm, 1),
                                  access_sequence=chain_desc),
                    text=(f"{l['emp']} performed {l['action']} on {l['acct']} "
                          f"at {f(l['ts'])}; the account sent ₹{t['amt']:,} "
                          f"{dtm:.0f} min later")))
    return sig


# ── Correlation — improved: temporal window + shared-employee + signal weighting
def correlate(sig, tx, logs, inc):
    # Union-find over accounts, with temporal gating:
    # Two signals sharing an account are only merged if their timestamps are within
    # CORRELATION_WINDOW_HOURS of each other.

    def _sig_ts(s):
        """Representative timestamp for a signal."""
        if "ts" in s:
            try:
                return dt.datetime.fromisoformat(s["ts"])
            except Exception:
                pass
        # Fallback: earliest transaction timestamp in the signal
        tx_ids = set(s.get("tx", []))
        matches = [t["ts"] for t in tx if t["id"] in tx_ids]
        return min(matches) if matches else T0

    par = {}
    sig_ts = [_sig_ts(s) for s in sig]

    def find(x):
        par.setdefault(x, x)
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x

    def union(a, b):
        par[find(a)] = find(b)

    # Build adjacency with temporal gating
    for i, si in enumerate(sig):
        for j, sj in enumerate(sig):
            if i >= j: continue
            # Check shared account
            shared_acct = set(si["accts"]) & set(sj["accts"])
            # Check shared employee
            shared_emp  = (si.get("emp") and si.get("emp") == sj.get("emp"))
            # Check shared transaction
            shared_tx   = set(si.get("tx",[])) & set(sj.get("tx",[]))
            if not (shared_acct or shared_emp or shared_tx): continue
            # Temporal gating
            dt_gap = abs((sig_ts[i] - sig_ts[j]).total_seconds()) / 3600
            if dt_gap > CORRELATION_WINDOW_HOURS: continue
            # Use first account as union key
            ai = sorted(si["accts"])[0]
            aj = sorted(sj["accts"])[0]
            union(ai, aj)
            # Also union on employee if shared
            if shared_emp:
                union(si["emp"], aj)

    groups = defaultdict(list)
    for s in sig:
        key = find(sorted(s["accts"])[0])
        groups[key].append(s)

    rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    out  = []

    for g in groups.values():
        accts_set = set().union(*[set(s["accts"]) for s in g])
        kinds     = {s["kind"] for s in g}
        accts     = sorted(accts_set)

        # ── Signal weighting for risk determination
        total_weight = sum(SIGNAL_WEIGHTS.get(s["kind"], 1) for s in g)
        has_strong   = bool({"UNAUTHORIZED_ACCESS","ACCESS_CHAIN","INSIDER_LINK",
                             "CIRCULAR_TRANSFER"} & kinds)
        n_kinds      = len(kinds)

        if has_strong and total_weight >= 4:
            risk = "HIGH"
        elif has_strong or total_weight >= 4 or n_kinds >= 2:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        emp    = next((s["emp"] for s in g if "emp" in s), None)
        bad_ids = {i for s in g for i in s.get("tx", [])}
        badtx  = [t for t in tx if t["id"] in bad_ids]
        ctx    = sorted([t for t in tx
                         if (t["src"] in accts_set or t["dst"] in accts_set)
                         and t["id"] not in bad_ids],
                        key=lambda t: -t["amt"])[:12]

        links = [dict(source=t["src"], target=t["dst"], amt=t["amt"],
                      bad=t["id"] in bad_ids, id=t["id"]) for t in badtx + ctx]
        nodes = {a: dict(id=a, kind="acct", bad=a in accts_set and risk!="LOW")
                 for a in accts_set}
        for l in links:
            for e in (l["source"], l["target"]):
                nodes.setdefault(e, dict(id=e, kind="acct", bad=False))
        if emp:
            nodes[emp] = dict(id=emp, kind="emp", bad=False)
            tgt_log = next((ll for ll in logs
                            if ll["emp"]==emp and ll["acct"] in accts_set), None)
            if tgt_log:
                tgt = tgt_log["acct"]
                auth_res = is_authorized(emp, tgt, tgt_log["action"])
                links.append(dict(source=emp, target=tgt, amt=0,
                                  bad=not auth_res["allowed"], kind="emp",
                                  authorized=auth_res["allowed"]))

        line = ([dict(ts=t["ts"].isoformat(),
                      text=f"{f(t['ts'])}  {t['src']} → {t['dst']}  ₹{t['amt']:,}",
                      kind="tx") for t in badtx]
                + [dict(ts=l["ts"].isoformat(),
                        text=f"{f(l['ts'])}  {l['emp']} {l['action']} on {l['acct']}",
                        kind="emp")
                   for l in logs
                   if l["acct"] in accts_set and l.get("emp")
                   and (l["action"].startswith("EDIT") or (emp and l["emp"]==emp))])

        # ── Synthesize correlated narrative
        narrative = _synthesize_narrative(g, emp)

        out.append(dict(
            risk=risk, kinds=sorted(kinds), accounts=accts, employee=emp,
            nodes=list(nodes.values()), links=links,
            evidence=g, timeline=sorted(line, key=lambda x: x["ts"]),
            total=sum(t["amt"] for t in badtx),
            tail=(badtx[-1]["dst"] if badtx else accts[0]),
            correlation_weight=total_weight,
            narrative=narrative,
        ))

    out.sort(key=lambda a: (rank[a["risk"]], -a["total"]))
    for i, a in enumerate(out):
        a.update(
            id=f"ALR-{i+1:03d}", status="OPEN", assignee=None,
            disposition="NONE", reviewer_notes=[],
            audit=[dict(ts=T0.isoformat(), by="correlation-engine",
                        note=f"Alert created: {', '.join(a['kinds'])}")],
        )
        a["explanation"] = (a["narrative"]
                            if a["narrative"]
                            else " ".join(s["text"] for s in a["evidence"][:3]))
    return out


def _synthesize_narrative(signals, emp):
    """
    Generate a human-readable correlated explanation from multiple signals.
    Generic — works for any employee or account combination.
    Only synthesizes when 2+ strong signals are present.
    """
    kinds     = {s["kind"] for s in signals}
    strong    = {"UNAUTHORIZED_ACCESS","ACCESS_CHAIN","INSIDER_LINK"} & kinds
    if not strong or not emp:
        return ""

    parts     = []
    unauth    = next((s for s in signals if s["kind"]=="UNAUTHORIZED_ACCESS"), None)
    chain     = next((s for s in signals if s["kind"] in ("ACCESS_CHAIN","INSIDER_LINK")), None)
    circ      = next((s for s in signals if s["kind"]=="CIRCULAR_TRANSFER"), None)

    if unauth:
        acct = unauth["accts"][0] if unauth.get("accts") else "an account"
        act  = unauth.get("authorization",{}).get("action","edited")
        parts.append(f"{emp} accessed {acct} outside their entitlement "
                     f"and performed {act}")

    if chain:
        seq  = chain.get("observed",{}).get("access_sequence","")
        acct = chain["accts"][0] if chain.get("accts") else "the account"
        amt  = chain.get("observed",{}).get("amount")
        dly  = chain.get("observed",{}).get("delay_minutes")
        seq_str = f" ({seq})" if seq else ""
        amt_str = f", which sent ₹{amt:,}" if amt else ""
        dly_str = f" {round(dly)} min later" if dly else ""
        parts.append(f"after repeated access{seq_str} on {acct}{amt_str}{dly_str}")

    if circ:
        total = circ.get("observed",{}).get("total")
        hops  = circ.get("observed",{}).get("hops")
        if total and hops:
            parts.append(f"the funds then moved through a {hops}-hop circular loop "
                         f"totalling ₹{total:,}")

    if not parts:
        return ""
    return "; ".join(parts) + "."


def triage(alerts, tx):
    """Auto-triage LOW-risk alerts that match known-legitimate behaviour patterns."""
    for a in alerts:
        if a["risk"] != "LOW" or len(a["accounts"]) != 1: continue
        acc = a["accounts"][0]
        out = [t for t in tx if t["src"] == acc]
        pay = defaultdict(list)
        for t in out: pay[t["dst"]].append(t["amt"])
        if (len(pay) >= 5
                and all(len(v) >= 2 and len(set(v)) == 1 for v in pay.values())):
            note = (f"Auto-triaged: consistent with routine payroll behavior "
                    f"({len(pay)} payees, identical payouts across repeated cycles, "
                    f"no linked insider activity).")
        else:
            continue
        a["status"] = "AUTO_TRIAGED"
        a["audit"].append(dict(
            ts=(T0 + 8*24*H).isoformat(), by="triage-agent", note=note))


def hop_model(tx, inc):
    G = nx.DiGraph()
    G.add_edges_from((t["src"], t["dst"]) for t in tx)
    U = G.to_undirected(); nodes = list(inc); r = random.Random(1)

    def ft(s, d):
        return [len(set(U[s]) & set(U[d])), G.in_degree(d),
                G.out_degree(s), int(inc[d] <= 30000), int(G.has_edge(d, s))]

    pos = list(G.edges); neg = []
    while len(neg) < 3 * len(pos):
        s, d = r.sample(nodes, 2)
        if not G.has_edge(s, d): neg.append((s, d))
    clf = LogisticRegression(max_iter=1000).fit(
        [ft(*e) for e in pos + neg], [1]*len(pos) + [0]*len(neg))

    def predict(tail, k=3):
        c = [d for d in nodes if d != tail and d != "CORP1"]
        p = clf.predict_proba([ft(tail, d) for d in c])[:, 1]
        return [dict(id=c[i], prob=round(float(p[i]), 3),
                     why=(f"{ft(tail,c[i])[0]} shared counterparties, "
                          f"low-income mule profile")
                     if inc[c[i]] <= 30000
                     else f"{ft(tail,c[i])[0]} shared counterparties")
                for i in np.argsort(-p)[:k]]
    return predict


def sar(a):
    prompt = (
        "You are a bank financial-crime analyst. Write a formal Suspicious Activity "
        "Report narrative (Subject, Summary of activity, Red flags, Recommended action). "
        "Use only these facts:\n" + json.dumps(dict(
            id=a["id"], risk=a["risk"], accounts=a["accounts"],
            employee=a.get("employee"),
            narrative=a.get("narrative",""),
            evidence=[x["text"] for x in a["evidence"]],
            timeline=[x["text"] for x in a["timeline"]],
            total=a["total"])))
    key = os.getenv("GEMINI_API_KEY")
    if key:
        try:
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.0-flash:generateContent?key={key}",
                json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode(),
                {"Content-Type": "application/json"})
            return (json.load(urllib.request.urlopen(req, timeout=25))
                    ["candidates"][0]["content"]["parts"][0]["text"],
                    "gemini-2.0-flash")
        except Exception as e:
            err = f" (Gemini unavailable: {e})"
    else:
        err = ""
    ins = (f" Internal access by {a['employee']} preceded the activity, "
           f"suggesting possible insider privilege misuse.") if a.get("employee") else ""
    narr = f"\n\nCorrelated finding: {a['narrative']}" if a.get("narrative") else ""
    return (
        f"SUSPICIOUS ACTIVITY REPORT — {a['id']}\n\n"
        f"Subject: accounts {', '.join(a['accounts'])}.\n\n"
        f"Summary: Between the dates shown in the timeline, the accounts moved "
        f"₹{a['total']:,} in a pattern flagged as "
        f"{', '.join(a['kinds']).lower().replace('_',' ')}.{ins}{narr}\n\n"
        f"Red flags:\n- " + "\n- ".join(x["text"] for x in a["evidence"]) +
        f"\n\nRecommended action: escalate to the financial-intelligence unit, "
        f"freeze outbound transfers pending review, and preserve access logs.{err}"
    ), "template-fallback"


def build():
    inc, tx, logs = gen()
    alerts = correlate(detect(tx, logs, inc), tx, logs, inc)
    triage(alerts, tx)
    ev = run_evaluation(alerts)
    return dict(
        alerts=alerts, predict=hop_model(tx, inc),
        eval=ev, n_tx=len(tx), n_logs=len(logs),
        inc=inc, tx=tx, logs=logs,
        now=T0 + dt.timedelta(days=7),
        seq=len(alerts)+1, queue=[], events=[], live=False, tick=0,
    )


# ── Real-time simulation ───────────────────────────────────────────────────────

def refresh(S):
    old, used = S["alerts"], set()
    new = correlate(detect(S["tx"], S["logs"], S["inc"]),
                    S["tx"], S["logs"], S["inc"])
    for n in new:
        o = next((a for a in old
                  if a["id"] not in used and set(a["accounts"]) & set(n["accounts"])), None)
        if o:
            used.add(o["id"])
            n.update(id=o["id"], status=o["status"], assignee=o["assignee"],
                     disposition=o.get("disposition","NONE"),
                     reviewer_notes=o.get("reviewer_notes",[]),
                     audit=o["audit"])
            if n["risk"] != o["risk"]:
                n["audit"].append(dict(
                    ts=S["now"].isoformat(), by="correlation-engine",
                    note=f"Risk changed {o['risk']} → {n['risk']} as new activity arrived"))
        else:
            n["id"]    = f"ALR-{S['seq']:03d}"; S["seq"] += 1
            n["audit"] = [dict(ts=S["now"].isoformat(), by="correlation-engine",
                               note="Detected live: " + ", ".join(n["kinds"]))]
            n.setdefault("disposition","NONE"); n.setdefault("reviewer_notes",[])
            if n["risk"] != "LOW":
                S["events"].append(dict(id=n["id"], risk=n["risk"]))
    triage([a for a in new if a["status"]=="OPEN"], S["tx"])
    S["alerts"] = new
    S["eval"]   = run_evaluation(new)


def live_step(S, r):
    S["tick"] += 1; S["now"] += dt.timedelta(minutes=r.randint(1, 4))
    if S["queue"]:
        t = S["queue"].pop(0); t["ts"] = S["now"]
    else:
        s, d = r.sample([a for a in S["inc"] if a != "CORP1"], 2)
        t = dict(src=s, dst=d,
                 amt=round(min(S["inc"][s]*r.uniform(.03,.4), 60000)),
                 ts=S["now"])
    t["id"] = f"T{len(S['tx'])+1:04d}"; S["tx"].append(t)
    if S["tick"] % 4 == 0: refresh(S)


def inject(S, kind, r):
    """Inject a synthetic attack scenario and ensure entitlement context is set."""
    pool = [a for a in S["inc"] if a not in BAD and a != "CORP1"]
    n    = 4 if kind == "ring" else 2
    a    = r.sample(pool, n)
    BAD.update(a)
    for x in (a if kind == "ring" else a[:1]):
        S["inc"][x] = 25000

    if kind == "ring":
        emp = "EMP4"
        acct = a[0]
        # EMP4 is NOT authorized to EDIT_LIMIT on these new accounts — intentional
        # Ensure is_authorized() has a deterministic answer: no entitlement → unauthorized
        # (do NOT grant entitlement so the detector correctly flags it)
        S["logs"].append(dict(emp=emp, action="EDIT_LIMIT", acct=acct, ts=S["now"]))
        S["queue"] += [dict(src=a[i], dst=a[(i+1)%n], amt=520_000-i*9_000)
                       for i in range(n)]
    else:
        S["queue"] += [dict(src=a[0], dst=a[1], amt=188_000+i*1_700)
                       for i in range(3)]
