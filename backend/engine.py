"""Sentinel engine: synthetic data -> detectors -> IsolationForest -> correlation -> triage agent -> next-hop -> SAR."""
import random, json, os, urllib.request, datetime as dt
from collections import defaultdict
import numpy as np, networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

T0 = dt.datetime(2026, 9, 21, 9, 0); THR = 200000; H = dt.timedelta(hours=1)
RING = [f"AC{n}" for n in range(1050, 1055)]
BAD = set(RING) | {"AC1040", "AC1045"}          # ground truth for evaluation
f = lambda t: t.strftime("%d %b %H:%M")

def gen(seed=7):
    r = random.Random(seed)
    inc = {f"AC{1000+i}": r.choice([25000, 40000, 60000, 90000, 150000]) for i in range(60)}
    for a in RING: inc[a] = 25000
    inc["AC1040"] = 40000; inc["CORP1"] = 500000
    accts = [a for a in inc if a != "CORP1"]; tx = []
    add = lambda s, d, a, t: tx.append(dict(id=f"T{len(tx)+1:04d}", src=s, dst=d, amt=round(a), ts=t))
    for _ in range(450):                                            # normal retail traffic
        s, d = r.sample(accts, 2)
        add(s, d, min(inc[s] * r.uniform(.03, .4), 60000), T0 + dt.timedelta(minutes=r.randint(0, 7 * 1440)))
    for a in accts[:12]:                                            # LEGIT: weekly payroll
        for k in (0, 6): add("CORP1", a, inc[a], T0 + dt.timedelta(days=k, hours=1))
    for k in range(3): add("AC1005", "AC1006", 15000, T0 + dt.timedelta(days=2 * k, hours=5))  # LEGIT: rent
    base = T0 + dt.timedelta(days=3, hours=1)                       # SCENARIO 1: laundering ring + insider
    for i in range(5):
        add(RING[i], RING[(i + 1) % 5], 480000 - i * 8000, base + dt.timedelta(minutes=31 + 22 * i))
    for i in range(3):                                              # SCENARIO 2: structuring
        add("AC1040", "AC1045", 190000 + i * 1500, T0 + dt.timedelta(days=4, hours=3, minutes=50 * i))
    logs = [dict(emp="EMP3", action="EDIT_LIMIT", acct=RING[0], ts=base + dt.timedelta(minutes=5))]
    logs.append(dict(emp="EMP2", action="EDIT_PROFILE", acct="AC1010", ts=T0 + dt.timedelta(days=2)))  # LEGIT edit
    for _ in range(300):                                            # LEGIT: 'busy clinician' teller, heavy reads
        logs.append(dict(emp="EMP5", action="VIEW", acct=r.choice(accts), ts=T0 + dt.timedelta(minutes=r.randint(0, 7 * 1440))))
    return inc, sorted(tx, key=lambda t: t["ts"]), sorted(logs, key=lambda l: l["ts"])

def detect(tx, logs, inc):
    sig = []; big = nx.DiGraph()
    for t in tx:
        if t["amt"] >= 100000: big.add_edge(t["src"], t["dst"], **t)
    for c in nx.simple_cycles(big, length_bound=5):                 # 1. circular transfers
        if len(c) < 3: continue
        es = [big[c[i]][c[(i + 1) % len(c)]] for i in range(len(c))]
        span = max(e["ts"] for e in es) - min(e["ts"] for e in es)
        if span <= 24 * H:
            sig.append(dict(kind="CIRCULAR_TRANSFER", accts=set(c), tx=[e["id"] for e in es],
                text=f"{len(c)}-hop loop returned money to its origin: ₹{sum(e['amt'] for e in es):,} moved in {span.total_seconds()/3600:.1f}h"))
    near = defaultdict(list)                                        # 2. structuring
    for t in tx:
        if .8 * THR <= t["amt"] < THR: near[(t["src"], t["dst"])].append(t)
    for (s, d), ts in near.items():
        if len(ts) >= 3 and ts[-1]["ts"] - ts[0]["ts"] <= 24 * H:
            sig.append(dict(kind="STRUCTURING", accts={s, d}, tx=[t["id"] for t in ts],
                text=f"{len(ts)} transfers of ₹{ts[0]['amt']:,}-₹{ts[-1]['amt']:,}, each just under the ₹{THR:,} reporting threshold (total ₹{sum(t['amt'] for t in ts):,})"))
    feats = {}                                                      # 3. profile mismatch: Isolation Forest
    for a in inc:
        o = [t["amt"] for t in tx if t["src"] == a]
        feats[a] = [sum(o) / inc[a], (max(o) if o else 0) / inc[a], len(o)]
    X = np.array(list(feats.values()))
    m = IsolationForest(n_estimators=200, contamination=0.12, random_state=0).fit(X)
    for a, p, s in zip(feats, m.predict(X), -m.score_samples(X)):
        if p == -1:
            sig.append(dict(kind="PROFILE_MISMATCH", accts={a}, tx=[],
                text=f"{a} activity is isolated as anomalous vs declared income ₹{inc[a]:,}/mo (outflow {feats[a][0]:.1f}x income, anomaly score {s:.2f})"))
    for l in logs:                                                  # 4. insider linkage
        if not l["action"].startswith("EDIT"): continue
        for t in tx:
            dtm = (t["ts"] - l["ts"]).total_seconds() / 60
            if t["src"] == l["acct"] and 0 <= dtm <= 60 and t["amt"] >= 100000:
                sig.append(dict(kind="INSIDER_LINK", accts={l["acct"], t["dst"]}, tx=[t["id"]], emp=l["emp"],
                    text=f"{l['emp']} performed {l['action']} on {l['acct']} at {f(l['ts'])}; the account sent ₹{t['amt']:,} {dtm:.0f} min later"))
    return sig

def correlate(sig, tx, logs, inc):
    par = {}
    def find(x):
        par.setdefault(x, x)
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for s in sig:
        a = sorted(s["accts"])
        for x in a[1:]: par[find(x)] = find(a[0])
    groups = defaultdict(list)
    for s in sig: groups[find(sorted(s["accts"])[0])].append(s)
    rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}; out = []
    for g in groups.values():
        accts = set().union(*[s["accts"] for s in g]); kinds = {s["kind"] for s in g}
        risk = "HIGH" if ("INSIDER_LINK" in kinds or len(kinds) >= 3) else "MEDIUM" if len(kinds) == 2 else "LOW"
        emp = next((s["emp"] for s in g if "emp" in s), None)
        bad_ids = {i for s in g for i in s["tx"]}
        badtx = [t for t in tx if t["id"] in bad_ids]
        ctx = sorted([t for t in tx if (t["src"] in accts or t["dst"] in accts) and t["id"] not in bad_ids], key=lambda t: -t["amt"])[:12]
        links = [dict(source=t["src"], target=t["dst"], amt=t["amt"], bad=t["id"] in bad_ids, id=t["id"]) for t in badtx + ctx]
        nodes = {a: dict(id=a, kind="acct", bad=a in accts and risk != "LOW") for a in accts}
        for l in links:
            for e in (l["source"], l["target"]): nodes.setdefault(e, dict(id=e, kind="acct", bad=False))
        if emp:
            nodes[emp] = dict(id=emp, kind="emp", bad=False)
            tgt = next(x["acct"] for x in [dict(acct=l["acct"]) for l in logs if l["emp"] == emp and l["acct"] in accts])
            links.append(dict(source=emp, target=tgt, amt=0, bad=False, kind="emp"))
        line = [dict(ts=t["ts"].isoformat(), text=f"{f(t['ts'])}  {t['src']} → {t['dst']}  ₹{t['amt']:,}", kind="tx") for t in badtx]
        line += [dict(ts=l["ts"].isoformat(), text=f"{f(l['ts'])}  {l['emp']} {l['action']} on {l['acct']}", kind="emp") for l in logs if l["action"].startswith("EDIT") and l["acct"] in accts]
        out.append(dict(risk=risk, kinds=sorted(kinds), accounts=sorted(accts), employee=emp, nodes=list(nodes.values()), links=links,
            evidence=[s["text"] for s in g], timeline=sorted(line, key=lambda x: x["ts"]),
            total=sum(t["amt"] for t in badtx), tail=(badtx[-1]["dst"] if badtx else sorted(accts)[0])))
    out.sort(key=lambda a: (rank[a["risk"]], -a["total"]))
    for i, a in enumerate(out):
        a.update(id=f"ALR-{i+1:03d}", status="OPEN", assignee=None,
                 audit=[dict(ts=T0.isoformat(), by="correlation-engine", note=f"Alert created: {', '.join(a['kinds'])}")])
        a["explanation"] = " ".join(a["evidence"][:3])
    return out

def triage(alerts, tx):
    """Agentic pre-triage: closes LOW alerts that match a known-legitimate behavioural pattern, leaves an audit note."""
    for a in alerts:
        if a["risk"] != "LOW" or len(a["accounts"]) != 1: continue
        acc = a["accounts"][0]; out = [t for t in tx if t["src"] == acc]
        pay = defaultdict(list)
        for t in out: pay[t["dst"]].append(t["amt"])
        if len(pay) >= 5 and all(len(v) >= 2 and len(set(v)) == 1 for v in pay.values()):
            note = f"Auto-triaged: consistent with routine payroll behavior ({len(pay)} payees, identical payouts across repeated cycles, no linked insider activity)."
        else: continue
        a["status"] = "AUTO_TRIAGED"; a["audit"].append(dict(ts=(T0 + 8 * 24 * H).isoformat(), by="triage-agent", note=note))

def hop_model(tx, inc):
    G = nx.DiGraph(); G.add_edges_from((t["src"], t["dst"]) for t in tx); U = G.to_undirected(); nodes = list(inc); r = random.Random(1)
    def ft(s, d): return [len(set(U[s]) & set(U[d])), G.in_degree(d), G.out_degree(s), int(inc[d] <= 30000), int(G.has_edge(d, s))]
    pos = list(G.edges); neg = []
    while len(neg) < 3 * len(pos):
        s, d = r.sample(nodes, 2)
        if not G.has_edge(s, d): neg.append((s, d))
    clf = LogisticRegression(max_iter=1000).fit([ft(*e) for e in pos + neg], [1] * len(pos) + [0] * len(neg))
    def predict(tail, k=3):
        c = [d for d in nodes if d != tail and d != "CORP1"]
        p = clf.predict_proba([ft(tail, d) for d in c])[:, 1]
        return [dict(id=c[i], prob=round(float(p[i]), 3), why=f"{ft(tail, c[i])[0]} shared counterparties, low-income mule profile" if inc[c[i]] <= 30000 else f"{ft(tail, c[i])[0]} shared counterparties")
                for i in np.argsort(-p)[:k]]
    return predict

def sar(a):
    prompt = ("You are a bank financial-crime analyst. Write a formal Suspicious Activity Report narrative "
              "(Subject, Summary of activity, Red flags, Recommended action). Use only these facts:\n" + json.dumps(
              dict(id=a["id"], risk=a["risk"], accounts=a["accounts"], employee=a["employee"], evidence=a["evidence"], timeline=[x["text"] for x in a["timeline"]], total=a["total"])))
    key = os.getenv("GEMINI_API_KEY")
    if key:
        try:
            req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
                json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode(), {"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=25))["candidates"][0]["content"]["parts"][0]["text"], "gemini-2.0-flash"
        except Exception as e: err = f" (Gemini unavailable: {e})"
    else: err = ""
    ins = f" Internal access by {a['employee']} preceded the activity, suggesting possible insider privilege misuse." if a["employee"] else ""
    return (f"SUSPICIOUS ACTIVITY REPORT — {a['id']}\n\nSubject: accounts {', '.join(a['accounts'])}.\n\nSummary: Between the dates shown in the timeline, the accounts "
            f"moved ₹{a['total']:,} in a pattern flagged as {', '.join(a['kinds']).lower().replace('_', ' ')}.{ins}\n\nRed flags:\n- " + "\n- ".join(a["evidence"]) +
            f"\n\nRecommended action: escalate to the financial-intelligence unit, freeze outbound transfers pending review, and preserve access logs.{err}"), "template-fallback"

def evaluate(alerts):
    act = [a for a in alerts if a["risk"] != "LOW"]; tp = [a for a in act if set(a["accounts"]) & BAD]
    low_fp = [a for a in alerts if a["risk"] == "LOW" and not set(a["accounts"]) & BAD]
    return dict(alerts=len(alerts), actionable=len(act), true_positive=len(tp), false_positive=len(act) - len(tp),
        precision=round(len(tp) / max(len(act), 1), 2), ring_found=any(len(set(a["accounts"]) & set(RING)) >= 4 for a in alerts),
        structuring_found=any(a["risk"] != "LOW" and {"AC1040", "AC1045"} <= set(a["accounts"]) for a in alerts),
        insider_linked=any(a["employee"] == "EMP3" for a in alerts), legit_flagged_low=len(low_fp),
        auto_triaged=sum(a["status"] == "AUTO_TRIAGED" for a in alerts), busy_teller_alerts=sum(a["employee"] == "EMP5" for a in alerts))

def build():
    inc, tx, logs = gen(); alerts = correlate(detect(tx, logs, inc), tx, logs, inc); triage(alerts, tx)
    return dict(alerts=alerts, predict=hop_model(tx, inc), eval=evaluate(alerts), n_tx=len(tx), n_logs=len(logs), inc=inc, tx=tx, logs=logs,
        now=T0 + dt.timedelta(days=7), seq=len(alerts) + 1, queue=[], events=[], live=False, tick=0)


# ---------------- real-time simulation ----------------
def refresh(S):
    old, used = S["alerts"], set()
    new = correlate(detect(S["tx"], S["logs"], S["inc"]), S["tx"], S["logs"], S["inc"])
    for n in new:
        o = next((a for a in old if a["id"] not in used and set(a["accounts"]) & set(n["accounts"])), None)
        if o:
            used.add(o["id"]); n.update(id=o["id"], status=o["status"], assignee=o["assignee"], audit=o["audit"])
            if n["risk"] != o["risk"]: n["audit"].append(dict(ts=S["now"].isoformat(), by="correlation-engine", note=f"Risk changed {o['risk']} -> {n['risk']} as new activity arrived"))
        else:
            n["id"] = f"ALR-{S['seq']:03d}"; S["seq"] += 1
            n["audit"] = [dict(ts=S["now"].isoformat(), by="correlation-engine", note="Detected live: " + ", ".join(n["kinds"]))]
            if n["risk"] != "LOW": S["events"].append(dict(id=n["id"], risk=n["risk"]))
    triage([a for a in new if a["status"] == "OPEN"], S["tx"]); S["alerts"] = new; S["eval"] = evaluate(new)

def live_step(S, r):
    S["tick"] += 1; S["now"] += dt.timedelta(minutes=r.randint(1, 4))
    if S["queue"]: t = S["queue"].pop(0); t["ts"] = S["now"]
    else:
        s, d = r.sample([a for a in S["inc"] if a != "CORP1"], 2)
        t = dict(src=s, dst=d, amt=round(min(S["inc"][s] * r.uniform(.03, .4), 60000)), ts=S["now"])
    t["id"] = f"T{len(S['tx'])+1:04d}"; S["tx"].append(t)
    if S["tick"] % 4 == 0: refresh(S)

def inject(S, kind, r):
    pool = [a for a in S["inc"] if a not in BAD and a != "CORP1"]; n = 4 if kind == "ring" else 2; a = r.sample(pool, n); BAD.update(a)
    for x in (a if kind == "ring" else a[:1]): S["inc"][x] = 25000
    if kind == "ring":
        S["logs"].append(dict(emp="EMP4", action="EDIT_LIMIT", acct=a[0], ts=S["now"]))
        S["queue"] += [dict(src=a[i], dst=a[(i + 1) % n], amt=520000 - i * 9000) for i in range(n)]
    else: S["queue"] += [dict(src=a[0], dst=a[1], amt=188000 + i * 1700) for i in range(3)]
