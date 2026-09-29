import os, datetime as dt, json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import engine, persistence

# ── JSON encoder ─────────────────────────────────────────────────────────────
class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, dt.datetime): return o.isoformat()
        if isinstance(o, set):         return sorted(o)
        return super().default(o)

def safe(obj):
    return json.loads(json.dumps(obj, cls=_Enc))

# ── Bootstrap ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Sentinel FIN-04")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
S   = engine.build()
persistence.apply(S["alerts"])   # restore case state from disk on startup

# ── Models ────────────────────────────────────────────────────────────────────
class Body(BaseModel): value: str = ""
class NoteBody(BaseModel): note: str = ""; actor: str = "analyst"

# ── Helpers ───────────────────────────────────────────────────────────────────
def get(i):
    a = next((x for x in S["alerts"] if x["id"] == i), None)
    if not a: raise HTTPException(404, "Unknown alert")
    return a

def log(a, by, note):
    a["audit"].append(dict(ts=dt.datetime.now().isoformat(), by=by, note=note))
    persistence.save(S["alerts"])

# ── Informational endpoints ───────────────────────────────────────────────────
@app.get("/api/summary")
def summary():
    ev = S["eval"]
    return dict(
        eval=ev,
        n_tx=len(S["tx"]),
        n_logs=len(S["logs"]),
        gemini=bool(os.getenv("GEMINI_API_KEY")),
    )

@app.get("/api/alerts")
def alerts():
    return [{k: a[k] for k in (
        "id","risk","kinds","accounts","status","assignee",
        "total","explanation","disposition","narrative",
    )} for a in S["alerts"]]

@app.get("/api/alerts/{i}")
def detail(i: str): return safe(get(i))

@app.get("/api/alerts/{i}/predict")
def predict(i: str):
    a = get(i)
    log(a, "next-hop-model", f"Predicted next targets from {a['tail']}")
    return dict(tail=a["tail"],
                candidates=S["predict"](a["tail"]),
                model="logistic link-prediction on graph features")

@app.get("/api/alerts/{i}/sar")
def sar(i: str):
    a = get(i)
    text, model = engine.sar(a)
    log(a, model, "SAR narrative generated")
    return dict(text=text, model=model)

# ── Case mutation endpoints ────────────────────────────────────────────────────

@app.post("/api/alerts/{i}/assign")
def assign(i: str, b: Body):
    a = get(i)
    a["assignee"] = b.value
    a["status"]   = "ASSIGNED"
    log(a, "analyst", f"Assigned to {b.value}")
    return safe(a)

@app.post("/api/alerts/{i}/status")
def set_status(i: str, b: Body):
    allowed = {"OPEN","ASSIGNED","UNDER_REVIEW","ESCALATED","CLOSED"}
    if b.value not in allowed:
        raise HTTPException(400, f"Status must be one of {sorted(allowed)}")
    a = get(i)
    old = a["status"]
    a["status"] = b.value
    log(a, "analyst", f"Status changed {old} → {b.value}")
    return safe(a)

@app.post("/api/alerts/{i}/disposition")
def set_disposition(i: str, b: Body):
    allowed = {"NONE","CONFIRMED_SUSPICIOUS","FALSE_POSITIVE",
               "BENIGN_LEGITIMATE","ESCALATED"}
    if b.value not in allowed:
        raise HTTPException(400, f"Disposition must be one of {sorted(allowed)}")
    a = get(i)
    a["disposition"] = b.value
    log(a, "analyst", f"Disposition set to {b.value}")
    return safe(a)

@app.post("/api/alerts/{i}/escalate")
def escalate(i: str, b: Body):
    """Escalate: status → ESCALATED, disposition → ESCALATED, optional reason in b.value."""
    a = get(i)
    a["status"]      = "ESCALATED"
    a["disposition"] = "ESCALATED"
    note = f"Escalated{': ' + b.value if b.value else ''}"
    log(a, "analyst", note)
    return safe(a)

@app.post("/api/alerts/{i}/note")
def add_note(i: str, b: NoteBody):
    """Add a free-text reviewer note (actor + timestamp + content)."""
    if not b.note.strip():
        raise HTTPException(400, "Note cannot be empty")
    a = get(i)
    entry = dict(ts=dt.datetime.now().isoformat(), by=b.actor, note=b.note.strip())
    a["reviewer_notes"].append(entry)
    a["audit"].append(dict(ts=entry["ts"], by=b.actor, note=f"Note added: {b.note.strip()[:80]}"))
    persistence.save(S["alerts"])
    return safe(a)

# ── Export (enriched) ─────────────────────────────────────────────────────────
@app.get("/api/alerts/{i}/export")
def export(i: str):
    a = get(i); text, _ = engine.sar(a)
    # Build enriched correlation summary
    rules     = sorted({e.get("rule","") for e in a["evidence"] if e.get("rule")})
    tx_ids    = sorted({tid for e in a["evidence"] for tid in e.get("tx",[])})
    thresholds = {}
    observed   = {}
    for e in a["evidence"]:
        if e.get("parameters"): thresholds.update(e["parameters"])
        if e.get("observed"):   observed.update(e["observed"])

    bundle = dict(
        sentinel_export_version="2.0",
        exported_at=dt.datetime.now().isoformat(),
        alert=dict(
            id=a["id"],
            case_id=a["id"],
            risk=a["risk"],
            status=a["status"],
            assignee=a["assignee"],
            disposition=a.get("disposition","NONE"),
            employee=a.get("employee"),
            accounts=a["accounts"],
            signal_kinds=a["kinds"],
            rules=rules,
            transaction_ids=tx_ids,
            total_amount=a["total"],
            correlation_weight=a.get("correlation_weight"),
        ),
        narrative=a.get("narrative",""),
        thresholds_used=thresholds,
        observed_values=observed,
        timeline=[x["text"] for x in a["timeline"]],
        evidence=safe(a["evidence"]),
        reviewer_notes=a.get("reviewer_notes",[]),
        audit_history=safe(a["audit"]),
        sar_draft=text,
    )
    return JSONResponse(
        bundle,
        headers={"Content-Disposition": f"attachment; filename={i}-evidence.json"})

@app.get("/")
def index():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html"))

# ── Live simulation ────────────────────────────────────────────────────────────
import asyncio, random
R = random.Random(11)

@app.get("/api/live")
def live(since: int = 0):
    tx  = S["tx"]
    hot = {a for al in S["alerts"] if al["risk"] != "LOW" for a in al["accounts"]}
    new = [dict(id=t["id"], src=t["src"], dst=t["dst"], amt=t["amt"],
                hot=t["src"] in hot and t["dst"] in hot)
           for t in tx[max(since, len(tx)-25):]]
    ev  = S["events"][:]; S["events"].clear()
    return dict(n=len(tx), txs=new, events=ev, live=S["live"],
                alerts=alerts(), summary=summary())

@app.post("/api/live/toggle")
def toggle():
    S["live"] = not S["live"]; return dict(live=S["live"])

@app.post("/api/live/inject")
def inject(b: Body):
    engine.inject(S, b.value, R); S["live"] = True; return dict(ok=True)

@app.on_event("startup")
async def boot():
    async def loop():
        while True:
            await asyncio.sleep(1.2)
            if S["live"]: engine.live_step(S, R)
    asyncio.create_task(loop())
