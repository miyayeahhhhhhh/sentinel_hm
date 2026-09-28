import os, datetime as dt
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import engine

app = FastAPI(title="Sentinel FIN-04"); S = engine.build()
class Body(BaseModel): value: str = ""
def get(i):
    a = next((x for x in S["alerts"] if x["id"] == i), None)
    if not a: raise HTTPException(404, "Unknown alert")
    return a
def log(a, by, note): a["audit"].append(dict(ts=dt.datetime.now().isoformat(), by=by, note=note))

@app.get("/api/summary")
def summary(): return dict(eval=S["eval"], n_tx=len(S["tx"]), n_logs=S["n_logs"], gemini=bool(os.getenv("GEMINI_API_KEY")))
@app.get("/api/alerts")
def alerts(): return [{k: a[k] for k in ("id", "risk", "kinds", "accounts", "status", "assignee", "total", "explanation")} for a in S["alerts"]]
@app.get("/api/alerts/{i}")
def detail(i: str): return get(i)
@app.get("/api/alerts/{i}/predict")
def predict(i: str):
    a = get(i); log(a, "next-hop-model", f"Predicted next targets from {a['tail']}")
    return dict(tail=a["tail"], candidates=S["predict"](a["tail"]), model="logistic link-prediction on graph features")
@app.get("/api/alerts/{i}/sar")
def sar(i: str):
    a = get(i); text, model = engine.sar(a); log(a, model, "SAR narrative generated"); return dict(text=text, model=model)
@app.post("/api/alerts/{i}/assign")
def assign(i: str, b: Body):
    a = get(i); a["assignee"] = b.value; a["status"] = "ASSIGNED"; log(a, "analyst", f"Assigned to {b.value}"); return a
@app.post("/api/alerts/{i}/status")
def status(i: str, b: Body):
    a = get(i); a["status"] = b.value; log(a, "analyst", f"Status set to {b.value}"); return a
@app.get("/api/alerts/{i}/export")
def export(i: str):
    a = get(i); text, _ = engine.sar(a)
    return JSONResponse(dict(alert=a, sar=text), headers={"Content-Disposition": f"attachment; filename={i}-evidence.json"})
@app.get("/")
def index(): return FileResponse(os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html"))

import asyncio, random
R = random.Random(11)
@app.get("/api/live")
def live(since: int = 0):
    tx = S["tx"]; hot = {a for al in S["alerts"] if al["risk"] != "LOW" for a in al["accounts"]}
    new = [dict(id=t["id"], src=t["src"], dst=t["dst"], amt=t["amt"], hot=t["src"] in hot and t["dst"] in hot) for t in tx[max(since, len(tx) - 25):]]
    ev = S["events"][:]; S["events"].clear()
    return dict(n=len(tx), txs=new, events=ev, live=S["live"], alerts=alerts(), summary=summary())
@app.post("/api/live/toggle")
def toggle(): S["live"] = not S["live"]; return dict(live=S["live"])
@app.post("/api/live/inject")
def inject(b: Body): engine.inject(S, b.value, R); S["live"] = True; return dict(ok=True)
@app.on_event("startup")
async def boot():
    async def loop():
        while True:
            await asyncio.sleep(1.2)
            if S["live"]: engine.live_step(S, R)
    asyncio.create_task(loop())
