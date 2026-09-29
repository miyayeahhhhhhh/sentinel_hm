# Sentinel — Financial Crime & Insider Risk Intelligence (HackMatrix 5.0, FIN-04)

Links **employee access events** to **money movement** in one investigation graph, with an evidence panel on every alert.

## Run
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key      # optional; without it the SAR uses an offline template
cd backend && uvicorn app:app --port 8000
# open http://localhost:8000
```
(The UI loads React, Tailwind and react-force-graph-2d from CDNs, so the first load needs internet.)

## Pipeline
`engine.py`: synthetic data (labelled ground truth) -> 4 detectors -> correlation into connected alerts -> triage agent -> next-hop model -> SAR.
- Detectors: circular transfers (NetworkX cycles), structuring (sub-threshold splits), **profile mismatch (scikit-learn Isolation Forest)**, insider linkage (EDIT action followed by large outflow within 60 min)
- Risk: HIGH = insider link or 3+ signal types, MEDIUM = 2, LOW = 1 (never a bare score; evidence lists every rule that fired)
- Triage agent: auto-closes LOW alerts matching routine payroll patterns and writes an audit note
- Next-hop: logistic-regression link prediction on graph features (shared counterparties, degree, low-income profile)
- SAR: Gemini API when a key is set, otherwise a template; both use only the structured alert facts

## Evaluation (synthetic, seed 7)
Scenarios: laundering ring + insider edit, structuring, weekly payroll, recurring rent, a "busy teller" with 300 reads. See the header KPIs in the UI or `GET /api/summary`.

## Limits
Synthetic data only; the next-hop model is trained on the same graph it predicts on (prototype, no held-out split); Isolation Forest contamination is fixed at 0.12.

## Layout
```
backend/engine.py  backend/app.py  frontend/index.html  requirements.txt
```

## Real-time mode
`Go live` (left panel) starts a simulated stream: a new transaction every ~1.2s, detectors re-run every ~5s, and new alerts appear with a toast. `Simulate an attack` injects a laundering ring (with an insider edit) or a structuring run into the stream. The frontend polls `/api/live`.
Themes: Ledger, Midnight, Ember, Aurora (dots in the header, or press `t`). Animated components (aurora + particle background, click sparks, count-up, blur text, spotlight cards, shiny text) are dependency-free re-creations in the style of React Bits.

## Deploy:
Backend (Python) on Render via `render.yaml`; frontend (static) on Netlify. Netlify proxies `/api/*` to the backend through `frontend/_redirects` (edit the URL). Or deploy only the backend on Render: it also serves the UI at its root URL.
