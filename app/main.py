"""PhysSandbox web app (FastAPI)."""
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import agent, ledger, report
from .pipeline import CATALOG, EXPERIMENTS, analyse
from .sandbox import MODE, RUNTIME, WALLS, image_digest, read_log, run_job
from .summariser import summarise

ROOT = Path(__file__).parent.parent
app = FastAPI(title="PhysSandbox")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

MAX_UPLOAD = 1_000_000          # bytes
# Per-visitor limits on the public demo: (requests, seconds). The model calls
# cost inference credit, so they get the tightest limits.
LIMITS = {"/api/agent": (8, 300), "/api/model-check": (4, 60), "/api/analyse": (40, 300),
          "/api/containment": (20, 300), "/api/report": (20, 300)}
_hits = defaultdict(deque)


@app.middleware("http")
async def guard(request: Request, call_next):
    path = request.url.path
    rule = next((v for k, v in LIMITS.items() if path.startswith(k)), None)
    if rule and os.getenv("RATE_LIMIT", "on") != "off":
        ip = request.client.host if request.client else "?"
        if ip in ("127.0.0.1", "::1"):
            # Behind Caddy: it sets X-Forwarded-For, and the app only listens on loopback.
            ip = (request.headers.get("x-forwarded-for") or ip).split(",")[-1].strip()
        if len(_hits) > 5000:
            _hits.clear()
        n, window = rule
        q, now = _hits[(ip, path.split("/")[2])], time.time()
        while q and q[0] < now - window:
            q.popleft()
        if len(q) >= n:
            return JSONResponse({"detail": "Too many requests from you just now. Try again in a minute."},
                                status_code=429)
        q.append(now)
    if int(request.headers.get("content-length") or 0) > MAX_UPLOAD + 10_000:
        return JSONResponse({"detail": "File too large: the demo accepts up to 1 MB."}, status_code=413)
    return await call_next(request)


def _read(file):
    raw = file.file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "File too large: the demo accepts up to 1 MB.")
    return raw


ATTACKS = {"egress", "runaway", "host_read", "forge_receipt"}
SAMPLES = sorted(p.name for p in (ROOT / "data" / "seeded").glob("*.csv"))


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status():
    return {
        "mode": MODE, "runtime": RUNTIME, "walls": WALLS, "image": image_digest(),
        "model": agent.MODEL if agent.model_configured() else None,
        "experiments": EXPERIMENTS, "catalog": CATALOG, "samples": SAMPLES,
        "ledger": ledger.verify(),          # includes head_record: keep a copy to detect rollback
        "public_key": "/api/ledger/public-key",
    }


@app.get("/api/samples/{name}")
def sample(name: str):
    if name not in SAMPLES:
        raise HTTPException(404)
    return FileResponse(ROOT / "data" / "seeded" / name, media_type="text/csv")


@app.post("/api/analyse")
def analyse_upload(file: UploadFile = File(...), experiment: str = Form(...),
                         expected: str = Form("{}"), context: str = Form("")):
    raw = _read(file)
    try:
        return analyse(raw, experiment, json.loads(expected), json.loads(context) if context else None)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/report")
def investigation_report(payload: dict = Body(...)):
    """PDF of one investigation. The body is the browser's copy of it and is
    treated as untrusted (see app/report.py)."""
    try:
        pdf, filename = report.build(payload)
    except report.ReportError as e:
        raise HTTPException(400, str(e)) from e
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"',
                             "Cache-Control": "no-store"})


@app.post("/api/agent")
def agent_run(file: UploadFile = File(...), question: str = Form(...),
                    experiment: str = Form("resistance_4wire"), raw_mode: str = Form("false"),
                    compromised: str = Form("false")):
    raw = _read(file)
    summary = summarise(raw)
    raw_head = None
    if raw_mode == "true":
        raw_head = "\n".join(raw.decode("utf-8", "replace").splitlines()[:12])
    question = question[:500]
    is_red = compromised == "true"
    code, author = agent.write_code(question, experiment, summary, raw_head, compromised=is_red)
    attempts = []
    # The agent loop: write, run sealed, read the sandbox's error, repair.
    for n in range(1, agent.MAX_ATTEMPTS + 1):
        flags = agent.scan(code)
        report, events, receipt = run_job(["code"], raw, code=code, label=f"agent-written code (attempt {n})")
        out, err = (report or {}).get("stdout", ""), (report or {}).get("error")
        contained = any(e["kind"] in ("blocked", "killed", "denied", "tamper") for e in events)
        attempts.append({"n": n, "code": code, "flags": flags, "output": out, "error": err,
                         "containment": events, "receipt": receipt, "walls": (report or {}).get("probe")})
        if not err or contained or is_red or n == agent.MAX_ATTEMPTS:
            break
        fixed = agent.fix_code(question, experiment, summary, code, err, out)
        if not fixed or fixed == code:
            break
        code = fixed
    last = attempts[-1]
    return {
        "question": question, "code": last["code"], "author": author, "flags": last["flags"],
        "raw_mode": raw_mode == "true", "compromised": is_red, "summary_sent": summary,
        "output": last["output"], "error": last["error"], "attempts": [
            {"n": a["n"], "error": a["error"], "receipt": (a["receipt"] or {}).get("seq")} for a in attempts],
        "containment": [e for a in attempts for e in a["containment"]],
        "receipt": last["receipt"], "walls": last["walls"],
    }


@app.post("/api/containment/{attack}")
def containment_demo(attack: str):
    if attack not in ATTACKS:
        raise HTTPException(404)
    sample_bytes = (ROOT / "data" / "seeded" / "inject_1.csv").read_bytes()
    report, events, receipt = run_job(["attack", attack], sample_bytes, label=f"break-out attempt: {attack}")
    return {"attack": attack, "output": (report or {}).get("stdout", ""),
            "containment": events, "receipt": receipt, "walls": (report or {}).get("probe")}


@app.get("/api/containment-log")
def containment_log():
    return {"events": read_log()}


@app.get("/api/ledger")
def ledger_entries(limit: int = 20, start: int = 0):
    """The latest receipts (or `limit` receipts from #start), with the signed
    genesis and head records. Save it and check it with
    tools/verify_receipts.py and the public key below."""
    limit = max(1, min(int(limit), 100))
    items = ledger.entries()
    if start > 0:
        items = [e for e in items if isinstance(e.get("seq"), int) and e["seq"] >= start][:limit]
    else:
        items = items[-limit:]
    return {"receipts": items, "genesis": ledger.genesis(), "head_record": ledger.head_record(),
            "verify": ledger.verify(), "public_key": "/api/ledger/public-key",
            "window": {"start": start if start > 0 else None, "limit": limit}}


@app.get("/api/ledger/public-key")
def ledger_public_key(format: str = "json"):
    """The notebook's Ed25519 public key: anyone can check the receipts with it."""
    key = ledger.public_key()
    if format == "pem":
        return Response(key["pem"], media_type="text/plain; charset=utf-8",
                        headers={"Content-Disposition": f'inline; filename="physsandbox-notebook-{key["key_id"]}.pem"'})
    return key


@app.get("/api/ledger/tamper-demo")
def ledger_tamper_demo():
    """Edits a copy of the notebook as an attacker without the private key
    would, and shows the check failing. The real notebook is untouched."""
    return ledger.tamper_demo()


@app.get("/api/model-check")
def model_check():
    """Tiny test call to the configured model. Reports the error type only."""
    from . import llm
    text, err = llm.chat("Reply with the single word: ready", "Are you there?", max_tokens=400, timeout=45)
    return {"model": llm.MODEL or None, "ok": text is not None, "error": err,
            "seconds": llm.LAST.get("seconds"), "reply": (text or "")[:60]}


@app.get("/healthz")
def health():
    return {"ok": True}
