"""PhysSandbox web app (FastAPI)."""
import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import agent, ledger
from .pipeline import EXPERIMENTS, analyse
from .sandbox import MODE, RUNTIME, WALLS, image_digest, read_log, run_job
from .summariser import summarise

ROOT = Path(__file__).parent.parent
app = FastAPI(title="PhysSandbox")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

ATTACKS = {"egress", "runaway", "host_read"}
SAMPLES = sorted(p.name for p in (ROOT / "data" / "seeded").glob("*.csv"))


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status():
    return {
        "mode": MODE, "runtime": RUNTIME, "walls": WALLS, "image": image_digest(),
        "model": agent.MODEL if agent.model_configured() else None,
        "experiments": EXPERIMENTS, "samples": SAMPLES,
        "ledger": ledger.verify(),
    }


@app.get("/api/samples/{name}")
def sample(name: str):
    if name not in SAMPLES:
        raise HTTPException(404)
    return FileResponse(ROOT / "data" / "seeded" / name, media_type="text/csv")


@app.post("/api/analyse")
async def analyse_upload(file: UploadFile = File(...), experiment: str = Form(...),
                         expected: str = Form("{}"), context: str = Form("")):
    raw = await file.read()
    try:
        return analyse(raw, experiment, json.loads(expected), json.loads(context) if context else None)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/agent")
async def agent_run(file: UploadFile = File(...), question: str = Form(...),
                    experiment: str = Form("resistance_4wire"), raw_mode: str = Form("false")):
    raw = await file.read()
    summary = summarise(raw)
    raw_head = None
    if raw_mode == "true":
        raw_head = "\n".join(raw.decode("utf-8", "replace").splitlines()[:12])
    code, author = agent.write_code(question[:500], experiment, summary, raw_head)
    flags = agent.scan(code)
    report, events, receipt = run_job(["code"], raw, code=code, label="agent-written code")
    return {
        "question": question, "code": code, "author": author, "flags": flags,
        "raw_mode": raw_mode == "true", "summary_sent": summary,
        "output": (report or {}).get("stdout", ""), "error": (report or {}).get("error"),
        "containment": events, "receipt": receipt, "walls": (report or {}).get("probe"),
    }


@app.post("/api/containment/{attack}")
async def containment_demo(attack: str):
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
def ledger_entries(limit: int = 20):
    return {"receipts": ledger.entries()[-limit:], "verify": ledger.verify()}


@app.get("/healthz")
def health():
    return {"ok": True}
