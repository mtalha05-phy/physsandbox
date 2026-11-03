"""PhysSandbox web app (FastAPI)."""
import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import EXPERIMENTS, analyse
from .sandbox import read_log, run_job

ROOT = Path(__file__).parent.parent
app = FastAPI(title="PhysSandbox")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

ATTACKS = {
    "egress": "/attacks/egress.py",
    "runaway": "/attacks/runaway.py",
    "host_read": "/attacks/host_read.py",
}
SAMPLES = sorted(p.name for p in (ROOT / "data" / "seeded").glob("*.csv"))


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/experiments")
def experiments():
    return {"experiments": EXPERIMENTS, "samples": SAMPLES}


@app.get("/api/samples/{name}")
def sample(name: str):
    if name not in SAMPLES:
        raise HTTPException(404)
    return FileResponse(ROOT / "data" / "seeded" / name, media_type="text/csv")


@app.post("/api/analyse")
async def analyse_upload(file: UploadFile = File(...), experiment: str = Form(...),
                         expected: str = Form("{}")):
    raw = await file.read()
    try:
        return analyse(raw, experiment, json.loads(expected))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/containment/{attack}")
async def containment_demo(attack: str):
    if attack not in ATTACKS:
        raise HTTPException(404)
    sample_bytes = (ROOT / "data" / "seeded" / "inject_1.csv").read_bytes()
    out, events = run_job(["python", ATTACKS[attack]], sample_bytes, label=f"attack:{attack}")
    return {"attack": attack, "output": out[-2000:], "containment": events}


@app.get("/api/containment-log")
def containment_log():
    return {"events": read_log()}


@app.get("/healthz")
def health():
    return {"ok": True}
