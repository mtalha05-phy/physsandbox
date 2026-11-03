"""The glovebox: run one job in a throwaway, locked-down container.

Every job gets a fresh container that has:
  - the gVisor runtime (runsc): syscalls hit a user-space kernel, not the host
  - no network at all (--network none)
  - a read-only root filesystem and a small noexec tmpfs
  - 512 MB memory, 1 CPU, 64 processes and a 30 s wall-clock limit
  - every Linux capability dropped, no-new-privileges, uid 65534 (nobody)
  - no environment variables from the host, so no keys or tokens
  - the job's input (and any agent-written code) mounted read-only at /job
The container is destroyed when the job ends. The job measures its own walls
from inside (sandbox/tools/runner.py) and the host turns that into a receipt
in the tamper-evident ledger.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from functools import lru_cache
from pathlib import Path

from . import ledger

ROOT = Path(__file__).parent.parent
IMAGE = os.getenv("SANDBOX_IMAGE", "physsandbox-runner:latest")
RUNTIME = os.getenv("SANDBOX_RUNTIME", "runsc")
MODE = os.getenv("SANDBOX_MODE", "docker")            # "local" = development only, NOT isolated
TIMEOUT_S = int(os.getenv("SANDBOX_TIMEOUT_S", "30"))
LOG_PATH = Path(os.getenv("CONTAINMENT_LOG", ROOT / "containment_log.jsonl"))

WALLS = {
    "runtime": "gVisor user-space kernel" if RUNTIME == "runsc" else "Docker (runc)",
    "network": "none",
    "filesystem": "read-only root, 64 MB scratch",
    "memory": "512 MB",
    "cpu": "1 vCPU, 64 processes",
    "time": f"{TIMEOUT_S} s",
    "identity": "uid 65534 (nobody), no capabilities",
    "secrets": "none passed in",
}

NETWORK_MARKERS = ("Network is unreachable", "Temporary failure in name resolution",
                   "Name or service not known", "No address associated",
                   "Errno -3", "Errno 101", "Errno -2", "timed out")
DENIAL_MARKERS = ("Permission denied", "PermissionError", "Read-only file system")


def log_event(kind, detail, job_id, wall=None):
    event = {"time": time.strftime("%H:%M:%S"), "job": job_id, "kind": kind,
             "detail": detail, "wall": wall}
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(event) + "\n")
    return event


def read_log(limit=60):
    if not LOG_PATH.exists():
        return []
    return [json.loads(l) for l in LOG_PATH.read_text().splitlines()[-limit:]]


@lru_cache(maxsize=1)
def image_digest():
    if MODE == "local":
        return "local-dev (not isolated)"
    try:
        out = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _docker_cmd(name, workdir, argv):
    return [
        "docker", "run", "--rm", "--name", name,
        f"--runtime={RUNTIME}",
        "--network", "none",
        "--read-only", "--tmpfs", "/tmp:rw,size=64m,noexec",
        "--memory", "512m", "--memory-swap", "512m",
        "--cpus", "1", "--pids-limit", "64",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "--env-file", "/dev/null",
        "-v", f"{workdir}:/job:ro",
        IMAGE, "python", "/tools/runner.py", *argv,
    ]


def _local_cmd(workdir, argv):
    env = {"PS_JOB_DIR": str(workdir), "PATH": os.environ.get("PATH", "")}
    return [sys.executable, str(ROOT / "sandbox" / "tools" / "runner.py"), *argv], env


def run_job(argv, input_bytes=b"", code=None, label="analysis"):
    """Run one job. argv is passed to runner.py, e.g. ["analyze", exp, json].

    Returns (report, events, receipt). report is the runner's JSON or None.
    """
    job_id = uuid.uuid4().hex[:8]
    events = [log_event("start", f"{label}: new container ({WALLS['runtime']})", job_id)]
    workdir = Path(tempfile.mkdtemp(prefix="ps-"))
    (workdir / "input.csv").write_bytes(input_bytes)
    if code is not None:
        (workdir / "code.py").write_text(code)
    for p in workdir.iterdir():
        os.chmod(p, 0o444)
    os.chmod(workdir, 0o755)

    if MODE == "local" and argv[0] == "attack":
        shutil.rmtree(workdir, ignore_errors=True)
        events.append(log_event("refused", "attack demos need the real sandbox (SANDBOX_MODE=docker)", job_id))
        return None, events, None

    env = None
    if MODE == "local":
        cmd, env = _local_cmd(workdir, argv)
    else:
        cmd = _docker_cmd(f"ps-{job_id}", workdir, argv)

    started = time.time()
    timed_out = False
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, env=env)
        out, err, code_rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", f"ps-{job_id}"], capture_output=True)
        out, err, code_rc, timed_out = "", "", -1, True
    elapsed = round(time.time() - started, 2)

    report = None
    try:
        report = json.loads(out.strip().splitlines()[-1]) if out.strip() else None
    except (ValueError, IndexError):
        report = None
    seen = " ".join(filter(None, [out, err, (report or {}).get("stdout"), (report or {}).get("error")]))

    if timed_out:
        events.append(log_event("killed", f"stopped at the {TIMEOUT_S} s time limit", job_id, "time"))
    if code_rc == 137 or "out of memory" in err.lower():
        events.append(log_event("killed", "stopped at the 512 MB memory limit", job_id, "memory"))
    if any(m in seen for m in NETWORK_MARKERS):
        events.append(log_event("blocked", "tried to reach the internet; there is no network", job_id, "network"))
    if any(m in seen for m in DENIAL_MARKERS):
        events.append(log_event("denied", "tried to write outside scratch space; filesystem is read-only",
                                job_id, "filesystem"))
    if report and report.get("error") and not any(e["kind"] in ("blocked", "denied") for e in events):
        events.append(log_event("error", report["error"][:200], job_id))
    events.append(log_event("end", f"container destroyed after {elapsed} s", job_id))
    shutil.rmtree(workdir, ignore_errors=True)

    probe = (report or {}).get("probe") or {}
    receipt = ledger.append({
        "job": job_id,
        "label": label,
        "mode": MODE,
        "image": image_digest(),
        "walls_configured": WALLS,
        "walls_measured": probe,
        "input_sha256": ledger.sha256(input_bytes),
        "code_sha256": ledger.sha256(code) if code is not None else None,
        "output_sha256": ledger.sha256(out),
        "outcome": [e["kind"] for e in events if e["kind"] not in ("start", "end")] or ["clean"],
        "seconds": elapsed,
        "max_rss_mb": ((report or {}).get("usage") or {}).get("max_rss_mb"),
    })
    return report, events, receipt
