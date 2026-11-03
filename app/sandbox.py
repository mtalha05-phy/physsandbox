"""Run one analysis job in a throwaway, locked-down container.

Every job gets a fresh container that has:
  - no network (--network none)
  - the gVisor runtime (runsc) by default, so syscalls hit a user-space kernel
  - a read-only root filesystem and a small tmpfs for scratch space
  - CPU, memory and process limits, and a hard wall-clock timeout
  - all Linux capabilities dropped, no-new-privileges, no secrets or env vars
  - the uploaded file mounted read-only
The container is removed when the job ends. Every block, kill and denial is
written to the containment log.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
IMAGE = os.getenv("SANDBOX_IMAGE", "physsandbox-runner:latest")
RUNTIME = os.getenv("SANDBOX_RUNTIME", "runsc")       # gVisor; "runc" = plain Docker
MODE = os.getenv("SANDBOX_MODE", "docker")            # "local" = dev only, NOT isolated
TIMEOUT_S = int(os.getenv("SANDBOX_TIMEOUT_S", "30"))
LOG_PATH = Path(os.getenv("CONTAINMENT_LOG", ROOT / "containment_log.jsonl"))

NETWORK_MARKERS = ("Network is unreachable", "Temporary failure in name resolution",
                   "Name or service not known", "No address associated")
DENIAL_MARKERS = ("Permission denied", "PermissionError", "Read-only file system")


def log_event(kind, detail, job_id):
    event = {"time": time.strftime("%H:%M:%S"), "job": job_id, "kind": kind, "detail": detail}
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(event) + "\n")
    return event


def read_log(limit=50):
    if not LOG_PATH.exists():
        return []
    return [json.loads(l) for l in LOG_PATH.read_text().splitlines()[-limit:]]


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
        "-v", f"{workdir}/input.csv:/data/input.csv:ro",
        IMAGE, *argv,
    ]


def run_job(argv, input_bytes=b"", label="analysis"):
    """Run argv inside the sandbox. Returns (stdout, events)."""
    job_id = uuid.uuid4().hex[:8]
    events = [log_event("start", f"{label} job started ({MODE}, runtime={RUNTIME})", job_id)]
    workdir = Path(tempfile.mkdtemp(prefix="ps-"))
    (workdir / "input.csv").write_bytes(input_bytes)
    os.chmod(workdir, 0o755)
    os.chmod(workdir / "input.csv", 0o444)

    if MODE == "local":
        # Development only: runs on the host with no isolation.
        cmd = [sys.executable, str(ROOT / "sandbox" / argv[1].lstrip("/")), *argv[2:],
               str(workdir / "input.csv")]
        if argv[1].startswith("/attacks"):
            shutil.rmtree(workdir, ignore_errors=True)
            events.append(log_event("refused", "attack demos need SANDBOX_MODE=docker", job_id))
            return "", events
    else:
        cmd = _docker_cmd(f"ps-{job_id}", workdir, argv)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
        out, err, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", f"ps-{job_id}"], capture_output=True)
        events.append(log_event("killed", f"timeout after {TIMEOUT_S} s", job_id))
        shutil.rmtree(workdir, ignore_errors=True)
        return "", events

    if code == 137:
        events.append(log_event("killed", "memory limit (512 MB) exceeded", job_id))
    if any(m in err or m in out for m in NETWORK_MARKERS):
        events.append(log_event("blocked", "network call blocked: sandbox has no network", job_id))
    if any(m in err or m in out for m in DENIAL_MARKERS):
        events.append(log_event("denied", "filesystem access denied", job_id))
    if code not in (0, 137):
        events.append(log_event("error", (err.strip().splitlines() or ["exit " + str(code)])[-1][:200], job_id))
    events.append(log_event("end", "container destroyed", job_id))
    shutil.rmtree(workdir, ignore_errors=True)
    return out, events
