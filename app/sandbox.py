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
import threading
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
                   "Errno -3", "Errno 101", "Errno -2")
DENIAL_MARKERS = ("Permission denied", "PermissionError", "Read-only file system")


# At most this many sandboxes at once, so a burst of visitors cannot exhaust
# the VM (each container may use 512 MB).
MAX_JOBS = int(os.getenv("SANDBOX_MAX_JOBS", "2"))
_slots = threading.BoundedSemaphore(MAX_JOBS)


def log_event(kind, detail, job_id, wall=None, source="host"):
    """source: "host" = observed by this server (Docker's record, the timer);
    "code" = what the job's own output says it tried. The walls themselves are
    always checked by the host; "code" events only describe the attempt."""
    event = {"time": time.strftime("%H:%M:%S"), "job": job_id, "kind": kind,
             "detail": detail, "wall": wall, "source": source}
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
    # No --rm: the host inspects the stopped container (real config and exit
    # state) before removing it, so the receipt never relies on the job's word.
    return [
        "docker", "run", "--name", name,
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


def _inspect(name):
    """What Docker itself says about the container: the walls it really had
    and how it ended. Measured by the host, so the job cannot forge it."""
    try:
        out = subprocess.run(["docker", "inspect", name], capture_output=True, text=True, timeout=10)
        info = json.loads(out.stdout)[0]
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None
    hc, st = info.get("HostConfig", {}), info.get("State", {})
    return {
        "runtime": hc.get("Runtime"),
        "network": hc.get("NetworkMode"),
        "read_only_root": hc.get("ReadonlyRootfs"),
        "memory_mb": (hc.get("Memory") or 0) // (1024 * 1024),
        "pids_limit": hc.get("PidsLimit"),
        "cap_drop": hc.get("CapDrop"),
        "no_new_privileges": "no-new-privileges" in (hc.get("SecurityOpt") or []),
        "user": info.get("Config", {}).get("User"),
        "exit_code": st.get("ExitCode"),
        "oom_killed": st.get("OOMKilled"),
    }


def host_walls_ok(h):
    return bool(h) and h["runtime"] == "runsc" and h["network"] == "none" and h["read_only_root"] \
        and h["memory_mb"] == 512 and h["cap_drop"] == ["ALL"] and h["no_new_privileges"] \
        and h["user"] == "65534:65534"


def _local_cmd(workdir, argv):
    env = {"PS_JOB_DIR": str(workdir), "PATH": os.environ.get("PATH", "")}
    return [sys.executable, str(ROOT / "sandbox" / "tools" / "runner.py"), *argv], env


def run_job(argv, input_bytes=b"", code=None, label="analysis", seal=None):
    """Run one job. argv is passed to runner.py, e.g. ["analyze", exp, json].

    seal(report) -> extra receipt fields (e.g. {"result_sha256": ...}): called
    after the container is gone and before the receipt is signed and appended,
    outside the sandbox slot (it may wait on the model). The receipt is
    appended even if seal raises.

    Returns (report, events, receipt). report is the runner's JSON or None.
    """
    with _slots:
        report, events, body = _run_job(argv, input_bytes, code, label)
    if body is None:
        return report, events, None
    extra = {}
    try:
        if seal:
            extra = seal(report) or {}
    finally:
        receipt = ledger.append({**body, **extra})
    return report, events, receipt


def _run_job(argv, input_bytes, code, label):
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
    name = f"ps-{job_id}"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, env=env)
        out, err, code_rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        out, err, code_rc, timed_out = "", "", -1, True
    elapsed = round(time.time() - started, 2)
    host = None
    if MODE != "local":
        host = _inspect(name)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    # Line 1 = probe (written before untrusted code existed); last line = result.
    lines = [l for l in out.strip().splitlines() if l.strip()]
    report = None
    try:
        first = json.loads(lines[0]) if lines else {}
        last = json.loads(lines[-1]) if len(lines) > 1 else {}
        report = {**last, "probe": first.get("probe")} if first.get("probe") else None
        if report is not None and len(lines) > 2:
            report["extra_lines"] = len(lines) - 2
    except (ValueError, IndexError):
        report = None
    seen = " ".join(filter(None, [err, (report or {}).get("stdout"), (report or {}).get("error")]))

    # Kills: from Docker's own record and the host's timer only.
    if timed_out:
        events.append(log_event("killed", f"stopped at the {TIMEOUT_S} s time limit", job_id, "time"))
    if (host or {}).get("oom_killed") or code_rc == 137:
        events.append(log_event("killed", "stopped at the 512 MB memory limit (Docker's record)", job_id, "memory"))
    # Attempts: described by the code's own output. The wall that stopped each
    # one is verified separately by the host check below.
    if any(m in seen for m in NETWORK_MARKERS):
        events.append(log_event("blocked", "tried to reach the internet and found no network "
                                "(the code's own error; host confirms network none)", job_id, "network", "code"))
    if any(m in seen for m in DENIAL_MARKERS):
        events.append(log_event("denied", "tried to write outside scratch space and was refused "
                                "(the code's own error; host confirms a read-only root)", job_id, "filesystem", "code"))
    if report and (report.get("extra_lines") or report.get("forged_probe")):
        events.append(log_event("tamper", "code printed a fake receipt; ignored, the real probe was taken "
                                "before the code ran", job_id, "receipt", "code"))
    if MODE != "local" and not host_walls_ok(host):
        events.append(log_event("alert", "host check failed: the container did not have the expected walls",
                                job_id, "host"))
    if report and report.get("error") and not any(e["kind"] in ("blocked", "denied", "killed") for e in events):
        events.append(log_event("error", report["error"][:200], job_id))
    events.append(log_event("end", f"container destroyed after {elapsed} s", job_id))
    shutil.rmtree(workdir, ignore_errors=True)

    probe = (report or {}).get("probe") or {}
    # The receipt body; run_job signs and appends it.
    return report, events, {
        "job": job_id,
        "label": label,
        "mode": MODE,
        "image": image_digest(),
        "walls_configured": WALLS,
        "walls_measured": probe,
        "walls_host_verified": host,
        "input_sha256": ledger.sha256(input_bytes),
        "code_sha256": ledger.sha256(code) if code is not None else None,
        "output_sha256": ledger.sha256(out),
        "outcome": [e["kind"] for e in events if e["kind"] not in ("start", "end")] or ["clean"],
        "seconds": elapsed,
        "max_rss_mb": ((report or {}).get("usage") or {}).get("max_rss_mb"),
    }
