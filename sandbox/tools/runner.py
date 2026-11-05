"""Entry point for every sandboxed job. Runs INSIDE the container.

1. Probes the walls of the box it is running in (kernel, network, filesystem,
   identity, secrets) so the host can issue a containment receipt based on
   what the job could actually see, not on what we configured.
2. Runs the job: a physics tool, agent-written code, or an attack demo.
3. Prints two JSON lines on stdout.

Trust boundary: the probe is measured and written (line 1) BEFORE any
untrusted code runs (agent code is already mounted read-only at /job, but
nothing has executed it yet). Agent code and attack demos then run in a
separate child process whose output is captured through a pipe, so they can
shape their own stdout but never the probe. The host additionally checks the
container's real configuration and exit state with `docker inspect`, so the
receipt never rests on what the job says about itself.

Usage:
  python /tools/runner.py analyze <experiment> <expected-json>
  python /tools/runner.py code
  python /tools/runner.py attack <name>
"""
import contextlib
import io
import json
import os
import resource
import runpy
import socket
import subprocess
import sys
import time
import traceback

JOB_DIR = os.environ.get("PS_JOB_DIR", "/job")
DATA = os.path.join(JOB_DIR, "input.csv")
GVISOR_KERNEL = "4.4.0 #1 SMP Sun Jan 10 15:06:54 PST 2016"


def probe():
    p = {}
    try:
        kernel = open("/proc/version").read().strip()
    except OSError:
        kernel = "unknown"
    p["kernel"] = kernel
    p["gvisor"] = "gvisor" in kernel.lower() or GVISOR_KERNEL in kernel
    try:
        lines = open("/proc/net/dev").read().splitlines()[2:]
        p["interfaces"] = sorted(l.split(":")[0].strip() for l in lines if ":" in l)
    except OSError:
        p["interfaces"] = []
    socket.setdefaulttimeout(2)
    try:
        socket.getaddrinfo("github.com", 443)
        p["dns"] = "resolves"
    except OSError:
        p["dns"] = "blocked"
    try:
        s = socket.create_connection(("1.1.1.1", 443), timeout=2)
        s.close()
        p["egress"] = "open"
    except OSError:
        p["egress"] = "blocked"
    # Is the root filesystem mounted read-only? Ask the kernel (statvfs), rather than
    # trying to write a root-owned path, which fails for uid nobody on any disk.
    try:
        ro = bool(os.statvfs("/").f_flag & os.ST_RDONLY)
    except (OSError, AttributeError):
        ro = False
    p["root_fs"] = "read-only" if ro else "writable"
    p["uid"] = os.getuid()
    secret_words = ("KEY", "TOKEN", "SECRET", "PASSWORD", "VULTR", "AWS")
    # GPG_KEY is set by the python base image: the public ID of the key that
    # signs Python releases, not a credential.
    public = {"GPG_KEY"}
    p["secrets_visible"] = sorted(k for k in os.environ
                                  if k not in public and any(w in k.upper() for w in secret_words))
    return p


ERROR_LINE = None   # line in the agent's code where it failed, if it did


def run_target(kind, args):
    global ERROR_LINE
    out = io.StringIO()
    error = None
    payload = None
    with contextlib.redirect_stdout(out):
        try:
            if kind == "analyze":
                sys.argv = ["analyze.py", args[0], args[1], DATA]
                runpy.run_path(os.path.join(os.path.dirname(__file__), "analyze.py"),
                               run_name="__main__")
            elif kind == "code":
                code = open(os.path.join(JOB_DIR, "code.py")).read()
                ns = {"__name__": "__main__", "DATA": DATA}
                exec(compile(code, "agent_code.py", "exec"), ns)  # noqa: S102 - contained by design
            elif kind == "attack":
                runpy.run_path(f"/attacks/{os.path.basename(args[0])}.py", run_name="__main__")
        except SystemExit:
            pass
        except BaseException as e:  # noqa: BLE001 - report everything
            error = "".join(traceback.format_exception_only(type(e), e)).strip()
            frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "agent_code.py"]
            ERROR_LINE = frames[-1].lineno if frames else None
    text = out.getvalue()
    if kind == "analyze" and not error:
        try:
            payload = json.loads(text.strip().splitlines()[-1])
            text = ""
        except (ValueError, IndexError):
            error = "analysis produced no result"
    return payload, text[-6000:], error


def _has_probe(text):
    for line in text.splitlines():
        try:
            j = json.loads(line)
        except ValueError:
            continue
        if isinstance(j, dict) and "probe" in j:
            return True
    return False


def child():
    """Runs untrusted work (agent code, attack demos) in its own process."""
    kind, args = sys.argv[2], sys.argv[3:]
    payload, stdout, error = run_target(kind, args)
    print(json.dumps({"result": payload, "stdout": stdout, "error": error, "error_line": ERROR_LINE}, default=float))


def main():
    if sys.argv[1] == "_child":
        return child()
    kind, args = sys.argv[1], sys.argv[2:]
    started = time.time()
    # Line 1: the walls, measured before any untrusted code is in the box.
    print(json.dumps({"probe": probe()}), flush=True)
    forged, error_line = False, None
    if kind == "analyze":
        # Our own physics code reading the uploaded file.
        payload, stdout, error = run_target(kind, args)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    else:
        try:
            # Not dumpable: the child (same uid) cannot open /proc/<runner>/fd/1
            # to write into the runner's own report.
            import ctypes
            ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE = 4
        except (OSError, AttributeError):
            pass
        try:
            proc = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "_child", kind, *args],
                                  capture_output=True, text=True, timeout=25, stdin=subprocess.DEVNULL)
            lines = proc.stdout.strip().splitlines()
            try:
                rep = json.loads(lines[-1][:20000])
                # Untrusted: keep only short text fields, never a "result" payload.
                payload = None
                stdout = str(rep.get("stdout") or "")[-6000:]
                error = str(rep["error"])[:2000] if rep.get("error") else None
                error_line = rep.get("error_line") if isinstance(rep.get("error_line"), int) else None
            except (ValueError, IndexError, TypeError, AttributeError):
                payload, stdout, error_line = None, proc.stdout[-6000:], None
                error = (proc.stderr.strip().splitlines() or [f"child exited with code {proc.returncode}"])[-1][:2000]
            # The child has no business printing a probe: that is a forgery attempt.
            forged = _has_probe(proc.stdout)
            if proc.returncode < 0 or proc.returncode == 137:
                error = error or f"killed (signal {-proc.returncode if proc.returncode < 0 else 9})"
        except subprocess.TimeoutExpired:
            payload, stdout, error, forged, error_line = None, "", "timed out", False, None
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # Line 2: what the job produced (untrusted for agent code: it is the code's own output).
    print(json.dumps({
        "result": payload,
        "stdout": stdout,
        "error": error,
        "forged_probe": forged,
        "error_line": error_line,
        "usage": {"seconds": round(time.time() - started, 3), "max_rss_mb": round(rss / 1024, 1)},
    }, default=float))


if __name__ == "__main__":
    main()
