"""Entry point for every sandboxed job. Runs INSIDE the container.

1. Probes the walls of the box it is running in (kernel, network, filesystem,
   identity, secrets) so the host can issue a containment receipt based on
   what the job could actually see, not on what we configured.
2. Runs the job: a physics tool, agent-written code, or an attack demo.
3. Prints one JSON object on stdout.

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
    p["gvisor"] = GVISOR_KERNEL in kernel
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
    try:
        with open("/tools/.write-test", "w") as f:
            f.write("x")
        p["root_fs"] = "writable"
    except OSError:
        p["root_fs"] = "read-only"
    p["uid"] = os.getuid()
    secret_words = ("KEY", "TOKEN", "SECRET", "PASSWORD", "VULTR", "AWS")
    p["secrets_visible"] = sorted(k for k in os.environ if any(w in k.upper() for w in secret_words))
    return p


def run_target(kind, args):
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
    text = out.getvalue()
    if kind == "analyze" and not error:
        try:
            payload = json.loads(text.strip().splitlines()[-1])
            text = ""
        except (ValueError, IndexError):
            error = "analysis produced no result"
    return payload, text[-6000:], error


def main():
    kind, args = sys.argv[1], sys.argv[2:]
    started = time.time()
    walls = probe()
    payload, stdout, error = run_target(kind, args)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(json.dumps({
        "probe": walls,
        "result": payload,
        "stdout": stdout,
        "error": error,
        "usage": {"seconds": round(time.time() - started, 3),
                  "max_rss_mb": round(usage.ru_maxrss / 1024, 1)},
    }, default=float))


if __name__ == "__main__":
    main()
