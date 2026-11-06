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
import re
import resource
import runpy
import shutil
import signal
import socket
import subprocess
import sys
import threading
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


# --- System-call tracer ---------------------------------------------------------
# Agent code and attack demos run under strace, started by this runner before the
# untrusted code exists. What the code TRIED (reach the network, write outside
# scratch space, read files it may not, start programs, signal other processes)
# is then taken from the kernel-level record of its system calls, not from the
# code's own error messages, so wrapping an attempt in try/except hides nothing.
# The code runs as the same user as the tracer, so it could kill it: that kill is
# itself recorded, and a tracer that dies early is reported as tampering.
TRACE_CALLS = ("socket,connect,sendto,sendmsg,bind,listen,open,openat,creat,mkdir,mkdirat,"
               "unlink,unlinkat,rename,renameat,renameat2,chmod,fchmodat,symlink,symlinkat,link,linkat,"
               "execve,ptrace,kill,tkill,tgkill")
TRACE_CAP = 2_000_000          # bytes of trace kept; the rest is drained and counted
SCRATCH = ("/tmp/", "/dev/null", "/dev/shm/")
_LINE = re.compile(r"^(\d+)\s+(?:<\.\.\.\s+)?(\w+)(?:\s+resumed>)?\(?(.*)$")
_RESULT = re.compile(r"\)\s+=\s+(-?\d+|\?)(?:\s+(E[A-Z0-9]+))?")


def tracer():
    """Path of a working strace, or None (then attempts fall back to the code's own output)."""
    st = shutil.which("strace")
    if not st:
        return None
    try:
        r = subprocess.run([st, "-qq", "-e", "trace=none", "/bin/true"], capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return st if r.returncode == 0 else None


def _inet_target(args):
    m = re.search(r'inet_addr\("([0-9.]+)"\)|inet_pton\(AF_INET6,\s*"([0-9a-fA-F:.]+)"', args)
    if not m:
        return None
    addr = m.group(1) or m.group(2)
    port = re.search(r"htons\((\d+)\)", args)
    return addr, int(port.group(1)) if port else None


def _loopback(addr):
    return addr.startswith("127.") or addr in ("::1", "0.0.0.0", "::")


def summarise_trace(text, self_exec, tracer_pid=None):
    """Turn strace output into counts per kind of attempt. Paths and addresses are
    the code's own choice, so they go only into `examples` (returned to the person
    who ran the job), never into the shared log or the public receipt."""
    out = {"network": 0, "network_reached": 0, "loopback": 0, "writes": 0, "writes_succeeded": 0,
           "reads_refused": 0, "programs": 0, "signals": 0, "tracer_signalled": False, "examples": []}
    seen_first_exec = False

    def ex(s):
        if len(out["examples"]) < 8:
            out["examples"].append(s[:160])

    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        call, rest = m.group(2), m.group(3)
        r = _RESULT.search(rest)
        ret, err = (r.group(1), r.group(2)) if r else ("?", None)
        failed = ret.startswith("-") or err is not None
        if call in ("socket",) and ("AF_INET" in rest):
            continue  # making a socket is harmless; using it is what counts
        if call in ("connect", "sendto", "sendmsg", "bind", "listen"):
            tgt = _inet_target(rest)
            if not tgt:
                continue  # local (unix) sockets: not a way out of the box
            addr, port = tgt
            if _loopback(addr):
                out["loopback"] += 1
                continue
            out["network"] += 1
            if not failed or err == "EINPROGRESS":
                out["network_reached"] += 1
            ex(f"{call} to {addr}:{port} -> {err or ret}")
        elif call in ("open", "openat", "creat"):
            pm = re.search(r'"([^"]*)"', rest)
            path = pm.group(1) if pm else "?"
            writing = call == "creat" or re.search(r"O_WRONLY|O_RDWR|O_CREAT|O_TRUNC|O_APPEND", rest)
            if writing:
                if path.startswith(SCRATCH) or not path.startswith("/"):
                    continue  # scratch space (relative paths are inside /tmp, the working directory)
                out["writes"] += 1
                out["writes_succeeded"] += 0 if failed else 1
                ex(f"write {path} -> {err or 'opened'}")
            elif err in ("EACCES", "EPERM"):
                out["reads_refused"] += 1
                ex(f"read {path} -> {err}")
        elif call in ("mkdir", "mkdirat", "unlink", "unlinkat", "rename", "renameat", "renameat2",
                      "chmod", "fchmodat", "symlink", "symlinkat", "link", "linkat"):
            pm = re.search(r'"([^"]*)"', rest)
            path = pm.group(1) if pm else "?"
            if path.startswith(SCRATCH) or not path.startswith("/"):
                continue
            out["writes"] += 1
            out["writes_succeeded"] += 0 if failed else 1
            ex(f"{call} {path} -> {err or 'done'}")
        elif call == "execve":
            if not seen_first_exec and self_exec in rest:
                seen_first_exec = True   # the tracer starting the job itself
                continue
            out["programs"] += 1
            pm = re.search(r'"([^"]*)"', rest)
            ex(f"start {pm.group(1) if pm else '?'} -> {err or 'started'}")
        elif call in ("kill", "tkill", "tgkill", "ptrace"):
            tm = re.match(r"\s*(-?\d+)", rest)
            target = int(tm.group(1)) if tm else None
            if call == "ptrace" or (target is not None and target != int(m.group(1))):
                out["signals"] += 1
                if tracer_pid is not None and target == tracer_pid:
                    out["tracer_signalled"] = True
                ex(f"{call} {target} -> {err or ret}")
    return out


def run_traced(st, argv, timeout):
    """Run argv under strace. Returns (proc-like result, trace summary)."""
    r, w = os.pipe()
    cmd = [st, "-f", "-qq", "-s", "120", "-e", f"trace={TRACE_CALLS}", "-e", "signal=none",
           "-o", f"/dev/fd/{w}", *argv]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                            pass_fds=(w,), start_new_session=True, text=True)
    os.close(w)
    chunks, size = [], [0, 0]

    def drain():
        while True:
            b = os.read(r, 65536)
            if not b:
                break
            if size[0] < TRACE_CAP:
                chunks.append(b[:TRACE_CAP - size[0]])
            size[0] += len(b)
    t = threading.Thread(target=drain, daemon=True)
    t.start()
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
    t.join(timeout=5)
    os.close(r)
    summary = summarise_trace(b"".join(chunks).decode("utf-8", "replace"), argv[0], tracer_pid=proc.pid)
    # strace exits with the tracee's status; killed by a signal means someone stopped it.
    summary["tracer_killed"] = proc.returncode is not None and proc.returncode < 0 and not timed_out \
        and (summary["tracer_signalled"] or -proc.returncode == signal.SIGKILL)
    summary["truncated"] = size[0] > TRACE_CAP
    summary["tracer"] = "strace"
    if timed_out:
        exc = subprocess.TimeoutExpired(cmd, timeout, output=out, stderr=err)
        exc.trace = summary
        raise exc
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err), summary


def main():
    if sys.argv[1] == "_child":
        return child()
    kind, args = sys.argv[1], sys.argv[2:]
    started = time.time()
    # Line 1: the walls, measured before any untrusted code is in the box.
    print(json.dumps({"probe": probe()}), flush=True)
    forged, error_line, trace = False, None, None
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
        st = tracer()
        child_argv = [sys.executable, "-I", "-B", os.path.abspath(__file__), "_child", kind, *args]
        try:
            if st:
                proc, trace = run_traced(st, child_argv, timeout=25)
            else:
                trace = {"tracer": None}
                proc = subprocess.run(child_argv, capture_output=True, text=True, timeout=25,
                                      stdin=subprocess.DEVNULL)
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
        except subprocess.TimeoutExpired as e:
            payload, stdout, error, forged, error_line = None, "", "timed out", False, None
            trace = {**(getattr(e, "trace", None) or {"tracer": "strace" if st else None}), "timed_out": True}
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # Line 2: what the job produced (untrusted for agent code: it is the code's own output).
    print(json.dumps({
        "result": payload,
        "stdout": stdout,
        "error": error,
        "forged_probe": forged,
        "error_line": error_line,
        "trace": trace,
        "usage": {"seconds": round(time.time() - started, 3), "max_rss_mb": round(rss / 1024, 1)},
    }, default=float))


if __name__ == "__main__":
    main()
