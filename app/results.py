"""Server-side store of every physics analysis step, for the PDF report.

The PDF report is built only from what this server computed, never from the
browser's copy. When analyse() finishes a step it builds a canonical record of
it (experiment, expected values, file name, test run, status, causes, next
test, values, plot, explanation, ...). The SHA-256 of that record goes into the
job's receipt as `result_sha256` *before* the receipt is signed and appended to
the lab notebook; the record itself is written here, one JSON file per receipt
hash. To use a record, the server finds its receipt in the notebook and
recomputes the record's hash:

  - a forged record has no receipt in the notebook;
  - a record edited on disk no longer matches the hash sealed in its signed
    receipt, and the receipt can't be changed to match without the signing key;
  - a real receipt from another job points to that job's own record (or to
    none, for agent and break-out jobs).

Records also carry the browser session that ran the analysis (`owner`, a hash
of a random cookie), so the public receipt hashes at GET /api/ledger are not
enough to read someone else's results.
"""
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", ROOT / "results"))
MAX_BYTES = int(float(os.getenv("RESULTS_MAX_MB", "200")) * 1024 * 1024)
MAX_POINTS = 5000          # per plot series
MAX_TEXT = 2000
VERSION = 1
HASH = re.compile(r"[0-9a-f]{64}")
TOKEN = re.compile(r"[A-Za-z0-9_-]{20,64}")
# The receipt fields the report shows; a copy is kept with the record for anyone
# auditing the store offline (the report itself reads them from the notebook).
RECEIPT_FIELDS = ("seq", "time", "hash", "prev", "sig", "mac", "input_sha256", "mode",
                  "walls_measured", "walls_host_verified", "result_sha256")
_lock = threading.Lock()


class StoreError(ValueError):
    pass


# ---------------------------------------------------------------- canonical form
def _plain(v):
    """JSON-safe copy: str keys, lists for tuples, non-finite floats -> None."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def canonical(record) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(record) -> str:
    return hashlib.sha256(canonical(record).encode()).hexdigest()


# ---------------------------------------------------------------- record
def owner_id(token):
    """Opaque id of the browser session that ran a step (None without a valid token)."""
    if not isinstance(token, str) or not TOKEN.fullmatch(token):
        return None
    return hashlib.sha256(b"physsandbox-session:" + token.encode()).hexdigest()


def clean_filename(name):
    name = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[\x00-\x1f\x7f​-‏  ﻿]", "", name).strip()
    return name if len(name) <= 120 else name[:119] + "…"


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _cap_series(xs, ys, peaks=False):
    """Pairs with finite x and y, at most MAX_POINTS of them. Long series keep
    evenly spaced points (first and last included); a spectrum keeps the
    largest point in each bin, so its peaks survive."""
    if not isinstance(xs, list) or not isinstance(ys, list):
        return [], []
    pts = [(x, y) for x, y in zip(xs, ys) if _finite(x) and _finite(y)]
    n = len(pts)
    if n > MAX_POINTS:
        if peaks:
            pts = [max(pts[n * i // MAX_POINTS: n * (i + 1) // MAX_POINTS], key=lambda p: abs(p[1]))
                   for i in range(MAX_POINTS)]
        else:
            pts = [pts[round(i * (n - 1) / (MAX_POINTS - 1))] for i in range(MAX_POINTS)]
    return [x for x, _ in pts], [y for _, y in pts]


def _cap_plot(plot):
    if not isinstance(plot, dict):
        return None
    p = dict(plot)
    p["x"], p["y"] = _cap_series(plot.get("x"), plot.get("y"), peaks=plot.get("mode") == "spectrum")
    if isinstance(plot.get("lines"), list):
        p["lines"] = []
        for ln in plot["lines"][:12]:
            if isinstance(ln, dict):
                ln = dict(ln)
                ln["x"], ln["y"] = _cap_series(ln.get("x"), ln.get("y"))
                p["lines"].append(ln)
    return p


def record(result, filename=None, owner=None, after=None, input_sha256=None):
    """The canonical record of one analysis step (what the report may show)."""
    nxt = (result.get("next_tests") or [None])[0]
    rec = {
        "v": VERSION,
        "experiment": result["experiment"],
        "expected": result.get("expected") or {},
        "file": clean_filename(filename) or None,
        "owner": owner,
        "after": after,
        "test_run": result.get("test_run"),
        "done": list(result.get("done") or []),
        "status": result["status"],
        "causes": [{"id": c["id"], "name": c["name"], "probability": c["probability"],
                    "typical_size": c.get("typical_size")} for c in result["causes"]],
        "next": None if not nxt else {k: nxt.get(k) for k in
                                      ("id", "name", "minutes", "info_gain_bits", "bits_per_minute")},
        "values": result.get("values") or {},
        "evidence": result.get("evidence") or {},
        "plot": _cap_plot(result.get("plot")),
        "explanation": (result.get("explanation") or "")[:MAX_TEXT] or None,
        "explained_by": (result.get("explained_by") or "")[:80] or None,
        "input_sha256": input_sha256,
    }
    # Normalised through JSON once, so the hash is the hash of exactly what is stored.
    return json.loads(json.dumps(_plain(rec), allow_nan=False))


# ---------------------------------------------------------------- store
def _path(receipt_hash):
    if not isinstance(receipt_hash, str) or not HASH.fullmatch(receipt_hash):
        return None
    return RESULTS_DIR / f"{receipt_hash}.json"


def _used():
    try:
        return sum(e.stat().st_size for e in os.scandir(RESULTS_DIR) if e.is_file())
    except OSError:
        return 0


def save(receipt, rec):
    """Write the record of one step, keyed by its receipt's hash. Atomic; refuses
    (returns False) once the store holds MAX_BYTES."""
    path = _path((receipt or {}).get("hash"))
    if path is None or receipt.get("result_sha256") != digest(rec):
        return False
    doc = {"receipt_hash": receipt["hash"], "result_sha256": receipt["result_sha256"], "result": rec,
           "receipt": {k: receipt[k] for k in RECEIPT_FIELDS if k in receipt}}
    data = json.dumps(doc, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    with _lock:
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
            if _used() + len(data) > MAX_BYTES:
                print(f"results: store full ({MAX_BYTES / 2**20:g} MB), receipt #{receipt.get('seq')} "
                      "not stored", file=sys.stderr)
                return False
            fd, tmp = tempfile.mkstemp(dir=RESULTS_DIR, prefix=".tmp-", suffix=".json")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        except OSError as e:
            print(f"results: could not store receipt #{receipt.get('seq')}: {e}", file=sys.stderr)
            return False
    return True


def load_verified(receipt_hash, entry):
    """The stored record for this receipt, checked against `entry` (the receipt
    as this server's lab notebook holds it). None if nothing is stored."""
    path, entry = _path(receipt_hash), entry or {}
    if path is None:
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        raise StoreError(f"The stored result for receipt #{entry.get('seq')} can't be read.") from e
    rec = doc.get("result") if isinstance(doc, dict) else None
    sealed = entry.get("result_sha256")
    if not isinstance(rec, dict) or doc.get("receipt_hash") != receipt_hash or entry.get("hash") != receipt_hash:
        raise StoreError(f"The stored result for receipt #{entry.get('seq')} is damaged.")
    if not sealed:
        raise StoreError(f"Receipt #{entry.get('seq')} has no result sealed in it, so nothing stored for it "
                         "can be trusted.")
    if digest(rec) != sealed:
        raise StoreError(f"The stored result for receipt #{entry.get('seq')} doesn't match the result hash "
                         "sealed in its signed receipt, so it was changed after the analysis ran.")
    return rec


def receipt_in_notebook(receipt_hash, items):
    return next((e for e in items if e.get("hash") == receipt_hash), None)
