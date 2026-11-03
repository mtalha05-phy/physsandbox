"""Tamper-evident lab notebook of containment receipts.

Every sandboxed job appends one receipt. Each receipt carries the SHA-256 of
the input, the code, the output and the sandbox image, the walls the job
measured from inside the box, and the hash of the previous receipt. Editing
or deleting any entry breaks every hash after it, which verify() detects.
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
LEDGER = Path(os.getenv("RECEIPT_LEDGER", ROOT / "receipts.jsonl"))
GENESIS = "0" * 64
_lock = threading.Lock()


def sha256(data) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data or b"").hexdigest()


def _canonical(entry) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)


def entries():
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def append(receipt: dict) -> dict:
    with _lock:
        prior = entries()
        receipt = dict(receipt)
        receipt["seq"] = len(prior) + 1
        receipt["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        receipt["prev"] = prior[-1]["hash"] if prior else GENESIS
        receipt["hash"] = sha256(_canonical(receipt))
        with LEDGER.open("a") as f:
            f.write(json.dumps(receipt, default=str) + "\n")
        return receipt


def verify():
    prev = GENESIS
    items = entries()
    for e in items:
        if e.get("prev") != prev or sha256(_canonical(e)) != e.get("hash"):
            return {"intact": False, "receipts": len(items), "broken_at": e.get("seq")}
        prev = e["hash"]
    return {"intact": True, "receipts": len(items), "head": prev}
