"""Tamper-evident lab notebook of containment receipts.

Every sandboxed job appends one receipt. Each receipt carries the SHA-256 of
the input, the code, the output and the sandbox image, the walls the job
measured from inside the box, and the hash of the previous receipt. Editing
or deleting any entry breaks every hash after it, which verify() detects.

A hash chain alone can be rebuilt by anyone who can edit the file, so each
receipt is also signed: mac = HMAC-SHA256(key, hash). The key lives outside
the repo and outside every sandbox (LEDGER_HMAC_KEY, or a 0600 key file), so
rewriting history needs the key as well as the file. The head hash also goes
into every downloaded investigation record, so a copy held elsewhere pins it.
"""
import hmac
import secrets
import hashlib
import json
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
LEDGER = Path(os.getenv("RECEIPT_LEDGER", ROOT / "receipts.jsonl"))
GENESIS = "0" * 64
KEY_FILE = Path(os.getenv("LEDGER_KEY_FILE", ROOT / ".ledger_key"))
_lock = threading.Lock()


def _key() -> bytes:
    env = os.getenv("LEDGER_HMAC_KEY")
    if env:
        return env.encode()
    if not KEY_FILE.exists():
        KEY_FILE.write_text(secrets.token_hex(32))
        os.chmod(KEY_FILE, 0o600)
    return KEY_FILE.read_text().strip().encode()


def _mac(h: str) -> str:
    return hmac.new(_key(), h.encode(), "sha256").hexdigest()


def sha256(data) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data or b"").hexdigest()


def _canonical(entry) -> str:
    body = {k: v for k, v in entry.items() if k not in ("hash", "mac")}
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
        receipt["mac"] = _mac(receipt["hash"])
        with LEDGER.open("a") as f:
            f.write(json.dumps(receipt, default=str) + "\n")
        return receipt


def verify():
    """Checks the chain and the signatures. Receipts written before signing
    was added (legacy) are chain-checked only; once a signed receipt appears,
    every later one must be signed."""
    prev, signed, signing = GENESIS, 0, False
    items = entries()
    for e in items:
        bad = e.get("prev") != prev or sha256(_canonical(e)) != e.get("hash")
        if "mac" in e:
            signing = True
            bad = bad or not hmac.compare_digest(e["mac"], _mac(e["hash"]))
            signed += 1
        elif signing:
            bad = True
        if bad:
            return {"intact": False, "receipts": len(items), "broken_at": e.get("seq")}
        prev = e["hash"]
    return {"intact": True, "receipts": len(items), "signed": signed, "head": prev}
