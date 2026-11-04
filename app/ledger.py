"""Tamper-evident lab notebook of containment receipts.

Every sandboxed job appends one receipt: SHA-256 of the input, code, output
and sandbox image, the walls measured inside the box and checked by the host,
and the hash of the previous receipt.

Two keyed protections sit on top of the hash chain:
  - every receipt carries mac = HMAC-SHA256(key, hash). Unsigned receipts are
    rejected, so stripping the signatures and rebuilding the chain fails.
  - a signed head record (count + last hash) is rewritten on every append, so
    deleting receipts off the end fails too.
The key lives outside the repository and outside every sandbox (default
~/.physsandbox/ledger.key, mode 0600, or LEDGER_HMAC_KEY). It is on the same
host, so someone with root on the VM could still read it: for that threat,
keep a copy of the head hash elsewhere (every investigation record carries its
receipts' hashes and signatures).
"""
import copy
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
LEDGER = Path(os.getenv("RECEIPT_LEDGER", ROOT / "receipts.jsonl"))
HEAD = LEDGER.with_name(LEDGER.stem + ".head.json")
GENESIS = "0" * 64
KEY_FILE = Path(os.getenv("LEDGER_KEY_FILE", Path.home() / ".physsandbox" / "ledger.key"))
_lock = threading.Lock()


def _key() -> bytes:
    env = os.getenv("LEDGER_HMAC_KEY")
    if env:
        return env.encode()
    if not KEY_FILE.exists():
        KEY_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        KEY_FILE.write_text(secrets.token_hex(32))
        os.chmod(KEY_FILE, 0o600)
    return KEY_FILE.read_text().strip().encode()


def _mac(text: str) -> str:
    return hmac.new(_key(), text.encode(), "sha256").hexdigest()


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


def _head():
    try:
        return json.loads(HEAD.read_text())
    except (OSError, ValueError):
        return None


def append(receipt: dict) -> dict:
    with _lock:
        head = _head()
        last_seq, last_hash = (head["seq"], head["hash"]) if head else (0, GENESIS)
        receipt = dict(receipt)
        receipt["seq"] = last_seq + 1
        receipt["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        receipt["prev"] = last_hash
        receipt["hash"] = sha256(_canonical(receipt))
        receipt["mac"] = _mac(receipt["hash"])
        with LEDGER.open("a") as f:
            f.write(json.dumps(receipt, default=str) + "\n")
        h = {"seq": receipt["seq"], "hash": receipt["hash"]}
        h["mac"] = _mac(f"{h['seq']}:{h['hash']}")
        HEAD.write_text(json.dumps(h))
        return receipt


def _verify(items, head):
    prev = GENESIS
    for e in items:
        if "mac" not in e:
            return {"intact": False, "receipts": len(items), "broken_at": e.get("seq"),
                    "reason": "receipt is not signed"}
        if e.get("prev") != prev or sha256(_canonical(e)) != e.get("hash"):
            return {"intact": False, "receipts": len(items), "broken_at": e.get("seq"),
                    "reason": "hash chain broken"}
        if not hmac.compare_digest(e["mac"], _mac(e["hash"])):
            return {"intact": False, "receipts": len(items), "broken_at": e.get("seq"),
                    "reason": "signature does not match"}
        prev = e["hash"]
    if items or head:
        ok = bool(head) and hmac.compare_digest(head.get("mac", ""), _mac(f"{head.get('seq')}:{head.get('hash')}")) \
            and head.get("seq") == (items[-1]["seq"] if items else 0) and head.get("hash") == prev
        if not ok:
            return {"intact": False, "receipts": len(items), "broken_at": "end",
                    "reason": "receipts missing from the end, or the head record was changed"}
    return {"intact": True, "receipts": len(items), "signed": len(items), "head": prev}


def verify():
    return _verify(entries(), _head())


def tamper_demo():
    """Edits a COPY of the notebook the way an attacker without the key would
    (change one receipt's outcome, recompute its hash and every later link),
    then runs the same check. The real notebook is not touched."""
    items = copy.deepcopy(entries())
    if len(items) < 2:
        return {"edited": None, "verify": verify()}
    # The edit a cover-up would make: a contained job rewritten as clean.
    contained = [e for e in items if set(e.get("outcome") or []) & {"blocked", "killed", "denied", "tamper"}]
    target = contained[-1] if contained else items[len(items) // 2]
    before = list(target.get("outcome") or [])
    after = ["clean"] if contained else ["blocked"]
    target["outcome"] = after
    target["label"] = str(target.get("label", "")) + " (edited)"
    prev = items[items.index(target) - 1]["hash"] if items.index(target) else GENESIS
    for e in items[items.index(target):]:
        e["prev"] = prev
        e["hash"] = sha256(_canonical(e))
        prev = e["hash"]
    head = dict(_head() or {}, hash=prev)
    return {"edited": {"seq": target["seq"], "outcome_before": before, "outcome_after": after},
            "verify": _verify(items, head)}
