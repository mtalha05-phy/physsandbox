"""Tamper-evident lab notebook of containment receipts.

Every sandboxed job appends one receipt: SHA-256 of the input, code, output
and sandbox image, the walls measured inside the box and checked by the host,
and the hash of the previous record.

Records. The notebook file (RECEIPT_LEDGER, default receipts.jsonl) holds one
JSON record per line: first a genesis record (seq 0, kind "genesis", the
notebook's creation time and a random notebook id), then the receipts (seq 1,
2, ...), each chained to the record before it by "prev". A head record (kind
"head": count, last_hash, genesis_hash, genesis_time, notebook_id) sits in a
second file and is rewritten on every append, so deleting receipts off the
end is caught.

Signatures. Every record, the head included, carries
    hash   = SHA-256 of its canonical JSON (every field except "hash" and
             "sig"; json.dumps(sort_keys=True, separators=(",", ":"))),
             64 lowercase hex characters
    key_id = first 16 hex characters of SHA-256 of the raw 32-byte public key
             (inside the hashed body, so it is signed too)
    sig    = Ed25519 signature over the ASCII bytes of "hash",
             128 lowercase hex characters
Ed25519 is public-key: anyone with the public key (GET /api/ledger/public-key)
can check every signature with tools/verify_receipts.py, without trusting this
server or its own check. Unsigned records are rejected, so stripping the
signatures and rebuilding the chain fails.

What the notebook cannot catch on its own (see the README's threat model):
  - rollback: restoring an older copy of both files gives an older notebook
    that still verifies. An earlier head kept elsewhere catches it (the count
    went down, or receipt #count no longer has that head's hash).
  - wipe: if the files are deleted, or the head is missing or unverifiable,
    the next job starts a new notebook (old files are archived beside it, not
    deleted) with a new genesis time and id. verify() reports that time, so a
    wipe shows as "notebook started <time>" and as a different genesis to
    anyone holding an earlier head.
  - the private key lives outside the repository and every sandbox (default
    ~/.physsandbox/ledger_ed25519.pem, mode 0600, or LEDGER_SIGNING_KEY_FILE)
    but on the same host: root on the VM could sign a rewritten history. An
    earlier head kept elsewhere still catches any rewrite of receipts before it.
"""
import copy
import hashlib
import json
import os
import secrets
import threading
import time
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).parent.parent
LEDGER = Path(os.getenv("RECEIPT_LEDGER", ROOT / "receipts.jsonl"))
HEAD = LEDGER.with_name(LEDGER.stem + ".head.json")
KEY_FILE = Path(os.getenv("LEDGER_SIGNING_KEY_FILE",
                          Path.home() / ".physsandbox" / "ledger_ed25519.pem")).expanduser()
ALG = "Ed25519"
ZERO = "0" * 64                     # "prev" of the genesis record
_lock = threading.RLock()
_keys = {}


# ---------- keys and signatures

def _private_key():
    path = Path(KEY_FILE)
    key = _keys.get(path)
    if key is None:
        with _lock:
            key = _keys.get(path) or _load_or_create(path)
            _keys[path] = key
    return key


def _load_or_create(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        pem = Ed25519PrivateKey.generate().private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        try:    # created 0600 from the start, never readable by others even briefly
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as f:
                f.write(pem)
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"{path} is not an Ed25519 private key")
    return key


def _public():
    return _private_key().public_key()


def key_id(pub=None) -> str:
    raw = (pub or _public()).public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]


def public_key() -> dict:
    pub = _public()
    raw = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return {
        "alg": ALG, "key_id": key_id(pub), "sha256": hashlib.sha256(raw).hexdigest(),
        "pem": pub.public_bytes(serialization.Encoding.PEM,
                                serialization.PublicFormat.SubjectPublicKeyInfo).decode(),
        "signs": "the ASCII bytes of each record's hash (SHA-256 hex of its canonical JSON "
                 "without hash and sig); sig is the 64-byte signature in hex",
    }


def _sign(h: str) -> str:
    return _private_key().sign(h.encode("ascii")).hex()


def _sig_ok(pub, h, sig) -> bool:
    try:
        pub.verify(bytes.fromhex(sig), h.encode("ascii"))
        return True
    except (InvalidSignature, ValueError, TypeError, AttributeError):
        return False


def sha256(data) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data or b"").hexdigest()


def _canonical(entry) -> str:
    body = {k: v for k, v in entry.items() if k not in ("hash", "sig")}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)


def _seal(record: dict) -> dict:
    record["key_id"] = key_id()
    record["hash"] = sha256(_canonical(record))
    record["sig"] = _sign(record["hash"])
    return record


def _problem(rec, pub, kid):
    """Why a record's own hash or signature does not check out, or None."""
    if not isinstance(rec.get("sig"), str):
        return "is not signed"
    if sha256(_canonical(rec)) != rec.get("hash"):
        return "does not match its hash"
    if rec.get("key_id") != kid:
        return f"was signed with a different key ({rec.get('key_id')}, not {kid})"
    if not _sig_ok(pub, rec["hash"], rec["sig"]):
        return "has a signature that does not match, so it was changed after it was signed"
    return None


# ---------- reading

def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _records():
    """Every line of the notebook, genesis first. Unreadable lines are kept
    as markers so verify() reports them instead of skipping them."""
    if not LEDGER.exists():
        return []
    out = []
    for n, line in enumerate(LEDGER.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            rec = None
        out.append(rec if isinstance(rec, dict) else {"_unreadable": n})
    return out


def entries():
    """The receipts, oldest first (the genesis record is not a receipt)."""
    return [r for r in _records() if r.get("kind") != "genesis" and "_unreadable" not in r]


def genesis():
    recs = _records()
    return recs[0] if recs and recs[0].get("kind") == "genesis" else None


def head_record():
    try:
        h = json.loads(HEAD.read_text())
    except (OSError, ValueError):
        return None
    return h if isinstance(h, dict) else None


# ---------- writing

def _head_usable(h, pub, kid):
    return bool(h) and h.get("kind") == "head" and _problem(h, pub, kid) is None \
        and isinstance(h.get("count"), int) and isinstance(h.get("last_hash"), str)


def _write_head(count, last_hash, g):
    h = _seal({"kind": "head", "count": count, "last_hash": last_hash, "time": _now(),
               "genesis_hash": g["genesis_hash"], "genesis_time": g["genesis_time"],
               "notebook_id": g["notebook_id"]})
    tmp = HEAD.with_name(HEAD.name + ".tmp")
    tmp.write_text(json.dumps(h))
    os.replace(tmp, HEAD)
    return h


def _start_notebook():
    """A new notebook: archive whatever is there (never delete it), then
    write and sign the genesis record and a head with count 0."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for path, suffix in ((LEDGER, LEDGER.suffix), (HEAD, ".head.json")):
        if path.exists():
            n, dest = 0, LEDGER.with_name(f"{LEDGER.stem}.archived-{stamp}{suffix}")
            while dest.exists():
                n += 1
                dest = LEDGER.with_name(f"{LEDGER.stem}.archived-{stamp}-{n}{suffix}")
            path.rename(dest)
    g = _seal({"seq": 0, "kind": "genesis", "time": _now(), "notebook_id": secrets.token_hex(8),
               "alg": ALG, "prev": ZERO,
               "about": "PhysSandbox lab notebook: containment receipts, hash-chained and Ed25519-signed"})
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("w") as f:
        f.write(json.dumps(g) + "\n")
    return _write_head(0, g["hash"], {"genesis_hash": g["hash"], "genesis_time": g["time"],
                                      "notebook_id": g["notebook_id"]})


def append(receipt: dict) -> dict:
    with _lock:
        pub = _public()
        head = head_record()
        if not _head_usable(head, pub, key_id(pub)):
            # First job, or the head is gone, edited, signed by another key or
            # left by the old HMAC notebook: start a new notebook, visibly.
            head = _start_notebook()
        receipt = {k: v for k, v in receipt.items() if k not in ("hash", "sig", "mac", "key_id")}
        receipt["seq"] = head["count"] + 1
        receipt["time"] = _now()
        receipt["prev"] = head["last_hash"]
        _seal(receipt)
        with LEDGER.open("a") as f:
            f.write(json.dumps(receipt, default=str) + "\n")
        _write_head(receipt["seq"], receipt["hash"], head)
        return receipt


# ---------- checking

def _verify(records, head, pub=None):
    pub = pub or _public()
    kid = key_id(pub)
    receipts = [r for r in records if r.get("kind") != "genesis"]
    n = len(receipts)
    info = {"alg": ALG, "key_id": kid, "started": None, "notebook_id": None, "genesis": None,
            "head_record": head}

    def broken(at, reason):
        return {"intact": False, "receipts": n, "broken_at": at, "reason": reason, **info}

    if not records and not head:
        return {"intact": True, "receipts": 0, "signed": 0, "head": None, **info,
                "note": "no receipts yet: the notebook starts with the first job"}
    g = records[0] if records else None
    if not g or g.get("kind") != "genesis" or g.get("seq") != 0 or g.get("prev") != ZERO:
        return broken(0, "the genesis record (the notebook's signed start) is missing")
    why = _problem(g, pub, kid)
    if why:
        return broken(0, "the genesis record " + why)
    info.update(started=g.get("time"), notebook_id=g.get("notebook_id"),
                genesis={k: g.get(k) for k in ("time", "notebook_id", "hash", "key_id")})
    prev, seq = g["hash"], 0
    for e in records[1:]:
        seq += 1
        if "_unreadable" in e:
            return broken(seq, f"line {e['_unreadable']} of the notebook is not valid JSON")
        if "sig" not in e:
            return broken(e.get("seq"), "receipt is not signed")
        if e.get("seq") != seq or e.get("kind") is not None:
            return broken(e.get("seq"), f"receipt out of order (expected #{seq})")
        if e.get("prev") != prev:
            return broken(seq, "hash chain broken")
        why = _problem(e, pub, kid)
        if why:
            return broken(seq, "receipt " + why)
        prev = e["hash"]
    ok = _head_usable(head, pub, kid) and head.get("count") == seq and head.get("last_hash") == prev \
        and head.get("genesis_hash") == g["hash"]
    if not ok:
        return broken("end", "receipts missing from the end, or the head record was changed")
    return {"intact": True, "receipts": n, "signed": n, "head": prev, **info}


def verify():
    """The server's own check. Same rules as tools/verify_receipts.py."""
    with _lock:
        return _verify(_records(), head_record())


def tamper_demo():
    """Edits a COPY of the notebook the way an attacker without the private
    key would (change one receipt's outcome, recompute its hash, every later
    link and the head), then runs the same check. The real notebook is not
    touched."""
    with _lock:
        records = copy.deepcopy(_records())
        head = copy.deepcopy(head_record())
    items = [r for r in records if r.get("kind") != "genesis" and "_unreadable" not in r]
    if len(items) < 2:
        return {"edited": None, "verify": verify()}
    # The edit a cover-up would make: a contained job rewritten as clean.
    contained = [e for e in items if set(e.get("outcome") or []) & {"blocked", "killed", "denied", "tamper"}]
    target = contained[-1] if contained else items[len(items) // 2]
    before = list(target.get("outcome") or [])
    after = ["clean"] if contained else ["blocked"]
    target["outcome"] = after
    target["label"] = str(target.get("label", "")) + " (edited)"
    i = next(k for k, r in enumerate(records) if r is target)
    prev = records[i - 1].get("hash") if i else ZERO
    for e in records[i:]:
        e["prev"] = prev
        e["hash"] = sha256(_canonical(e))      # the old signatures stay: no key to re-sign
        prev = e["hash"]
    head = dict(head or {}, last_hash=prev)
    head["hash"] = sha256(_canonical(head))
    return {"edited": {"seq": target.get("seq"), "outcome_before": before, "outcome_after": after},
            "verify": _verify(records, head)}
