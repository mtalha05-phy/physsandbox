#!/usr/bin/env python3
"""Check a PhysSandbox lab notebook yourself, without trusting the server.

    curl -s 'https://HOST/api/ledger?limit=100' -o export.json
    curl -s 'https://HOST/api/ledger/public-key?format=pem' -o notebook.pem
    python verify_receipts.py export.json notebook.pem
    python verify_receipts.py export.json notebook.pem --expect-head physsandbox-record.md

It recomputes every hash and checks every Ed25519 signature in the export
(the genesis record, each receipt and the head), and the hash chain between
them. The server's own verdict inside the export is ignored.

--expect-head takes a head you saved earlier: a downloaded investigation
record (.md), a saved /api/status or /api/ledger response, a head record on
its own, or "COUNT:LAST_HASH". It catches what the notebook cannot catch on
its own: a rollback to an older copy (the count went down, or receipt #COUNT
no longer has your hash) and a wipe (a different genesis: a new notebook).

The export holds at most 100 receipts. If your head is older than that,
export from it: /api/ledger?start=COUNT&limit=100.

Needs Python 3.8+ and the `cryptography` package (pip install cryptography).
Exit status: 0 all checks pass, 1 tampering, rollback or wipe found,
2 the check could not be completed (bad input, or the export does not cover it).
"""
import argparse
import hashlib
import json
import re
import sys

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
except ImportError:
    sys.exit("This check needs the cryptography package: pip install cryptography")

ZERO = "0" * 64
PASS, FAIL, NOTE = "ok  ", "FAIL", "note"


def canonical(rec):
    """The bytes the hash covers: every field except hash and sig."""
    body = {k: v for k, v in rec.items() if k not in ("hash", "sig")}
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def load_public_key(path):
    text = open(path, encoding="utf-8").read()
    if text.lstrip().startswith("{"):      # the JSON from /api/ledger/public-key
        text = json.loads(text)["pem"]
    key = serialization.load_pem_public_key(text.encode())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("not an Ed25519 public key")
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return key, hashlib.sha256(raw).hexdigest()[:16]


def record_problem(rec, pub, kid):
    """None if the record's hash and Ed25519 signature check out."""
    if not isinstance(rec, dict):
        return "is not a JSON object"
    if not isinstance(rec.get("sig"), str):
        return "is not signed"
    if hashlib.sha256(canonical(rec).encode()).hexdigest() != rec.get("hash"):
        return "does not match its hash (it was edited)"
    if rec.get("key_id") != kid:
        return f"was signed by key {rec.get('key_id')}, not by this public key ({kid})"
    try:
        pub.verify(bytes.fromhex(rec["sig"]), rec["hash"].encode("ascii"))
    except (InvalidSignature, ValueError):
        return "has a signature that does not match (it was changed after it was signed)"
    return None


def find_head(obj):
    """A head record inside whatever the user saved."""
    if isinstance(obj, dict):
        if obj.get("kind") == "head":
            return obj
        for k in ("head_record", "ledger", "verify"):
            h = find_head(obj.get(k))
            if h:
                return h
    return None


def load_expected(arg):
    m = re.fullmatch(r"#?(\d+):([0-9a-f]{64})", arg.strip())
    if m:
        return {"count": int(m.group(1)), "last_hash": m.group(2)}
    text = open(arg, encoding="utf-8").read()
    candidates = [text] + re.findall(r"```(?:json)?\s*\n(.*?)\n```", text, re.S)
    for c in candidates:
        try:
            h = find_head(json.loads(c))
        except ValueError:
            continue
        if h:
            return h
    raise ValueError(f"no head record found in {arg}")


def short(h):
    return (h or "")[:16] + "…" if h else "n/a"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check a PhysSandbox lab notebook export with its public key.")
    ap.add_argument("export", help="JSON saved from GET /api/ledger?limit=100")
    ap.add_argument("public_key", help="PEM saved from GET /api/ledger/public-key?format=pem (or its JSON)")
    ap.add_argument("--expect-head", metavar="HEAD",
                    help="a head you saved earlier (record .md, status/export JSON, head JSON, or COUNT:HASH)")
    a = ap.parse_args(argv)

    lines, failed, incomplete = [], [], []

    def say(mark, what, text):
        lines.append(f"  [{mark}] {what:<10} {text}")
        if mark == FAIL:
            failed.append(f"{what}: {text}")

    try:
        pub, kid = load_public_key(a.public_key)
        export = json.load(open(a.export, encoding="utf-8"))
        expected = load_expected(a.expect_head) if a.expect_head else None
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"Could not read the input: {e}")
        return 2
    if not isinstance(export, dict) or not isinstance(export.get("receipts"), list):
        print("This doesn't look like a /api/ledger export (no receipts list).")
        return 2

    print("PhysSandbox lab notebook check (this script's own check, not the server's)")
    say(NOTE, "Key", f"Ed25519, key_id {kid} (from {a.public_key})")

    # ---- genesis: the signed start of the notebook
    g = export.get("genesis")
    g_hash = None
    if not isinstance(g, dict) or g.get("kind") != "genesis" or g.get("seq") != 0 or g.get("prev") != ZERO:
        say(FAIL, "Genesis", "missing from the export: the notebook's signed start can't be checked")
    else:
        why = record_problem(g, pub, kid)
        if why:
            say(FAIL, "Genesis", "the genesis record " + why)
        else:
            g_hash = g["hash"]
            say(PASS, "Genesis", f"notebook started {g.get('time')} (id {g.get('notebook_id')}), signed")

    # ---- receipts: each one's hash and signature, and the links between them
    rs = export["receipts"]
    first = rs[0].get("seq") if rs and isinstance(rs[0], dict) else None
    prev, seq, bad = None, None, None
    for i, r in enumerate(rs):
        if not isinstance(r, dict) or not isinstance(r.get("seq"), int):
            bad = f"entry {i + 1} has no receipt number"
            break
        n = r["seq"]
        why = record_problem(r, pub, kid)
        if why:
            bad = f"receipt #{n} {why}"
            break
        if r.get("kind") is not None or (seq is not None and n != seq + 1):
            bad = f"receipt #{n} is out of order (expected #{seq + 1 if seq is not None else 1})"
            break
        if prev is None:
            if n == 1 and g_hash and r.get("prev") != g_hash:
                bad = "receipt #1 does not link to the genesis record"
                break
        elif r.get("prev") != prev:
            bad = f"hash chain broken at receipt #{n}: it does not link to #{seq}"
            break
        prev, seq = r["hash"], n
    if bad:
        say(FAIL, "Receipts", bad)
    elif not rs:
        say(NOTE, "Receipts", "none in this export")
    else:
        span = f"#{first} to #{seq} ({len(rs)} receipts)"
        say(PASS, "Receipts", f"{span}: hashes, chain links and Ed25519 signatures check out")
        if first != 1:
            say(NOTE, "Receipts", f"#1 to #{first - 1} are not in this export; receipt #{first}'s signed "
                                  f"link (prev {short(rs[0].get('prev'))}) covers them")

    # ---- head: the signed count and last hash
    h = export.get("head_record")
    head_ok = False
    if not isinstance(h, dict) or h.get("kind") != "head":
        say(FAIL, "Head", "missing from the export: receipts deleted off the end would not show")
    else:
        why = record_problem(h, pub, kid)
        if why:
            say(FAIL, "Head", "the head record " + why)
        elif g_hash and h.get("genesis_hash") != g_hash:
            say(FAIL, "Head", "belongs to a different notebook than the genesis record")
        elif not bad:
            count, last = h.get("count"), h.get("last_hash")
            if not isinstance(count, int):
                say(FAIL, "Head", "has no receipt count")
            elif rs and seq == count:
                if last == prev:
                    head_ok = True
                    say(PASS, "Head", f"count {count}, last hash {short(last)}: matches the last receipt, signed")
                else:
                    say(FAIL, "Head", f"last hash {short(last)} is not receipt #{count}'s hash")
            elif rs and seq > count:
                say(FAIL, "Head", f"says {count} receipts, but the export goes on to #{seq}")
            elif not rs and count == 0 and last == g_hash:
                head_ok = True
                say(PASS, "Head", "count 0: the notebook has no receipts yet, signed")
            elif not rs and count > 0:
                incomplete.append("head")
                say(NOTE, "Head", f"count {count}, but the export has no receipts to compare it with")
            elif (export.get("window") or {}).get("start"):
                incomplete.append("head")
                say(NOTE, "Head", f"count {count}, but this export (from ?start=) ends at #{seq}: "
                                  "receipts after it were not checked")
            else:
                gone = f"#{count} was" if count == seq + 1 else f"#{seq + 1} to #{count} were"
                say(FAIL, "Head", f"says {count} receipts, but the latest receipt is #{seq}: "
                                  f"{gone} deleted off the end")

    # ---- an earlier head you kept: rollback and wipe
    if expected:
        ec, eh = expected.get("count"), expected.get("last_hash")
        if not isinstance(ec, int) or not isinstance(eh, str):
            say(FAIL, "Your head", "has no count and last hash")
        else:
            say(NOTE, "Your head", f"count {ec}, last hash {short(eh)}"
                + (f", notebook started {expected.get('genesis_time')}" if expected.get("genesis_time") else ""))
            if expected.get("sig") is not None:
                why = record_problem(expected, pub, kid)
                if why and expected.get("key_id") != kid:
                    say(FAIL, "Your head", f"was signed by key {expected.get('key_id')}; this notebook is signed "
                                           f"by {kid}: the signing key was replaced")
                elif why:
                    say(FAIL, "Your head", "your saved copy " + why)
            if expected.get("genesis_hash") and g_hash and expected["genesis_hash"] != g_hash:
                say(FAIL, "Wipe", f"this is a different notebook: yours started {expected.get('genesis_time')}, "
                                  f"this one started {g.get('time')}. The notebook was wiped or replaced.")
            elif isinstance(h, dict) and isinstance(h.get("count"), int) and h["count"] < ec:
                say(FAIL, "Rollback", f"the notebook now has {h['count']} receipts; your head had {ec}. "
                                      "It was rolled back to an older copy, or receipts were removed.")
            else:
                got = next((r for r in rs if isinstance(r, dict) and r.get("seq") == ec), None)
                if ec == 0 and g_hash:
                    got = {"hash": g_hash}
                if got is None:
                    incomplete.append("expect")
                    say(NOTE, "Rollback", f"receipt #{ec} is not in this export, so your head can't be compared: "
                                          f"export with /api/ledger?start={ec}&limit=100")
                elif got.get("hash") != eh:
                    say(FAIL, "Rollback", f"receipt #{ec} now has hash {short(got.get('hash'))}, not your "
                                          f"{short(eh)}: the history up to your head was rewritten")
                else:
                    say(PASS, "Rollback", f"receipt #{ec} still has your hash, so nothing up to your head changed, "
                                          "and the notebook has not been rolled back past it")

    print("\n".join(lines))
    if failed:
        print(f"\nRESULT: NOT INTACT. {failed[0]}")
        return 1
    if incomplete or not head_ok:
        print("\nRESULT: INCOMPLETE. Everything this export contains checks out, but it does not cover "
              "the whole check (see the notes above).")
        return 2
    msg = "Every record in this export is signed by this key, the chain is unbroken and the head matches."
    if not expected:
        msg += (" On its own this cannot rule out a rollback or a wipe: keep this export's head "
                "and pass it with --expect-head next time.")
    print(f"\nRESULT: INTACT. {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
