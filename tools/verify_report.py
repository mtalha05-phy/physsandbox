#!/usr/bin/env python3
"""Check a PhysSandbox PDF report yourself, offline, without trusting the server.

    python tools/verify_report.py physsandbox-report.pdf notebook/ed25519-public-key.pem

Every report carries its evidence as an attached file, physsandbox-evidence.json:
each step's signed receipt from the lab notebook and the exact stored result the
step's pages were built from. This script checks:

  1. each receipt's hash and Ed25519 signature, with the public key you give it
     (use the copy published in the repository, not one taken from the PDF);
  2. each result against the SHA-256 sealed in its signed receipt;
  3. the order of the steps (each follow-up names the step before it);
  4. that the page text still says what the signed result says (the last step's
     explanation and its leading cause), so an edited page is caught.

It then prints what the signed record says, so you can compare it with the pages.

Needs Python 3.8+, `cryptography` and `pypdf` (pip install cryptography pypdf).
Exit status: 0 all checks pass, 1 something was changed, 2 the check could not be done.
"""
import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from pypdf import PdfReader
    from verify_receipts import load_public_key, record_problem
except ImportError as e:
    sys.exit(f"This check needs pypdf and cryptography (pip install pypdf cryptography): {e}")

EVIDENCE = "physsandbox-evidence.json"
PASS, FAIL = "ok  ", "FAIL"


def result_digest(rec):
    """SHA-256 of the result's canonical JSON, exactly as the server seals it."""
    text = json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def squash(text):
    return re.sub(r"\s+", " ", text or "").strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check a PhysSandbox PDF report offline.")
    ap.add_argument("pdf")
    ap.add_argument("public_key", help="the notebook's Ed25519 public key (PEM, or the JSON from /api/ledger/public-key)")
    a = ap.parse_args(argv)
    try:
        reader = PdfReader(a.pdf)
        files = reader.attachments
        pub, kid = load_public_key(a.public_key)
    except Exception as e:  # noqa: BLE001
        print(f"Could not read the inputs: {e}")
        return 2
    if EVIDENCE not in files:
        print(f"{FAIL} the PDF has no {EVIDENCE} attached: it is not an original PhysSandbox report, "
              "or the attachment was removed")
        return 1
    try:
        doc = json.loads(files[EVIDENCE][0])
        steps = doc["steps"]
        assert isinstance(steps, list) and steps
    except (ValueError, KeyError, AssertionError, TypeError):
        print(f"{FAIL} the attached evidence is not readable")
        return 1

    bad = 0
    print(f"Public key {kid}" + ("" if doc.get("public_key", {}).get("key_id") == kid else
                                 f" (the report names key {doc.get('public_key', {}).get('key_id')})"))
    prev_hash, exp = None, None
    for i, st in enumerate(steps, 1):
        r, res = st.get("receipt") or {}, st.get("result") or {}
        problem = record_problem(r, pub, kid)
        if problem:
            print(f"{FAIL} step {i}: receipt #{r.get('seq')} {problem}")
            bad += 1
        else:
            print(f"{PASS} step {i}: receipt #{r.get('seq')} ({r.get('time')}) is signed by key {kid}")
        sealed = r.get("result_sha256")
        if not sealed or result_digest(res) != sealed:
            print(f"{FAIL} step {i}: the result does not match the SHA-256 sealed in its receipt (it was edited)")
            bad += 1
        else:
            print(f"{PASS} step {i}: the result matches the hash sealed in the receipt ({sealed[:16]}…)")
        if i > 1 and res.get("after") != prev_hash:
            print(f"{FAIL} step {i}: is not the follow-up to step {i - 1}")
            bad += 1
        if exp is not None and res.get("experiment") != exp:
            print(f"{FAIL} step {i}: is from a different experiment")
            bad += 1
        prev_hash, exp = r.get("hash"), res.get("experiment")

    # The pages: the text a reader sees must still say what the signed result says.
    try:
        page_text = squash(" ".join(p.extract_text() or "" for p in reader.pages))
    except Exception:  # noqa: BLE001
        page_text = ""
    last = steps[-1].get("result") or {}
    expl = squash(last.get("explanation"))
    if expl:
        # Compare a run of words from the explanation (layout may split long lines).
        words = re.sub(r"[^\w.%±−-]+", " ", expl).split()[:25]
        probe = " ".join(words)
        flat = " ".join(re.sub(r"[^\w.%±−-]+", " ", page_text).split())
        if probe and probe in flat:
            print(f"{PASS} the pages show the signed explanation of the last step")
        else:
            print(f"{FAIL} the pages do not show the signed explanation of the last step: "
                  "the visible text was changed")
            bad += 1

    print("\nWhat the signed record says (the pages must agree):")
    for i, st in enumerate(steps, 1):
        res = st.get("result") or {}
        causes = sorted(res.get("causes") or [], key=lambda c: -(c.get("probability") or 0))[:3]
        top = ", ".join(f"{c.get('id')} {100 * (c.get('probability') or 0):.0f}%" for c in causes)
        print(f"  step {i}: {res.get('experiment')} · {res.get('file')} · {res.get('status')}"
              + (f" · after test {res.get('test_run')}" if res.get("test_run") else ""))
        print(f"          causes: {top}")
        vals = res.get("values") or {}
        shown = []
        for k, v in list(vals.items())[:8]:
            v = v if isinstance(v, dict) else {"value": v}
            u = f" ± {v['uncertainty']:.4g}" if isinstance(v.get("uncertainty"), (int, float)) else ""
            if isinstance(v.get("value"), (int, float)):
                shown.append(f"{k} = {v['value']:.6g}{u}")
        if shown:
            print("          " + "; ".join(shown))
    print()
    if bad:
        print(f"{bad} problem(s): this report was changed after the server produced it.")
        return 1
    print("All checks pass: the report is what the server computed and signed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
