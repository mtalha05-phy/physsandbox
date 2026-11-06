"""Score the troubleshooting loop on seeded datasets with known faults.

Pass bar: correct top cause on every case, the expected status where one is
given (clear / confirmed), and the injection file diagnosed like its clean
twin. Follow-up cases replay the loop: analyse the first file, run the
recommended test, add its outcome, re-rank.

Some first measurements are ambiguous by design: two faults predict the same
data, so the data alone can't tell them apart and the higher-prior one leads.
A case whose true cause is the LOWER-prior partner says so with `step1_top`
(the fault the data alone favour). It passes at step 1 when that fault leads,
the true cause is second, and the recommended test is one whose outcomes the
two faults predict differently; its follow-up case must then confirm the true
cause. Its follow-up (marked `partner: true`) must put the true cause first;
whether it is already confirmed above 90% is reported, not required, because
the data alone favoured the other fault. The summary reports the first
measurement and the follow-ups separately.

Run:  python eval/run_eval.py      (uses SANDBOX_MODE from the environment)
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from app.pipeline import analyse  # noqa: E402
from app.ranking import LIB  # noqa: E402

SEEDED = ROOT / "data" / "seeded"
manifest = yaml.safe_load((ROOT / "data" / "manifest.yaml").read_text())
for _p in sorted((ROOT / "data" / "manifest.d").glob("*.yaml")):
    manifest["datasets"] += yaml.safe_load(_p.read_text())["datasets"]
rows, passed = [], 0
tally = {k: [0, 0] for k in ("first", "partner", "follow-up", "healthy")}
confirmed = [0, 0]   # partner follow-ups: true cause first, and how many reach "confirmed"
for d in manifest["datasets"]:
    context = None
    label = d["file"]
    if d.get("after"):
        first = analyse((SEEDED / d["after"]).read_bytes(), d["experiment"], d.get("expected", {}), use_model=False)
        # The follow-up names the earlier step by its receipt; the server reads that
        # step's evidence from its own stored record.
        context = {"after": first["receipt"]["hash"], "test": d["test"]}
        label = f"{d['after']} + {d['test']}"
    res = analyse((SEEDED / d["file"]).read_bytes(), d["experiment"], d.get("expected", {}), context, use_model=False)
    if "error" in res:
        rows.append((label, d["truth"], "ERROR", "", "", False))
        tally["follow-up" if d.get("after") else "first"][1] += 1
        continue
    top = res["causes"][0]
    nxt = res["next_tests"][0]["id"] if res["next_tests"] else res["status"]
    kind = "follow-up" if d.get("after") else ("healthy" if d["truth"] == "no_fault" else "first")
    if d.get("step1_top") and not d.get("after"):
        # Ambiguous by design, true cause is the lower-prior partner.
        kind = "partner"
        second = res["causes"][1]["id"] if len(res["causes"]) > 1 else None
        tests = LIB["experiments"][d["experiment"]]["tests"]
        outs = (tests.get(nxt) or {}).get("outcomes", {})
        separates = outs.get(d["truth"]) is not None and outs.get(d["truth"]) != outs.get(d["step1_top"])
        ok = top["id"] == d["step1_top"] and second == d["truth"] and separates
    else:
        ok = top["id"] == d["truth"] and (d.get("status") in (None, res["status"]))
    if kind == "follow-up" and d.get("partner"):
        kind = "partner follow-up"
        tally.setdefault(kind, [0, 0])
        confirmed[1] += 1
        confirmed[0] += ok and res["status"] == "confirmed"
    passed += ok
    tally[kind][0] += ok
    tally[kind][1] += 1
    rows.append((label, d["truth"], top["id"], f"{top['probability']:.0%}", nxt, ok))

w = max(len(r[0]) for r in rows)
print(f"{'case':<{w}}  {'truth':<22} {'top cause':<22} {'p':>4}  next")
for r in rows:
    print(f"{r[0]:<{w}}  {r[1]:<22} {r[2]:<22} {r[3]:>4}  {r[4]:<22} {'PASS' if r[5] else 'FAIL'}")
f, pa, fu, he = (tally[k] for k in ("first", "partner", "follow-up", "healthy"))
print(f"\nfirst measurement, fault cases: {f[0]}/{f[1]} name the true cause first")
if pa[1]:
    print(f"first measurement, lower-prior partner of an ambiguous pair: {pa[0]}/{pa[1]} rank it second, "
          "behind the partner the data can't rule out, and recommend the test that separates them")
print(f"after the recommended test: {fu[0]}/{fu[1]} confirm the true cause")
pf = tally.get("partner follow-up")
if pf:
    print(f"after the test, lower-prior partners: {pf[0]}/{pf[1]} now rank the true cause first "
          f"({confirmed[0]} of them above 90%, confirmed; the rest lead and get a second test)")
print(f"healthy runs: {he[0]}/{he[1]} come back clear")
print(f"\n{passed}/{len(rows)} correct")
sys.exit(0 if passed == len(rows) else 1)
