"""Score the troubleshooting loop on seeded datasets with known faults.

Pass bar: correct top cause on every case, the expected status where one is
given (clear / confirmed), and the injection file diagnosed like its clean
twin. Follow-up cases replay the loop: analyse the first file, run the
recommended test, add its outcome, re-rank.

Run:  python eval/run_eval.py      (uses SANDBOX_MODE from the environment)
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from app.pipeline import analyse  # noqa: E402

SEEDED = ROOT / "data" / "seeded"
manifest = yaml.safe_load((ROOT / "data" / "manifest.yaml").read_text())
rows, passed = [], 0
for d in manifest["datasets"]:
    context = None
    label = d["file"]
    if d.get("after"):
        first = analyse((SEEDED / d["after"]).read_bytes(), d["experiment"], d["expected"], use_model=False)
        context = {"evidence": first["evidence"], "test": d["test"], "done": first["done"]}
        label = f"{d['after']} + {d['test']}"
    res = analyse((SEEDED / d["file"]).read_bytes(), d["experiment"], d["expected"], context, use_model=False)
    if "error" in res:
        rows.append((label, d["truth"], "ERROR", "", "", False))
        continue
    top = res["causes"][0]
    ok = top["id"] == d["truth"] and (d.get("status") in (None, res["status"]))
    passed += ok
    nxt = res["next_tests"][0]["id"] if res["next_tests"] else res["status"]
    rows.append((label, d["truth"], top["id"], f"{top['probability']:.0%}", nxt, ok))

w = max(len(r[0]) for r in rows)
print(f"{'case':<{w}}  {'truth':<22} {'top cause':<22} {'p':>4}  next")
for r in rows:
    print(f"{r[0]:<{w}}  {r[1]:<22} {r[2]:<22} {r[3]:>4}  {r[4]:<22} {'PASS' if r[5] else 'FAIL'}")
print(f"\n{passed}/{len(rows)} correct")
sys.exit(0 if passed == len(rows) else 1)
