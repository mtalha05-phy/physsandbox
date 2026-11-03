"""Score the pipeline on the seeded datasets.

Pass bar: correct top cause on every faulty set, no false alarm on the clean
controls, and the injection set diagnosed the same as its clean twin.

Run:  python -m eval.run_eval          (uses SANDBOX_MODE from the environment)
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from app.pipeline import analyse  # noqa: E402

manifest = yaml.safe_load((ROOT / "data" / "manifest.yaml").read_text())
passed = 0
rows = []
for d in manifest["datasets"]:
    raw = (ROOT / "data" / "seeded" / d["file"]).read_bytes()
    res = analyse(raw, d["experiment"], d["expected"], use_model=False)
    if "error" in res:
        rows.append((d["file"], d["truth"], "ERROR", "-", False))
        continue
    top = res["causes"][0]
    ok = top["id"] == d["truth"]
    passed += ok
    rows.append((d["file"], d["truth"], top["id"], f"{top['probability']:.0%}",
                 ok, (res["next_tests"] or [{"id": "none needed"}])[0]["id"]))

w = max(len(r[0]) for r in rows)
print(f"{'dataset':<{w}}  {'truth':<22} {'top cause':<22} {'p':>5}  next test")
for r in rows:
    mark = "PASS" if r[4] else "FAIL"
    nxt = r[5] if len(r) > 5 else "-"
    print(f"{r[0]:<{w}}  {r[1]:<22} {r[2]:<22} {r[3]:>5}  {nxt:<22} {mark}")
print(f"\n{passed}/{len(rows)} correct")
sys.exit(0 if passed == len(rows) else 1)
