"""Rank likely causes and choose the next test.

Causes: prior x likelihood of the observed evidence, normalised.
Next test: highest expected information gain (bits) per minute of effort.
All deterministic; the language model never sets these numbers.
"""
import math
from pathlib import Path

import yaml

LIB = yaml.safe_load((Path(__file__).parent.parent / "faults" / "faults.yaml").read_text())


def _holds(cond, evidence):
    x = evidence.get(cond["feature"])
    if x is None:
        return None
    op, v = cond["op"], cond["value"]
    return {
        "<": lambda: x < v,
        ">": lambda: x > v,
        "abs_lt": lambda: abs(x) < v,
        "abs_gt": lambda: abs(x) > v,
        "is": lambda: bool(x) is bool(v),
    }[op]()


def rank_causes(experiment, evidence):
    faults = LIB["experiments"][experiment]["faults"]
    scores = {}
    for key, f in faults.items():
        like = f.get("flat_likelihood", 1.0) if not f["conditions"] else 1.0
        for c in f["conditions"]:
            h = _holds(c, evidence)
            if h is None:
                continue
            like *= c["p_true"] if h else c["p_false"]
        scores[key] = f["prior"] * like
    total = sum(scores.values()) or 1.0
    ranked = sorted(
        ({"id": k, "name": faults[k]["name"], "probability": s / total,
          "typical_size": faults[k].get("typical_size")} for k, s in scores.items()),
        key=lambda r: -r["probability"],
    )
    return ranked


def _entropy(ps):
    return -sum(p * math.log2(p) for p in ps if p > 0)


def next_tests(experiment, ranked, top_n=3):
    tests = LIB["experiments"][experiment]["tests"]
    probs = {r["id"]: r["probability"] for r in ranked}
    out = []
    for key, t in tests.items():
        outcome_mass = {}
        for fault, p in probs.items():
            outcome = t["outcomes"].get(fault, "unpredicted")
            outcome_mass[outcome] = outcome_mass.get(outcome, 0) + p
        gain = _entropy(outcome_mass.values())
        out.append({
            "id": key, "name": t["name"], "how": t["how"], "minutes": t["minutes"],
            "info_gain_bits": round(gain, 3),
            "bits_per_minute": round(gain / t["minutes"], 3),
            "safety": t.get("safety"),
            "predicted_outcomes": t["outcomes"],
        })
    out.sort(key=lambda r: -r["bits_per_minute"])
    return out[:top_n]
