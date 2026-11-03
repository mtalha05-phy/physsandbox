"""Rank likely causes and choose the next test.

Causes: prior x likelihood of the observed evidence, normalised.
Next test: highest expected information gain (bits) per minute of bench time.
Follow-up data: only the OUTCOME of the test that was run is added to the
evidence already collected, so the same observation is never counted twice.
All deterministic; the language model never sets these numbers.
"""
import math
from pathlib import Path

import yaml

LIB = yaml.safe_load((Path(__file__).parent.parent / "faults" / "faults.yaml").read_text())
CONFIRM_AT = 0.90


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
    return sorted(
        ({"id": k, "name": faults[k]["name"], "probability": s / total,
          "typical_size": faults[k].get("typical_size")} for k, s in scores.items()),
        key=lambda r: -r["probability"],
    )


def test_outcome(test_id, new_evidence, prior_evidence):
    """Features that the test just run adds to the investigation."""
    if test_id == "current_reversal":
        return {"offset_survives_reversal": new_evidence.get("offset_survives_reversal"),
                "has_current_reversal": new_evidence.get("has_current_reversal")}
    if test_id == "zero_current_reading":
        return {"offset_only_on_sample": new_evidence.get("offset_only_on_sample")}
    if test_id == "single_point_ground":
        return {"hum_gone_after_fix": new_evidence.get("mains_snr", 0) < 10}
    if test_id == "twist_and_move_leads":
        before = prior_evidence.get("mains_snr", 0) or 1
        return {"hum_dropped_after_twist": new_evidence.get("mains_snr", 0) < before / 3}
    if test_id == "raise_sample_rate":
        return {"signal_at_expected_freq": new_evidence.get("signal_at_expected_freq")}
    if test_id == "reference_resistor":
        return {"slope_error_pct": new_evidence.get("slope_error_pct")}
    return {}


def _entropy(ps):
    return -sum(p * math.log2(p) for p in ps if p > 0)


def next_tests(experiment, ranked, done=(), top_n=3):
    tests = LIB["experiments"][experiment]["tests"]
    probs = {r["id"]: r["probability"] for r in ranked}
    out = []
    for key, t in tests.items():
        if key in done:
            continue
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
    out = [t for t in out if t["info_gain_bits"] > 0.05]
    out.sort(key=lambda r: -r["bits_per_minute"])
    return out[:top_n]
