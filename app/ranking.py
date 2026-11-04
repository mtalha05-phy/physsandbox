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
        "abs_dev1_lt": lambda: abs(x - 1) < v,
        "abs_dev1_gt": lambda: abs(x - 1) > v,
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


def _copy(*names):
    return lambda new, prior: {n: new.get(n) for n in names}


# What each bench test adds to the evidence. Only the outcome of the test is
# added, so an observation already counted is never counted twice.
OUTCOMES = {
    # low-resistance measurement
    "current_reversal": _copy("offset_survives_reversal", "has_current_reversal"),
    "zero_current_reading": _copy("offset_only_on_sample"),
    "reference_resistor": _copy("slope_error_pct"),
    # oscilloscope
    "single_point_ground": lambda new, prior: {"hum_gone_after_fix": new.get("mains_snr", 0) < 10},
    "twist_and_move_leads": lambda new, prior: {
        "hum_dropped_after_twist": new.get("mains_snr", 0) < (prior.get("mains_snr", 0) or 1) / 3},
    "raise_sample_rate": _copy("signal_at_expected_freq"),
    # Helmholtz coil
    "zero_gauss_chamber": _copy("offset_gone_in_shield"),
    # laser alignment
    "repeat_scan_after_warmup": lambda new, prior: {
        "centre_stable": abs(new.get("centre_offset_mm", 0) - prior.get("centre_offset_mm", 0)) < 0.1},
    # Faraday induction
    "short_amplifier_input": _copy("offset_with_input_shorted"),
    # Beer-Lambert
    "enclosure_rescan": lambda new, prior: {"background_gone_enclosed": new.get("background_sigma", 99) < 3},
    # Malus's law
    "block_beam_reading": _copy("light_with_beam_blocked"),
    # cooling
    "ice_point_check": _copy("offset_at_ice_point"),
    # RC charging
    "x10_probe_repeat": lambda new, prior: {
        "loading_reduced_with_x10": new.get("final_ratio", 0) > prior.get("final_ratio", 1) + 0.2},
}


def test_outcome(test_id, new_evidence, prior_evidence):
    """Features that the test just run adds to the investigation."""
    hook = OUTCOMES.get(test_id)
    return hook(new_evidence, prior_evidence) if hook else {}


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
