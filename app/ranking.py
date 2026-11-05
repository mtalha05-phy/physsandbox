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

_FAULTS = Path(__file__).parent.parent / "faults"
LIB = yaml.safe_load((_FAULTS / "faults.yaml").read_text())
# Further experiment groups: faults/bench_<group>.yaml, each with an `experiments:` map.
for _p in sorted(_FAULTS.glob("bench_*.yaml")):
    LIB["experiments"].update(yaml.safe_load(_p.read_text())["experiments"])
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
    "enclosure_rescan": lambda new, prior: {"background_gone_enclosed": new.get("background_sigma", 99) < 4},
    # Malus's law
    "block_beam_reading": _copy("light_with_beam_blocked"),
    # cooling
    "ice_point_check": _copy("offset_at_ice_point"),
    # RC charging
    "x10_probe_repeat": lambda new, prior: {
        "loading_reduced_with_x10": new.get("final_ratio", 0) > prior.get("final_ratio", 1) + 0.2},
    # pendulum
    "small_amplitude_repeat": lambda new, prior: {"g_fixed_at_small_amplitude": abs(new.get("g_error_pct", 99)) < 1},
    # Hooke's law
    "include_hanger_repeat": lambda new, prior: {"offset_gone_with_hanger": new.get("zero_force_sigma", 99) < 4},
    # diode
    "voltmeter_on_diode": lambda new, prior: {"droop_gone_after_rewire": new.get("rs_excess_sigma", 99) < 4},
    # diffraction grating
    "reference_line_check": lambda new, prior: {
        "grating_confirmed_wrong": abs(new.get("reference_error_pct", 0)) > 2},
    # Hall effect
    "reverse_field": lambda new, prior: {"misalignment_found": new.get("misalignment_fraction", 0) > 0.05},
}


def test_outcome(test_id, new_evidence, prior_evidence):
    """Features that the test just run adds to the investigation."""
    hook = OUTCOMES.get(test_id)
    if hook:
        return hook(new_evidence, prior_evidence)
    # Declarative form in a fault library test:
    #   adds: {feature: offset_gone, from: offset_sigma, op: "<", value: 4}
    # (op "copy" passes the follow-up feature through unchanged)
    for exp in LIB["experiments"].values():
        spec = (exp.get("tests", {}).get(test_id) or {}).get("adds")
        if spec:
            v = new_evidence.get(spec["from"])
            if spec.get("op", "copy") == "copy":
                return {spec["feature"]: v} if v is not None else {}
            if v is None:
                return {}
            return {spec["feature"]: bool(_holds({"feature": "v", "op": spec["op"], "value": spec["value"]}, {"v": v}))}
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
        # Expected information = mutual information between the cause and the test's
        # outcome. Faults whose outcome the library doesn't predict for this test carry
        # no information: assume their outcome follows the same mix as the predicted
        # ones. Then MI = (share of probability that is predicted) x H(predicted mix).
        outcome_mass, predicted = {}, 0.0
        for fault, p in probs.items():
            outcome = t["outcomes"].get(fault)
            if outcome is None:
                continue
            outcome_mass[outcome] = outcome_mass.get(outcome, 0) + p
            predicted += p
        gain = predicted * _entropy([m / predicted for m in outcome_mass.values()]) if predicted > 0 else 0.0
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
