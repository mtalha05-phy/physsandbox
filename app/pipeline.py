"""The troubleshooting loop: summarise -> sandboxed physics -> rank -> next test -> explain."""
import json

from .explain import explain
from .ranking import next_tests, rank_causes
from .sandbox import run_job
from .summariser import summarise

EXPERIMENTS = {
    "resistance_4wire": "Low-resistance 4-wire measurement (V vs I)",
    "waveform": "Oscilloscope waveform (time vs voltage)",
}


def analyse(raw: bytes, experiment: str, expected: dict, use_model=True):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{experiment}'")
    summary = summarise(raw)
    out, events = run_job(
        ["python", "/tools/analyze.py", experiment, json.dumps(expected)], raw, label="physics"
    )
    if not out:
        return {"error": "Analysis did not complete in the sandbox", "containment": events}
    physics = json.loads(out)
    causes = rank_causes(experiment, physics["evidence"])
    # Nothing to chase if the data already match expectations.
    if causes[0]["id"] == "no_fault" and causes[0]["probability"] >= 0.8:
        tests = []
    else:
        tests = next_tests(experiment, causes)
    result = {
        "experiment": experiment,
        "expected": expected,
        "summary": summary,
        "values": physics["values"],
        "evidence": physics["evidence"],
        "plot": physics.get("plot"),
        "causes": causes,
        "next_tests": tests,
        "containment": events,
    }
    if use_model:
        result["explanation"], result["explained_by"] = explain(result)
    return result
