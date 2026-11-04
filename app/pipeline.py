"""The troubleshooting loop: summarise -> glovebox physics -> rank -> next test -> explain."""
import json
from pathlib import Path

import yaml

from .explain import explain
from .ranking import CONFIRM_AT, next_tests, rank_causes, test_outcome
from .sandbox import run_job
from .summariser import summarise

CATALOG = yaml.safe_load((Path(__file__).parent / "catalog.yaml").read_text())
EXPERIMENTS = {k: v["label"] for k, v in CATALOG["experiments"].items()}


def defaults(experiment):
    return {k: v for k, _, v in CATALOG["experiments"][experiment]["expected"]}


def analyse(raw: bytes, experiment: str, expected: dict, context=None, use_model=True):
    """context (follow-up data): {"evidence": {...}, "test": "<test id>", "done": [...]}"""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{experiment}'")
    expected = {**defaults(experiment), **{k: v for k, v in (expected or {}).items() if v is not None}}
    summary = summarise(raw)
    report, events, receipt = run_job(
        ["analyze", experiment, json.dumps(expected)], raw, label=f"physics: {experiment}")
    if not report or not report.get("result"):
        err = (report or {}).get("error") or "The analysis did not finish in the sandbox."
        return {"error": err.replace("DataError: ", "").replace("__main__.", ""),
                "containment": events, "receipt": receipt}
    physics = report["result"]

    context = context or {}
    test = context.get("test")
    done = list(context.get("done") or [])
    if test:
        evidence = dict(context.get("evidence") or {})
        evidence.update({k: v for k, v in test_outcome(test, physics["evidence"], evidence).items()
                         if v is not None})
        done.append(test)
    else:
        evidence = physics["evidence"]

    causes = rank_causes(experiment, evidence)
    top = causes[0]
    if top["probability"] >= CONFIRM_AT:
        status, tests = ("clear" if top["id"] == "no_fault" else "confirmed"), []
    else:
        status, tests = "investigating", next_tests(experiment, causes, done)

    result = {
        "experiment": experiment,
        "expected": expected,
        "summary": summary,
        "values": physics["values"],
        "evidence": evidence,
        "plot": physics.get("plot"),
        "causes": causes,
        "status": status,
        "next_tests": tests,
        "done": done,
        "test_run": test,
        "containment": events,
        "receipt": receipt,
        "walls": report.get("probe"),
    }
    if use_model:
        result["explanation"], result["explained_by"] = explain(result)
    return result
