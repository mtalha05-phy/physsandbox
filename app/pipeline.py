"""The troubleshooting loop: summarise -> glovebox physics -> rank -> next test -> explain."""
import json
from pathlib import Path

import yaml

from . import ledger, results
from .explain import explain
from .ranking import CONFIRM_AT, LIB, next_tests, rank_causes, test_outcome
from .sandbox import run_job
from .summariser import summarise

CATALOG = yaml.safe_load((Path(__file__).parent / "catalog.yaml").read_text())
CATALOG.setdefault("labels", {})
# Further experiment groups: app/catalog.d/<group>.yaml with domains, experiments,
# sample_info and labels (display names for the values the analyses return).
for _p in sorted((Path(__file__).parent / "catalog.d").glob("*.yaml")):
    _c = yaml.safe_load(_p.read_text())
    known = {d["id"] for d in CATALOG["domains"]}
    CATALOG["domains"] += [d for d in _c.get("domains", []) if d["id"] not in known]
    CATALOG["experiments"].update(_c.get("experiments", {}))
    CATALOG["sample_info"].update(_c.get("sample_info", {}))
    CATALOG["labels"].update(_c.get("labels", {}))
EXPERIMENTS = {k: v["label"] for k, v in CATALOG["experiments"].items()}


def defaults(experiment):
    return {k: v for k, _, v in CATALOG["experiments"][experiment]["expected"]}


def analyse(raw: bytes, experiment: str, expected: dict, context=None, use_model=True,
            filename=None, owner=None):
    """context (follow-up data): {"after": <receipt hash of the step before>, "test": <test id>}.
    The earlier step's evidence and the tests already done are read from this
    server's stored record of that step, never from the caller.

    Every finished step is stored (app/results.py) for the PDF report, with
    filename (the uploaded file's name) and owner (the browser session that ran
    it); the SHA-256 of that record is sealed into the job's signed receipt."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{experiment}'")
    expected = {**defaults(experiment), **{k: v for k, v in (expected or {}).items() if v is not None}}
    context = context or {}
    test, after = context.get("test"), context.get("after")
    prior = None
    if test or after:
        prior = _prior_step(experiment, after, test, owner)
    summary = summarise(raw)
    sealed = {}

    def seal(report):
        """Rank, choose the next test and explain, then seal the record into the receipt."""
        if not report or not report.get("result"):
            return {}
        physics = report["result"]
        if prior:
            evidence = dict(prior.get("evidence") or {})
            evidence.update({k: v for k, v in test_outcome(test, physics["evidence"], evidence).items()
                             if v is not None})
            done = list(prior.get("done") or []) + [test]
        else:
            evidence, done = physics["evidence"], []
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
        }
        if use_model:
            result["explanation"], result["explained_by"] = explain(result)
        rec = results.record(result, filename=filename, owner=owner, after=after,
                             input_sha256=ledger.sha256(raw))
        sealed.update(result=result, record=rec)
        return {"result_sha256": results.digest(rec)}

    report, events, receipt = run_job(
        ["analyze", experiment, json.dumps(expected)], raw, label=f"physics: {experiment}", seal=seal)
    if "result" not in sealed:
        err = (report or {}).get("error") or "The analysis did not finish in the sandbox."
        return {"error": err.replace("DataError: ", "").replace("__main__.", ""),
                "containment": events, "receipt": receipt}
    result = sealed["result"]
    result.update(containment=events, receipt=receipt, walls=report.get("probe"),
                  stored=results.save(receipt, sealed["record"]))
    return result


def _prior_step(experiment, after, test, owner):
    """The verified stored record of the step a follow-up builds on."""
    tests = LIB["experiments"].get(experiment, {}).get("tests", {})
    if not isinstance(test, str) or test not in tests:
        raise ValueError("Unknown bench test for this experiment.")
    entry = results.receipt_in_notebook(after, ledger.entries()) if isinstance(after, str) else None
    try:
        prior = results.load_verified(after, entry) if entry else None
    except results.StoreError as e:
        raise ValueError(str(e)) from e
    if prior is None or prior.get("owner") != owner:
        raise ValueError("The earlier step of this investigation isn't stored on this server. "
                         "Start again from the first measurement.")
    if prior.get("experiment") != experiment:
        raise ValueError("A follow-up must be for the same experiment as the step before it.")
    if test in (prior.get("done") or []):
        raise ValueError("The result of that test is already in this investigation.")
    return prior
