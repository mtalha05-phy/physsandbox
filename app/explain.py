"""Plain-language explanation from the model on Vultr Serverless Inference.

The model receives ONLY: the schema summary, the tool-computed values, the
ranked causes and the chosen next test. It never receives raw file content.
Its text is checked by the safety filter, and it is told not to state any
number that is not in the tool output. If no API key is set, a template
explanation is used so the app still works offline.
"""
import json

from . import llm
from .safety import check_text

SYSTEM = (
    "You are PhysSandbox, a careful lab troubleshooting assistant talking to a student or "
    "researcher. In at most 70 words of plain prose, say what the measurement shows, the leading "
    "suspects with their percentages, and why the recommended test is the best next step. Write "
    "values with SI prefixes and at most 3 significant figures (e.g. 40.0 µV, 9.99 mΩ). Use only "
    "numbers from the tool output. Causes are hypotheses, not verdicts. Never suggest removing a "
    "safety earth or bypassing an interlock. No markdown, no lists."
)


PROPER = ("Earth", "Hall", "Faraday", "Malus", "Newton", "Beer")


def _lc(name):
    # keep proper nouns and acronyms (LED, EMF) as they are
    if name.startswith(PROPER) or (len(name) > 1 and name[1].isupper()):
        return name
    return name[0].lower() + name[1:]


def _template(result):
    top = result["causes"][0]
    status = result.get("status")
    if status == "clear":
        return ("Your data match what you expected within uncertainty. "
                "There is nothing to chase here.")
    if status == "confirmed":
        return (f"Cause confirmed: {top['name']} ({top['probability']:.0%}). "
                "The test you ran ruled out the alternatives.")
    if not result["next_tests"]:
        return f"Most likely: {top['name'].lower()} ({top['probability']:.0%})."
    test = result["next_tests"][0]
    second = result["causes"][1]
    return (f"Your data don't match what you expected. The leading suspect is "
            f"{_lc(top['name'])} ({top['probability']:.0%}), with "
            f"{_lc(second['name'])} ({second['probability']:.0%}) close behind. "
            f"{test['name']} takes about {test['minutes']} min and tells them apart best.")


def explain(result):
    if not llm.configured():
        return _template(result), "template"
    def tidy(v):
        if isinstance(v, float):
            return float(f"{v:.4g}")
        if isinstance(v, dict):
            return {k: tidy(x) for k, x in v.items()}
        if isinstance(v, list):
            return [tidy(x) for x in v]
        return v
    payload = {
        "experiment": result["experiment"], "status": result["status"],
        "values": tidy(result["values"]),
        "causes": [{"name": c["name"], "probability": f"{c['probability']:.0%}"} for c in result["causes"][:3]],
        "next_test": tidy(result["next_tests"][:1]),
    }
    text, err = llm.chat(SYSTEM, json.dumps(payload, default=str)[:12000], max_tokens=1500)
    if text is None:
        return _template(result), f"template (model call failed: {err})"
    ok, reason = check_text(text)
    if not ok:
        return _template(result) + f" [model text withheld: {reason}]", "template (safety)"
    return text, llm.MODEL
