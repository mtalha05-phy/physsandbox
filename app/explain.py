"""Plain-language explanation from the model on Vultr Serverless Inference.

The model receives ONLY: the schema summary, the tool-computed values, the
ranked causes and the chosen next test. It never receives raw file content.
Its text is checked by the safety filter, and it is told not to state any
number that is not in the tool output. If no API key is set, a template
explanation is used so the app still works offline.
"""
import json
import os

from .safety import check_text

BASE_URL = os.getenv("VULTR_INFERENCE_BASE_URL", "https://api.vultrinference.com/v1")
API_KEY = os.getenv("VULTR_INFERENCE_API_KEY")
MODEL = os.getenv("VULTR_INFERENCE_MODEL", "")

SYSTEM = (
    "You are PhysSandbox, a careful lab troubleshooting assistant. Explain, in at most "
    "120 words, why the measurement differs from what was expected and why the "
    "recommended test is the best next step. Use ONLY numbers that appear in the tool "
    "output; never invent values. Present causes as hypotheses, not verdicts. Never "
    "suggest removing a safety earth or bypassing an interlock."
)


def _template(result):
    top = result["causes"][0]
    if top["id"] == "no_fault" or not result["next_tests"]:
        return (f"The data match what you expected within uncertainty. "
                f"Most likely: {top['name'].lower()} ({top['probability']:.0%}).")
    test = result["next_tests"][0]
    return (f"The data do not match what you expected. Most likely cause: {top['name']} "
            f"({top['probability']:.0%}). Next: {test['name'].lower()} — it takes about "
            f"{test['minutes']} min and best separates the leading causes "
            f"({test['info_gain_bits']} bits).")


def explain(result):
    if not (API_KEY and MODEL):
        return _template(result), "template"
    from openai import OpenAI

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=30)
    payload = {k: result[k] for k in ("experiment", "summary", "values", "causes", "next_tests")}
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": json.dumps(payload, default=str)[:12000]}],
            temperature=0.2,
            max_tokens=300,
        )
        text = resp.choices[0].message.content.strip()
    except Exception:  # noqa: BLE001 - fall back rather than fail the demo
        return _template(result), "template (model unavailable)"
    ok, reason = check_text(text)
    if not ok:
        return _template(result) + f" [model text withheld: {reason}]", "template (safety)"
    return text, MODEL
