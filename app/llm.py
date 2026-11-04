"""One place for calls to Vultr Serverless Inference (OpenAI-compatible).

Reasoning models can spend their token budget thinking and return the final
answer late, or put it in a separate field, so budgets are generous and both
fields are read. Errors are reported by type only, never with the key.
"""
import os
import re
import time

BASE_URL = os.getenv("VULTR_INFERENCE_BASE_URL", "https://api.vultrinference.com/v1")
API_KEY = os.getenv("VULTR_INFERENCE_API_KEY")
MODEL = os.getenv("VULTR_INFERENCE_MODEL", "")
LAST = {"ok": None, "error": None, "seconds": None}


def configured():
    return bool(API_KEY and MODEL)


def _strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def chat(system, user, max_tokens=2000, temperature=0.2, timeout=60):
    """Returns (text, error). text is None when the call fails."""
    if not configured():
        return None, "no model key set"
    from openai import OpenAI

    started = time.time()
    try:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=timeout)
        resp = client.chat.completions.create(
            model=MODEL, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        msg = resp.choices[0].message
        text = _strip_thinking(msg.content)
        if not text:
            extra = getattr(msg, "reasoning_content", None) or (getattr(msg, "model_extra", None) or {}).get("reasoning_content")
            text = _strip_thinking(extra)
        if not text:
            raise ValueError("model returned an empty answer")
        LAST.update(ok=True, error=None, seconds=round(time.time() - started, 2))
        return text, None
    except Exception as e:  # noqa: BLE001 - the demo falls back to templates
        status = getattr(e, "status_code", None)
        err = f"{type(e).__name__}{f' {status}' if status else ''}"
        LAST.update(ok=False, error=err, seconds=round(time.time() - started, 2))
        return None, err
