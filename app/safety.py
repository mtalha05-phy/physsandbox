"""Rule-based safety checks on any text shown to the user."""
from pathlib import Path

import yaml

_LIB = yaml.safe_load((Path(__file__).parent.parent / "faults" / "faults.yaml").read_text())
FORBIDDEN = [p.lower() for p in _LIB["safety"]["forbidden_phrases"]]


def check_text(text: str):
    low = text.lower()
    for phrase in FORBIDDEN:
        if phrase in low:
            return False, f"unsafe advice ('{phrase}')"
    return True, ""
