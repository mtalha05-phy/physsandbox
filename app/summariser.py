"""Turn an uploaded CSV into a schema summary for the language model.

The model never sees raw rows or free text from the file. Comment lines and
any non-numeric text are dropped and only counted, which removes the
prompt-injection channel through uploaded data.
"""
import csv
import io
import re

MAX_BYTES = 5_000_000
_SAFE_COL = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,40}$")


def summarise(raw: bytes) -> dict:
    if len(raw) > MAX_BYTES:
        raise ValueError("File too large (limit 5 MB)")
    text = raw.decode("utf-8", errors="replace")
    comment_lines, rows, header = 0, [], None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            comment_lines += 1
            continue
        if header is None:
            header = next(csv.reader(io.StringIO(s)))
            continue
        rows.append(s)

    columns = []
    for i, name in enumerate(header or []):
        name = name.strip()
        vals = []
        for r in rows:
            parts = r.split(",")
            try:
                vals.append(float(parts[i]))
            except (ValueError, IndexError):
                pass
        columns.append({
            # Column names are the only text that reaches the model, so they
            # must look like identifiers; anything else is replaced.
            "name": name if _SAFE_COL.match(name) else f"column_{i}",
            "count": len(vals),
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        })

    return {
        "rows": len(rows),
        "columns": columns,
        "comment_lines_dropped": comment_lines,
        "note": "Free text in the file is never shown to the model.",
    }
