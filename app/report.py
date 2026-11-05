"""PDF report of one investigation.

The browser sends the investigation it holds (each step's plot, causes, next
test, preformatted values and receipt). All of it is untrusted: every string is
truncated and escaped before it reaches reportlab's Paragraph markup, every
number must be finite, sizes are capped, and the experiment must be in the
catalogue. Names, fixes, test instructions and safety notes are looked up here
from the fault library rather than taken from the request, and each receipt is
checked against this server's lab notebook. Nothing in the input is ever used
as a URL or a file path.
"""
import io
import math
import os
import re
import threading
import time
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": True})
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import Formatter, MaxNLocator, MultipleLocator, ScalarFormatter  # noqa: E402

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_RIGHT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.pdfgen import canvas as rl_canvas  # noqa: E402
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, Paragraph,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)
from reportlab.platypus.doctemplate import LayoutError  # noqa: E402

from . import ledger
from .pipeline import CATALOG
from .ranking import LIB

# ---------------------------------------------------------------- limits
MAX_STEPS = 10
MAX_STR = 500
MAX_POINTS = 5000
MAX_LINES = 12
MAX_VLINES = 10
MAX_VALUES = 40
MAX_CAUSES = 12
_busy = threading.BoundedSemaphore(2)     # charts are CPU work; two reports at once is plenty


class ReportError(ValueError):
    pass


# ---------------------------------------------------------------- brand
INK, LASER, COBALT, GREEN, GREY = "#13202B", "#E0312B", "#2754D6", "#17805C", "#E6EAED"
MUTED, RULE = "#5B6873", "#C9D1D7"
C = {k: colors.HexColor(v) for k, v in
     dict(ink=INK, laser=LASER, cobalt=COBALT, green=GREEN, grey=GREY, muted=MUTED, rule=RULE).items()}

_FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
for _name, _file in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                     ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(_name, os.path.join(_FONT_DIR, _file)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")


def _style(name, **kw):
    base = dict(fontName="DV", fontSize=9, leading=12.5, textColor=C["ink"])
    base.update(kw)
    return ParagraphStyle(name, **base)


ST = {
    "body": _style("body"),
    "small": _style("small", fontSize=7.8, leading=10.5, textColor=C["muted"]),
    "cell": _style("cell", fontSize=8.3, leading=11),
    "cellb": _style("cellb", fontName="DV-B", fontSize=8.3, leading=11),
    "cellr": _style("cellr", fontSize=8.3, leading=11, alignment=TA_RIGHT),
    "head": _style("head", fontName="DV-B", fontSize=7.6, leading=10, textColor=C["muted"]),
    "headr": _style("headr", fontName="DV-B", fontSize=7.6, leading=10, textColor=C["muted"], alignment=TA_RIGHT),
    "mono": _style("mono", fontName="DVM", fontSize=7.2, leading=9.6),
    "h1": _style("h1", fontName="DV-B", fontSize=19, leading=23),
    "eq": _style("eq", fontName="DV", fontSize=11, leading=15, textColor=C["cobalt"]),
    "h2": _style("h2", fontName="DV-B", fontSize=12.5, leading=16, spaceBefore=4, spaceAfter=2),
    "h3": _style("h3", fontName="DV-B", fontSize=9.6, leading=13, spaceBefore=6, spaceAfter=3),
    "label": _style("label", fontName="DV-B", fontSize=7.4, leading=10, textColor=C["muted"]),
}


# ---------------------------------------------------------------- validation
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f  ﻿]")


def _s(v, n=MAX_STR):
    """Untrusted value -> short plain string (not yet escaped)."""
    if v is None:
        return ""
    if isinstance(v, bool):
        v = "true" if v else "false"
    if not isinstance(v, (str, int, float)):
        raise ReportError("Expected text in the report data.")
    v = _CTRL.sub("", str(v)).strip()
    return v if len(v) <= n else v[: n - 1] + "…"


def _e(v, n=MAX_STR):
    """Untrusted value -> text safe to place inside Paragraph markup."""
    return escape(_s(v, n), {'"': "&quot;", "'": "&#39;"})


def _num(v, name="number"):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ReportError(f"The report data has a {name} that isn't a number.")
    v = float(v)
    if not math.isfinite(v):
        raise ReportError(f"The report data has a {name} that isn't finite.")
    return v


def _opt_num(v, name="number"):
    return None if v is None else _num(v, name)


def _series(v, name):
    if not isinstance(v, list):
        raise ReportError(f"The plot's {name} must be a list of numbers.")
    if len(v) > MAX_POINTS:
        raise ReportError(f"The plot has more than {MAX_POINTS} points in one series.")
    return [_num(x, f"plot {name} value") for x in v]


def _dict(v):
    return v if isinstance(v, dict) else {}


def _list(v, cap):
    if v is None:
        return []
    if not isinstance(v, list):
        raise ReportError("The report data has a list in the wrong shape.")
    return v[:cap]


def _clean_plot(p):
    if not isinstance(p, dict):
        return None
    x, y = _series(p.get("x", []), "x"), _series(p.get("y", []), "y")
    n = min(len(x), len(y))
    lines = []
    for ln in _list(p.get("lines"), MAX_LINES):
        ln = _dict(ln)
        lx, ly = _series(ln.get("x", []), "line x"), _series(ln.get("y", []), "line y")
        k = min(len(lx), len(ly))
        if k >= 2 and ln.get("role") in ("fit", "expected"):
            lines.append({"role": ln["role"], "x": lx[:k], "y": ly[:k]})
    vlines = []
    for v in _list(p.get("vlines"), MAX_VLINES):
        v = _dict(v)
        vlines.append({"x": _num(v.get("x"), "marker position"), "label": _s(v.get("label"), 80)})
    an = p.get("annotation")
    if isinstance(an, dict):
        an = {"value": _num(an.get("value"), "annotation value"), "unit": _s(an.get("unit"), 12),
              "text": _s(an.get("text"), 160)}
    else:
        an = None
    return {
        "x": x[:n], "y": y[:n], "lines": lines, "vlines": vlines, "annotation": an,
        "harmonics": [_num(h, "harmonic") for h in _list(p.get("harmonics"), 20)],
        "mode": "spectrum" if p.get("mode") == "spectrum" else "points",
        "connect": p.get("connect") is True,
        "x_label": _s(p.get("x_label"), 90), "y_label": _s(p.get("y_label"), 90),
        "unit": _s(p.get("unit"), 12), "x_unit": _s(p.get("x_unit"), 12),
    }


def _hexish(v, n=64):
    v = _s(v, max(n, 80))
    return v if re.fullmatch(r"[0-9a-f]{%d}" % n, v) else ""


def _clean_receipt(r):
    r = _dict(r)
    seq = r.get("seq")
    return {
        "seq": int(seq) if isinstance(seq, int) and not isinstance(seq, bool) and 0 < seq < 10**9 else None,
        "time": _s(r.get("time"), 40),
        "hash": _hexish(r.get("hash")), "prev": _hexish(r.get("prev")),
        "sig": _hexish(r.get("sig") or r.get("mac"), 128 if r.get("sig") else 64), "key_id": _hexish(r.get("key_id"), 16),
        "input_sha256": _hexish(r.get("input_sha256")),
        "mode": _s(r.get("mode"), 20),
        "walls_measured": _dict(r.get("walls_measured")),
        "walls_host_verified": r.get("walls_host_verified") if isinstance(r.get("walls_host_verified"), dict) else None,
    }


def validate(payload):
    """Untrusted JSON from the browser -> a clean investigation, or ReportError."""
    if not isinstance(payload, dict):
        raise ReportError("The report data must be a JSON object.")
    exp = payload.get("experiment")
    if not isinstance(exp, str) or exp not in CATALOG["experiments"]:
        raise ReportError("Unknown experiment.")
    lib = LIB["experiments"].get(exp, {"faults": {}, "tests": {}})
    steps_in = payload.get("steps")
    if not isinstance(steps_in, list) or not steps_in:
        raise ReportError("The investigation has no steps yet. Analyse a measurement first.")
    if len(steps_in) > MAX_STEPS:
        raise ReportError(f"A report covers at most {MAX_STEPS} steps.")
    expected_in = _dict(payload.get("expected"))
    expected = []
    for key, label, default in CATALOG["experiments"][exp]["expected"]:
        v = expected_in.get(key, default)
        expected.append((label, _num(v, "expected value") if v is not None else default))
    steps = []
    for st in steps_in:
        st = _dict(st)
        status = st.get("status") if st.get("status") in ("investigating", "confirmed", "clear") else "investigating"
        causes = []
        for c in _list(st.get("causes"), MAX_CAUSES):
            c = _dict(c)
            cid = c.get("id")
            if not isinstance(cid, str) or cid not in lib["faults"]:
                continue
            p = _num(c.get("probability"), "probability")
            if not 0 <= p <= 1:
                raise ReportError("A cause probability is outside 0 to 1.")
            f = lib["faults"][cid]
            causes.append({"id": cid, "name": f["name"], "p": p, "size": f.get("typical_size") or "",
                           "fix": f.get("fix") or ""})
        causes.sort(key=lambda c: -c["p"])
        nxt = _dict(st.get("next"))
        test = None
        if isinstance(nxt.get("id"), str) and nxt["id"] in lib["tests"]:
            t = lib["tests"][nxt["id"]]
            test = {"id": nxt["id"], "name": t["name"], "how": t.get("how", ""), "minutes": t.get("minutes"),
                    "safety": t.get("safety") or "",
                    "bits": _opt_num(nxt.get("info_gain_bits"), "information gain"),
                    "bpm": _opt_num(nxt.get("bits_per_minute"), "bits per minute")}
        ran = st.get("test")
        values = []
        for row in _list(st.get("values"), MAX_VALUES):
            if isinstance(row, list) and len(row) == 2:
                values.append((_s(row[0], 80), _s(row[1], 80)))
        steps.append({
            "file": _s(st.get("file"), 200) or "uploaded file",
            "test_id": ran if isinstance(ran, str) and ran in lib["tests"] else None,
            "test_given": bool(ran),
            "status": status, "causes": causes, "next": test, "values": values,
            "explanation": _s(st.get("explanation"), 1200),
            "explained_by": _s(st.get("explained_by"), 60),
            "plot": _clean_plot(st.get("plot")),
            "receipt": _clean_receipt(st.get("receipt")),
        })
    return {"experiment": exp, "cat": CATALOG["experiments"][exp], "lib": lib,
            "expected": expected, "steps": steps}


# ---------------------------------------------------------------- numbers
_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
SI_UNITS = {"V", "A", "Ω", "Hz", "s", "F"}


def _g(v, prec=4):
    return f"{float(f'{v:.{prec}g}'):g}" if abs(v) < 1e15 else f"{v:.{prec}g}"


def plain(v, unit="", prec=4):
    a = abs(v)
    if a >= 1e4 or (0 < a < 1e-3):
        m, e = f"{v:.{max(1, min(prec, 4) - 1)}e}".split("e")
        n = f"{m}×10{str(int(e)).translate(_SUP)}"
    else:
        n = _g(v, prec)
    return (f"{n}°" if unit == "°" else f"{n} {unit}") if unit else n


def si(v, unit=""):
    a = abs(v)
    if a == 0:
        return f"0 {unit}".strip()
    if unit in ("Hz", "s") and a >= 1:
        return f"{_g(v)} {unit}"
    for f, p in [(1e9, "G"), (1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n")]:
        if a >= f * 0.9999999:
            return f"{_g(v / f)} {p}{unit}".strip()
    return f"{v:.2e} {unit}".strip()


def fmt(v, unit):
    return si(v, unit) if unit in SI_UNITS else plain(v, unit)


def pct(p):
    return f"{round(p * 100)}%"


# ---------------------------------------------------------------- chart
def _sorted(xs, ys):
    pts = sorted(zip(xs, ys))
    return [a for a, _ in pts], [b for _, b in pts]


class _SIAxis(Formatter):
    """Tick labels with one SI prefix for the whole axis (800 mV, 1.2 V -> 0.8 V, 1.2 V)."""

    def __init__(self, unit):
        self.unit = unit

    def __call__(self, v, pos=None):
        big = max((abs(t) for t in self.locs), default=abs(v)) or abs(v)
        if self.unit == "s" and big >= 1:
            f, p = 1, ""
        else:
            f, p = next(((f, p) for f, p in [(1e9, "G"), (1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"),
                                             (1e-6, "µ"), (1e-9, "n"), (1e-12, "p")] if big >= f * 0.9999999),
                        (1e-12, "p"))
        n = 0.0 if abs(v) < big * 1e-9 else v / f
        return f"{_g(n)} {p}{self.unit}".replace("-", "−")


def _formatter(unit):
    if unit in SI_UNITS and unit != "Hz":
        return _SIAxis(unit)
    f = ScalarFormatter(useMathText=True, useOffset=False)
    f.set_powerlimits((-3, 4))
    return f


def chart_png(p, width_in=6.9, height_in=2.55):
    fig = Figure(figsize=(width_in, height_in), dpi=200, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.tick_params(labelsize=7.5, colors=MUTED, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.grid(axis="y", color=GREY, lw=0.8)
    ax.set_axisbelow(True)
    handles = []
    x, y = p["x"], p["y"]
    if p["mode"] == "spectrum":
        if x:
            sx, sy = _sorted(x, y)
            handles += ax.plot(sx, sy, color=COBALT, lw=1.1, label="Spectrum of the trace")
        xmax = max(x) if x else 0
        first = True
        for f in p["harmonics"]:
            if f > xmax:
                continue
            h = ax.axvline(f, color=LASER, lw=0.9, ls=(0, (3, 4)), alpha=0.6,
                           label="Mains frequency and harmonics" if first else None)
            if first:
                handles.append(h)
            first = False
        ax.set_ylim(bottom=0)
        if x:
            ax.set_xlim(min(x), max(x))
    else:
        for v in p["vlines"]:
            ax.axvline(v["x"], color=GREEN, lw=1, ls=(0, (2, 3)))
            ax.annotate(v["label"], xy=(v["x"], 1), xycoords=("data", "axes fraction"), xytext=(4, -2),
                        textcoords="offset points", va="top", fontsize=7, color=GREEN, annotation_clip=False)
        exp_l = [l for l in p["lines"] if l["role"] == "expected"]
        fit_l = [l for l in p["lines"] if l["role"] == "fit"]
        for i, l in enumerate(exp_l):
            sx, sy = _sorted(l["x"], l["y"])
            h, = ax.plot(sx, sy, color=GREEN, lw=1.5, ls=(0, (5, 4)), label="What theory predicts" if i == 0 else None)
            if i == 0:
                handles.append(h)
        for i, l in enumerate(fit_l):
            sx, sy = _sorted(l["x"], l["y"])
            h, = ax.plot(sx, sy, color=COBALT, lw=1.6, label="Best fit" if i == 0 else None)
            if i == 0:
                handles.append(h)
        if p["connect"] and x:
            sx, sy = _sorted(x, y)
            ax.plot(sx, sy, color=LASER, lw=0.8, alpha=0.45)
        if x:
            sc = ax.scatter(x, y, s=5 if len(x) > 60 else 14, color=LASER, edgecolors="none", zorder=3,
                            label="Readings")
            handles.insert(0, sc)
        an = p["annotation"]
        if an and fit_l and exp_l:
            def at_min(l):
                i = l["x"].index(min(l["x"]))
                return l["x"][i], l["y"][i]
            xf, yf = at_min(fit_l[0])
            _, ye = at_min(exp_l[0])
            span = (max(x) - min(x)) if x else 1
            xa = xf + 0.025 * span
            ax.annotate("", xy=(xa, yf), xytext=(xa, ye),
                        arrowprops=dict(arrowstyle="|-|", color=INK, lw=1.3, shrinkA=0, shrinkB=0, mutation_scale=3.5))
            # Label in the empty upper-left corner, tied to the bracket by a vertical leader (as on the page).
            label = f"{fmt(an['value'], an['unit'])} offset between fit and expected"
            ax.annotate(label, xy=(xa, max(yf, ye)), xytext=(xa, 0.97), textcoords=("data", "axes fraction"),
                        va="top", ha="left", fontsize=7.3, color=INK, fontweight="bold", parse_math=False,
                        arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, alpha=0.45, shrinkA=0, shrinkB=5,
                                        relpos=(0, 0)))
            if an["text"]:
                ax.annotate(an["text"], xy=(xa, 0.97), xycoords=("data", "axes fraction"), xytext=(5, -11),
                            textcoords="offset points", va="top", ha="left", fontsize=7, color=MUTED,
                            parse_math=False)
        if x and all(float(v).is_integer() for v in x):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        flat = [l for l in fit_l if len(set(l["y"])) == 1]
        if x and len(fit_l) == 2 and len(flat) == 2 and x == list(map(float, range(len(x)))):
            # Two-condition follow-up: shade the second condition's readings.
            lo, hi = min(flat[1]["x"]), max(flat[1]["x"])
            ax.axvspan(lo - 0.5, hi + 0.5, color=GREY, alpha=0.6, lw=0, zorder=0)
        if p["x_unit"] == "°" and x and max(x) - min(x) >= 90:
            ax.xaxis.set_major_locator(MultipleLocator(45))
    ax.margins(x=0.03, y=0.08)
    if p["mode"] == "spectrum":
        ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(_formatter("" if p["x_unit"] == "Hz" else p["x_unit"]))
    ax.yaxis.set_major_formatter(_formatter(p["unit"]))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    for a in (ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()):
        a.set_fontsize(7)
        a.set_color(MUTED)
    ax.set_xlabel(p["x_label"], fontsize=8, color=INK, parse_math=False)
    ax.set_ylabel(p["y_label"], fontsize=8, color=INK, parse_math=False)
    if handles:
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(handles), frameon=False,
                  fontsize=7.5, handlelength=2.2, borderaxespad=0.2, columnspacing=1.6, markerscale=2 if len(x) > 60 else 1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    return buf.getvalue()


# ---------------------------------------------------------------- page furniture
def _wrap(text, font, size, width):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and pdfmetrics.stringWidth(trial, font, size) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur] if cur else lines


class _Canvas(rl_canvas.Canvas):
    """Adds 'Page n of N' and the honest footer line to every page."""

    def __init__(self, *a, footer=("", ""), running="", **kw):
        super().__init__(*a, **kw)
        self._pages, self._footer, self._running = [], footer, running

    def showPage(self):
        self._pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._pages)
        for st in self._pages:
            self.__dict__.update(st)
            self._furniture(total)
            super().showPage()
        super().save()

    def _furniture(self, total):
        w, _ = A4
        left, right = 18 * mm, w - 18 * mm
        self.setStrokeColor(C["rule"])
        self.setLineWidth(0.6)
        self.line(left, 17 * mm, right, 17 * mm)
        self.setFillColor(C["muted"])
        self.setFont("DV", 6.8)
        y = 13.4 * mm
        for line in _wrap(" ".join(self._footer), "DV", 6.8, right - left - 22 * mm):
            self.drawString(left, y, line)
            y -= 3.1 * mm
        self.setFont("DV-B", 7.2)
        self.drawRightString(right, 13.4 * mm, f"Page {self._pageNumber} of {total}")
        if self._pageNumber > 1:
            self.setFont("DV", 7.2)
            self.drawString(left, A4[1] - 11 * mm, self._running)
            self.setFillColor(C["laser"])
            self.rect(right - 2.2 * mm, A4[1] - 11.4 * mm, 2.2 * mm, 2.2 * mm, stroke=0, fill=1)


# ---------------------------------------------------------------- building blocks
def P(text, style="body"):
    return Paragraph(text, ST[style])


def _table(rows, widths, head=True, zebra=False, style=None):
    t = Table(rows, colWidths=widths, repeatRows=1 if head else 0, hAlign="LEFT")
    cmds = [("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, C["rule"])]
    if head:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), C["grey"]), ("LINEBELOW", (0, 0), (-1, 0), 0.8, C["ink"])]
    t.setStyle(TableStyle(cmds + (style or [])))
    return t


def _kv(rows, w1, w2):
    """Two-column label/value list; values are markup already escaped by the caller."""
    out = [[P(_e(k, 80), "label"), P(v, "cell")] for k, v in rows]
    t = Table(out, colWidths=[w1, w2], hAlign="LEFT")
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6)]))
    return t


def _box(flowables, width, bg=GREY, accent=INK):
    t = Table([[flowables]], colWidths=[width], hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(accent)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def _pill(text, colour):
    t = Table([[Paragraph(text, _style("pill", fontName="DV-B", fontSize=9, leading=12, textColor=colors.white))]],
              hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(colour)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5)]))
    return t


def _step_title(i, st):
    if i == 0 and not st["test_given"]:
        return "First measurement"
    if st["test_id"]:
        return "Result of: " + st["lib_test_name"]
    return "Follow-up measurement"


def _walls_rows(w, host, mode):
    """[(wall, text, ok)] for the walls measured inside the box."""
    w = _dict(w)
    ifaces = [_s(i, 20) for i in _list(w.get("interfaces"), 20) if i != "lo"]
    secrets = [_s(i, 40) for i in _list(w.get("secrets_visible"), 20)]
    uid = w.get("uid")
    rows = [
        ("Kernel", "gVisor (sandboxed)" if w.get("gvisor") is True else "host kernel: " + _s(w.get("kernel"), 90),
         w.get("gvisor") is True),
        ("Network", ", ".join(ifaces) if ifaces else "loopback only", not ifaces),
        ("DNS", _s(w.get("dns"), 40) or "n/a", w.get("dns") == "blocked"),
        ("Outbound", _s(w.get("egress"), 40) or "n/a", w.get("egress") == "blocked"),
        ("Disk", _s(w.get("root_fs"), 40) or "n/a", w.get("root_fs") == "read-only"),
        ("Runs as", f"uid {_s(uid, 12)}" if uid is not None else "n/a", isinstance(uid, int) and uid != 0),
        ("Secrets seen", ", ".join(secrets) if secrets else "none", not secrets),
    ]
    return rows


def _host_rows(h):
    if not h:
        return None
    checks = [
        ("Runtime", h.get("runtime"), h.get("runtime") == "runsc", "runsc (gVisor)"),
        ("Network mode", h.get("network"), h.get("network") == "none", "none"),
        ("Read-only root", h.get("read_only_root"), h.get("read_only_root") is True, "true"),
        ("Memory limit", f"{h.get('memory_mb')} MB", h.get("memory_mb") == 512, "512 MB"),
        ("User", h.get("user"), h.get("user") == "65534:65534", "65534:65534 (nobody)"),
        ("Capabilities dropped", ", ".join(_s(c, 20) for c in _list(h.get("cap_drop"), 10)) or "none",
         h.get("cap_drop") == ["ALL"], "ALL"),
        ("no-new-privileges", h.get("no_new_privileges"), h.get("no_new_privileges") is True, "true"),
        ("Killed for memory", h.get("oom_killed"), None, ""),
        ("Exit code", h.get("exit_code"), None, ""),
    ]
    return [(k, _s(v, 40), ok, want) for k, v, ok, want in checks]


def _mark(ok):
    if ok is None:
        return '<font color="%s">•</font>' % MUTED
    return ('<font color="%s"><b>✓</b></font>' % GREEN) if ok else ('<font color="%s"><b>✕</b></font>' % LASER)


# ---------------------------------------------------------------- the report
def build(payload):
    """Returns (pdf bytes, filename). Raises ReportError for bad input."""
    inv = validate(payload)
    if not _busy.acquire(timeout=30):
        raise ReportError("The server is busy building other reports. Try again in a moment.")
    try:
        return _render(inv)
    except LayoutError as e:
        raise ReportError("The investigation is too large to lay out as a PDF.") from e
    finally:
        _busy.release()


def _notebook_check(steps):
    """Match each receipt against this server's lab notebook. A match means the
    receipt's walls and host check below are the server's own record, not the
    browser's copy."""
    by_seq = {}
    want = {st["receipt"]["seq"] for st in steps if st["receipt"]["seq"]}
    if want:
        for e in ledger.entries():
            if e.get("seq") in want:
                by_seq[e["seq"]] = e
    for st in steps:
        r = st["receipt"]
        e = by_seq.get(r["seq"])
        if e and e.get("hash") == r["hash"] and (not r["sig"] or (e.get("sig") or e.get("mac")) == r["sig"]):
            r.update({"matched": True, "sig": _hexish(e.get("sig"), 128) or _hexish(e.get("mac")),
                      "key_id": _hexish(e.get("key_id"), 16),
                      "walls_measured": _dict(e.get("walls_measured")),
                      "walls_host_verified": e.get("walls_host_verified") if isinstance(
                          e.get("walls_host_verified"), dict) else None,
                      "input_sha256": _hexish(e.get("input_sha256")), "prev": _hexish(e.get("prev")),
                      "time": _s(e.get("time"), 40), "mode": _s(e.get("mode"), 20)})
        else:
            r["matched"] = False


def _render(inv):
    exp, cat, steps = inv["experiment"], inv["cat"], inv["steps"]
    for st in steps:
        st["lib_test_name"] = inv["lib"]["tests"][st["test_id"]]["name"] if st["test_id"] else ""
    _notebook_check(steps)
    now = time.gmtime()
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", now)
    last = steps[-1]
    top = last["causes"][0] if last["causes"] else None
    status = last["status"]
    gvisor = all(st["receipt"]["mode"] == "docker"
                 and (st["receipt"]["walls_host_verified"] or {}).get("runtime") == "runsc" for st in steps)
    docker = all(st["receipt"]["mode"] == "docker" for st in steps)

    W = A4[0] - 36 * mm
    likely = "Fault likelihoods are hand-set from bench experience, not calibrated on data."
    if gvisor:
        footer = ("Numbers come from deterministic physics code run in a gVisor sandbox with no network.", likely)
    elif docker:
        footer = ("Numbers come from deterministic physics code run in a Docker container with no network "
                  "(not every job had the gVisor kernel).", likely)
    else:
        footer = ("Numbers come from deterministic physics code, here run in development mode outside the "
                  "sandbox, so the jobs were not isolated.", likely)
    running = f"PhysSandbox investigation report · {cat['label']} · {stamp}"

    flow = []
    # ---- header band
    band = Table([[Paragraph('<font name="DV-B" size="15" color="white">PhysSandbox</font>'
                             '<font size="9" color="#AFC0CC">   investigation report</font>', ST["body"]),
                   Paragraph(f'<font color="#AFC0CC" size="8">Generated {stamp}</font>', ST["cellr"])]],
                 colWidths=[W * 0.62, W * 0.38])
    band.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C["ink"]),
                              ("LINEBELOW", (0, 0), (-1, -1), 2.5, C["laser"]),
                              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                              ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    flow += [band, Spacer(1, 12)]
    domain = next((d["label"] for d in CATALOG["domains"] if d["id"] == cat.get("domain")), "")
    flow += [P(_e(domain.upper() + " · EXPERIMENT"), "label"), P(_e(cat["label"]), "h1"),
             P(_e(cat.get("model", "")), "eq"), Spacer(1, 4)]
    if cat.get("blurb"):
        flow.append(P(_e(cat["blurb"]), "small"))
    flow.append(Spacer(1, 8))

    if status == "confirmed" and top:
        pill, colour = f"Cause confirmed: {_e(top['name'])} ({pct(top['p'])})", LASER
    elif status == "clear":
        pill, colour = f"No fault found ({pct(top['p']) if top else ''} no fault)", GREEN
    else:
        pill, colour = (f"Still investigating · leading cause: {_e(top['name'])} ({pct(top['p'])})"
                        if top else "Still investigating"), COBALT
    flow += [_pill(pill, colour), Spacer(1, 8)]

    expected = ", ".join(f"{_e(l)} = {_e(plain(v, '', 6))}" for l, v in inv["expected"])
    meta = [("Expected", expected or "n/a"),
            ("Steps", f"{len(steps)} ({'first measurement' if len(steps) == 1 else 'first measurement and '}"
                      f"{'' if len(steps) == 1 else str(len(steps) - 1) + ' follow-up test' + ('s' if len(steps) > 2 else '')})"),
            ("Receipts", ", ".join(f"#{st['receipt']['seq']}" for st in steps if st["receipt"]["seq"]) or "n/a")]
    flow += [_kv(meta, 28 * mm, W - 28 * mm), Spacer(1, 10)]

    # ---- summary box
    box = [P("<b>Summary</b>", "h3")]
    if status == "confirmed" and top:
        found = f"The evidence points to <b>{_e(lc(top['name']))}</b> with {pct(top['p'])} probability, above the 90% confirmation threshold."
    elif status == "clear":
        found = "The measurement matches what was expected. No fault was found and no bench test is needed."
    elif top:
        second = last["causes"][1] if len(last["causes"]) > 1 else None
        found = (f"Not settled yet. The leading cause is <b>{_e(lc(top['name']))}</b> at {pct(top['p'])}"
                 + (f", with {_e(lc(second['name']))} at {pct(second['p'])}." if second else "."))
    else:
        found = "No causes were ranked."
    box.append(P(found))
    if len(steps) > 1 and top:
        hist = [next((c["p"] for c in st["causes"] if c["id"] == top["id"]), 0.0) for st in steps]
        box.append(P(f"{_e(top['name'])}: " + " → ".join(pct(h) for h in hist) + " across the steps.", "small"))
    if last["explanation"]:
        who = "built-in wording" if last["explained_by"] in ("", "template") else last["explained_by"]
        box += [Spacer(1, 3), P(f"<i>{_e(last['explanation'], 1200)}</i>", "body"),
                P(f"Explanation by {_e(who)}; every number in it comes from the physics code.", "small")]
    if top and top.get("fix"):
        label = {"confirmed": "Fix", "clear": "Fix"}.get(status, "Fix if this cause is confirmed")
        box += [Spacer(1, 5), P(f'<font color="{GREEN if status == "clear" else LASER}"><b>{label}.</b></font> '
                                f"{_e(top['fix'])}")]
    if status == "investigating" and last["next"]:
        t = last["next"]
        bits = f", {t['bits']:.2f} bits of information" if t["bits"] is not None else ""
        box += [Spacer(1, 5), P(f'<font color="{COBALT}"><b>Next test.</b></font> {_e(t["name"])} '
                                f"(about {t['minutes']} min{bits}). {_e(t['how'])}")]
        if t["safety"]:
            box.append(P(f"<b>Safety:</b> {_e(t['safety'])}", "small"))
    elif status == "confirmed":
        box += [Spacer(1, 3), P("Fix the cause, then repeat the measurement to check it.", "small")]
    flow += [_box(box, W, accent=colour), Spacer(1, 6)]

    # ---- the investigation at a glance
    rows = [[P("Step", "head"), P("Measurement", "head"), P("Leading cause afterwards", "head"),
             P("Outcome", "head")]]
    for i, st in enumerate(steps):
        c0 = st["causes"][0] if st["causes"] else None
        outcome = {"confirmed": f'<font color="{LASER}"><b>Cause confirmed</b></font>',
                   "clear": f'<font color="{GREEN}"><b>No fault found</b></font>'}.get(st["status"])
        if not outcome:
            outcome = (f'<font color="{COBALT}"><b>Next:</b></font> {_e(st["next"]["name"])}'
                       if st["next"] else "Still investigating")
        rows.append([P(str(i + 1), "cellb"),
                     P(f'{_e(_step_title(i, st))}<br/><font size="7" color="{MUTED}">{_e(st["file"])}</font>', "cell"),
                     P(f"{_e(c0['name'])} <b>{pct(c0['p'])}</b>" if c0 else "n/a", "cell"),
                     P(outcome, "cell")])
    flow += [Spacer(1, 6), P("The investigation at a glance", "h3"),
             _table(rows, [W * 0.07, W * 0.33, W * 0.32, W * 0.28]), Spacer(1, 6)]

    # ---- steps: each kept on one page where it fits
    for i, st in enumerate(steps):
        part = [Spacer(1, 4), P(f'<font color="{LASER}">Step {i + 1}</font>  {_e(_step_title(i, st))}', "h2")]
        r = st["receipt"]
        meta = [("Data file", _e(st["file"])),
                ("Input SHA-256", f'<font name="DVM" size="7.2">{r["input_sha256"] or "n/a"}</font>'),
                ("Receipt", f"#{r['seq'] or 'n/a'}" + (f" · {_e(r['time'])}" if r["time"] else "")
                 + ("" if r["matched"] else ' · <font color="%s">not found in this server\'s lab notebook</font>' % LASER))]
        part += [_kv(meta, 28 * mm, W - 28 * mm), Spacer(1, 6)]
        if st["plot"] and (st["plot"]["x"] or st["plot"]["lines"]):
            png = chart_png(st["plot"])
            part += [Image(io.BytesIO(png), width=W, height=W * 2.55 / 6.9), Spacer(1, 2)]
        vals = [[P("Measured value", "head"), P("Value ± uncertainty", "head")]] + \
               [[P(_e(k, 80), "cell"), P(_e(v, 80), "cell")] for k, v in st["values"]]
        vt = _table(vals, [W * 0.23, W * 0.25]) if st["values"] else P("No values reported.", "small")
        cz = [[P("Likely cause", "head"), P("Prob.", "headr")]]
        for j, c in enumerate(st["causes"][:6]):
            name = _e(c["name"]) + (f'<br/><font size="7" color="{MUTED}">Typical size: {_e(c["size"])}</font>'
                                    if c["size"] else "")
            cz.append([P(name, "cellb" if j == 0 else "cell"), P(pct(c["p"]), "cellr")])
        ct = _table(cz, [W * 0.41, W * 0.08])
        side = Table([[[P("Measured values", "h3"), vt], [P("Likely causes", "h3"), ct]]],
                     colWidths=[W * 0.51, W * 0.49], hAlign="LEFT")
        side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                  ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        if side.wrap(W, A4[1])[1] < 120 * mm:
            part.append(side)
        else:   # too tall to sit side by side on one page: stack, so each table can split across pages
            part += [P("Measured values", "h3"), vt, P("Likely causes", "h3"), ct]
        t = st["next"]
        if t:
            rows = [P(f'<font color="{COBALT}"><b>Recommended test.</b></font> <b>{_e(t["name"])}</b>'),
                    P(f"About {t['minutes']} min"
                      + (f" · {t['bits']:.2f} bits of information" if t["bits"] is not None else "")
                      + (f" · {t['bpm']:.2f} bits per minute" if t["bpm"] is not None else ""), "small"),
                    P(_e(t["how"]))]
            if t["safety"]:
                rows.append(P(f'<font color="{LASER}"><b>Safety:</b></font> {_e(t["safety"])}', "small"))
            part += [Spacer(1, 8), _box(rows, W, bg="#F2F5F7", accent=COBALT)]
        elif st["status"] != "investigating":
            msg = "No further test needed: the cause is confirmed." if st["status"] == "confirmed" \
                else "No test needed: no fault found."
            part += [Spacer(1, 6), P(msg, "small")]
        flow += [KeepTogether(part), Spacer(1, 10)]

    # ---- containment
    flow.append(CondPageBreak(60 * mm))
    flow += [P("Containment", "h2"),
             P("Every step ran in a fresh container that was destroyed afterwards. Each job leaves a receipt, "
               "hash-chained to the one before it and signed with this server's Ed25519 key, which anyone can "
               "check with its public key (GET /api/ledger/public-key).", "small"),
             Spacer(1, 4)]
    for i, st in enumerate(steps):
        r = st["receipt"]
        rows = [("Receipt", f"#{r['seq'] or 'n/a'} · step {i + 1}"
                 + (f' · <font color="{GREEN}">matches this server\'s lab notebook</font>' if r["matched"]
                    else f' · <font color="{LASER}">not found in this server\'s lab notebook; shown as supplied</font>')),
                ("Hash", f'<font name="DVM" size="7.2">{r["hash"] or "n/a"}</font>'),
                ("Ed25519 signature", f'<font name="DVM" size="7.2">{r["sig"] or "n/a"}</font>'),
                ("Signing key", f'<font name="DVM" size="7.2">{r["key_id"] or "n/a"}</font>'),
                ("Previous hash", f'<font name="DVM" size="7.2">{r["prev"] or "n/a"}</font>')]
        flow += [KeepTogether([_kv(rows, 28 * mm, W - 28 * mm), Spacer(1, 5)])]

    walls = [(st["receipt"]["walls_measured"], st["receipt"]["walls_host_verified"], st["receipt"]["mode"])
             for st in steps]
    w_last, h_last, m_last = walls[-1]
    inside = [[P("", "head"), P("Wall, measured inside the box", "head"), P("Result", "head")]]
    for k, v, ok in _walls_rows(w_last, h_last, m_last):
        inside.append([P(_mark(ok), "cell"), P(_e(k), "cellb"), P(_e(v, 120), "cell")])
    host = _host_rows(h_last)
    hrows = [[P("", "head"), P("Host check (docker inspect)", "head"), P("Recorded", "head"), P("Required", "head")]]
    if host:
        for k, v, ok, want in host:
            hrows.append([P(_mark(ok), "cell"), P(_e(k), "cellb"), P(_e(v), "cell"), P(_e(want), "cell")])
    blocks = [P(f"Walls of the last job (receipt #{steps[-1]['receipt']['seq'] or 'n/a'})", "h3"),
              _table(inside, [W * 0.05, W * 0.3, W * 0.65]), Spacer(1, 8)]
    if host:
        blocks += [_table(hrows, [W * 0.05, W * 0.3, W * 0.3, W * 0.35])]
    elif m_last == "docker":
        blocks += [_box([P(f'<font color="{LASER}"><b>No host check.</b></font> The host could not inspect this '
                           "job's container, so its walls are unverified.", "cell")], W, bg="#FBE9E8", accent=LASER)]
    else:
        blocks += [_box([P(f'<font color="{LASER}"><b>No host check.</b></font> This job ran in development mode '
                           f"({_e(m_last or 'local')}), without Docker or gVisor, so the host had no container to "
                           "inspect and the job was not isolated.", "cell")], W, bg="#FBE9E8", accent=LASER)]
    if len(steps) > 1:
        differ = [i + 1 for i, w in enumerate(walls[:-1]) if w[0] != w_last or w[1] != h_last]
        note = ("The earlier steps' receipts record the same walls and host check." if not differ else
                "Steps " + ", ".join(map(str, differ)) + " recorded different walls; see their receipts in the "
                "lab notebook.")
        blocks += [Spacer(1, 4), P(note, "small")]
    flow += [KeepTogether(blocks), Spacer(1, 10),
             P("To check these receipts later, compare the hashes above with the lab notebook on the server "
               "(GET /api/ledger) and run its tamper check. Keep this report as an independent copy of the "
               "hashes and signatures.", "small")]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=22 * mm, title=f"PhysSandbox report: {cat['label']}",
                            author="PhysSandbox", subject=f"Investigation of {cat['label']}")
    doc.build(flow, canvasmaker=lambda *a, **kw: _Canvas(*a, footer=footer, running=running, **kw))
    return buf.getvalue(), f"physsandbox-{exp}-{time.strftime('%Y-%m-%d', now)}.pdf"


_PROPER = re.compile(r"^(Earth|Hall|Faraday|Malus|Newton|Beer|LED)")


def lc(t):
    return t if _PROPER.match(t or "") else (t or "")[:1].lower() + (t or "")[1:]
