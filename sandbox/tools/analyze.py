"""Deterministic physics analysis. Runs INSIDE the sandbox container.

Reads the job's input CSV (read-only mount) and writes one JSON object to
stdout. Every number the researcher sees comes from here, never from the
language model.

Usage: python analyze.py <experiment> '<expected-json>' [input_path]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiments  # noqa: E402


class DataError(ValueError):
    pass


COLUMNS = {
    "resistance_4wire": (["current_A", "voltage_V"], ["shorted", "voltage_V"]),
    "waveform": (["time_s", "voltage_V"],),
    "helmholtz": (["current_A", "measured_B_uT"], ["in_shield", "measured_B_uT"]),
    "laser_alignment": (["position_mm", "detector_voltage_mV"],),
    "faraday": (["time_s", "magnetic_flux_Wb", "induced_voltage_V"], ["input_shorted", "voltage_V"]),
    "beer_lambert": (["thickness_cm", "transmitted_intensity"],),
    "malus": (["angle_deg", "intensity"], ["beam_blocked", "intensity"]),
    "cooling": (["time_s", "temperature_C"],),
    "rc_charging": (["time_s", "voltage_V"],),
    "pendulum": (["length_m", "period_s"],),
    "hooke": (["mass_kg", "extension_m"],),
    "diode": (["voltage_V", "current_A"],),
    "grating": (["order_m", "angle_deg"],),
    "hall": (["current_A", "hall_voltage_V"],),
}


def load_csv(path):
    """Load a numeric CSV. Lines starting with '#' are comments and are skipped."""
    rows, header, dropped = [], None, 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                dropped += 1
                continue
            if header is None:
                header = [h.strip() for h in line.split(",")]
                continue
            try:
                row = [float(x) for x in line.split(",")]
            except ValueError:
                raise DataError(f"Line {n} has a value that isn't a number.")
            if len(row) != len(header):
                raise DataError(f"Line {n} has {len(row)} values but the header has {len(header)} columns.")
            if not all(np.isfinite(row)):
                raise DataError(f"Line {n} contains NaN or infinity.")
            rows.append(row)
    if header is None or len(rows) < 3:
        raise DataError("The file needs a header row and at least 3 rows of numbers.")
    return header, np.array(rows), dropped


def check_columns(experiment, header):
    options = COLUMNS[experiment]
    if any(all(c in header for c in opt) for opt in options):
        return
    want = " or ".join(", ".join(o) for o in options)
    raise DataError(f"This experiment needs the columns {want}. Your file has: {', '.join(header)}.")


# --- Resistance (4-wire) ----------------------------------------------------
def analyze_resistance(header, data, expected):
    from lmfit.models import LinearModel

    i = data[:, header.index("current_A")]
    v = data[:, header.index("voltage_V")]
    r_exp = float(expected.get("resistance_ohm", 0.010))

    model = LinearModel()
    result = model.fit(v, model.make_params(slope=r_exp, intercept=0.0), x=i)
    slope = result.params["slope"]
    icpt = result.params["intercept"]
    icpt_err = icpt.stderr or float("nan")
    resid = v - result.best_fit

    v_expected = i * r_exp
    lowest = np.argmin(np.abs(i))
    apparent_r_low = v[lowest] / i[lowest]
    has_reversal = bool((i > 0).any() and (i < 0).any())
    offset_sigma = float(abs(icpt.value) / icpt_err) if icpt_err else 0.0
    # Self-heating makes R rise with power, so V curves upward with |I|.
    q = np.polyfit(i, v, 2, cov=True) if len(i) > 4 else None
    curvature_sigma = float(abs(q[0][0]) / np.sqrt(q[1][0, 0])) if q is not None and q[1][0, 0] > 0 else 0.0
    xs = np.linspace(float(i.min()), float(i.max()), 2)
    return {
        "values": {
            "resistance_fit_ohm": {"value": slope.value, "uncertainty": slope.stderr},
            "offset_V": {"value": icpt.value, "uncertainty": icpt_err},
            "apparent_resistance_at_lowest_current_ohm": {"value": apparent_r_low},
            "expected_resistance_ohm": {"value": r_exp},
            "residual_rms_V": {"value": float(np.sqrt(np.mean(resid**2)))},
        },
        "evidence": {
            "offset_sigma": offset_sigma,
            "slope_error_pct": float(100 * (slope.value - r_exp) / r_exp),
            "apparent_r_low_error_pct": float(100 * (apparent_r_low - r_exp) / r_exp),
            "has_current_reversal": has_reversal,
            "offset_survives_reversal": bool(has_reversal and offset_sigma > 5),
            "curvature_sigma": curvature_sigma,
        },
        "plot": {"x": i.tolist(), "y": v.tolist(), "x_label": "current (A)", "y_label": "voltage (V)",
                 "unit": "V", "x_unit": "A",
                 "lines": [{"role": "fit", "x": xs.tolist(), "y": (slope.value * xs + icpt.value).tolist()},
                           {"role": "expected", "x": xs.tolist(), "y": (r_exp * xs).tolist()}],
                 "annotation": {"value": icpt.value, "unit": "V",
                                "text": "the same at every current, so it is not a resistance error"}
                 if offset_sigma > 5 else None},
    }


def analyze_zero_current(header, data, expected):
    """Follow-up: voltmeter read at I = 0, leads on the sample (shorted=0) and
    with the meter input shorted (shorted=1). A thermal EMF at the sample
    contacts appears only on the sample; a meter zero offset appears in both."""
    (ms, es, mm, em), plot = experiments.groups_result(
        header, data, "shorted", "voltage_V", "V", "voltage at zero current (V)",
        "reading (leads on sample, then meter shorted)")
    return {
        "values": {"offset_on_sample_V": {"value": ms, "uncertainty": es},
                   "offset_with_meter_shorted_V": {"value": mm, "uncertainty": em}},
        "evidence": {"offset_sigma": experiments.sig(ms, es),
                     "offset_only_on_sample": bool(experiments.sig(ms, es) > 5 and experiments.sig(mm, em) < 3)},
        "plot": plot,
    }


# --- Waveform (oscilloscope export) ------------------------------------------
def analyze_waveform(header, data, expected):
    t = data[:, header.index("time_s")]
    v = data[:, header.index("voltage_V")]
    f_sig = float(expected.get("signal_hz", 1000))
    f_mains = float(expected.get("mains_hz", 50))

    fs = 1.0 / np.median(np.diff(t))
    window = np.hanning(v.size)
    spec = np.abs(np.fft.rfft((v - v.mean()) * window)) * 2 / window.sum()
    freqs = np.fft.rfftfreq(v.size, 1 / fs)

    def amp_at(f):
        k = np.argmin(np.abs(freqs - f))
        return float(spec[max(k - 1, 0): k + 2].max())

    noise_floor = float(np.median(spec))
    mains = {f"{int(f_mains * n)}_Hz": amp_at(f_mains * n) for n in range(1, 6)}
    mains_total = float(np.sqrt(sum(a**2 for a in mains.values())))
    sig_amp = amp_at(f_sig)
    peak_freq = float(freqs[np.argmax(spec)])
    mains_snr = amp_at(f_mains) / noise_floor if noise_floor else 0.0
    keep = int(min(len(freqs), 2.2 * f_sig * v.size / fs))
    return {
        "values": {
            "dominant_frequency_Hz": {"value": peak_freq},
            "signal_amplitude_V": {"value": sig_amp},
            "mains_total_amplitude_V": {"value": mains_total},
            "noise_floor_V": {"value": noise_floor},
            "sample_rate_Hz": {"value": fs},
        },
        "evidence": {
            "mains_to_signal_ratio": mains_total / sig_amp if sig_amp else 0.0,
            "mains_snr": mains_snr,
            "hum_absent": bool(mains_snr < 10),
            "odd_harmonics_present": bool(mains[f"{int(3 * f_mains)}_Hz"] > 5 * noise_floor),
            "signal_at_expected_freq": bool(abs(peak_freq - f_sig) < 2 * fs / v.size),
        },
        "plot": {"mode": "spectrum", "x": freqs[:keep].tolist(), "y": spec[:keep].tolist(),
                 "x_label": "frequency (Hz)", "y_label": "amplitude (V)", "unit": "V", "x_unit": "Hz",
                 "harmonics": [f_mains * n for n in range(1, 6)]},
    }


def analyze_resistance_any(header, data, expected):
    if "shorted" in header:
        return analyze_zero_current(header, data, expected)
    return analyze_resistance(header, data, expected)


ANALYSES = {"resistance_4wire": analyze_resistance_any, "waveform": analyze_waveform, **experiments.ANALYSES}


def main():
    experiment = sys.argv[1]
    expected = json.loads(sys.argv[2])
    path = sys.argv[3] if len(sys.argv) > 3 else "/job/input.csv"
    if experiment not in ANALYSES:
        raise DataError(f"Unknown experiment '{experiment}'.")
    header, data, dropped = load_csv(path)
    check_columns(experiment, header)
    try:
        out = ANALYSES[experiment](header, data, expected)
    except RuntimeError as e:  # curve_fit failed to converge
        raise DataError(f"The model fit didn't converge on this data ({e}). Check the columns and units.")
    out["input"] = {"rows": int(data.shape[0]), "columns": header, "comment_lines_ignored": dropped}
    print(json.dumps(out, default=float))


if __name__ == "__main__":
    main()
