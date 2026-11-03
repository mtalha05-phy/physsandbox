"""Deterministic physics analysis. Runs INSIDE the sandbox container.

Reads /data/input.csv (read-only mount), writes one JSON object to stdout.
Every number the user sees comes from here, never from the language model.

Usage: python analyze.py <experiment> '<expected-json>' [input_path]
"""
import json
import sys

import numpy as np


def load_csv(path):
    """Load numeric CSV, ignoring comment lines (lines starting with '#')."""
    rows, header, dropped = [], None, 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                dropped += 1
                continue
            if header is None:
                header = [h.strip() for h in line.split(",")]
                continue
            rows.append([float(x) for x in line.split(",")])
    return header, np.array(rows), dropped


# --- Resistance (4-wire) ----------------------------------------------------
def analyze_resistance(header, data, expected):
    from lmfit.models import LinearModel

    i = data[:, header.index("current_A")]
    v = data[:, header.index("voltage_V")]
    r_exp = float(expected["resistance_ohm"])

    model = LinearModel()
    result = model.fit(v, model.make_params(slope=r_exp, intercept=0.0), x=i)
    slope = result.params["slope"]
    icpt = result.params["intercept"]
    icpt_err = icpt.stderr or float("nan")
    resid = v - result.best_fit

    # Expected-signal check first: what should the voltmeter have read?
    v_expected = i * r_exp
    lowest = np.argmin(np.abs(i))
    apparent_r_low = v[lowest] / i[lowest]

    has_reversal = bool((i > 0).any() and (i < 0).any())
    return {
        "values": {
            "resistance_fit_ohm": {"value": slope.value, "uncertainty": slope.stderr},
            "offset_V": {"value": icpt.value, "uncertainty": icpt_err},
            "apparent_resistance_at_lowest_current_ohm": {"value": apparent_r_low},
            "expected_resistance_ohm": {"value": r_exp},
            "residual_rms_V": {"value": float(np.sqrt(np.mean(resid**2)))},
            "max_gap_from_expected_V": {"value": float(np.max(np.abs(v - v_expected)))},
        },
        "evidence": {
            "offset_sigma": float(abs(icpt.value) / icpt_err) if icpt_err else 0.0,
            "slope_error_pct": float(100 * (slope.value - r_exp) / r_exp),
            "apparent_r_low_error_pct": float(100 * (apparent_r_low - r_exp) / r_exp),
            "has_current_reversal": has_reversal,
            "offset_survives_reversal": bool(
                has_reversal and abs(icpt.value) / icpt_err > 5 if icpt_err else False
            ),
        },
        "plot": {"x": i.tolist(), "y": v.tolist(), "fit": result.best_fit.tolist(),
                 "expected": v_expected.tolist(), "x_label": "current (A)",
                 "y_label": "voltage (V)"},
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

    return {
        "values": {
            "sample_rate_Hz": {"value": fs},
            "dominant_frequency_Hz": {"value": peak_freq},
            "signal_amplitude_V": {"value": sig_amp},
            "mains_harmonics_V": {"value": mains},
            "mains_total_amplitude_V": {"value": mains_total},
            "noise_floor_V": {"value": noise_floor},
        },
        "evidence": {
            "mains_to_signal_ratio": mains_total / sig_amp if sig_amp else 0.0,
            "mains_snr": amp_at(f_mains) / noise_floor if noise_floor else 0.0,
            "odd_harmonics_present": bool(mains[f"{int(3 * f_mains)}_Hz"] > 5 * noise_floor),
            "signal_at_expected_freq": bool(abs(peak_freq - f_sig) < 2 * fs / v.size),
        },
        "plot": {"x": freqs[: int(2 * f_sig * v.size / fs)].tolist(),
                 "y": spec[: int(2 * f_sig * v.size / fs)].tolist(),
                 "x_label": "frequency (Hz)", "y_label": "amplitude (V)"},
    }


ANALYSES = {"resistance_4wire": analyze_resistance, "waveform": analyze_waveform}


def main():
    experiment = sys.argv[1]
    expected = json.loads(sys.argv[2])
    path = sys.argv[3] if len(sys.argv) > 3 else "/data/input.csv"
    header, data, dropped = load_csv(path)
    out = ANALYSES[experiment](header, data, expected)
    out["input"] = {"rows": int(data.shape[0]), "columns": header,
                    "comment_lines_ignored": dropped}
    print(json.dumps(out, default=float))


if __name__ == "__main__":
    main()
