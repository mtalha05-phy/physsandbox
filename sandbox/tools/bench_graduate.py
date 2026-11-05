"""Graduate advanced-lab experiments. Runs INSIDE the sandbox; loaded by experiments.py.

  nmr_echo            echo_time_ms, echo_amplitude_mV          (follow-up: pulse_length_us, fid_amplitude_mV)
  sc_transition       temperature_K, resistance_mOhm           (follow-up: + warming, a warming sweep too)
  compton             angle_deg, energy_keV                    (follow-up: reference_keV, measured_keV)
  optical_pumping     coil_current_A, frequency_kHz, isotope   (follow-up: ammeter, coil_current_A)

Where a run has few points, thresholds use a stated reading precision (an
`expected` parameter) with absolute sigma, not the fit's own scatter.
"""
import numpy as np
from scipy.optimize import curve_fit

from experiments import col, dense, fit, groups_result, linfit, mean_err, sig  # noqa: F401


def _wfit(f, x, y, p0, prec, bounds=(-np.inf, np.inf)):
    """Least squares with a stated, absolute reading precision."""
    p, cov = curve_fit(f, x, y, p0=p0, sigma=np.full(len(y), prec), absolute_sigma=True,
                       bounds=bounds, maxfev=20000)
    return p, np.sqrt(np.clip(np.diag(cov), 0, None))


# --- Pulsed NMR spin echo: A(2τ) = A₀ exp(−2τ/T₂ − (2τ/T_D)³) ---------------------
def echo_f(t, a0, r, b):
    # r = 1/T₂ in /ms; b is the diffusion term in units of (1/10 ms)³
    return a0 * np.exp(-r * t - b * (t / 10.0) ** 3)


def nutation_f(p, a, p90, d):
    return a * np.abs(np.sin(np.pi / 2 * p / p90)) + d


def nmr_echo(header, data, expected):
    t2_exp = float(expected.get("t2_ms", 20.0))
    a_exp = float(expected.get("expected_amplitude_mV", 500.0))
    p90_set = float(expected.get("pulse90_us", 10.0))
    if "pulse_length_us" in header:
        # follow-up: nutation sweep, FID amplitude against pulse length; its first maximum is the true 90° pulse
        p = col(header, data, "pulse_length_us")
        v = col(header, data, "fid_amplitude_mV")
        guess = float(p[np.argmax(v)])
        (a, p90, d), (ea, ep90, ed) = fit(nutation_f, p, v, [float(v.max()), guess, 0.0],
                                          bounds=([0, 0.2 * guess, -np.inf], [np.inf, 5 * guess, np.inf]))
        err = 100 * (p90 / p90_set - 1)
        ps = np.linspace(float(p.min()), float(p.max()), 200)
        return {
            "values": {"pulse90_measured_us": {"value": p90, "uncertainty": ep90},
                       "pulse90_set_us": {"value": p90_set},
                       "fid_maximum_mV": {"value": a, "uncertainty": ea}},
            "evidence": {"pulse_error_pct": err,
                         "pulse_misset": bool(abs(err) > 5 and sig(p90 - p90_set, ep90) > 5)},
            "plot": {"x": p.tolist(), "y": v.tolist(), "x_label": "RF pulse length (µs)",
                     "y_label": "FID amplitude after one pulse (mV)", "unit": "mV", "x_unit": "µs",
                     "lines": [{"role": "fit", "x": ps.tolist(), "y": nutation_f(ps, a, p90, d).tolist()}],
                     "vlines": [{"x": p90_set, "label": "pulse set as 90°"},
                                {"x": p90, "label": "measured 90°"}]},
        }
    t = col(header, data, "echo_time_ms")
    v = col(header, data, "echo_amplitude_mV")
    prec = float(expected.get("precision_mV", 3.0))
    (a0, r, b), (ea0, er, eb) = _wfit(echo_f, t, v, [float(v.max()) * 1.1, 1 / t2_exp, 0.0], prec)
    # pure exponential for the reported T₂ when the curvature is not there
    (a1, r1), (ea1, er1) = _wfit(lambda t, a, r: a * np.exp(-r * t), t, v, [float(v.max()) * 1.1, 1 / t2_exp], prec)
    t2, et2 = 1 / r1, er1 / r1 ** 2
    xs = dense(np.append(t, 0))
    curved = sig(b, eb) > 5 and b > 0
    values = {
        "t2_ms": {"value": t2, "uncertainty": et2},
        "expected_t2_ms": {"value": t2_exp},
        "echo_amplitude_at_zero_mV": {"value": a1, "uncertainty": ea1},
        "expected_amplitude_mV": {"value": a_exp},
        "diffusion_term": {"value": b, "uncertainty": eb},
    }
    if curved:
        # with the (2τ)³ diffusion term in the fit, T₂ and A₀ are the sample's own; the
        # single exponential's T₂ is what the decay looks like
        values = {"t2_ms": {"value": 1 / r, "uncertainty": er / r ** 2},
                  "apparent_t2_ms": {"value": t2, "uncertainty": et2},
                  "expected_t2_ms": {"value": t2_exp},
                  "echo_amplitude_at_zero_mV": {"value": a0, "uncertainty": ea0},
                  "expected_amplitude_mV": {"value": a_exp},
                  "diffusion_term": {"value": b, "uncertainty": eb}}
    return {
        "values": values,
        "evidence": {
            "t2_error_pct": 100 * (t2 / t2_exp - 1),
            "amplitude_ratio": a1 / a_exp,
            "curvature_sigma": float(b / eb) if eb > 0 else 0.0,
        },
        "plot": {"x": t.tolist(), "y": v.tolist(), "x_label": "echo time 2τ (ms)",
                 "y_label": "echo amplitude (mV)", "unit": "mV", "x_unit": "ms",
                 "lines": [{"role": "fit", "x": xs.tolist(),
                            "y": (echo_f(xs, a0, r, b) if curved else a1 * np.exp(-r1 * xs)).tolist()},
                           {"role": "expected", "x": xs.tolist(), "y": (a_exp * np.exp(-xs / t2_exp)).tolist()}]},
    }


# --- Superconducting transition, four-probe R(T) ------------------------------------
def sc_f(dT, rn, s, tc, w, r0):
    # normal-state resistance rn + s·ΔT switched on by a logistic step of width w around tc (ΔT from the expected Tc)
    z = np.clip((dT - tc) / w, -60, 60)
    return (rn + s * dT) / (1 + np.exp(-z)) + r0


def _sc_fit(T, R, tc_exp, prec):
    dT = T - tc_exp
    rn = float(np.median(R[dT > 3])) if np.any(dT > 3) else float(R.max())
    above = R > 0.5 * rn
    tc0 = float(dT[above].min()) if np.any(above) else 0.0
    p, e = _wfit(sc_f, dT, R, [rn, 0.0, tc0, 0.3, 0.0], prec,
                 bounds=([0, -np.inf, -30, 0.02, -np.inf], [np.inf, np.inf, 30, 10, np.inf]))
    return p, e


def sc_transition(header, data, expected):
    tc_exp = float(expected.get("tc_K", 92.0))
    w_exp = float(expected.get("width_K", 1.0))
    prec = float(expected.get("precision_mOhm", 0.01))
    T = col(header, data, "temperature_K")
    R = col(header, data, "resistance_mOhm")
    if "warming" in header:
        # follow-up: the same sample swept down and then back up
        wm = col(header, data, "warming")
        (pc, ec), (pw, ew) = _sc_fit(T[wm == 0], R[wm == 0], tc_exp, prec), _sc_fit(T[wm == 1], R[wm == 1], tc_exp, prec)
        hyst = float(pw[2] - pc[2])
        eh = float(np.hypot(ec[2], ew[2]))
        xs = dense(T)
        return {
            "values": {"tc_cooling_K": {"value": pc[2] + tc_exp, "uncertainty": ec[2]},
                       "tc_warming_K": {"value": pw[2] + tc_exp, "uncertainty": ew[2]},
                       "warming_minus_cooling_K": {"value": hyst, "uncertainty": eh}},
            "evidence": {"sweeps_disagree": bool(abs(hyst) > 0.5 and sig(hyst, eh) > 5)},
            "plot": {"x": T.tolist(), "y": R.tolist(), "x_label": "thermometer reading (K), cooling then warming",
                     "y_label": "four-probe resistance (mΩ)", "unit": "mΩ", "x_unit": "K",
                     "lines": [{"role": "fit", "x": xs.tolist(), "y": sc_f(xs - tc_exp, *pc).tolist()},
                               {"role": "fit", "x": xs.tolist(), "y": sc_f(xs - tc_exp, *pw).tolist()}],
                     "vlines": [{"x": pc[2] + tc_exp, "label": "cooling"}, {"x": pw[2] + tc_exp, "label": "warming"}]},
        }
    (rn, s, tc, w, r0), (ern, es, etc, ew, er0) = _sc_fit(T, R, tc_exp, prec)
    width = 2 * np.log(9) * w                       # 10–90 % width of the logistic step
    xs = dense(T)
    return {
        "values": {
            "tc_K": {"value": tc + tc_exp, "uncertainty": etc},
            "expected_tc_K": {"value": tc_exp},
            "transition_width_K": {"value": width, "uncertainty": 2 * np.log(9) * ew},
            "normal_resistance_mOhm": {"value": rn, "uncertainty": ern},
            "resistance_below_tc_mOhm": {"value": r0, "uncertainty": er0},
        },
        "evidence": {
            "tc_shift_K": float(tc),
            "width_ratio": float(width / w_exp),
            "residual_sigma": sig(r0, er0),
        },
        "plot": {"x": T.tolist(), "y": R.tolist(), "x_label": "temperature (K)",
                 "y_label": "four-probe resistance (mΩ)", "unit": "mΩ", "x_unit": "K",
                 "lines": [{"role": "fit", "x": xs.tolist(), "y": sc_f(xs - tc_exp, rn, s, tc, w, r0).tolist()},
                           {"role": "expected", "x": xs.tolist(),
                            "y": sc_f(xs - tc_exp, rn, s, 0.0, w_exp / (2 * np.log(9)), 0.0).tolist()}],
                 "vlines": [{"x": tc_exp, "label": f"expected Tc {tc_exp:g} K"}],
                 "annotation": {"x": float(T.min()), "y_from": 0.0, "y_to": r0, "value": r0, "unit": "mΩ",
                                "text": "resistance left below Tc, where it should be zero"}
                 if sig(r0, er0) > 5 else None},
    }


# --- Compton scattering: E' = E / (1 + (E/mc²)(1 − cos θ)) --------------------------
def compton_f(th, e0, mc2):
    # energies in MeV so both parameters are of order one
    return e0 / (1 + e0 / mc2 * (1 - np.cos(np.radians(th))))


def compton(header, data, expected):
    e_src = float(expected.get("source_keV", 661.66))
    me_exp = float(expected.get("electron_rest_energy_keV", 511.0))
    prec = float(expected.get("precision_keV", 2.0))
    if "reference_keV" in header:
        # follow-up: known calibration lines read on the same MCA settings
        ref = col(header, data, "reference_keV")
        meas = col(header, data, "measured_keV")
        (g, c), (eg, ec) = _wfit(lambda x, g, c: g * x + c, ref, meas, [1.0, 0.0], prec)
        xs = dense(np.append(ref, 0))
        gain_pct = 100 * (g - 1)
        return {
            "values": {"mca_gain": {"value": g, "uncertainty": eg},
                       "mca_zero_offset_keV": {"value": c, "uncertainty": ec}},
            "evidence": {"gain_error_pct": gain_pct, "zero_offset_sigma": sig(c, ec),
                         "mca_gain_off": bool(abs(gain_pct) > 1 and sig(g - 1, eg) > 5)},
            "plot": {"x": ref.tolist(), "y": (meas - ref).tolist(), "x_label": "true line energy (keV)",
                     "y_label": "MCA reading − true (keV)", "unit": "keV", "x_unit": "keV", "connect": False,
                     "lines": [{"role": "fit", "x": xs.tolist(), "y": ((g - 1) * xs + c).tolist()},
                               {"role": "expected", "x": [float(xs.min()), float(xs.max())], "y": [0, 0]}]},
        }
    th = col(header, data, "angle_deg")
    E = col(header, data, "energy_keV")
    f = lambda th, e0, mc2: 1e3 * compton_f(th, e0, mc2)
    (e0, mc2), (ee0, emc2) = _wfit(f, th, E, [e_src / 1e3, me_exp / 1e3], prec)
    # one point against a fit through the others (a misidentified peak), and how well the rest
    # fit once the worst point is dropped (a systematic misfit, such as an angle zero error, stays)
    worst, odd, rest_chi2 = 0.0, 0, float(np.sum(((E - f(th, e0, mc2)) / prec) ** 2) / max(1, len(th) - 2))
    if len(th) >= 5:
        for i in range(len(th)):
            m = np.ones(len(th), bool)
            m[i] = False
            (a, b), _ = _wfit(f, th[m], E[m], [e0, mc2], prec)
            z = abs(float(E[i] - f(th[i], a, b))) / prec
            if z > worst:
                worst, odd = z, i
                rest_chi2 = float(np.sum(((E[m] - f(th[m], a, b)) / prec) ** 2) / max(1, m.sum() - 2))
    xs = np.linspace(0, float(th.max()), 160)
    values = {
        "electron_rest_energy_keV": {"value": mc2 * 1e3, "uncertainty": emc2 * 1e3},
        "accepted_rest_energy_keV": {"value": me_exp},
        "incident_energy_fit_keV": {"value": e0 * 1e3, "uncertainty": ee0 * 1e3},
        "source_energy_keV": {"value": e_src},
    }
    fe0, fmc2 = e0, mc2
    if worst > 5 and rest_chi2 < 2:
        # one point off a curve the rest follow: report the fit without it, and which angle it was
        m = np.ones(len(th), bool)
        m[odd] = False
        (fe0, fmc2), (re0, rmc2) = _wfit(f, th[m], E[m], [e0, mc2], prec)
        values["electron_rest_energy_keV"] = {"value": fmc2 * 1e3, "uncertainty": rmc2 * 1e3}
        values["incident_energy_fit_keV"] = {"value": fe0 * 1e3, "uncertainty": re0 * 1e3}
        values["odd_point_angle_deg"] = {"value": float(th[odd])}
    return {
        "values": values,
        "evidence": {
            "me_error_pct": 100 * (mc2 * 1e3 / me_exp - 1),
            "me_error_sigma": sig(mc2 * 1e3 - me_exp, emc2 * 1e3),
            "e0_error_sigma": sig(e0 * 1e3 - e_src, ee0 * 1e3),
            "outlier_sigma": worst,
            "rest_chi2": rest_chi2,
        },
        "plot": {"x": th.tolist(), "y": E.tolist(), "x_label": "scattering angle θ (degrees)",
                 "y_label": "scattered photopeak energy (keV)", "unit": "keV", "x_unit": "°", "connect": False,
                 "lines": [{"role": "fit", "x": xs.tolist(), "y": f(xs, fe0, fmc2).tolist()},
                           {"role": "expected", "x": xs.tolist(), "y": f(xs, e_src / 1e3, me_exp / 1e3).tolist()}]},
    }


# --- Optical pumping of rubidium: f = g_F (μ_B/h) |k·I + B₀| ------------------------
MUB_H_KHZ_PER_UT = 13.99624            # μ_B/h in kHz per µT
G_F = {87: 0.5, 85: 1 / 3}


def optical_pumping(header, data, expected):
    k = float(expected.get("coil_constant_uT_per_A", 60.0))
    if "ammeter" in header:
        # follow-up: one coil current read on the lab ammeter, then on a calibrated ammeter in series
        (ml, el, mr, er), plot = groups_result(header, data, "ammeter", "coil_current_A", "A",
                                               "coil current (A)", "reading (lab ammeter, then calibrated ammeter)",
                                               expected_level=float(np.mean(col(header, data, "coil_current_A"))))
        plot["lines"] = plot["lines"][:2]
        ratio = ml / mr
        return {
            "values": {"lab_ammeter_A": {"value": ml, "uncertainty": el},
                       "reference_ammeter_A": {"value": mr, "uncertainty": er},
                       "lab_over_reference_current": {"value": ratio}},
            "evidence": {"ammeter_disagrees": bool(abs(ratio - 1) > 0.02 and sig(ml - mr, float(np.hypot(el, er))) > 5)},
            "plot": plot,
        }
    I = col(header, data, "coil_current_A")
    f = col(header, data, "frequency_kHz")
    iso = col(header, data, "isotope").astype(int)
    prec = float(expected.get("precision_kHz", 1.0))
    is87 = (iso == 87).astype(float)

    def model(X, g87, g85, b0):
        i, s = X
        return (s * g87 + (1 - s) * g85) * MUB_H_KHZ_PER_UT * np.abs(k * i + b0)

    (g87, g85, b0), (eg87, eg85, eb0) = _wfit(model, np.vstack([I, is87]), f, [0.5, 1 / 3, 0.0], prec)
    scale = (g87 / G_F[87] + g85 / G_F[85]) / 2
    escale = float(np.hypot(eg87 / G_F[87], eg85 / G_F[85]) / 2)
    ratio = g87 / g85
    eratio = ratio * float(np.hypot(eg87 / g87, eg85 / g85))
    xs = dense(np.append(I, 0))
    lines = []
    for s, g in ((1.0, g87), (0.0, g85)):
        lines.append({"role": "fit", "x": xs.tolist(), "y": model(np.vstack([xs, np.full_like(xs, s)]),
                                                                  g87, g85, b0).tolist()})
    for g in (G_F[87], G_F[85]):
        lines.append({"role": "expected", "x": xs.tolist(), "y": (g * MUB_H_KHZ_PER_UT * k * xs).tolist()})
    return {
        "values": {
            "lande_g_87": {"value": g87, "uncertainty": eg87},
            "lande_g_85": {"value": g85, "uncertainty": eg85},
            "g_ratio_87_over_85": {"value": ratio, "uncertainty": eratio},
            "residual_field_uT": {"value": b0, "uncertainty": eb0},
        },
        "evidence": {
            "g_scale_error_pct": 100 * (scale - 1),
            "g_scale_sigma": sig(scale - 1, escale),
            "ratio_error_pct": 100 * (ratio / 1.5 - 1),
            "ratio_sigma": sig(ratio - 1.5, eratio),
            "residual_field_sigma": sig(b0, eb0),
        },
        "plot": {"x": I.tolist(), "y": f.tolist(), "x_label": "coil current (A); upper line ⁸⁷Rb, lower ⁸⁵Rb",
                 "y_label": "resonance frequency (kHz)", "unit": "kHz", "x_unit": "A", "connect": False,
                 "lines": lines},
    }


ANALYSES = {
    "nmr_echo": nmr_echo,
    "sc_transition": sc_transition,
    "compton": compton,
    "optical_pumping": optical_pumping,
}

COLUMNS = {
    "nmr_echo": (["echo_time_ms", "echo_amplitude_mV"], ["pulse_length_us", "fid_amplitude_mV"]),
    "sc_transition": (["temperature_K", "resistance_mOhm"],),
    "compton": (["angle_deg", "energy_keV"], ["reference_keV", "measured_keV"]),
    "optical_pumping": (["coil_current_A", "frequency_kHz", "isotope"], ["ammeter", "coil_current_A"]),
}
