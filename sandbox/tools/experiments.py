"""Physics analyses for the extended experiment bench. Runs INSIDE the sandbox.

Each analysis takes (header, data, expected) and returns:
  values   - fitted quantities with uncertainties, for the researcher
  evidence - the features the fault library reasons about
  plot     - points, model lines and an optional annotation for the chart

Experiments and CSV columns (column names follow the formats used in the
team's earlier prototype, so those files work unchanged):
  helmholtz       current_A, measured_B_uT            (follow-up: in_shield, measured_B_uT)
  laser_alignment position_mm, detector_voltage_mV    (follow-up: a repeat scan)
  faraday         time_s, magnetic_flux_Wb, induced_voltage_V
                                                      (follow-up: input_shorted, voltage_V)
  beer_lambert    thickness_cm, transmitted_intensity (follow-up: a scan in the enclosure)
  malus           angle_deg, intensity                (follow-up: beam_blocked, intensity)
  cooling         time_s, temperature_C               (follow-up: ice-point readings)
  rc_charging     time_s, voltage_V                   (follow-up: a scan with a x10 probe)
"""
import numpy as np
from scipy.optimize import curve_fit

MU0 = 4e-7 * np.pi


def col(header, data, name):
    return data[:, header.index(name)]


def mean_err(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0


def linfit(x, y):
    """y = a*x + b with standard errors."""
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    dof = max(1, len(x) - 2)
    s2 = float(np.sum((y - (a * x + b)) ** 2)) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(a), float(b), float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))


def sig(value, err):
    return float(abs(value) / err) if err and err > 0 else (0.0 if value == 0 else 99.0)


def fit(f, x, y, p0, bounds=(-np.inf, np.inf)):
    p, cov = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=20000)
    return p, np.sqrt(np.clip(np.diag(cov), 0, None))


def dense(x, n=160):
    return np.linspace(float(np.min(x)), float(np.max(x)), n)


def groups_result(header, data, group_col, value_col, unit, y_label, x_label, expected_level=0.0):
    """Two-condition follow-up (e.g. leads on sample vs meter shorted)."""
    g = col(header, data, group_col)
    v = col(header, data, value_col)
    a, b = v[g == 0], v[g == 1]
    ma, ea = mean_err(a)
    mb, eb = mean_err(b)
    idx = np.arange(v.size)
    lines = [{"role": "fit", "x": [float(idx[g == 0].min()), float(idx[g == 0].max())], "y": [ma, ma]},
             {"role": "fit", "x": [float(idx[g == 1].min()), float(idx[g == 1].max())], "y": [mb, mb]},
             {"role": "expected", "x": [0, float(v.size - 1)], "y": [expected_level, expected_level]}]
    return (ma, ea, mb, eb), {"x": idx.tolist(), "y": v.tolist(), "lines": lines,
                              "x_label": x_label, "y_label": y_label, "unit": unit}


# --- Helmholtz coil -----------------------------------------------------------
def helmholtz(header, data, expected):
    R = float(expected.get("radius_m", 0.15))
    N = float(expected.get("turns", 100))
    k_th = (4 / 5) ** 1.5 * MU0 * N / R * 1e6          # µT per A
    if "in_shield" in header:
        (mo, eo, ms, es), plot = groups_result(
            header, data, "in_shield", "measured_B_uT", "µT", "field at zero current (µT)",
            "reading (probe in the open, then inside the zero-gauss chamber)")
        return {
            "values": {"field_in_open_uT": {"value": mo, "uncertainty": eo},
                       "field_in_shield_uT": {"value": ms, "uncertainty": es}},
            "evidence": {"offset_gone_in_shield": bool(sig(mo, eo) > 5 and sig(ms, es) < 3)},
            "plot": plot,
        }
    i = col(header, data, "current_A")
    b = col(header, data, "measured_B_uT")
    k, b0, ek, eb0 = linfit(i, b)
    xs = dense(np.append(i, 0))
    return {
        "values": {
            "coil_constant_uT_per_A": {"value": k, "uncertainty": ek},
            "theoretical_constant_uT_per_A": {"value": k_th},
            "field_offset_uT": {"value": b0, "uncertainty": eb0},
        },
        "evidence": {
            "offset_sigma": sig(b0, eb0),
            "offset_uT": b0,
            "gain_error_pct": 100 * (k - k_th) / k_th,
        },
        "plot": {"x": i.tolist(), "y": b.tolist(), "x_label": "coil current (A)", "y_label": "field at centre (µT)",
                 "unit": "µT", "x_unit": "A",
                 "lines": [{"role": "fit", "x": xs.tolist(), "y": (k * xs + b0).tolist()},
                           {"role": "expected", "x": xs.tolist(), "y": (k_th * xs).tolist()}],
                 "annotation": {"x": 0.0, "y_from": 0.0, "y_to": b0, "value": b0, "unit": "µT",
                                "text": "field present with no current in the coils"}
                 if sig(b0, eb0) > 5 else None},
    }


# --- Laser and photodetector alignment ---------------------------------------
def gauss(x, d, a, x0, w):
    return d + a * np.exp(-2 * ((x - x0) / w) ** 2)


def laser_alignment(header, data, expected):
    x = col(header, data, "position_mm")
    v = col(header, data, "detector_voltage_mV")
    x_exp = float(expected.get("expected_center_mm", 0.0))
    a_exp = float(expected.get("expected_peak_mV", 600.0))
    w_exp = float(expected.get("waist_mm", 1.0))
    p0 = [float(v.min()), float(v.max() - v.min()), float(x[np.argmax(v)]), w_exp]
    (d, a, x0, w), (ed, ea, ex0, ew) = fit(gauss, x, v, p0)
    w = abs(w)
    near_top = int(np.sum(v > v.max() - 0.01 * (v.max() - v.min())))
    resid = v - gauss(x, d, a, x0, w)
    noise = float(np.std(resid))
    xs = dense(x)
    return {
        "values": {
            "beam_centre_mm": {"value": x0, "uncertainty": ex0},
            "peak_signal_mV": {"value": a, "uncertainty": ea},
            "beam_radius_mm": {"value": w, "uncertainty": ew},
            "dark_level_mV": {"value": d, "uncertainty": ed},
        },
        "evidence": {
            "centre_offset_mm": x0 - x_exp,
            "centre_offset_sigma": sig(x0 - x_exp, ex0),
            "peak_ratio": a / a_exp,
            "width_ratio": w / w_exp,
            "flat_top": bool(near_top >= 3 and float(np.max(np.abs(resid))) > 4 * max(noise, 1e-9) / 2),
            "fit_rms_mV": noise,
        },
        "plot": {"x": x.tolist(), "y": v.tolist(), "x_label": "detector position (mm)",
                 "y_label": "photodetector signal (mV)", "unit": "mV", "x_unit": "mm",
                 "lines": [{"role": "fit", "x": xs.tolist(), "y": gauss(xs, d, a, x0, w).tolist()},
                           {"role": "expected", "x": xs.tolist(), "y": gauss(xs, d, a_exp, x_exp, w_exp).tolist()}],
                 "vlines": [{"x": x_exp, "label": "where the beam should be"}]},
    }


# --- Faraday's law -------------------------------------------------------------
def faraday(header, data, expected):
    if "input_shorted" in header:
        (mc, ec, ms, es), plot = groups_result(
            header, data, "input_shorted", "voltage_V", "V", "amplifier output, magnet still (V)",
            "reading (coil connected, then amplifier input shorted)")
        return {
            "values": {"offset_coil_connected_V": {"value": mc, "uncertainty": ec},
                       "offset_input_shorted_V": {"value": ms, "uncertainty": es}},
            "evidence": {"offset_with_input_shorted": bool(sig(ms, es) > 5)},
            "plot": plot,
        }
    t = col(header, data, "time_s")
    phi = col(header, data, "magnetic_flux_Wb")
    v = col(header, data, "induced_voltage_V")
    N = float(expected.get("turns", 200))
    emf = -N * np.gradient(phi, t)
    g, v0, eg, ev0 = linfit(emf, v)
    return {
        "values": {
            "measured_over_predicted": {"value": g, "uncertainty": eg},
            "dc_offset_V": {"value": v0, "uncertainty": ev0},
            "peak_predicted_emf_V": {"value": float(np.max(np.abs(emf)))},
        },
        "evidence": {
            "gain_ratio": g,
            "offset_sigma": sig(v0, ev0),
            "polarity_reversed": bool(g < 0),
        },
        "plot": {"x": t.tolist(), "y": v.tolist(), "x_label": "time (s)", "y_label": "induced EMF (V)",
                 "unit": "V", "x_unit": "s", "connect": True,
                 "lines": [{"role": "expected", "x": t.tolist(), "y": emf.tolist()}]},
    }


# --- Beer-Lambert absorption ---------------------------------------------------
def beer(x, i0, alpha, bg):
    return i0 * np.exp(-alpha * x) + bg


def beer_lambert(header, data, expected):
    x = col(header, data, "thickness_cm")
    y = col(header, data, "transmitted_intensity")
    a_exp = float(expected.get("expected_alpha", 0.5))
    i_exp = float(expected.get("expected_i0", 1.0))
    (i0, alpha, bg), (ei0, ea, ebg) = fit(beer, x, y, [float(y.max()), a_exp, 0.0])
    xs = dense(x)
    return {
        "values": {
            "absorption_coefficient_per_cm": {"value": alpha, "uncertainty": ea},
            "unattenuated_intensity": {"value": i0, "uncertainty": ei0},
            "background_light": {"value": bg, "uncertainty": ebg},
        },
        "evidence": {
            "background_sigma": sig(bg, ebg),
            "background_fraction": bg / max(i0, 1e-12),
            "alpha_error_pct": 100 * (alpha - a_exp) / a_exp,
        },
        "plot": {"x": x.tolist(), "y": y.tolist(), "x_label": "sample thickness (cm)",
                 "y_label": "transmitted intensity (relative)", "unit": "", "x_unit": "cm",
                 "lines": [{"role": "fit", "x": xs.tolist(), "y": beer(xs, i0, alpha, bg).tolist()},
                           {"role": "expected", "x": xs.tolist(), "y": beer(xs, i_exp, a_exp, 0).tolist()}]},
    }


# --- Malus's law -----------------------------------------------------------------
def malus_f(th, i0, th0, bg):
    return i0 * np.cos(np.radians(th - th0)) ** 2 + bg


def malus(header, data, expected):
    if "beam_blocked" in header:
        (mx, ex, mb, eb), plot = groups_result(
            header, data, "beam_blocked", "intensity", "", "detector reading at the crossed position",
            "reading (polarisers crossed, then laser beam blocked)")
        return {
            "values": {"reading_crossed": {"value": mx, "uncertainty": ex},
                       "reading_beam_blocked": {"value": mb, "uncertainty": eb}},
            "evidence": {"light_with_beam_blocked": bool(sig(mb, eb) > 5 and mb > 0.3 * mx)},
            "plot": plot,
        }
    th = col(header, data, "angle_deg")
    y = col(header, data, "intensity")
    th0_exp = float(expected.get("expected_offset_deg", 0.0))
    (i0, th0, bg), (ei0, eth0, ebg) = fit(malus_f, th, y, [float(y.max() - y.min()), float(th[np.argmax(y)]), float(y.min())])
    th0 = ((th0 + 90) % 180) - 90
    xs = dense(th)
    return {
        "values": {
            "peak_transmission": {"value": i0, "uncertainty": ei0},
            "angle_of_maximum_deg": {"value": th0, "uncertainty": eth0},
            "light_at_extinction": {"value": bg, "uncertainty": ebg},
        },
        "evidence": {
            "angle_offset_deg": th0 - th0_exp,
            "angle_offset_sigma": sig(th0 - th0_exp, eth0),
            "extinction_leak_fraction": bg / max(i0 + bg, 1e-12),
            "extinction_sigma": sig(bg, ebg),
        },
        "plot": {"x": th.tolist(), "y": y.tolist(), "x_label": "analyser angle (degrees)",
                 "y_label": "transmitted intensity (relative)", "unit": "", "x_unit": "°",
                 "lines": [{"role": "fit", "x": xs.tolist(), "y": malus_f(xs, i0, th0, bg).tolist()},
                           {"role": "expected", "x": xs.tolist(), "y": malus_f(xs, i0 + bg, th0_exp, 0).tolist()}]},
    }


# --- Newton's law of cooling with a thermocouple ----------------------------------
def cool_f(t, ta, dt, tau):
    return ta + dt * np.exp(-t / tau)


def cooling(header, data, expected):
    t = col(header, data, "time_s")
    temp = col(header, data, "temperature_C")
    ambient = float(expected.get("ambient_C", 21.0))
    if np.ptp(temp) < 2.0 and abs(np.mean(temp)) < 15:
        # follow-up: thermocouple held in an ice-water bath (should read 0 °C)
        m, e = mean_err(temp)
        return {
            "values": {"ice_point_reading_C": {"value": m, "uncertainty": e}},
            "evidence": {"offset_at_ice_point": bool(sig(m, e) > 5 and abs(m) > 0.3)},
            "plot": {"x": t.tolist(), "y": temp.tolist(), "x_label": "time in ice-water bath (s)",
                     "y_label": "thermocouple reading (°C)", "unit": "°C", "x_unit": "s",
                     "lines": [{"role": "fit", "x": [float(t.min()), float(t.max())], "y": [m, m]},
                               {"role": "expected", "x": [float(t.min()), float(t.max())], "y": [0, 0]}]},
        }
    p0 = [float(temp[-1]), float(temp[0] - temp[-1]), float((t.max() - t.min()) / 3)]
    (ta, dT, tau), (eta, edT, etau) = fit(cool_f, t, temp, p0)
    tau_exp = expected.get("expected_tau_s")
    xs = dense(t)
    ev = {"asymptote_offset_C": ta - ambient, "asymptote_sigma": sig(ta - ambient, eta)}
    if tau_exp:
        ev["tau_ratio"] = tau / float(tau_exp)
    return {
        "values": {
            "settles_at_C": {"value": ta, "uncertainty": eta},
            "room_temperature_C": {"value": ambient},
            "cooling_time_constant_s": {"value": tau, "uncertainty": etau},
        },
        "evidence": ev,
        "plot": {"x": t.tolist(), "y": temp.tolist(), "x_label": "time (s)", "y_label": "temperature (°C)",
                 "unit": "°C", "x_unit": "s",
                 "lines": [{"role": "fit", "x": xs.tolist(), "y": cool_f(xs, ta, dT, tau).tolist()},
                           {"role": "expected", "x": [float(t.min()), float(t.max())], "y": [ambient, ambient]}]},
    }


# --- RC charging curve ------------------------------------------------------------
def rc_f(t, vf, tau):
    return vf * (1 - np.exp(-t / tau))


def rc_charging(header, data, expected):
    t = col(header, data, "time_s")
    v = col(header, data, "voltage_V")
    vs = float(expected.get("supply_V", 5.0))
    tau_exp = float(expected.get("resistance_ohm", 1e6)) * float(expected.get("capacitance_F", 100e-9))
    (vf, tau), (evf, etau) = fit(rc_f, t, v, [float(v.max()), tau_exp])
    xs = dense(t)
    return {
        "values": {
            "final_voltage_V": {"value": vf, "uncertainty": evf},
            "time_constant_s": {"value": tau, "uncertainty": etau},
            "expected_time_constant_s": {"value": tau_exp},
        },
        "evidence": {
            "final_ratio": vf / vs,
            "tau_ratio": tau / tau_exp,
        },
        "plot": {"x": t.tolist(), "y": v.tolist(), "x_label": "time (s)", "y_label": "capacitor voltage (V)",
                 "unit": "V", "x_unit": "s",
                 "lines": [{"role": "fit", "x": xs.tolist(), "y": rc_f(xs, vf, tau).tolist()},
                           {"role": "expected", "x": xs.tolist(), "y": rc_f(xs, vs, tau_exp).tolist()}]},
    }


ANALYSES = {
    "helmholtz": helmholtz,
    "laser_alignment": laser_alignment,
    "faraday": faraday,
    "beer_lambert": beer_lambert,
    "malus": malus,
    "cooling": cooling,
    "rc_charging": rc_charging,
}
