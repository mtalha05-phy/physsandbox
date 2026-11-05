"""Classic lab experiments: Michelson interferometer, speed of sound in a
resonance tube, Stefan-Boltzmann law with a lamp filament. Runs INSIDE the sandbox.

Same contract as experiments.py: (header, data, expected) -> values, evidence, plot.
Few points, so every fit uses the precision the researcher states (absolute
sigma), not the scatter of the points themselves.

  michelson      micrometer_um, fringes                 (follow-up: + recount, counted again slowly)
  resonance_tube frequency_Hz, resonance_length_m       (follow-up: depth_cm, air_temperature_C)
  stefan_lamp    voltage_V, current_A                   (follow-up: reading, open_circuit_A)
"""
import numpy as np
from scipy.stats import chi2 as _chi2, norm as _norm

from experiments import dense, mean_err, sig


def wfit(x, y, s, deg=1):
    """Polynomial fit with absolute sigma s: coefficients (highest power first) and covariance."""
    p, cov = np.polyfit(x, y, deg, w=1 / np.broadcast_to(s, np.shape(x)), cov="unscaled")
    return p, cov


def misfit_sigma(resid, s, n_par):
    """Chi-square of the residuals in units of the stated precision, as a Gaussian significance."""
    dof = max(1, resid.size - n_par)
    c2 = float(np.sum((resid / s) ** 2))
    z = float(_norm.isf(_chi2.sf(c2, dof)))
    return float(np.clip(np.nan_to_num(z, posinf=99.0, neginf=0.0), 0.0, 99.0))


# --- Michelson interferometer: N = 2d/λ ------------------------------------------
def michelson(header, data, expected):
    d = col_(header, data, "micrometer_um")
    n = col_(header, data, "fringes")
    lam_exp = float(expected.get("wavelength_nm", 632.8))
    prec = float(expected.get("precision_um", 0.2))
    # Counting is exact (integers); the reading error is on the micrometer, so fit d against N.
    # N in hundreds keeps the fit well conditioned.
    p, cov = wfit(n / 100, d, prec)
    s, d0 = p[0] / 100, p[1]                            # µm per fringe, µm at zero fringes
    es, ed0 = float(np.sqrt(cov[0, 0])) / 100, float(np.sqrt(cov[1, 1]))
    lam, elam = 2000 * s, 2000 * es
    resid = d - np.polyval(p, n / 100)
    mis = misfit_sigma(resid, prec, 2)
    lam_sig = sig(lam - lam_exp, elam)
    # Travel before the first counted fringe: the starting reading against a line through
    # the others. A dead zone in the screw is several fringes' worth; a skipped count is one.
    k = int(np.argmin(n))
    m = np.arange(n.size) != k
    pk, ck = wfit(n[m] / 100, d[m], prec)
    g = np.array([n[k] / 100, 1.0])
    dead = float(np.polyval(pk, n[k] / 100) - d[k])
    e_dead = float(np.hypot(prec, np.sqrt(max(g @ ck @ g, 0))))
    ds = dense(np.append(d, 0))
    plot = {"x": d.tolist(), "y": n.tolist(), "x_label": "mirror travel read on the micrometer (µm)",
            "y_label": "fringes counted", "unit": "", "x_unit": "µm", "connect": False,
            "lines": [{"role": "fit", "x": ds.tolist(), "y": ((ds - d0) / s).tolist()},
                      {"role": "expected", "x": ds.tolist(), "y": (2000 * ds / lam_exp).tolist()}]}
    values = {
        "laser_wavelength_fit_nm": {"value": lam, "uncertainty": elam},
        "laser_wavelength_expected_nm": {"value": lam_exp},
        "dead_travel_at_start_um": {"value": dead, "uncertainty": e_dead},
        "scatter_about_line_um": {"value": float(np.sqrt(np.mean(resid ** 2)))},
    }
    if "recount" in header:
        # follow-up: the same travel counted again, slowly, turning one way only
        return {"values": values,
                "evidence": {"recount_on_line": bool(mis < 4 and lam_sig < 4)},
                "plot": {**plot, "x_label": "mirror travel, recounted slowly one way (µm)"}}
    return {"values": values,
            "evidence": {"lambda_error_pct": 100 * (lam / lam_exp - 1), "lambda_error_sigma": lam_sig,
                         "misfit_sigma": mis, "start_jump_fringes": dead / (lam_exp / 2000)},
            "plot": plot}


def col_(header, data, name):
    return data[:, header.index(name)]


# --- Speed of sound, closed resonance tube: L + e = v / 4f ------------------------
def v_sound(t_c):
    return 331.3 * np.sqrt(1 + t_c / 273.15)


def resonance_tube(header, data, expected):
    t_air = float(expected.get("air_temperature_C", 20.0))
    if "air_temperature_C" in header:
        # follow-up: a thermometer lowered down the tube
        z = col_(header, data, "depth_cm")
        tt = col_(header, data, "air_temperature_C")
        m, e = mean_err(tt)
        e = max(e, 0.2)                                # a lab thermometer is good to about 0.2 °C
        return {
            "values": {"tube_air_temperature_C": {"value": m, "uncertainty": e},
                       "stated_air_temperature_C": {"value": t_air},
                       "speed_expected_in_tube_m_s": {"value": float(v_sound(m))}},
            "evidence": {"tube_air_differs": bool(abs(m - t_air) > 3 and sig(m - t_air, e) > 5)},
            "plot": {"x": z.tolist(), "y": tt.tolist(), "x_label": "thermometer depth below the open end (cm)",
                     "y_label": "air temperature in the tube (°C)", "unit": "°C", "x_unit": "cm", "connect": True,
                     "lines": [{"role": "fit", "x": [float(z.min()), float(z.max())], "y": [m, m]},
                               {"role": "expected", "x": [float(z.min()), float(z.max())], "y": [t_air, t_air]}]},
        }
    f = col_(header, data, "frequency_Hz")
    L = col_(header, data, "resonance_length_m")
    D = float(expected.get("tube_diameter_m", 0.035))
    prec = float(expected.get("precision_mm", 1.0)) / 1000
    x = 1000 / f                                       # period in ms, so the fit is well conditioned
    p, cov = wfit(x, L, prec)
    v, ev = 4000 * p[0], 4000 * float(np.sqrt(cov[0, 0]))
    e_fit, ee = -p[1], float(np.sqrt(cov[1, 1]))
    v_exp = float(v_sound(t_air))
    # humidity alone moves v by up to ~0.3%; the end correction lies between ~0.25 and 0.4 D
    ev_tot = float(np.hypot(ev, 0.002 * v_exp))
    e_exp, e_tol = 0.3 * D, 0.06 * D
    xs = dense(x)
    return {
        "values": {
            "speed_of_sound_fit_m_s": {"value": v, "uncertainty": ev},
            "speed_of_sound_expected_m_s": {"value": v_exp},
            "end_correction_fit_mm": {"value": 1000 * e_fit, "uncertainty": 1000 * ee},
            "end_correction_expected_mm": {"value": 1000 * e_exp},
        },
        "evidence": {
            "v_error_pct": 100 * (v / v_exp - 1),
            "v_error_sigma": sig(v - v_exp, ev_tot),
            "end_correction_sigma": sig(e_fit - e_exp, float(np.hypot(ee, e_tol))),
        },
        "plot": {"x": x.tolist(), "y": (100 * L).tolist(), "x_label": "period 1/f (ms)",
                 "y_label": "first resonance length (cm)", "unit": "cm", "x_unit": "ms", "connect": False,
                 "lines": [{"role": "fit", "x": xs.tolist(), "y": (100 * np.polyval(p, xs)).tolist()},
                           {"role": "expected", "x": xs.tolist(), "y": (100 * (v_exp / 4000 * xs - e_exp)).tolist()}]},
    }


# --- Stefan-Boltzmann law with a lamp filament: P ∝ T⁴ ------------------------------
BETA_W = 1.209          # tungsten: R/R₀ = (T/T₀)^1.209 from 300 to 3000 K, within about 2%


def stefan_lamp(header, data, expected):
    if "open_circuit_A" in header:
        # follow-up: the ammeter read with the lamp circuit open (true current zero)
        r = col_(header, data, "reading")
        i0 = col_(header, data, "open_circuit_A")
        m, e = mean_err(i0)
        e = max(e, 0.0002)                             # last digit of a 2 A range
        return {
            "values": {"ammeter_reading_open_circuit_A": {"value": m, "uncertainty": e}},
            "evidence": {"ammeter_offset_found": bool(abs(m) > 0.003 and sig(m, e) > 5)},
            "plot": {"x": r.tolist(), "y": (1000 * i0).tolist(), "x_label": "reading number, lamp circuit open",
                     "y_label": "ammeter reading (mA)", "unit": "mA", "x_unit": "", "connect": True,
                     "lines": [{"role": "fit", "x": [float(r.min()), float(r.max())], "y": [1000 * m, 1000 * m]},
                               {"role": "expected", "x": [float(r.min()), float(r.max())], "y": [0, 0]}]},
        }
    V = col_(header, data, "voltage_V")
    I = col_(header, data, "current_A")
    R0 = float(expected.get("cold_resistance_ohm", 1.60))
    T0 = float(expected.get("room_temperature_K", 293.0))
    prec = float(expected.get("precision_pct", 0.5)) / 100
    R, P = V / I, V * I
    T = T0 * (R / R0) ** (1 / BETA_W)
    tm = float(np.exp(np.mean(np.log(T))))
    x, y = np.log(T / tm), np.log(P)
    # V and I errors enter ln P and ln T together: ln P − 4 ln T carries (1 − 4/β) dlnV + (1 + 4/β) dlnI
    s = prec * float(np.hypot(1 - 4 / BETA_W, 1 + 4 / BETA_W))
    (n, c), cl = wfit(x, y, s, 1)
    (q, n2, c2), cq = wfit(x, y, s, 2)
    en = float(np.sqrt(cl[0, 0]))
    n_hot = n2 + 2 * q * float(x.max())
    # power ÷ T⁴, scaled to 1 at the hottest point (on the fitted curve, so one noisy
    # reading doesn't set the scale): flat if every watt leaves as radiation
    t_hi = float(T.max())
    p_hi = float(np.exp(np.polyval([q, n2, c2], np.log(t_hi / tm))))
    norm = lambda pp, tt: pp / p_hi * (t_hi / tt) ** 4
    ts = np.geomspace(float(T.min()), t_hi, 160)
    fit_p = np.exp(np.polyval([q, n2, c2], np.log(ts / tm)))
    return {
        "values": {
            "stefan_exponent_fit": {"value": n, "uncertainty": en},
            "stefan_exponent_hottest": {"value": n_hot},
            "hottest_filament_temperature_K": {"value": float(T.max())},
            "coldest_filament_temperature_K": {"value": float(T.min())},
            "power_excess_coldest_pct": {"value": float(100 * (norm(P[np.argmin(T)], T.min()) - 1))},
        },
        # signed: conduction losses and a high-reading ammeter pull the exponent below 4
        "evidence": {"exponent_shift_sigma": float((n - 4) / en), "exponent": n},
        "plot": {"x": T.tolist(), "y": norm(P, T).tolist(), "x_label": "filament temperature from its resistance (K)",
                 "y_label": "power ÷ T⁴, relative to the hottest point", "unit": "", "x_unit": "K", "connect": False,
                 "lines": [{"role": "fit", "x": ts.tolist(), "y": norm(fit_p, ts).tolist()},
                           {"role": "expected", "x": [float(T.min()), float(T.max())], "y": [1.0, 1.0]}]},
    }


ANALYSES = {"michelson": michelson, "resonance_tube": resonance_tube, "stefan_lamp": stefan_lamp}
COLUMNS = {
    "michelson": (["micrometer_um", "fringes"],),
    "resonance_tube": (["frequency_Hz", "resonance_length_m"], ["depth_cm", "air_temperature_C"]),
    "stefan_lamp": (["voltage_V", "current_A"], ["reading", "open_circuit_A"]),
}
