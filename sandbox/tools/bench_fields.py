"""Fields and scattering bench: e/m of the electron in a fine-beam tube, electron spin
resonance of DPPH, Rutherford scattering of alpha particles from gold, Bragg X-ray
diffraction from NaCl. Runs INSIDE the sandbox; loaded by experiments.py.

  e_over_m    accelerating_voltage_V, coil_current_A, beam_radius_cm
              (follow-ups: + mirror_scale; coil_current_A, probe_field_mT; turned, coil_current_A;
               meter_at_tube, voltage_V)
  esr         rf_frequency_MHz, resonance_field_mT
              (follow-ups: + field_reversed; + phase_set; + recheck; field_used_mT, probe_field_mT)
  rutherford  angle_deg, counts, live_time_s, angles on one side of the beam
              (follow-ups: the same on both sides of zero; + thin_foil; foil_out, counts, live_time_s;
               reading, pressure_mbar)
  bragg_xray  order_n, angle_deg, the Kα peak of each order
              (follow-ups: detector_angle_deg, count_rate; crystal_angle_deg, detector_angle_deg, count_rate;
               order_n, angle_deg, crystal; zr_filter, count_rate)

Few points, so every fit uses the precision the researcher states (absolute sigma), not
the scatter of the points themselves; counts use Poisson statistics (σ = √N, from the model).
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2 as _chi2, norm as _norm

from experiments import col, dense, fit, gauss, groups_result, mean_err, sig  # noqa: F401

MU0 = 4e-7 * np.pi
Q_E = 1.75882001076e11                 # e/m of the electron, C/kg
MUB_H = 13.99624493                    # μ_B/h in MHz per mT


def wfit(x, y, s, deg=1):
    """Polynomial fit with absolute sigma s: coefficients (highest power first) and covariance."""
    return np.polyfit(x, y, deg, w=1 / np.broadcast_to(s, np.shape(x)), cov="unscaled")


def wlstsq(A, y, s):
    """Weighted linear least squares y ≈ A·p with absolute sigma s: p and covariance."""
    w = 1 / np.broadcast_to(s, y.shape)
    Aw = A * w[:, None]
    cov = np.linalg.inv(Aw.T @ Aw)
    return cov @ Aw.T @ (y * w), cov


def misfit_sigma(resid, s, n_par):
    """Chi-square of the residuals in units of their errors, as a Gaussian significance."""
    dof = max(1, resid.size - n_par)
    c2 = float(np.sum((resid / s) ** 2))
    z = float(_norm.isf(_chi2.sf(c2, dof)))
    return float(np.clip(np.nan_to_num(z, posinf=99.0, neginf=0.0), 0.0, 99.0))


def _two_groups(header, data, flag, value, unit, y_label, x_label, floor):
    """Means of two groups of readings (flag 0, then 1), each error at least `floor`."""
    (ma, ea, mb, eb), plot = groups_result(header, data, flag, value, unit, y_label, x_label)
    plot["lines"] = plot["lines"][:2]
    g = col(header, data, flag)
    ea = max(ea, floor / np.sqrt(max(1, int(np.sum(g == 0)))))
    eb = max(eb, floor / np.sqrt(max(1, int(np.sum(g == 1)))))
    return ma, ea, mb, eb, plot


# =============================================================================
# --- e/m of the electron, fine-beam tube in Helmholtz coils: e/m = 2V/(B r)² ---------
def coil_constant(expected):
    """Field at the centre of a Helmholtz pair per amp, mT/A: (4/5)^(3/2) μ₀ N / R."""
    R = float(expected.get("coil_radius_m", 0.150))
    N = float(expected.get("coil_turns", 130))
    return (4 / 5) ** 1.5 * MU0 * N / R * 1e3


def e_over_m(header, data, expected):
    k = coil_constant(expected)
    q_acc = float(expected.get("e_over_m_C_per_kg", Q_E))
    if "probe_field_mT" in header:
        # follow-up: a calibrated Hall probe at the centre of the coils, field against current
        I = col(header, data, "coil_current_A")
        Bp = col(header, data, "probe_field_mT")
        pp = float(expected.get("probe_precision_mT", 0.01))
        (a, b), cv = wfit(I, Bp, pp)
        ea, eb = float(np.sqrt(cv[0, 0])), float(np.sqrt(cv[1, 1]))
        xs = np.array([0.0, float(I.max())])
        return {
            "values": {"coil_constant_measured_mT_per_A": {"value": a, "uncertainty": ea},
                       "coil_constant_used_mT_per_A": {"value": k},
                       "probe_field_at_zero_current_mT": {"value": b, "uncertainty": eb}},
            "evidence": {"coil_constant_off": bool(abs(a / k - 1) > 0.02 and sig(a - k, ea) > 5)},
            "plot": {"x": I.tolist(), "y": Bp.tolist(), "x_label": "coil current (A)",
                     "y_label": "field at the centre, calibrated Hall probe (mT)", "unit": "mT", "x_unit": "A",
                     "connect": False,
                     "lines": [{"role": "fit", "x": xs.tolist(), "y": (a * xs + b).tolist()},
                               {"role": "expected", "x": xs.tolist(), "y": (k * xs).tolist()}]},
        }
    if "turned" in header:
        # follow-up: the current that puts the beam on the same mark at the same voltage, with
        # the apparatus facing one way and then turned round; the field along the axis flips
        m0, e0, m1, e1, plot = _two_groups(header, data, "turned", "coil_current_A", "A",
                                           "coil current for the same beam diameter (A)",
                                           "reading (apparatus facing one way, then turned round)", 0.005)
        d, ed = m1 - m0, float(np.hypot(e0, e1))
        return {
            "values": {"current_facing_first_way_A": {"value": m0, "uncertainty": e0},
                       "current_turned_round_A": {"value": m1, "uncertainty": e1},
                       "axial_field_from_turning_uT": {"value": 1e3 * k * d / 2, "uncertainty": 1e3 * k * ed / 2}},
            "evidence": {"turning_changes_current": bool(abs(d) > 0.01 * abs(m0 + m1) / 2 and sig(d, ed) > 5)},
            "plot": plot,
        }
    if "meter_at_tube" in header:
        # follow-up: the voltmeter as wired for the run, then moved onto the tube's anode and cathode sockets
        m0, e0, m1, e1, plot = _two_groups(header, data, "meter_at_tube", "voltage_V", "V",
                                           "accelerating voltage reading (V)",
                                           "reading (voltmeter as wired, then on the anode and cathode sockets)", 0.5)
        d, ed = m0 - m1, float(np.hypot(e0, e1))
        return {
            "values": {"voltage_as_wired_V": {"value": m0, "uncertainty": e0},
                       "voltage_at_tube_sockets_V": {"value": m1, "uncertainty": e1}},
            "evidence": {"voltmeter_reads_extra": bool(abs(d) > max(2.0, 0.02 * abs(m1)) and sig(d, ed) > 5)},
            "plot": plot,
        }
    V = col(header, data, "accelerating_voltage_V")
    I = col(header, data, "coil_current_A")
    r = col(header, data, "beam_radius_cm")
    prec = float(expected.get("precision_cm", 0.05))
    if np.any(V <= 0) or np.any(I <= 0) or np.any(r <= 0):
        raise ValueError("Voltages, currents and radii must all be positive.")
    # r = g/√(e/m) with e/m in 10¹¹ C/kg, so the fit is well conditioned: a line through the origin in s = 1/√(e/m)
    g = 100 * np.sqrt(2 * V / 1e11) / (k * 1e-3 * I)
    s, es = float(np.sum(g * r) / np.sum(g * g)), float(prec / np.sqrt(np.sum(g * g)))
    q, eq = 1 / s ** 2, 2 * es / s ** 3
    q_exp = q_acc / 1e11
    mis = misfit_sigma(r - s * g, prec, 1)
    # electrons accelerated through less (or more) than the voltage read: V_true = V_read − ΔV.
    # Seen only when the voltages span a good range (ΔV matters most at the lowest voltage).
    dv = edv = qv = eqv = None
    dv_sig = 0.0
    if V.size >= 4 and np.ptp(V) > 0.3 * float(np.mean(V)):
        f2 = lambda X, qq, d10: 100 * np.sqrt(2 * np.clip(X[0] - 10 * d10, 1e-6, None) / (qq * 1e11)) / (k * 1e-3 * X[1])
        try:
            p2, c2 = curve_fit(f2, np.vstack([V, I]), r, p0=[q, 0.0], sigma=np.full(r.size, prec),
                               absolute_sigma=True, bounds=([0.05, -30.0], [50.0, 0.09 * float(V.min())]), maxfev=20000)
            qv, dv = float(p2[0]), 10 * float(p2[1])
            eqv, edv = float(np.sqrt(c2[0, 0])), 10 * float(np.sqrt(c2[1, 1]))
            dv_sig = float(dv / edv) if edv > 0 else 0.0
        except (RuntimeError, ValueError):
            pass
    q_each = 2 * V / ((k * 1e-3 * I) * (r / 100)) ** 2 / 1e11
    vs = dense(V)
    fit_line = np.full(vs.size, q)
    if dv is not None and abs(dv_sig) > 5:
        fit_line = qv * vs / (vs - dv)
    values = {
        "electron_e_over_m_C_per_kg": {"value": q * 1e11, "uncertainty": eq * 1e11},
        "accepted_e_over_m_C_per_kg": {"value": q_acc},
        "coil_constant_used_mT_per_A": {"value": k},
    }
    if dv is not None:
        values["accelerating_voltage_offset_V"] = {"value": dv, "uncertainty": edv}
        if abs(dv_sig) > 5:
            values["e_over_m_offset_corrected_C_per_kg"] = {"value": qv * 1e11, "uncertainty": eqv * 1e11}
    evidence = {"em_error_pct": 100 * (q / q_exp - 1), "em_error_sigma": sig(q - q_exp, eq),
                "voltage_offset_sigma": dv_sig, "misfit_sigma": mis}
    x_label = "accelerating voltage read (V)"
    if "mirror_scale" in header:
        # follow-up: every diameter read again on a mirror scale, beam lined up with its reflection
        evidence["reread_em_ok"] = bool(evidence["em_error_sigma"] < 4 and abs(dv_sig) < 4)
        x_label = "accelerating voltage read (V), diameters reread on the mirror scale"
    return {
        "values": values,
        "evidence": evidence,
        "plot": {"x": V.tolist(), "y": q_each.tolist(), "x_label": x_label,
                 "y_label": "e/m from each reading, 2V/(Br)² (×10¹¹ C/kg)", "unit": "×10¹¹ C/kg", "x_unit": "V",
                 "connect": False,
                 "lines": [{"role": "fit", "x": vs.tolist(), "y": fit_line.tolist()},
                           {"role": "expected", "x": [float(V.min()), float(V.max())], "y": [q_exp, q_exp]}]},
    }


# =============================================================================
# --- Electron spin resonance of DPPH: h f = g μ_B B -----------------------------------
def _esr_line(f, B, prec):
    """B = s·(f/100) + c with absolute sigma: s, c, covariance."""
    (s, c), cv = wfit(f / 100, B, prec)
    return float(s), float(c), cv


def _loo_line(x, y, prec):
    """Worst leave-one-out residual of a weighted line, in units of its own expected spread
    (stated precision plus the prediction uncertainty of the line through the others), the
    index of that point, and how well the others fit without it."""
    worst, odd, rest = 0.0, -1, 0.0
    if x.size < 5:
        return worst, odd, rest
    for i in range(x.size):
        m = np.ones(x.size, bool)
        m[i] = False
        p, cv = wfit(x[m], y[m], prec)
        gr = np.array([x[i], 1.0])
        sd = float(np.sqrt(max(gr @ cv @ gr, 0.0)))
        z = abs(float(y[i] - np.polyval(p, x[i]))) / float(np.hypot(prec, sd))
        if z > worst:
            worst, odd, rest = z, i, misfit_sigma(y[m] - np.polyval(p, x[m]), prec, 2)
    return worst, odd, rest


def esr(header, data, expected):
    g_exp = float(expected.get("g_factor", 2.0036))
    prec = float(expected.get("precision_mT", 0.01))
    if "probe_field_mT" in header:
        # follow-up: the field at the sample read on a calibrated Hall probe, against the field used in the run
        bu = col(header, data, "field_used_mT")
        bp = col(header, data, "probe_field_mT")
        (a, b), cv = wfit(bu, bp, prec)
        ea, eb = float(np.sqrt(cv[0, 0])), float(np.sqrt(cv[1, 1]))
        xs = np.array([0.0, float(bu.max())])
        return {
            "values": {"probe_over_used_field": {"value": a, "uncertainty": ea},
                       "probe_minus_used_at_zero_mT": {"value": b, "uncertainty": eb}},
            "evidence": {"field_scale_off": bool(abs(a - 1) > 0.015 and sig(a - 1, ea) > 5)},
            "plot": {"x": bu.tolist(), "y": bp.tolist(), "x_label": "field as used in the run (mT)",
                     "y_label": "field on the calibrated Hall probe (mT)", "unit": "mT", "x_unit": "mT",
                     "connect": False,
                     "lines": [{"role": "fit", "x": xs.tolist(), "y": (a * xs + b).tolist()},
                               {"role": "expected", "x": xs.tolist(), "y": xs.tolist()}]},
        }
    f = col(header, data, "rf_frequency_MHz")
    B = col(header, data, "resonance_field_mT")
    for flag, key, what in (("field_reversed", "reversal", "field forward, then reversed (magnitudes)"),
                            ("phase_set", "phase", "as before, then with the phase set")):
        if flag in header:
            # follow-up: the same frequencies in two conditions; one slope, an intercept for each
            gflag = col(header, data, flag)
            x = f / 100
            A = np.vstack([x, 1 - gflag, gflag]).T
            (s, c0, c1), cv = wlstsq(A, B, prec)
            dd = float(c0 - c1)
            edd = float(np.sqrt(max(cv[1, 1] + cv[2, 2] - 2 * cv[1, 2], 0.0)))
            gf = 100 / (MUB_H * s)
            found = bool(abs(dd) > 0.02 and sig(dd, edd) > 5)
            fs = dense(np.append(f, 0))
            values = {"g_factor": {"value": gf, "uncertainty": gf * float(np.sqrt(cv[0, 0])) / s}}
            if key == "reversal":
                values["field_shift_on_reversal_mT"] = {"value": dd, "uncertainty": edd}
                values["zero_offset_from_reversal_mT"] = {"value": dd / 2, "uncertainty": edd / 2}
                evidence = {"reversal_shift_found": found}
            else:
                values["field_shift_after_phasing_mT"] = {"value": dd, "uncertainty": edd}
                evidence = {"phase_shift_found": found}
            return {
                "values": values, "evidence": evidence,
                "plot": {"x": f.tolist(), "y": B.tolist(), "x_label": f"RF frequency (MHz): {what}",
                         "y_label": "resonance field (mT)", "unit": "mT", "x_unit": "MHz", "connect": False,
                         "lines": [{"role": "fit", "x": fs.tolist(), "y": (s * fs / 100 + c0).tolist()},
                                   {"role": "fit", "x": fs.tolist(), "y": (s * fs / 100 + c1).tolist()},
                                   {"role": "expected", "x": fs.tolist(), "y": (fs / (g_exp * MUB_H)).tolist()}]},
            }
    if f.size < 3:
        raise ValueError("The ESR fit needs at least three frequencies.")
    worst, odd, rest = _loo_line(f / 100, B, prec)
    keep = np.ones(f.size, bool)
    values = {}
    if worst > 5 and rest < 4:
        # one point off a line the rest follow: fit without it, and say which it was
        keep[odd] = False
        values["odd_point_frequency_MHz"] = {"value": float(f[odd])}
    s, c, cv = _esr_line(f[keep], B[keep], prec)
    es, ec = float(np.sqrt(cv[0, 0])), float(np.sqrt(cv[1, 1]))
    gf = 100 / (MUB_H * s)
    eg = gf * es / s
    fs = dense(np.append(f, 0))
    values = {"g_factor": {"value": gf, "uncertainty": eg}, "accepted_g_factor": {"value": g_exp},
              "esr_field_offset_mT": {"value": c, "uncertainty": ec}, **values}
    evidence = {"g_error_pct": 100 * (gf / g_exp - 1), "g_error_sigma": float((gf - g_exp) / eg),
                "field_offset_sigma": float(c / ec), "field_offset_mT": c, "outlier_sigma": worst}
    if "recheck" in header:
        # follow-up: the frequency counter read again at every setting
        evidence["recheck_outlier_gone"] = bool(worst < 4)
    return {
        "values": values,
        "evidence": evidence,
        "plot": {"x": f.tolist(), "y": B.tolist(), "x_label": "RF frequency (MHz)",
                 "y_label": "resonance field (mT)", "unit": "mT", "x_unit": "MHz", "connect": False,
                 "lines": [{"role": "fit", "x": fs.tolist(), "y": (s * fs / 100 + c).tolist()},
                           {"role": "expected", "x": fs.tolist(), "y": (fs / (g_exp * MUB_H)).tolist()}]},
    }


# =============================================================================
# --- Rutherford scattering: N(θ) ∝ 1/sin⁴(θ/2) ----------------------------------------
S20 = float(np.sin(np.radians(10.0)) ** 4)          # sin⁴(20°/2): A is the rate at 20°


def ruth_f(th, a, th0, b):
    return a * S20 / np.sin(np.radians(np.abs(th - th0)) / 2) ** 4 + b


def _poisson_fit(f, x, n, t, p0, bounds):
    """Rates n/t fitted with Poisson errors √N/t taken from the model (iterated), absolute sigma."""
    r = n / t
    s = np.sqrt(np.maximum(n, 1.0)) / t
    p = np.array(p0, float)
    for _ in range(4):
        p, cov = curve_fit(f, x, r, p0=p, sigma=s, absolute_sigma=True, bounds=bounds, maxfev=20000)
        s = np.sqrt(np.maximum(f(x, *p) * t, 1.0)) / t
    resid = (r - f(x, *p)) / s
    return p, np.sqrt(np.clip(np.diag(cov), 0, None)), resid


def _a_guess(th, n, t):
    k = int(np.argmin(np.abs(np.abs(th) - 20)))
    return max(float(n[k] / t[k] * np.sin(np.radians(abs(th[k])) / 2) ** 4 / S20), 1e-6)


def _log_rate(n, t):
    return np.log10(np.maximum(n, 0.5) / t)


def rutherford(header, data, expected):
    a_exp = float(expected.get("rate_at_20deg_per_s", 1.2))
    if "pressure_mbar" in header:
        # follow-up: the chamber's vacuum gauge read a few times
        rd = col(header, data, "reading")
        pr = col(header, data, "pressure_mbar")
        m, e = mean_err(pr)
        return {
            "values": {"chamber_pressure_mbar": {"value": m, "uncertainty": e}},
            # below about 50 mbar the gas between source, foil and detector takes little energy from the alphas
            "evidence": {"pressure_high": bool(m > 50)},
            "plot": {"x": rd.tolist(), "y": pr.tolist(), "x_label": "gauge reading number", "y_label": "chamber pressure (mbar)",
                     "unit": "mbar", "x_unit": "", "connect": True,
                     "lines": [{"role": "fit", "x": [float(rd.min()), float(rd.max())], "y": [m, m]}]},
        }
    n = col(header, data, "counts")
    t = col(header, data, "live_time_s")
    if np.any(t <= 0) or np.any(n < 0):
        raise ValueError("Live times must be positive and counts can't be negative.")
    if "foil_out" in header:
        # follow-up: counts at a scattering angle with the gold foil taken out of the holder
        tot, tt = float(n.sum()), float(t.sum())
        rate, er = tot / tt, float(np.sqrt(max(tot, 1.0))) / tt
        idx = np.arange(1, n.size + 1)
        return {
            "values": {"rate_with_foil_out_per_s": {"value": rate, "uncertainty": er}},
            # a surface-barrier detector in vacuum sees well under 0.005 counts/s with nothing to count
            "evidence": {"leak_without_foil": bool(rate > 0.005 and rate / er > 5)},
            "plot": {"x": idx.tolist(), "y": (n / t).tolist(), "x_label": "count number, foil removed",
                     "y_label": "count rate (/s)", "unit": "/s", "x_unit": "", "connect": False,
                     "lines": [{"role": "fit", "x": [1, int(n.size)], "y": [rate, rate]},
                               {"role": "expected", "x": [1, int(n.size)], "y": [0, 0]}]},
        }
    th = col(header, data, "angle_deg")
    if np.any(np.abs(th) < 3):
        raise ValueError("Angles within 3° of the beam are the unscattered beam itself; leave them out.")
    a0 = _a_guess(th, n, t)
    if np.any(th < 0) and np.any(th > 0):
        # both sides of the beam: the true zero is where the two sides match
        lim = min(5.0, float(np.min(np.abs(th))) - 1.0)
        (a, th0, b), (ea, eth0, eb), z = _poisson_fit(ruth_f, th, n, t, [a0, 0.0, 0.0],
                                                      ([0, -lim, -np.inf], [np.inf, lim, np.inf]))
        xs = np.linspace(float(th.min()), float(th.max()), 400)
        xs = xs[np.abs(xs - th0) > 2.5]
        return {
            "values": {"zero_from_both_sides_deg": {"value": th0, "uncertainty": eth0},
                       "rate_at_20deg_per_s": {"value": a, "uncertainty": ea},
                       "background_rate_per_s": {"value": b, "uncertainty": eb}},
            "evidence": {"sides_asymmetric": bool(abs(th0) > 0.3 and sig(th0, eth0) > 5),
                         "floor_sigma": float(b / eb) if eb > 0 else 0.0,
                         "misfit_sigma": misfit_sigma(z, 1.0, 3)},
            "plot": {"x": th.tolist(), "y": _log_rate(n, t).tolist(),
                     "x_label": "detector angle read (degrees), both sides of the beam",
                     "y_label": "log₁₀ count rate (/s)", "unit": "", "x_unit": "°", "connect": False,
                     "lines": [{"role": "fit", "x": xs[xs < th0].tolist(), "y": np.log10(ruth_f(xs[xs < th0], a, th0, b)).tolist()},
                               {"role": "fit", "x": xs[xs > th0].tolist(), "y": np.log10(ruth_f(xs[xs > th0], a, th0, b)).tolist()}],
                     "vlines": [{"x": th0, "label": "zero from both sides"}]},
        }
    th = np.abs(th)
    lim = float(th.min()) - 2.0
    if "thin_foil" in header:
        # follow-up: the small angles counted again with a thin foil fitted (shape only, no floor)
        (a, th0), (ea, eth0), z = _poisson_fit(lambda x, a, s: ruth_f(x, a, s, 0.0), th, n, t, [a0, 0.0],
                                               ([0, -5.0], [np.inf, lim]))
        xs = np.linspace(float(th.min()), float(th.max()), 200)
        return {
            "values": {"angle_shift_thin_foil_deg": {"value": th0, "uncertainty": eth0},
                       "rate_at_20deg_per_s": {"value": a, "uncertainty": ea}},
            "evidence": {"small_angle_excess_gone": bool(sig(th0, eth0) < 4)},
            "plot": {"x": th.tolist(), "y": _log_rate(n, t).tolist(), "x_label": "scattering angle read (degrees), thin foil",
                     "y_label": "log₁₀ count rate (/s)", "unit": "", "x_unit": "°", "connect": False,
                     "lines": [{"role": "fit", "x": xs.tolist(), "y": np.log10(ruth_f(xs, a, th0, 0.0)).tolist()},
                               {"role": "expected", "x": xs.tolist(), "y": np.log10(ruth_f(xs, a, 0.0, 0.0)).tolist()}]},
        }
    if th.size < 5:
        raise ValueError("The Rutherford fit needs counts at five or more angles.")
    # A (rate at 20°), a shift of the angle scale and a flat floor, all from the counts
    (a, th0, b), (ea, eth0, eb), z = _poisson_fit(ruth_f, th, n, t, [a0, 0.0, 0.0],
                                                  ([0, -5.0, -np.inf], [np.inf, lim, np.inf]))
    # the exponent a student gets from a straight line on log axes (no shift, no floor)
    (slope, icpt), cl = np.polyfit(np.log(np.sin(np.radians(th) / 2)), np.log(np.maximum(n, 0.5) / t), 1,
                                   w=np.sqrt(np.maximum(n, 1.0)), cov="unscaled")
    xs = np.linspace(float(th.min()), float(th.max()), 200)
    return {
        "values": {
            "rate_at_20deg_per_s": {"value": a, "uncertainty": ea},
            "expected_rate_at_20deg_per_s": {"value": a_exp},
            "scattering_angle_shift_deg": {"value": th0, "uncertainty": eth0},
            "background_rate_per_s": {"value": b, "uncertainty": eb},
            "rutherford_exponent": {"value": -float(slope), "uncertainty": float(np.sqrt(cl[0, 0]))},
        },
        "evidence": {
            # signed: above zero, the small angles count too fast for the angles read
            "angle_shift_sigma": float(th0 / eth0) if eth0 > 0 else 0.0,
            "angle_shift_deg": th0,
            "floor_sigma": float(b / eb) if eb > 0 else 0.0,
            "misfit_sigma": misfit_sigma(z, 1.0, 3),
            "rate_ratio": a / a_exp,
        },
        "plot": {"x": th.tolist(), "y": _log_rate(n, t).tolist(), "x_label": "scattering angle read (degrees)",
                 "y_label": "log₁₀ count rate (/s)", "unit": "", "x_unit": "°", "connect": False,
                 "lines": [{"role": "fit", "x": xs.tolist(), "y": np.log10(ruth_f(xs, a, th0, b)).tolist()},
                           {"role": "expected", "x": xs.tolist(), "y": np.log10(ruth_f(xs, a, 0.0, 0.0)).tolist()}]},
    }


# =============================================================================
# --- X-ray Bragg diffraction from NaCl: n λ = 2 d sin θ -------------------------------
def bragg_xray(header, data, expected):
    lam = float(expected.get("wavelength_pm", 71.07))
    lam_b = float(expected.get("kbeta_wavelength_pm", 63.26))
    d_exp = float(expected.get("d_spacing_pm", 282.0))
    prec = float(expected.get("precision_deg", 0.05))
    if "crystal_angle_deg" in header:
        # follow-up: detector parked on a peak, the crystal rocked through it on its own
        x = col(header, data, "crystal_angle_deg")
        y = col(header, data, "count_rate")
        half = float(np.mean(col(header, data, "detector_angle_deg"))) / 2
        (dd, aa, x0, w), (_, _, ex0, _) = fit(gauss, x, y, [float(y.min()), float(np.ptp(y)), float(x[np.argmax(y)]), 0.3])
        off = x0 - half
        xs = dense(x)
        return {
            "values": {"rocking_peak_crystal_deg": {"value": x0, "uncertainty": ex0},
                       "half_detector_angle_deg": {"value": half},
                       "crystal_offset_deg": {"value": off, "uncertainty": ex0}},
            "evidence": {"rocking_peak_off": bool(abs(off) > 0.15 and sig(off, ex0) > 5)},
            "plot": {"x": x.tolist(), "y": y.tolist(), "x_label": "crystal angle read (degrees), detector fixed",
                     "y_label": "count rate (/s)", "unit": "/s", "x_unit": "°",
                     "lines": [{"role": "fit", "x": xs.tolist(), "y": gauss(xs, dd, aa, x0, w).tolist()}],
                     "vlines": [{"x": half, "label": "half the detector angle"}]},
        }
    if "detector_angle_deg" in header:
        # follow-up: crystal out, the detector scanned through the direct beam (it should peak at 0°)
        x = col(header, data, "detector_angle_deg")
        y = col(header, data, "count_rate")
        (dd, aa, x0, w), (_, _, ex0, _) = fit(gauss, x, y, [float(y.min()), float(np.ptp(y)), float(x[np.argmax(y)]), 0.5])
        xs = dense(x)
        return {
            "values": {"direct_beam_centre_deg": {"value": x0, "uncertainty": ex0},
                       "goniometer_zero_from_beam_deg": {"value": x0 / 2, "uncertainty": ex0 / 2}},
            "evidence": {"direct_beam_off_zero": bool(abs(x0) > 0.2 and sig(x0, ex0) > 5)},
            "plot": {"x": x.tolist(), "y": y.tolist(), "x_label": "detector angle 2θ read (degrees), crystal removed",
                     "y_label": "count rate (/s)", "unit": "/s", "x_unit": "°",
                     "lines": [{"role": "fit", "x": xs.tolist(), "y": gauss(xs, dd, aa, x0, w).tolist()}],
                     "vlines": [{"x": 0.0, "label": "where the beam should be"}]},
        }
    if "zr_filter" in header:
        # follow-up: count rate at the odd peak without and with the zirconium filter in the beam
        m0, e0, m1, e1, plot = _two_groups(header, data, "zr_filter", "count_rate", "/s", "count rate at the odd peak (/s)",
                                           "reading (no filter, then the zirconium filter in)", 0.5)
        ratio = m1 / m0 if m0 > 0 else 1.0
        er = abs(ratio) * float(np.hypot(e0 / max(m0, 1e-9), e1 / max(abs(m1), 1e-9)))
        return {
            "values": {"peak_rate_without_filter": {"value": m0, "uncertainty": e0},
                       "peak_rate_with_filter": {"value": m1, "uncertainty": e1},
                       "filter_transmission": {"value": ratio, "uncertainty": er}},
            # Zr passes roughly half of Mo Kα but only a few per cent of Kβ
            "evidence": {"odd_peak_suppressed": bool(ratio < 0.15 and sig(m0 - m1, float(np.hypot(e0, e1))) > 5)},
            "plot": plot,
        }
    n = col(header, data, "order_n")
    th = col(header, data, "angle_deg")
    if np.any(n < 1) or np.any(np.abs(n - np.round(n)) > 1e-6):
        raise ValueError("Orders must be whole numbers 1, 2, 3 ...")
    f = lambda nn, d100, t0: np.degrees(np.arcsin(np.clip(nn * lam / (200 * d100), -1, 1))) + t0
    if "crystal" in header:
        # follow-up: two orders on the crystal used (0) and on the reference NaCl crystal (1), one goniometer zero
        cr = col(header, data, "crystal")
        f2 = lambda X, d0, d1, t0: f(X[0], np.where(X[1] == 0, d0, d1), t0)
        p, cv = curve_fit(f2, np.vstack([n, cr]), th, p0=[d_exp / 100, d_exp / 100, 0.0],
                          sigma=np.full(n.size, prec), absolute_sigma=True, maxfev=20000)
        d0, d1 = 100 * p[0], 100 * p[1]
        ed0, ed1 = 100 * float(np.sqrt(cv[0, 0])), 100 * float(np.sqrt(cv[1, 1]))
        edd = 100 * float(np.sqrt(max(cv[0, 0] + cv[1, 1] - 2 * cv[0, 1], 0.0)))
        ns = np.linspace(float(n.min()), float(n.max()), 60)
        return {
            "values": {"d_crystal_used_pm": {"value": d0, "uncertainty": ed0},
                       "d_reference_crystal_pm": {"value": d1, "uncertainty": ed1}},
            "evidence": {"crystals_differ": bool(abs(d0 / d1 - 1) > 0.02 and sig(d0 - d1, edd) > 5)},
            "plot": {"x": n.tolist(), "y": th.tolist(), "x_label": "order n: crystal used, then the reference NaCl crystal",
                     "y_label": "Bragg angle θ read (degrees)", "unit": "°", "x_unit": "", "connect": False,
                     "lines": [{"role": "fit", "x": ns.tolist(), "y": f(ns, p[0], p[2]).tolist()},
                               {"role": "fit", "x": ns.tolist(), "y": f(ns, p[1], p[2]).tolist()}]},
        }
    if n.size < 3:
        raise ValueError("The Bragg fit needs peaks from at least three orders.")

    def fitm(m, p0):
        p, cv = curve_fit(f, n[m], th[m], p0=p0, sigma=np.full(int(m.sum()), prec), absolute_sigma=True, maxfev=20000)
        return p, cv

    allm = np.ones(n.size, bool)
    p, cv = fitm(allm, [d_exp / 100, 0.0])
    # each order against a fit through the others (a peak taken from the wrong line), with the
    # prediction uncertainty of that fit at the left-out order added in quadrature
    worst, odd, rest = 0.0, -1, 0.0
    if n.size >= 4:
        for i in range(n.size):
            m = allm.copy()
            m[i] = False
            pi, ci = fitm(m, p)
            h = 1e-6
            gr = np.array([(f(n[i], pi[0] + h, pi[1]) - f(n[i], pi[0] - h, pi[1])) / (2 * h), 1.0])
            sd = float(np.sqrt(max(gr @ ci @ gr, 0.0)))
            zz = abs(float(th[i] - f(n[i], *pi))) / float(np.hypot(prec, sd))
            if zz > worst:
                worst, odd = zz, i
                rest = misfit_sigma(th[m] - f(n[m], *pi), prec, 2)
    values = {}
    kbeta = False
    if worst > 5 and rest < 4:
        m = allm.copy()
        m[odd] = False
        p, cv = fitm(m, p)
        lam_odd = float(2 * 100 * p[0] * np.sin(np.radians(th[odd] - p[1])) / n[odd])
        kbeta = bool(abs(lam_odd / lam_b - 1) < 0.015)
        values = {"odd_point_order": {"value": float(n[odd])}, "odd_point_wavelength_pm": {"value": lam_odd}}
    d, t0 = 100 * float(p[0]), float(p[1])
    ed, et0 = 100 * float(np.sqrt(cv[0, 0])), float(np.sqrt(cv[1, 1]))
    d_each = n * lam / (2 * np.sin(np.radians(th)))
    lo, hi = int(np.argmin(n)), int(np.argmax(n))
    ns = np.linspace(float(n.min()), float(n.max()), 120)
    return {
        "values": {
            "d_spacing_fit_pm": {"value": d, "uncertainty": ed},
            "d_spacing_accepted_pm": {"value": d_exp},
            "goniometer_zero_offset_deg": {"value": t0, "uncertainty": et0},
            "d_from_lowest_order_pm": {"value": float(d_each[lo])},
            "d_from_highest_order_pm": {"value": float(d_each[hi])},
            "xray_wavelength_pm": {"value": lam},
            **values,
        },
        "evidence": {
            "d_error_pct": 100 * (d / d_exp - 1),
            "d_error_sigma": float((d - d_exp) / ed),
            "zero_offset_sigma": float(t0 / et0),
            "outlier_sigma": worst,
            "odd_point_matches_kbeta": kbeta,
        },
        "plot": {"x": n.tolist(), "y": th.tolist(), "x_label": "diffraction order n (Kα peak)",
                 "y_label": "Bragg angle θ read (degrees)", "unit": "°", "x_unit": "", "connect": False,
                 "lines": [{"role": "fit", "x": ns.tolist(), "y": f(ns, p[0], p[1]).tolist()},
                           {"role": "expected", "x": ns.tolist(), "y": f(ns, d_exp / 100, 0.0).tolist()}]},
    }


ANALYSES = {"e_over_m": e_over_m, "esr": esr, "rutherford": rutherford, "bragg_xray": bragg_xray}
COLUMNS = {
    "e_over_m": (["accelerating_voltage_V", "coil_current_A", "beam_radius_cm"], ["coil_current_A", "probe_field_mT"],
                 ["turned", "coil_current_A"], ["meter_at_tube", "voltage_V"]),
    "esr": (["rf_frequency_MHz", "resonance_field_mT"], ["field_used_mT", "probe_field_mT"]),
    "rutherford": (["angle_deg", "counts", "live_time_s"], ["foil_out", "counts", "live_time_s"],
                   ["reading", "pressure_mbar"]),
    "bragg_xray": (["order_n", "angle_deg"], ["detector_angle_deg", "count_rate"], ["zr_filter", "count_rate"]),
}
