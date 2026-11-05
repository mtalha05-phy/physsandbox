"""Core undergraduate lab experiments. Runs INSIDE the sandbox (loaded by experiments.py).

  photoelectric       wavelength_nm, stopping_voltage_V        (follow-up: + anode_masked)
  franck_hertz        accelerating_voltage_V, anode_current_nA (follow-up: minute, oven_temperature_C)
  radioactive_decay   time_s, counts                           (follow-up: + source_removed)

Where there are few points (five mercury lines, five Franck-Hertz peaks) the
uncertainties come from a stated measurement precision, not from the scatter
of the fit; counts use Poisson statistics.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from experiments import col, mean_err, sig, dense  # noqa: F401  (loader runs after these exist)

H = 6.62607015e-34
E_CH = 1.602176634e-19
C = 299792458.0
LN2 = float(np.log(2))


class DataError(ValueError):
    pass


def _line(x, y, prec):
    """Straight line y = a x + b with a known per-point precision (absolute sigma)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    a, b = np.polyfit(x, y, 1)
    xm = float(np.mean(x))
    sxx = float(np.sum((x - xm) ** 2))
    ea = prec / np.sqrt(sxx)
    eb = prec * np.sqrt(1 / x.size + xm ** 2 / sxx)
    return float(a), float(b), float(ea), float(eb)


def _worst_outlier(x, y, prec):
    """Largest leave-one-out residual, in units of its own expected spread."""
    worst = 0.0
    if x.size < 4:
        return worst
    for i in range(x.size):
        m = np.ones(x.size, bool)
        m[i] = False
        a, b = np.polyfit(x[m], y[m], 1)
        xm = np.mean(x[m])
        var_pred = prec ** 2 * (1 / m.sum() + (x[i] - xm) ** 2 / np.sum((x[m] - xm) ** 2))
        worst = max(worst, abs(float(y[i] - (a * x[i] + b))) / float(np.sqrt(prec ** 2 + var_pred)))
    return worst


# --- Photoelectric effect: e V_s = h f − φ -------------------------------------
def photoelectric(header, data, expected):
    he_exp = float(expected.get("h_over_e_Vs", H / E_CH))
    prec = float(expected.get("precision_V", 0.01))
    lam = col(header, data, "wavelength_nm")
    v = col(header, data, "stopping_voltage_V")
    f = C / (lam * 1e-9) / 1e14                      # in 10^14 Hz, so the fit is well conditioned
    a_exp = he_exp * 1e14                            # V per 10^14 Hz
    if "anode_masked" in header:
        # follow-up: every line read again with a mask so the light reaches only the cathode
        g = col(header, data, "anode_masked")
        a0, b0, e0, _ = _line(f[g == 0], v[g == 0], prec)
        a1, b1, e1, _ = _line(f[g == 1], v[g == 1], prec)
        change = 100 * (a1 / a0 - 1)
        change_sig = (a1 - a0) / float(np.hypot(e0, e1))
        fs = dense(f)
        return {
            "values": {"h_over_e_open_Vs": {"value": a0 * 1e-14, "uncertainty": e0 * 1e-14},
                       "h_over_e_masked_Vs": {"value": a1 * 1e-14, "uncertainty": e1 * 1e-14},
                       "slope_change_masked_pct": {"value": change},
                       "accepted_h_over_e_Vs": {"value": he_exp}},
            # masking the anode can only remove a reverse current, so only a rise counts
            "evidence": {"anode_effect_found": bool(change > 3 and change_sig > 5),
                         "masked_slope_change_sigma": float(change_sig)},
            "plot": {"x": f.tolist(), "y": v.tolist(), "connect": False,
                     "x_label": "light frequency (×10¹⁴ Hz): lower line anode lit, upper line anode masked",
                     "y_label": "stopping voltage (V)", "unit": "V", "x_unit": "",
                     "lines": [{"role": "fit", "x": fs.tolist(), "y": (a0 * fs + b0).tolist()},
                               {"role": "fit", "x": fs.tolist(), "y": (a1 * fs + b1).tolist()},
                               {"role": "expected", "x": fs.tolist(),
                                "y": (a_exp * (fs - np.mean(f[g == 1])) + np.mean(v[g == 1])).tolist()}]},
        }
    if f.size < 3:
        raise DataError("The photoelectric fit needs at least three spectral lines.")
    a, b, ea, eb = _line(f, v, prec)
    fs = dense(f)
    return {
        "values": {
            "h_over_e_Vs": {"value": a * 1e-14, "uncertainty": ea * 1e-14},
            "accepted_h_over_e_Vs": {"value": he_exp},
            "planck_from_photoelectric_Js": {"value": a * 1e-14 * E_CH, "uncertainty": ea * 1e-14 * E_CH},
            "work_function_eV": {"value": -b, "uncertainty": eb},
            "threshold_frequency_1e14Hz": {"value": -b / a},
        },
        "evidence": {
            "h_error_pct": 100 * (a / a_exp - 1),
            "h_dev_sigma": (a - a_exp) / ea,           # signed: negative when the line is too shallow
            "outlier_sigma": _worst_outlier(f, v, prec),
        },
        "plot": {"x": f.tolist(), "y": v.tolist(), "connect": False,
                 "x_label": "light frequency (×10¹⁴ Hz, mercury lines)", "y_label": "stopping voltage (V)",
                 "unit": "V", "x_unit": "",
                 "lines": [{"role": "fit", "x": fs.tolist(), "y": (a * fs + b).tolist()},
                           {"role": "expected", "x": fs.tolist(),
                            "y": (a_exp * (fs - np.mean(f)) + np.mean(v)).tolist()}]},
    }


# --- Franck–Hertz (mercury): peaks every E₁/e volts ------------------------------
def _fh_peaks(V, I, e_exp):
    dv = float(np.median(np.diff(V)))
    k = max(1, int(round(0.3 / dv)))                 # smooth over about 0.3 V
    Is = np.convolve(I, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    Is[:k], Is[-k:] = I[:k], I[-k:]
    top = float(np.max(I))
    # a saturated amplifier sits exactly at its rail for several readings
    rail = I >= top * (1 - 2e-4)
    saturated = bool(rail.sum() >= 4)
    idx, props = find_peaks(Is, prominence=0.0, distance=max(1, int(0.5 * e_exp / dv)))
    keep = (props["prominences"] > 0.03 * Is[idx]) & (Is[idx] > 0.02 * top)
    idx = idx[keep]
    peaks = []
    for i in idx:
        w = np.abs(V - V[i]) <= 0.5
        clipped = bool(saturated and np.any(rail & (np.abs(V - V[i]) <= 1.0)))
        pos = float(V[i])
        if w.sum() >= 5 and not clipped:
            c2, c1, _ = np.polyfit(V[w] - V[i], I[w], 2)
            if c2 < 0 and abs(-c1 / (2 * c2)) < 0.5:
                pos = float(V[i] - c1 / (2 * c2))
        peaks.append({"i": int(i), "V": pos, "I": float(Is[i]), "clipped": clipped})
    return Is, peaks, saturated


def franck_hertz(header, data, expected):
    e_exp = float(expected.get("excitation_eV", 4.9))
    prec = float(expected.get("peak_precision_V", 0.15))
    if "oven_temperature_C" in header:
        # follow-up: the oven thermometer, read for a few minutes
        tmin = col(header, data, "minute")
        temp = col(header, data, "oven_temperature_C")
        setp = float(expected.get("oven_setpoint_C", 180))
        m, e = mean_err(temp)
        return {
            "values": {"oven_temperature_C": {"value": m, "uncertainty": e},
                       "oven_setpoint_C": {"value": setp}},
            # mercury vapour pressure roughly doubles every 20 °C near 180 °C
            "evidence": {"oven_cold": bool(setp - m > 10), "oven_shortfall_C": setp - m},
            "plot": {"x": tmin.tolist(), "y": temp.tolist(), "x_label": "time (minutes)",
                     "y_label": "oven temperature (°C)", "unit": "°C", "x_unit": "", "connect": True,
                     "lines": [{"role": "fit", "x": [float(tmin.min()), float(tmin.max())], "y": [m, m]},
                               {"role": "expected", "x": [float(tmin.min()), float(tmin.max())], "y": [setp, setp]}]},
        }
    o = np.argsort(col(header, data, "accelerating_voltage_V"))
    V = col(header, data, "accelerating_voltage_V")[o]
    I = col(header, data, "anode_current_nA")[o]
    if V.size < 30:
        raise DataError("The Franck–Hertz analysis needs a sweep of at least 30 readings.")
    Is, peaks, saturated = _fh_peaks(V, I, e_exp)
    good = [p for p in peaks if not p["clipped"]]
    # order of each peak, counting the gaps in units of the expected spacing
    # (the first peak's order assumes a contact potential of roughly 0–4 V)
    order = max(1, int(round((peaks[0]["V"] - 2.0) / e_exp))) if peaks else 1
    n = []
    for j, p in enumerate(peaks):
        if j:
            order += max(1, int(round((p["V"] - peaks[j - 1]["V"]) / e_exp)))
        n.append(order)
    contrasts = []
    for j in range(len(peaks) - 1):
        p, q = peaks[j], peaks[j + 1]
        if p["clipped"]:
            continue
        valley = float(np.min(Is[p["i"]:q["i"] + 1]))
        contrasts.append((p["I"] - valley) / p["I"])
    values = {"peaks_found": {"value": len(peaks)},
              "saturated_peaks": {"value": sum(p["clipped"] for p in peaks)}}
    evidence = {"amplifier_saturated": saturated, "peaks_found": len(peaks)}
    vlines = []
    if contrasts:
        values["peak_contrast"] = {"value": float(np.mean(contrasts))}
        evidence["peak_contrast"] = float(np.mean(contrasts))
    if len(good) >= 3:
        ng = np.array([n[j] for j, p in enumerate(peaks) if not p["clipped"]], float)
        vg = np.array([p["V"] for p in good])
        s, v0, es, ev0 = _line(ng, vg, prec)
        values.update({"excitation_energy_eV": {"value": s, "uncertainty": es},
                       "accepted_excitation_eV": {"value": e_exp},
                       "contact_potential_V": {"value": v0, "uncertainty": ev0},
                       "first_peak_V": {"value": float(vg[0])}})
        evidence["spacing_dev_sigma"] = (s - e_exp) / es
        evidence["spacing_error_pct"] = 100 * (s / e_exp - 1)
        label = f"peaks {s:.2f} V apart"
    else:
        label = "peaks"
    for j, p in enumerate(peaks):
        vlines.append({"x": p["V"], "label": (label if j == 0 else "") + (" (clipped)" if p["clipped"] else "")})
    return {
        "values": values,
        "evidence": evidence,
        "plot": {"x": V.tolist(), "y": I.tolist(), "x_label": "accelerating voltage (V)",
                 "y_label": "anode current (nA)", "unit": "nA", "x_unit": "V", "connect": True,
                 "vlines": vlines, "lines": []},
    }


# --- Radioactive decay with a Geiger–Müller counter: N = N₀ e^(−λt) + B ------------
def _decay_fit(t, c, dt, t_half_guess):
    tm = t - t.min() + dt / 2
    f = lambda tm, A, lam, B: A * np.exp(-lam * tm) + B
    p = [max(float(c[0]), 1.0), LN2 / t_half_guess, max(float(np.min(c)), 0.0)]
    # Poisson weights from the model, not the noisy counts. Counts that already had a
    # background subtracted still carry that background's noise, so an extra variance v0
    # is chosen so the fit's chi-squared per degree of freedom is 1 (it stays zero for raw counts).
    v0 = 0.0
    dof = max(1, c.size - 3)
    for _ in range(4):
        s = np.sqrt(np.maximum(f(tm, *p), 1.0) + v0)
        # a floor below about −0.5 counts/s can't come from over-subtracting a room background
        p, cov = curve_fit(f, tm, c, p0=p, sigma=s, absolute_sigma=True, maxfev=20000,
                           bounds=([0, 0, -0.5 * dt], [np.inf, np.inf, np.inf]))
        m, r2 = np.maximum(f(tm, *p), 1.0), (c - f(tm, *p)) ** 2
        if np.sum(r2 / m) <= dof:
            break
        lo, hi = 0.0, float(np.max(r2))
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if np.sum(r2 / (m + mid)) > dof else (lo, mid)
        v0 = (lo + hi) / 2
    return p, np.sqrt(np.clip(np.diag(cov), 0, None)), f


def radioactive_decay(header, data, expected):
    t_exp = float(expected.get("half_life_s", 153.1))
    tau = float(expected.get("dead_time_us", 150)) * 1e-6
    t = col(header, data, "time_s")
    c = col(header, data, "counts")
    bk = None
    if "source_removed" in header:
        g = col(header, data, "source_removed")
        bk, bt = c[g == 1], t[g == 1]
        t, c = t[g == 0], c[g == 0]
    o = np.argsort(t)
    t, c = t[o], c[o]
    if t.size < 8:
        raise DataError("The decay fit needs at least eight counting intervals.")
    dt = float(np.median(np.diff(t)))
    (A, lam, B), (eA, elam, eB), f = _decay_fit(t, c, dt, t_exp)
    T, eT = LN2 / lam, LN2 / lam ** 2 * elam
    tm = t - t.min() + dt / 2
    ts = np.linspace(float(tm.min()), float(tm.max()), 200)
    xs = ts + t.min() - dt / 2
    log = lambda y: np.log10(np.maximum(np.asarray(y, float), 0.5)).tolist()
    lines = [{"role": "fit", "x": xs.tolist(), "y": log(f(ts, A, lam, B))},
             {"role": "expected", "x": xs.tolist(), "y": log(A * np.exp(-LN2 / t_exp * ts))}]
    values = {
        "half_life_s": {"value": T, "uncertainty": eT},
        "expected_half_life_s": {"value": t_exp},
        "floor_cps": {"value": B / dt, "uncertainty": eB / dt},
        "peak_count_rate_cps": {"value": float(np.max(c)) / dt},
    }
    evidence = {
        "half_life_dev_sigma": (T - t_exp) / eT,      # signed: positive when the decay looks too slow
        "half_life_error_pct": 100 * (T / t_exp - 1),
        "floor_sigma": B / eB if eB > 0 else 0.0,
        # fraction of counts the counter loses at the highest rate, from its stated dead time
        "peak_loss_pct": 100 * float(np.max(c)) / dt * tau,
    }
    px, py = t.tolist(), log(c)
    x_label = "time since elution (s)"
    if bk is not None and bk.size >= 3:
        mb, _ = mean_err(bk)
        eb = float(np.sqrt(max(mb, 1.0) / bk.size))   # Poisson error on the mean count
        values["room_background_cps"] = {"value": mb / dt, "uncertainty": eb / dt}
        same = abs(B - mb) / float(np.hypot(eB, eb)) < 4
        evidence = {"floor_from_room_background": bool(B / eB > 5 and mb / eb > 5 and same),
                    "floor_minus_background_sigma": (B - mb) / float(np.hypot(eB, eb))}
        px, py = px + bt.tolist(), py + log(bk)
        x_label = "time (s): decay run, then the source removed"
        lines.append({"role": "expected", "x": [float(t.min()), float(bt.max())], "y": log([mb, mb])})
        lines = [lines[0], lines[2]]
    return {
        "values": values,
        "evidence": evidence,
        "plot": {"x": px, "y": py, "x_label": x_label, "y_label": "log₁₀ counts per interval",
                 "unit": "", "x_unit": "s", "connect": False, "lines": lines},
    }


ANALYSES = {
    "photoelectric": photoelectric,
    "franck_hertz": franck_hertz,
    "radioactive_decay": radioactive_decay,
}

COLUMNS = {
    "photoelectric": (["wavelength_nm", "stopping_voltage_V"],),
    "franck_hertz": (["accelerating_voltage_V", "anode_current_nA"], ["minute", "oven_temperature_C"]),
    "radioactive_decay": (["time_s", "counts"],),
}
