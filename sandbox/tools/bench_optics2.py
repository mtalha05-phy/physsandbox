"""Optics, semiconductor and thermocouple bench: thin-lens focal length, Brewster's angle,
band gap of germanium from its intrinsic resistance, type K thermocouple calibration.
Runs INSIDE the sandbox; loaded by experiments.py.

Same contract as experiments.py: (header, data, expected) -> values, evidence, plot.
Few points, so every fit uses the precision the researcher states (absolute sigma),
not the scatter of the points themselves.

  thin_lens     object_distance_cm, image_distance_cm     (follow-ups: index_mark_cm, lens_centre_cm;
                                                            + lens_reversed; distant_object_m, distant_image_cm)
  brewster      incidence_angle_deg, reflected_power_uW   (follow-ups: polariser_angle_deg; analyser_crossed,
                                                            transmitted_uW; reading, normal_incidence_deg;
                                                            + nd_filter; + iris_closed)
  band_gap      temperature_K, resistance_ohm             (follow-ups: + cooling; reading, oven_thermometer_K,
                                                            reference_thermometer_K; + current_mA;
                                                            temperature_K, two_terminal_ohm, four_terminal_ohm)
  thermocouple  hot_junction_C, emf_mV                    (follow-ups: reading, ice_bath_C; reading,
                                                            shorted_emf_mV; leg, magnetic; + deeper)
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2 as _chi2, norm as _norm

from experiments import col, dense, groups_result, mean_err, sig

K_B = 8.617333262e-5          # Boltzmann constant (eV/K)


def wfit(x, y, s, deg=1):
    """Polynomial fit with absolute sigma s: coefficients (highest power first) and covariance."""
    return np.polyfit(x, y, deg, w=1 / np.broadcast_to(s, np.shape(x)), cov="unscaled")


def misfit(resid, s, n_par):
    """Chi-square of the residuals against the stated precision: (Gaussian significance, chi²/dof)."""
    dof = max(1, resid.size - n_par)
    c2 = float(np.sum((resid / s) ** 2))
    z = float(_norm.isf(_chi2.sf(c2, dof)))
    return float(np.clip(np.nan_to_num(z, posinf=99.0, neginf=0.0), 0.0, 99.0)), c2 / dof


def flat(x, v):
    return [float(np.min(x)), float(np.max(x))], [float(v), float(v)]


def reading_plot(r, y, mean, ref, x_label, y_label, unit):
    """Repeated readings of one quantity, with their mean and the value expected."""
    xs, ym = flat(r, mean)
    return {"x": r.tolist(), "y": y.tolist(), "x_label": x_label, "y_label": y_label, "unit": unit,
            "x_unit": "", "connect": True,
            "lines": [{"role": "fit", "x": xs, "y": ym}, {"role": "expected", "x": xs, "y": flat(r, ref)[1]}]}


# =====================================================================================
# --- Thin converging lens on an optical bench: 1/u + 1/v = 1/f -----------------------
def _lens_fit(u, v, pu, pv):
    """1/v against 1/u in m⁻¹ (dioptres), weighted with the stated precision on u and v."""
    if u.size < 3:
        raise RuntimeError("at least three object distances are needed for each line")
    x, y = 100 / u, 100 / v
    sx, sy = 100 * pu / u ** 2, 100 * pv / v ** 2
    m = -1.0
    for _ in range(3):                                 # effective variance: the slope carries the u error
        s = np.hypot(sy, m * sx)
        p, cov = wfit(x, y, s)
        m = float(p[0])
    return x, y, s, p, cov


def thin_lens(header, data, expected):
    f_exp = float(expected.get("focal_length_cm", 10.0))
    pu = float(expected.get("precision_u_cm", 0.1))
    pv = float(expected.get("precision_v_cm", 0.2))
    if "index_mark_cm" in header:
        # follow-up: the holder's index mark and the lens's own mid-plane, read on the bench scale
        idx = col(header, data, "index_mark_cm")
        mid = col(header, data, "lens_centre_cm")
        d = idx - mid
        m, e = mean_err(d)
        e = max(e, pu * np.sqrt(2 / d.size))
        r = np.arange(1, d.size + 1)
        return {
            "values": {"index_mark_minus_lens_centre_cm": {"value": m, "uncertainty": e}},
            "evidence": {"lens_index_offset_found": bool(abs(m) > 0.3 and sig(m, e) > 5)},
            "plot": reading_plot(r, d, m, 0.0, "holder position along the bench (reading number)",
                                 "index mark − lens centre (cm)", "cm"),
        }
    if "distant_object_m" in header:
        # follow-up: a distant window or lamp focused on the screen, distance read from the lens itself
        U = 100 * col(header, data, "distant_object_m")
        V = col(header, data, "distant_image_cm")
        fi = 1 / (1 / U + 1 / V)
        m, e = mean_err(fi)
        e = max(e, pv / np.sqrt(fi.size))              # v ≈ f, so the focusing error carries straight into f
        r = np.arange(1, fi.size + 1)
        return {
            "values": {"focal_length_distant_object_cm": {"value": m, "uncertainty": e},
                       "focal_length_stated_cm": {"value": f_exp}},
            "evidence": {"lens_focal_length_off": bool(abs(m / f_exp - 1) > 0.06 and sig(m - f_exp, e) > 5)},
            "plot": reading_plot(r, fi, m, f_exp, "reading number, distant object focused on the screen",
                                 "focal length from 1/u + 1/v (cm)", "cm"),
        }
    u = col(header, data, "object_distance_cm")
    v = col(header, data, "image_distance_cm")
    if np.any(u <= 0) or np.any(v <= 0):
        raise RuntimeError("object and image distances must be positive (real object, real image)")
    if "lens_reversed" in header:
        # follow-up: the same kind of run with the lens turned round in its holder
        rv = col(header, data, "lens_reversed")
        fits = [_lens_fit(u[rv == k], v[rv == k], pu, pv) for k in (0, 1)]
        (m0, e0), (m1, e1) = [(float(p[0]), float(np.sqrt(c[0, 0]))) for _, _, _, p, c in fits]
        d, ed = m1 - m0, float(np.hypot(e0, e1))
        lines = []
        for x, _, _, p, _ in fits:
            xs = dense(x)
            lines.append({"role": "fit", "x": xs.tolist(), "y": np.polyval(p, xs).tolist()})
        x_all = np.concatenate([fits[0][0], fits[1][0]])
        xs = dense(x_all)
        lines.append({"role": "expected", "x": xs.tolist(), "y": (100 / f_exp - xs).tolist()})
        return {
            "values": {"line_slope_as_before": {"value": m0, "uncertainty": e0},
                       "line_slope_lens_reversed": {"value": m1, "uncertainty": e1}},
            # a thick lens's principal planes swap sides when it is turned round, so the slope's
            # departure from −1 changes sign; an index mark off the lens stays where it was
            "evidence": {"lens_slope_flips": bool(sig(d, ed) > 5 and (m0 + 1) * (m1 + 1) < 0)},
            "plot": {"x": x_all.tolist(), "y": np.concatenate([fits[0][1], fits[1][1]]).tolist(),
                     "x_label": "1/u (m⁻¹), lens as before then turned round", "y_label": "1/v (m⁻¹)",
                     "unit": "", "x_unit": "", "connect": False, "lines": lines},
        }
    x, y, s, p, cov = _lens_fit(u, v, pu, pv)
    m, c = float(p[0]), float(p[1])
    if m >= 0:
        raise RuntimeError("1/v should fall as 1/u rises; check the two columns")
    # power from the two intercepts, averaged: 1/v at 1/u = 0 is c, 1/u at 1/v = 0 is −c/m
    P = c * (1 - 1 / m) / 2
    g = np.array([c / (2 * m ** 2), (1 - 1 / m) / 2])
    eP = float(np.sqrt(max(g @ cov @ g, 0.0)))
    f, ef = 100 / P, 100 * eP / P ** 2
    mis, red = misfit(y - np.polyval(p, x), s, 2)
    # scatter larger than stated widens every uncertainty, so a noisy run is not read as a biased one
    birge = max(1.0, float(np.sqrt(red)))
    em = float(np.sqrt(cov[0, 0]))
    # a lens reference point off by δ (u long, v short) tilts the line: slope ≈ −(1 + δ/f)²
    delta = float(f * (np.sqrt(-m) - 1))
    e_delta = float(f * em / (2 * np.sqrt(-m)))
    xs = dense(np.append(x, 0))
    return {
        "values": {
            "focal_length_fit_cm": {"value": f, "uncertainty": ef * birge},
            "focal_length_stated_cm": {"value": f_exp},
            "line_slope": {"value": m, "uncertainty": em * birge},
            "reference_offset_from_slope_cm": {"value": delta, "uncertainty": e_delta * birge},
            "scatter_over_precision": {"value": float(np.sqrt(red))},
        },
        "evidence": {
            "slope_sigma": float((m + 1) / (em * birge)),
            "f_error_pct": 100 * (f / f_exp - 1),
            "f_error_sigma": sig(f - f_exp, ef * birge),
            "misfit_sigma": mis,
        },
        "plot": {"x": x.tolist(), "y": y.tolist(), "x_label": "1/u, reciprocal object distance (m⁻¹)",
                 "y_label": "1/v, reciprocal image distance (m⁻¹)", "unit": "", "x_unit": "", "connect": False,
                 "lines": [{"role": "fit", "x": xs.tolist(), "y": np.polyval(p, xs).tolist()},
                           {"role": "expected", "x": xs.tolist(), "y": (100 / f_exp - xs).tolist()}]},
    }


# =====================================================================================
# --- Brewster's angle: Fresnel reflection of p-polarised light from glass -------------
def fresnel(th_deg, n):
    """Power reflectances (R_p, R_s) at an air–glass surface, angle of incidence in degrees."""
    t = np.radians(np.clip(np.abs(th_deg), 0.0, 89.99))
    c, s2 = np.cos(t), np.sin(t) ** 2
    root = np.sqrt(n ** 2 - s2)
    rp = (n ** 2 * c - root) / (n ** 2 * c + root)
    rs = (c - root) / (c + root)
    return rp ** 2, rs ** 2


def _brewster_sigma(th, P, p_inc, n, th0, pu, pp, pd):
    """Stated power precision (absolute and relative) plus the angle reading's, through dR/dθ."""
    dR = (fresnel(th - th0 + 0.05, n)[0] - fresnel(th - th0 - 0.05, n)[0]) / 0.1
    return np.sqrt(pu ** 2 + (pp * P) ** 2 + (p_inc * dR * pd) ** 2)


def brewster(header, data, expected):
    n_exp = float(expected.get("refractive_index", 1.515))
    p0 = float(expected.get("incident_power_uW", 1000.0))
    pu = float(expected.get("precision_uW", 0.1))
    pp = float(expected.get("precision_pct", 1.0)) / 100
    pd = float(expected.get("precision_deg", 0.2))
    thB = float(np.degrees(np.arctan(n_exp)))
    if "polariser_angle_deg" in header:
        # follow-up: at the minimum, the polariser turned either side of its setting
        phi = col(header, data, "polariser_angle_deg")
        P = col(header, data, "reflected_power_uW")
        s = np.sqrt(pu ** 2 + (pp * P) ** 2)
        k = int(np.argmin(P))
        f = lambda ph, a, ph0, b: a * np.sin(np.radians(ph - ph0)) ** 2 + b
        (a, ph0, b), cov = curve_fit(f, phi, P, p0=[max(float(P.max() - P.min()), 1.0) * 30, float(phi[k]),
                                                    float(P.min())],
                                     sigma=s, absolute_sigma=True, maxfev=20000)
        a, ph0, b = float(a), float(ph0), float(b)
        e = [float(v) for v in np.sqrt(np.clip(np.diag(cov), 0, None))]
        xs = dense(phi)
        return {
            "values": {"polariser_turn_to_deepest_minimum_deg": {"value": ph0, "uncertainty": e[1]},
                       "power_at_deepest_minimum_uW": {"value": b, "uncertainty": e[2]},
                       "power_at_set_polariser_angle_uW": {"value": float(f(0.0, a, ph0, b))}},
            "evidence": {"brewster_polariser_off_p": bool(abs(ph0) > 1.0 and sig(ph0, e[1]) > 5)},
            "plot": {"x": phi.tolist(), "y": P.tolist(), "x_label": "polariser angle from its setting in the run (°)",
                     "y_label": "reflected power at Brewster's angle (µW)", "unit": "µW", "x_unit": "°",
                     "connect": False, "lines": [{"role": "fit", "x": xs.tolist(), "y": f(xs, a, ph0, b).tolist()}],
                     "vlines": [{"x": 0.0, "label": "setting used"}, {"x": float(ph0), "label": "deepest"}]},
        }
    if "analyser_crossed" in header:
        # follow-up: a second polariser after the first, parallel then crossed
        (mp, ep, mc, ec), plot = groups_result(header, data, "analyser_crossed", "transmitted_uW", "µW",
                                               "power through both polarisers (µW)",
                                               "reading (analyser parallel, then crossed)")
        plot["lines"] = plot["lines"][:2]
        ratio = mc / mp
        ec = max(ec, pu / np.sqrt(np.sum(col(header, data, "analyser_crossed") == 1)))
        return {
            "values": {"transmitted_parallel_uW": {"value": mp, "uncertainty": ep},
                       "transmitted_crossed_uW": {"value": mc, "uncertainty": ec},
                       "extinction_ratio": {"value": ratio}},
            "evidence": {"brewster_extinction_poor": bool(ratio > 2e-3 and sig(mc, ec) > 5)},
            "plot": plot,
        }
    if "normal_incidence_deg" in header:
        # follow-up: the stage turned until the front-face reflection goes straight back into the laser
        r = col(header, data, "reading")
        z = col(header, data, "normal_incidence_deg")
        m, e = mean_err(z)
        e = max(e, pd / np.sqrt(z.size))
        return {
            "values": {"stage_reading_at_normal_incidence_deg": {"value": m, "uncertainty": e}},
            "evidence": {"brewster_stage_zero_off": bool(abs(m) > 0.5 and sig(m, e) > 5)},
            "plot": reading_plot(r, z, m, 0.0, "reading number, reflection sent back into the laser",
                                 "rotation-stage reading (°)", "°"),
        }
    th = col(header, data, "incidence_angle_deg")
    P = col(header, data, "reflected_power_uW")
    p_inc = col(header, data, "incident_power_uW") if "incident_power_uW" in header else np.full(th.size, p0)
    if "nd_filter" in header or "iris_closed" in header:
        # follow-ups: the same angles read twice, without and with an ND filter (or the iris closed)
        key = "nd_filter" if "nd_filter" in header else "iris_closed"
        g = col(header, data, key)
        ang = sorted(set(th[g == 0].tolist()) & set(th[g == 1].tolist()))
        if len(ang) < 2:
            raise RuntimeError("each angle needs a reading both ways")
        r, er, po = [], [], []
        for a in ang:
            (m0, e0), (m1, e1) = mean_err(P[(th == a) & (g == 0)]), mean_err(P[(th == a) & (g == 1)])
            e0, e1 = max(e0, float(np.hypot(pu, pp * m0))), max(e1, float(np.hypot(pu, pp * m1)))
            r.append(m1 / m0)
            er.append(m1 / m0 * float(np.hypot(e0 / m0, e1 / m1)))
            po.append(m0)
        r, er, po = np.array(r), np.array(er), np.array(po)
        plot = {"x": ang, "y": r.tolist(), "x_label": "angle of incidence (°)", "unit": "", "x_unit": "°",
                "connect": True}
        if key == "nd_filter":
            # a linear detector gives the same ratio at every power; a saturating one reads the bright
            # unfiltered beam low, so the ratio climbs with power. The faintest angle calibrates the filter.
            lo, hi = int(np.argmin(po)), int(np.argmax(po))
            lin = float(r[hi] / r[lo])
            elin = lin * float(np.hypot(er[hi] / r[hi], er[lo] / r[lo]))
            plot.update(y_label="reading with the ND filter ÷ reading without",
                        lines=[{"role": "expected", "x": [ang[0], ang[-1]], "y": [float(r[lo])] * 2}])
            return {
                "values": {"filter_ratio_faintest_angle": {"value": float(r[lo]), "uncertainty": float(er[lo])},
                           "filter_ratio_brightest_angle": {"value": float(r[hi]), "uncertainty": float(er[hi])},
                           "brightest_over_faintest_ratio": {"value": lin, "uncertainty": elin}},
                "evidence": {"brewster_detector_saturates": bool(lin - 1 > 0.05 and sig(lin - 1, elin) > 5)},
                "plot": plot,
            }
        w = 1 / er ** 2
        rm, erm = float(np.sum(w * r) / np.sum(w)), float(1 / np.sqrt(np.sum(w)))
        plot.update(y_label="reading with the iris closed ÷ reading with it open",
                    lines=[{"role": "fit", "x": [ang[0], ang[-1]], "y": [rm, rm]},
                           {"role": "expected", "x": [ang[0], ang[-1]], "y": [1.0, 1.0]}])
        return {
            "values": {"iris_closed_over_open": {"value": rm, "uncertainty": erm}},
            "evidence": {"brewster_ghost_found": bool(1 - rm > 0.2 and sig(1 - rm, erm) > 5)},
            "plot": plot,
        }
    o = np.argsort(th)
    th, P, p_inc = th[o], P[o], p_inc[o]
    rt = np.sqrt(np.clip(P / p_inc, 0, None))
    # near the minimum: p reflection (scaled by g) plus a fraction ε of s-polarised light in the beam,
    # the angle scale shifted by θ₀; this finds Brewster's angle and the floor under it
    k = int(np.argmin(P))
    win = np.abs(th - th[k]) <= 8.0
    if win.sum() < 5:
        win = np.zeros(th.size, bool)
        win[np.argsort(np.abs(th - th[k]))[:5]] = True
    th0_guess = float(th[k]) - thB
    s = _brewster_sigma(th, P, p_inc, n_exp, th0_guess, pu, pp, pd)
    near = lambda X, g, t0, eps: X[1] * (g * fresnel(X[0] - t0, n_exp)[0] + eps * fresnel(X[0] - t0, n_exp)[1])
    X = np.vstack([th[win], p_inc[win]])
    eps0 = max(float(P[k] / p_inc[k] / fresnel(thB, n_exp)[1]), 0.0)
    pfit = [1.0, th0_guess, eps0]
    for _ in range(2):
        pfit, cov = curve_fit(near, X, P[win], p0=pfit, sigma=s[win], absolute_sigma=True,
                              bounds=([0.1, -15, -0.02], [10, 15, 0.5]), maxfev=20000)
        s = _brewster_sigma(th, P, p_inc, n_exp, pfit[1], pu, pp, pd)
    g, th0, eps = (float(v) for v in pfit)
    _, red = misfit(P[win] - near(X, *pfit), s[win], 3)
    e = np.sqrt(np.clip(np.diag(cov), 0, None)) * max(1.0, float(np.sqrt(red)))
    eth0, eeps = float(e[1]), float(e[2])
    # every reading against Fresnel's law for the glass entered, on the angle scale found above
    rp, rs = fresnel(th - th0, n_exp)
    Pe = p_inc * (rp + eps * rs)

    def ratio(mask):
        if not mask.any():
            return 1.0, 1.0
        r, sr = P[mask] / Pe[mask], s[mask] / Pe[mask]
        w = 1 / sr ** 2
        return float(np.sum(w * r) / np.sum(w)), float(np.hypot(1 / np.sqrt(np.sum(w)), pp))

    lo, elo = ratio(th <= 40.0)
    hi, ehi = ratio(th >= 75.0)
    thm = thB + th0
    n_meas = float(np.tan(np.radians(thm)))
    xs = np.linspace(float(th.min()), float(th.max()), 240)
    fit_rt = np.sqrt(np.clip(near(np.vstack([xs, np.full(xs.size, float(np.mean(p_inc)))]), g, th0, eps)
                             / np.mean(p_inc), 0, None))
    return {
        "values": {
            "brewster_angle_fit_deg": {"value": thm, "uncertainty": eth0},
            "brewster_angle_expected_deg": {"value": thB},
            "refractive_index_from_brewster": {"value": n_meas,
                                               "uncertainty": float((1 + n_meas ** 2) * np.radians(eth0))},
            "s_polarised_fraction": {"value": eps, "uncertainty": eeps},
            "reflectance_low_angles_over_fresnel": {"value": lo, "uncertainty": elo},
            "reflectance_high_angles_over_fresnel": {"value": hi, "uncertainty": ehi},
        },
        "evidence": {
            "brewster_shift_deg": th0,
            "brewster_shift_sigma": float(th0 / eth0) if eth0 > 0 else 0.0,
            "s_fraction": eps,
            "s_fraction_sigma": float(eps / eeps) if eeps > 0 else 0.0,
            "low_ratio": lo,
            "low_ratio_sigma": float((lo - 1) / elo),
            "high_ratio": hi,
            "high_ratio_sigma": float((hi - 1) / ehi),
        },
        "plot": {"x": th.tolist(), "y": rt.tolist(), "x_label": "angle of incidence on the rotation stage (°)",
                 "y_label": "|r_p| = √(reflected ÷ incident power)", "unit": "", "x_unit": "°", "connect": False,
                 "lines": [{"role": "fit", "x": xs.tolist(), "y": fit_rt.tolist()},
                           {"role": "expected", "x": xs.tolist(), "y": np.sqrt(fresnel(xs, n_exp)[0]).tolist()}],
                 "vlines": [{"x": thB, "label": f"tan θ = {n_exp:g}"}]},
    }


# =====================================================================================
# --- Band gap from intrinsic resistance: R ∝ T^(−3/2) e^(Eg/2kT) ----------------------
def _gap_fit(T, R, prec, pT, deg=1):
    """ln(R·T^{3/2}) against 1000/T, weighted with the resistance and temperature precision."""
    if T.size < deg + 2:
        raise RuntimeError("too few temperatures for the fit")
    x, y = 1000 / T, np.log(R * T ** 1.5)
    xm = float(np.mean(x))
    b = float(np.polyfit(x, y, 1)[0])
    s = np.sqrt(prec ** 2 + (abs(b) * 1000 / T ** 2 * pT) ** 2)   # dy/dT = −b·1000/T²
    p, cov = wfit(x - xm, y, s, deg)
    return x, y, s, xm, p, cov


def band_gap(header, data, expected):
    eg_exp = float(expected.get("band_gap_eV", 0.67))
    prec = float(expected.get("precision_pct", 0.5)) / 100
    pT = float(expected.get("precision_K", 0.5))
    to_ev = 2 * K_B * 1000                              # slope per (1000/T) -> eV
    if "reference_thermometer_K" in header:
        # follow-up: oven thermometer and a calibrated thermometer side by side, oven held steady
        r = col(header, data, "reading")
        d = col(header, data, "oven_thermometer_K") - col(header, data, "reference_thermometer_K")
        m, e = mean_err(d)
        e = max(e, pT * np.sqrt(2 / d.size))
        return {
            "values": {"oven_minus_reference_K": {"value": m, "uncertainty": e}},
            "evidence": {"bandgap_thermometer_reads_off": bool(abs(m) > 2 and sig(m, e) > 5)},
            "plot": reading_plot(r, d, m, 0.0, "reading number, oven held steady",
                                 "oven thermometer − reference thermometer (K)", "K"),
        }
    if "four_terminal_ohm" in header:
        # follow-up: the same temperatures read across the end contacts and across the inner pair
        T = col(header, data, "temperature_K")
        r2 = col(header, data, "two_terminal_ohm")
        r4 = col(header, data, "four_terminal_ohm")
        d = r2 - r4
        m, e = mean_err(d)
        e = max(e, prec * float(np.mean(np.hypot(r2, r4))) / np.sqrt(d.size))
        frac = m / float(np.mean(r4))
        o = np.argsort(T)
        return {
            "values": {"contact_resistance_ohm": {"value": m, "uncertainty": e},
                       "contact_share_of_reading_pct": {"value": 100 * m / float(np.mean(r2))}},
            "evidence": {"bandgap_contact_resistance_found": bool(frac > 0.05 and sig(m, e) > 5)},
            "plot": {"x": T[o].tolist(), "y": d[o].tolist(), "x_label": "temperature (K)",
                     "y_label": "two-terminal − four-terminal resistance (Ω)", "unit": "Ω", "x_unit": "K",
                     "connect": True, "lines": [{"role": "fit", "x": flat(T, m)[0], "y": flat(T, m)[1]},
                                                {"role": "expected", "x": flat(T, 0)[0], "y": [0.0, 0.0]}]},
        }
    T = col(header, data, "temperature_K")
    R = col(header, data, "resistance_ohm")
    if np.any(T <= 0) or np.any(R <= 0):
        raise RuntimeError("temperatures (K) and resistances must be positive")
    if "current_mA" in header:
        # follow-up: the coldest temperatures read again at a lower sample current
        I = col(header, data, "current_mA")
        hi_I, lo_I = float(I.max()), float(I.min())
        temps = sorted(set(T[I == hi_I].tolist()) & set(T[I == lo_I].tolist()))
        if len(temps) < 2:
            raise RuntimeError("each temperature needs a reading at both currents")
        rr = np.array([float(np.mean(R[(T == t) & (I == lo_I)]) / np.mean(R[(T == t) & (I == hi_I)]))
                       for t in temps])
        m, e = mean_err(rr)
        e = max(e, prec * np.sqrt(2 / rr.size))
        return {
            "values": {"resistance_low_over_high_current": {"value": m, "uncertainty": e}},
            "evidence": {"bandgap_resistance_depends_on_current": bool(abs(m - 1) > 0.02 and sig(m - 1, e) > 5)},
            "plot": {"x": temps, "y": rr.tolist(), "x_label": "temperature (K)",
                     "y_label": f"R at {lo_I:g} mA ÷ R at {hi_I:g} mA", "unit": "", "x_unit": "K",
                     "connect": True, "lines": [{"role": "fit", "x": [temps[0], temps[-1]], "y": [m, m]},
                                                {"role": "expected", "x": [temps[0], temps[-1]], "y": [1.0, 1.0]}]},
        }
    if "cooling" in header:
        # follow-up: the heating sweep and a cooling sweep at a similar rate
        cl = col(header, data, "cooling")
        res = {}
        for k in (0, 1):
            x, y, s, xm, p, cov = _gap_fit(T[cl == k], R[cl == k], prec, pT)
            res[k] = (x, y, xm, p, cov)
        # temperature each sweep gives for the same resistance (the middle of the heating sweep)
        ystar = float(np.mean(res[0][1]))
        tt, et = [], []
        for k in (0, 1):
            _, _, xm, (b, a), cov = res[k]
            xk = xm + (ystar - a) / b
            gr = np.array([-(ystar - a) / b ** 2, -1 / b])        # d x / d(b, a)
            ex = float(np.sqrt(max(gr @ cov @ gr, 0)))
            tt.append(1000 / xk)
            et.append(1000 / xk ** 2 * ex)
        shift, es = float(tt[0] - tt[1]), float(np.hypot(*et))
        egs = [to_ev * float(res[k][3][0]) for k in (0, 1)]
        egs_e = [to_ev * float(np.sqrt(res[k][4][0, 0])) for k in (0, 1)]
        lines = []
        for k in (0, 1):
            x, _, xm, p, _ = res[k]
            xs = dense(x)
            lines.append({"role": "fit", "x": xs.tolist(), "y": np.polyval(p, xs - xm).tolist()})
        return {
            "values": {"band_gap_heating_eV": {"value": egs[0], "uncertainty": egs_e[0]},
                       "band_gap_cooling_eV": {"value": egs[1], "uncertainty": egs_e[1]},
                       "heating_minus_cooling_K": {"value": shift, "uncertainty": es}},
            "evidence": {"bandgap_sweeps_disagree": bool(abs(shift) > 3 and sig(shift, es) > 5)},
            "plot": {"x": (1000 / T).tolist(), "y": np.log(R * T ** 1.5).tolist(),
                     "x_label": "1000/T (K⁻¹), heating then cooling", "y_label": "ln(R·T^3/2), R in Ω and T in K",
                     "unit": "", "x_unit": "", "connect": False, "lines": lines},
        }
    x, y, s, xm, p, cov = _gap_fit(T, R, prec, pT)
    b, a = (float(v) for v in p)
    eg, eeg = to_ev * b, to_ev * float(np.sqrt(cov[0, 0]))
    mis, red = misfit(y - np.polyval(p, x - xm), s, 2)
    _, _, _, _, q, cq = _gap_fit(T, R, prec, pT, deg=2)
    c2, ec2 = float(q[0]), float(np.sqrt(cq[0, 0]))
    # local gap at the hot and cold ends, from the quadratic: Eg is twice k times the local slope
    slope_at = lambda xx: to_ev * (q[1] + 2 * q[0] * (xx - xm))
    xs = dense(x)
    return {
        "values": {
            "band_gap_fit_eV": {"value": eg, "uncertainty": eeg * max(1.0, float(np.sqrt(red)))},
            "band_gap_expected_eV": {"value": eg_exp},
            "band_gap_hot_end_eV": {"value": float(slope_at(x.min()))},
            "band_gap_cold_end_eV": {"value": float(slope_at(x.max()))},
            "scatter_over_precision": {"value": float(np.sqrt(red))},
        },
        "evidence": {
            "eg_error_pct": 100 * (eg / eg_exp - 1),
            # signed: > 0 the line flattens at the hot end (a resistance in series), < 0 at the cold end
            "curvature_sigma": float(c2 / ec2) if ec2 > 0 else 0.0,
            "misfit_sigma": mis,
        },
        "plot": {"x": x.tolist(), "y": y.tolist(), "x_label": "1000/T (K⁻¹)",
                 "y_label": "ln(R·T^3/2), R in Ω and T in K", "unit": "", "x_unit": "", "connect": False,
                 "lines": [{"role": "fit", "x": xs.tolist(), "y": np.polyval(p, xs - xm).tolist()},
                           {"role": "expected", "x": xs.tolist(),
                            "y": (a + eg_exp / to_ev * (xs - xm)).tolist()}]},
    }


# =====================================================================================
# --- Type K thermocouple calibration --------------------------------------------------
# NIST ITS-90 reference function for type K, 0 to 1372 °C (emf in mV, t in °C).
_K_C = [-0.176004136860e-01, 0.389212049750e-01, 0.185587700320e-04, -0.994575928740e-07,
        0.318409457190e-09, -0.560728448890e-12, 0.560750590590e-15, -0.320207200030e-18,
        0.971511471520e-22, -0.121047212750e-25]
_K_A = (0.118597600000e+00, -0.118343200000e-03, 0.126968600000e+03)


def emf_type_k(t):
    """Type K emf (mV) against a 0 °C reference. Below 0 °C the first two terms of the
    sub-zero function are used, good to about 1% down to −50 °C."""
    t = np.asarray(t, dtype=float)
    pos = np.polyval(_K_C[::-1], t) + _K_A[0] * np.exp(_K_A[1] * (t - _K_A[2]) ** 2)
    neg = 3.9450128025e-02 * t + 2.3622373598e-05 * t ** 2
    return np.where(t >= 0, pos, neg)


def seebeck_k(t):
    """dE/dT of type K in mV/K."""
    return (emf_type_k(np.asarray(t) + 0.05) - emf_type_k(np.asarray(t) - 0.05)) / 0.1


def thermocouple(header, data, expected):
    t_ref = float(expected.get("reference_junction_C", 0.0))
    pv = float(expected.get("precision_mV", 0.01))
    pt = float(expected.get("precision_C", 0.2))
    if "ice_bath_C" in header:
        # follow-up: a thermometer in the reference flask beside the junction
        r = col(header, data, "reading")
        tb = col(header, data, "ice_bath_C")
        m, e = mean_err(tb)
        e = max(e, pt / np.sqrt(tb.size))
        return {
            "values": {"reference_bath_temperature_C": {"value": m, "uncertainty": e},
                       "reference_temperature_entered_C": {"value": t_ref}},
            "evidence": {"thermocouple_reference_warm": bool(abs(m - t_ref) > 0.5 and sig(m - t_ref, e) > 5)},
            "plot": reading_plot(r, tb, m, t_ref, "reading number, thermometer in the reference flask",
                                 "reference bath temperature (°C)", "°C"),
        }
    if "shorted_emf_mV" in header:
        # follow-up: the voltmeter read with its input shorted by a copper link
        r = col(header, data, "reading")
        z = col(header, data, "shorted_emf_mV")
        m, e = mean_err(z)
        e = max(e, pv / np.sqrt(z.size))
        return {
            "values": {"meter_reading_input_shorted_mV": {"value": m, "uncertainty": e}},
            "evidence": {"thermocouple_meter_offset": bool(abs(m) > 0.02 and sig(m, e) > 5)},
            "plot": reading_plot(r, z, m, 0.0, "reading number, meter input shorted",
                                 "voltmeter reading (mV)", "mV"),
        }
    if "magnetic" in header:
        # follow-up: a magnet held against each leg; leg +1 is the one on the meter's + terminal
        leg = col(header, data, "leg")
        mag = col(header, data, "magnetic")
        if not (np.any(leg > 0) and np.any(leg < 0)):
            raise RuntimeError("test both legs: +1 for the leg on the meter's + terminal, −1 for the other")
        fp, fn = float(np.mean(mag[leg > 0])), float(np.mean(mag[leg < 0]))
        idx = np.arange(1, leg.size + 1)
        return {
            "values": {"positive_leg_magnetic_fraction": {"value": fp},
                       "negative_leg_magnetic_fraction": {"value": fn}},
            # type K wired the right way round: alumel (−) is magnetic, chromel (+) is not
            "evidence": {"thermocouple_legs_not_type_k": bool(not (fn > 0.5 and fp < 0.5))},
            "plot": {"x": idx.tolist(), "y": (mag * np.sign(leg)).tolist(),
                     "x_label": "test number (+1 leg on the meter's + terminal, −1 the other)",
                     "y_label": "magnet pulls (+1 or −1 for the leg) or not (0)", "unit": "", "x_unit": "",
                     "connect": False, "lines": []},
        }
    T = col(header, data, "hot_junction_C")
    E = col(header, data, "emf_mV")
    ref = emf_type_k(T) - emf_type_k(t_ref)
    s = np.sqrt(pv ** 2 + (seebeck_k(T) * pt) ** 2)
    if "deeper" in header:
        # follow-up: the top temperatures with the junction as before, then pushed deeper into the oil
        dp = col(header, data, "deeper")
        dr = E - ref
        (m0, e0), (m1, e1) = mean_err(dr[dp == 0]), mean_err(dr[dp == 1])
        e0 = max(e0, float(np.mean(s[dp == 0])) / np.sqrt(np.sum(dp == 0)))
        e1 = max(e1, float(np.mean(s[dp == 1])) / np.sqrt(np.sum(dp == 1)))
        d, ed = m1 - m0, float(np.hypot(e0, e1))
        frac = d / float(np.mean(ref))
        o = np.argsort(T)
        return {
            "values": {"emf_gain_when_deeper_mV": {"value": d, "uncertainty": ed},
                       "emf_gain_when_deeper_pct": {"value": 100 * frac}},
            "evidence": {"thermocouple_emf_rises_when_deeper": bool(frac > 0.01 and sig(d, ed) > 5)},
            "plot": {"x": T[o].tolist(), "y": dr[o].tolist(), "x_label": "bath temperature (°C), as before then deeper",
                     "y_label": "emf − type K table (mV)", "unit": "mV", "x_unit": "°C", "connect": False,
                     "lines": [{"role": "expected", "x": flat(T, 0)[0], "y": [0.0, 0.0]}]},
        }
    # emf = a + g·(table emf): g is the Seebeck coefficient relative to type K, a an offset at T = T_ref
    p, cov = np.polyfit(ref, E, 1, w=1 / s, cov="unscaled")
    g, a = (float(v) for v in p)
    eg, ea = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
    mis, red = misfit(E - np.polyval(p, ref), s, 2)
    birge = max(1.0, float(np.sqrt(red)))
    span = float(T.max() - T.min()) or 1.0
    s_k = 1000 * float(emf_type_k(T.max()) - emf_type_k(T.min())) / span
    # emf = S(T − T_ref) + q(T − T_ref)², in units of 100 K so the fit is well conditioned
    dT = (T - t_ref) / 100
    (q2, s1, c0), cq = np.polyfit(dT, E, 2, w=1 / s, cov="unscaled")
    tm = float((T.max() + T.min()) / 2 - t_ref) / 100
    s_mid = 10 * float(s1 + 2 * q2 * tm)                # µV/K at the middle of the range
    es_mid = 10 * float(np.sqrt(max(np.array([2 * tm, 1, 0]) @ cq @ np.array([2 * tm, 1, 0]), 0)))
    s_ref0 = 1000 * float(seebeck_k(t_ref))
    ts = dense(np.append(T, t_ref))
    return {
        "values": {
            "seebeck_fit_mid_range_uV_per_K": {"value": s_mid, "uncertainty": es_mid * birge},
            "seebeck_type_k_mid_range_uV_per_K": {"value": 1000 * float(seebeck_k((T.max() + T.min()) / 2))},
            "emf_over_type_k_table": {"value": g, "uncertainty": eg * birge},
            "emf_offset_at_reference_uV": {"value": 1000 * a, "uncertainty": 1000 * ea * birge},
            "offset_as_temperature_K": {"value": -a / (s_ref0 / 1000) if s_ref0 else 0.0},
            "mean_seebeck_type_k_uV_per_K": {"value": s_k},
        },
        "evidence": {
            "emf_gain": g,
            "gain_sigma": float((g - 1) / (eg * birge)),
            "offset_sigma": float(a / (ea * birge)),
            "misfit_sigma": mis,
        },
        "plot": {"x": T.tolist(), "y": E.tolist(), "x_label": "hot-junction temperature (°C)",
                 "y_label": "thermocouple emf (mV)", "unit": "mV", "x_unit": "°C", "connect": False,
                 "lines": [{"role": "fit", "x": ts.tolist(), "y": (a + g * (emf_type_k(ts) - emf_type_k(t_ref))).tolist()},
                           {"role": "expected", "x": ts.tolist(), "y": (emf_type_k(ts) - emf_type_k(t_ref)).tolist()}]},
    }


ANALYSES = {"thin_lens": thin_lens, "brewster": brewster, "band_gap": band_gap, "thermocouple": thermocouple}
COLUMNS = {
    "thin_lens": (["object_distance_cm", "image_distance_cm"], ["index_mark_cm", "lens_centre_cm"],
                  ["distant_object_m", "distant_image_cm"]),
    "brewster": (["incidence_angle_deg", "reflected_power_uW"], ["polariser_angle_deg", "reflected_power_uW"],
                 ["analyser_crossed", "transmitted_uW"], ["reading", "normal_incidence_deg"]),
    "band_gap": (["temperature_K", "resistance_ohm"], ["reading", "oven_thermometer_K", "reference_thermometer_K"],
                 ["temperature_K", "two_terminal_ohm", "four_terminal_ohm"]),
    "thermocouple": (["hot_junction_C", "emf_mV"], ["reading", "ice_bath_C"], ["reading", "shorted_emf_mV"],
                     ["leg", "magnetic"]),
}
