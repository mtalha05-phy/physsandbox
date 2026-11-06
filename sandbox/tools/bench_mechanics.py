"""Mechanics benches: Kater's reversible pendulum, the Cavendish torsion balance,
a driven damped oscillator's resonance curve, Boyle's law in a syringe. Runs
INSIDE the sandbox; loaded by experiments.py.

Same contract as experiments.py: (header, data, expected) -> values, evidence, plot.
Every fit uses the precision the researcher states (absolute sigma). Where the
points scatter more than that precision allows, the uncertainty of the result is
scaled up by sqrt(chi2/dof) (Birge ratio), so a noisy run is reported as noisy
rather than as a precise wrong answer.

  kater_pendulum     knife_edge_mass_position_cm, period_1_s, period_2_s
                     (follow-ups: reading, amplitude_deg | reading, separation_mm |
                      the sweep again with a light_gate column)
  cavendish          time_s, beam_position_mm
                     (follow-ups: reading, mirror_to_scale_m | reading, sphere_gap_mm |
                      time_s, spot_masses_removed_mm)
  driven_oscillator  drive_frequency_Hz, amplitude_mm
                     (follow-ups: dial_frequency_Hz, timed_cycles, stopwatch_s |
                      free_cycles, stopwatch_s | reading, brake_current_A |
                      drive_frequency_Hz, early_mm, settled_mm)
  boyle_gas          volume_ml, pressure_kPa
                     (follow-ups: tube_length_mm, tube_bore_mm | time_s, held_pressure_kPa |
                      volume_ml, pressure_at_once_kPa, pressure_after_30s_kPa |
                      reading, open_port_kPa)
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2 as _chi2, norm as _norm

from experiments import col, dense, mean_err, sig


def wfit(x, y, s, deg=1):
    """Polynomial fit with absolute sigma s: coefficients (highest power first) and covariance."""
    return np.polyfit(x, y, deg, w=1 / np.broadcast_to(s, np.shape(x)), cov="unscaled")


def misfit_sigma(resid, s, n_par):
    """Chi-square of the residuals in units of the stated precision, as a Gaussian significance."""
    dof = max(1, resid.size - n_par)
    c2 = float(np.sum((resid / s) ** 2))
    z = float(_norm.isf(_chi2.sf(c2, dof)))
    return float(np.clip(np.nan_to_num(z, posinf=99.0, neginf=0.0), 0.0, 99.0))


def birge(resid, s, n_par):
    """max(1, chi2/dof): how far the scatter exceeds the stated precision."""
    dof = max(1, resid.size - n_par)
    return max(1.0, float(np.sum((resid / s) ** 2)) / dof)


def wlstsq(A, y, s):
    """Weighted linear least squares y ≈ A·p with absolute sigma s: p and covariance."""
    w = 1 / np.broadcast_to(s, y.shape)
    Aw = A * w[:, None]
    cov = np.linalg.inv(Aw.T @ Aw)
    return cov @ Aw.T @ (y * w), cov


def flat(x, v):
    return [float(np.min(x)), float(np.max(x))], [float(v), float(v)]


def repeated(header, data, name, floor):
    """Mean of repeated readings, with an uncertainty no smaller than the instrument allows."""
    m, e = mean_err(col(header, data, name))
    return m, max(e, floor)


def reading_plot(v, scale, entered, x_label, y_label, unit):
    idx = np.arange(1, v.size + 1)
    m = float(np.mean(v))
    return {"x": idx.tolist(), "y": (scale * v).tolist(), "x_label": x_label, "y_label": y_label,
            "unit": unit, "x_unit": "", "connect": False,
            "lines": [{"role": "fit", "x": [1, float(idx[-1])], "y": [scale * m, scale * m]},
                      {"role": "expected", "x": [1, float(idx[-1])], "y": [scale * entered, scale * entered]}]}


# --- Kater's reversible pendulum: T₁ = T₂ = T at the crossing, g = 4π² L / T² ------------
def amplitude_factor(theta_deg):
    """Period at amplitude θ over the small-swing period: 1 + θ²/16 + 11θ⁴/3072."""
    th = np.radians(theta_deg)
    return 1 + th ** 2 / 16 + 11 * th ** 4 / 3072


def _worst_left_out(u, y, prec, deg=2):
    """The point furthest from a fit through the others, in units of its combined
    uncertainty: the stated precision and the left-out fit's prediction uncertainty there."""
    worst, k, res = 0.0, 0, 0.0
    if u.size < deg + 3:
        return worst, k, res
    for i in range(u.size):
        m = np.arange(u.size) != i
        p, cov = wfit(u[m], y[m], prec, deg)
        g = u[i] ** np.arange(deg, -1, -1)
        sd = float(np.sqrt(max(g @ cov @ g, 0.0)))
        r = float(y[i] - np.polyval(p, u[i]))
        z = abs(r) / float(np.hypot(prec, sd))
        if z > worst:
            worst, k, res = z, i, r
    return worst, k, res


def _kater_fit(x, t1, t2, prec):
    """Quadratic through each edge's periods (ms, against decimetres from the scan's centre),
    a single miscounted point set aside, and the crossing T₁ = T₂ with its uncertainty."""
    xm = float(np.mean(x))
    u = (x - xm) / 10
    tref = round(float(np.mean(np.r_[t1, t2])), 2)
    sides, worst, rests, fulls, odd, jump = [], 0.0, [], [], None, 0.0
    for edge, t in ((1, t1), (2, t2)):
        y = 1000 * (t - tref)
        z, k, res = _worst_left_out(u, y, prec)
        pf, _ = wfit(u, y, prec, 2)
        fulls.append(misfit_sigma(y - np.polyval(pf, u), prec, 3))
        keep = np.ones(u.size, bool)
        keep[k] = False
        pk, _ = wfit(u[keep], y[keep], prec, 2)
        rest = misfit_sigma(y[keep] - np.polyval(pk, u[keep]), prec, 3)
        if not (z > 5 and rest < 4):
            keep[:] = True                 # no single odd point: keep them all
        elif z > worst:
            odd = (edge, k)
        if z > worst:
            jump = res / 1000 / (tref + (y[k] - res) / 1000)   # fractional jump of the worst point
        worst = max(worst, z)
        rests.append(rest)
        p, cov = wfit(u[keep], y[keep], prec, 2)
        r = y[keep] - np.polyval(p, u[keep])
        sides.append({"p": p, "cov": cov * birge(r, prec, 3), "y": y, "keep": keep,
                      "rms": float(np.sqrt(np.mean(r ** 2)))})
    d = sides[0]["p"] - sides[1]["p"]
    roots = np.roots(d if abs(d[0]) > 1e-12 else d[1:])
    roots = roots[np.isreal(roots)].real
    span = float(np.max(np.abs(u))) + 0.5
    roots = roots[np.abs(roots) <= span]
    if roots.size == 0:
        raise RuntimeError("the period curves about the two knife edges don't cross within the slider "
                           "positions measured; move the slider until the two periods pass each other")
    uc = float(roots[np.argmin(np.abs(roots))])
    gv = np.array([uc ** 2, uc, 1.0])
    s1, s2 = (float(np.polyval(np.polyder(s["p"]), uc)) for s in sides)
    v1, v2 = (float(gv @ s["cov"] @ gv) for s in sides)
    tc = float(np.polyval(sides[0]["p"], uc))
    # T = q₁(u*) where q₁(u*) = q₂(u*): moving either curve moves the crossing along the other
    etc = float(np.sqrt((s2 / (s1 - s2)) ** 2 * v1 + (s1 / (s1 - s2)) ** 2 * v2))
    euc = float(np.sqrt(v1 + v2) / abs(s1 - s2))
    return {"xm": xm, "tref": tref, "sides": sides, "uc": uc, "euc": euc,
            "T": tref + tc / 1000, "eT": etc / 1000, "worst": worst, "scatter": max(rests),
            "scatter_min": min(fulls), "odd": odd, "u": u, "jump": jump}


def kater_pendulum(header, data, expected):
    L = float(expected.get("separation_mm", 995.0))
    g_loc = float(expected.get("g_local_m_s2", 9.812))
    amp = float(expected.get("amplitude_deg", 2.0))
    if "amplitude_deg" in header:
        # follow-up: the swing amplitude read on a protractor scale behind the bar's lower end
        a = col(header, data, "amplitude_deg")
        m, e = repeated(header, data, "amplitude_deg", 0.5)      # a protractor read by eye, about 0.5°
        corr = 100 * (amplitude_factor(m) / amplitude_factor(amp) - 1)
        return {
            "values": {"amplitude_measured_deg": {"value": m, "uncertainty": e},
                       "amplitude_entered_deg": {"value": amp},
                       "period_excess_at_measured_amplitude_pct": {"value": float(corr)},
                       "g_shortfall_at_measured_amplitude_pct": {"value": float(100 * (1 - (1 + corr / 100) ** -2))}},
            "evidence": {"amplitude_above_stated": bool(m - amp > 2 and sig(m - amp, e) > 5)},
            "plot": reading_plot(a, 1, amp, "release, as in the run", "swing amplitude (degrees)", "°"),
        }
    if "separation_mm" in header:
        # follow-up: the knife-edge separation measured again, edge to edge
        s = col(header, data, "separation_mm")
        m, e = repeated(header, data, "separation_mm", 0.2)       # steel rule and lens, about 0.2 mm
        return {
            "values": {"separation_measured_mm": {"value": m, "uncertainty": e},
                       "separation_entered_mm": {"value": L},
                       "g_with_measured_separation_change_pct": {"value": float(100 * (m / L - 1))}},
            "evidence": {"separation_differs": bool(abs(m - L) > 1 and sig(m - L, e) > 5)},
            "plot": reading_plot(s, 1, L, "measurement, knife edge to knife edge",
                                 "knife-edge separation (mm)", "mm"),
        }
    prec = float(expected.get("precision_ms", 1.0))
    eL = float(expected.get("separation_precision_mm", 0.3))
    n_sw = float(expected.get("swings_timed", 50))
    x = col(header, data, "knife_edge_mass_position_cm")
    t1 = col(header, data, "period_1_s")
    t2 = col(header, data, "period_2_s")
    k = _kater_fit(x, t1, t2, prec)
    # the stated release amplitude lengthens every period alike; take it out
    corr = float(amplitude_factor(amp))
    T0, eT0 = k["T"] / corr, k["eT"] / corr
    g = 4 * np.pi ** 2 * (L / 1000) / T0 ** 2
    eg = g * float(np.hypot(2 * eT0 / T0, eL / L))
    values = {
        "g_kater_m_s2": {"value": g, "uncertainty": eg},
        "g_local_m_s2": {"value": g_loc},
        "crossing_period_s": {"value": k["T"], "uncertainty": k["eT"]},
        "crossing_position_cm": {"value": k["xm"] + 10 * k["uc"], "uncertainty": 10 * k["euc"]},
        "amplitude_correction_pct": {"value": 100 * (corr - 1)},
        "scatter_edge_1_ms": {"value": k["sides"][0]["rms"]},
        "scatter_edge_2_ms": {"value": k["sides"][1]["rms"]},
    }
    if k["odd"]:
        edge, i = k["odd"]
        s = k["sides"][edge - 1]
        curve = k["tref"] + float(np.polyval(s["p"], k["u"][i])) / 1000
        values["odd_point_position_cm"] = {"value": float(x[i])}
        values["odd_point_period_ratio"] = {"value": float((k["tref"] + s["y"][i] / 1000) / curve)}
    xs = dense(x)
    us = (xs - k["xm"]) / 10
    t_exp = 2 * np.pi * np.sqrt(L / 1000 / g_loc) * corr
    plot = {"x": np.r_[x, x].tolist(), "y": np.r_[t1, t2].tolist(),
            "x_label": "slider position on the bar's scale (cm)",
            "y_label": "period about knife edge 1 and about knife edge 2 (s)", "unit": "s", "x_unit": "cm",
            "connect": False,
            "lines": [{"role": "fit", "x": xs.tolist(), "y": (k["tref"] + np.polyval(s["p"], us) / 1000).tolist()}
                      for s in k["sides"]]
            + [{"role": "expected", "x": flat(xs, t_exp)[0], "y": flat(xs, t_exp)[1]}],
            "vlines": [{"x": k["xm"] + 10 * k["uc"], "label": "periods equal"}]}
    if "light_gate" in header:
        # follow-up: the whole sweep timed again with a light gate counting the swings
        return {"values": values,
                "evidence": {"light_gate_curves_clean": bool(k["worst"] < 4 and k["scatter"] < 4)},
                "plot": {**plot, "x_label": "slider position on the bar's scale (cm), retimed with a light gate"}}
    return {
        "values": values,
        "evidence": {
            "g_error_pct": 100 * (g / g_loc - 1),
            # signed: a large swing only ever lengthens the period, so g comes out low
            "g_shift_sigma": float((g - g_loc) / eg),
            "outlier_sigma": k["worst"],
            # the worst point's jump in swings: a miscount of one swing in n moves a period by 1/n
            "outlier_swings": float(abs(k["jump"]) * n_sw),
            # the worse edge's scatter once its single worst point is set aside (a lone
            # miscount is not scatter), and the better edge's with every point (both edges
            # scattered means the timing is poorer than the precision entered)
            "scatter_sigma": k["scatter"],
            "scatter_min_sigma": k["scatter_min"],
        },
        "plot": plot,
    }


# --- Cavendish torsion balance: G = 2π² L r² θ / (M T²), θ = ΔS / 4D ---------------------
G_NEWTON = 6.67430e-11


def cav_step(t, s_old, s_new, tau, period, v):
    """Beam at rest at s_old until the large spheres swing at t = 0, then a damped approach
    to s_new; plus a steady drift v. t, tau, period in units of 100 s; v in mm per 100 s."""
    w = 2 * np.pi / period
    tt = np.clip(t, 0, None)
    ring = (s_old - s_new) * np.exp(-tt / tau) * (np.cos(w * tt) + np.sin(w * tt) / (w * tau))
    return np.where(t >= 0, s_new + ring, s_old) + v * t


def natural_period(period, tau):
    """Undamped period from the damped one: ω₀² = ω_d² + 1/τ² (amplitude decays as e^(−t/τ))."""
    return 2 * np.pi / np.sqrt((2 * np.pi / period) ** 2 + 1 / tau ** 2)


def _far_sphere(r, d):
    """β: each large sphere also pulls, the other way, on the far small sphere."""
    return r ** 3 / (r ** 2 + 4 * d ** 2) ** 1.5


def cavendish_G(dS_mm, T0, M, d, r, D):
    """G from the swing of the spot between the two positions. Small spheres as points
    on a massless beam of length L = 2d; the mirror doubles the twist and the spheres go
    from −θ to +θ, so the spot moves ΔS = 4θD. Corrected for the far sphere by 1/(1 − β)."""
    theta = abs(dS_mm) / 1000 / (4 * D)
    return 2 * np.pi ** 2 * (2 * d) * r ** 2 * theta / (M * T0 ** 2) / (1 - _far_sphere(r, d))


def cavendish(header, data, expected):
    D = float(expected.get("scale_distance_m", 4.20))
    if "mirror_to_scale_m" in header:
        # follow-up: the mirror-to-scale distance measured with a steel tape along the beam
        Dm = col(header, data, "mirror_to_scale_m")
        m, e = repeated(header, data, "mirror_to_scale_m", 0.003)   # a tape over 4 m, about 3 mm
        return {
            "values": {"scale_distance_measured_m": {"value": m, "uncertainty": e},
                       "scale_distance_entered_m": {"value": D},
                       "G_change_with_measured_distance_pct": {"value": float(100 * (D / m - 1))}},
            "evidence": {"scale_distance_differs": bool(abs(m / D - 1) > 0.01 and sig(m - D, e) > 5)},
            "plot": reading_plot(Dm, 100, D, "tape reading, mirror to scale", "mirror-to-scale distance (cm)", "cm"),
        }
    if "sphere_gap_mm" in header:
        # follow-up: gap between each large sphere and the case wall, both positions
        gp = col(header, data, "sphere_gap_mm")
        m, e = repeated(header, data, "sphere_gap_mm", 0.05)         # feeler gauges, about 0.05 mm
        return {
            "values": {"sphere_gap_mean_mm": {"value": m, "uncertainty": e},
                       "sphere_gap_largest_mm": {"value": float(gp.max())}},
            "evidence": {"spheres_off_position": bool(m > 0.5 and sig(m, e) > 5)},
            "plot": reading_plot(gp, 1, 0.0, "sphere and position (two spheres, both positions)",
                                 "gap between large sphere and case (mm)", "mm"),
        }
    prec = float(expected.get("precision_mm", 1.0))
    if "spot_masses_removed_mm" in header:
        # follow-up: the settled spot logged with the large spheres lifted away
        t = col(header, data, "time_s") / 100
        s = col(header, data, "spot_masses_removed_mm")
        (v, c), cv = wfit(t, s, prec)
        r = s - (v * t + c)
        ev = float(np.sqrt(cv[0, 0] * birge(r, prec, 2)))
        rate = 0.6 * v                                           # mm per minute
        return {
            "values": {"drift_without_spheres_mm_per_min": {"value": rate, "uncertainty": 0.6 * ev},
                       "scatter_without_spheres_mm": {"value": float(np.sqrt(np.mean(r ** 2)))}},
            "evidence": {"drift_without_masses": bool(abs(rate) * float(np.ptp(t)) * 100 / 60 > 3 and sig(v, ev) > 5)},
            "plot": {"x": (100 * t / 60).tolist(), "y": s.tolist(), "x_label": "time, large spheres removed (min)",
                     "y_label": "spot position on the scale (mm)", "unit": "mm", "x_unit": "min", "connect": True,
                     "lines": [{"role": "fit", "x": flat(100 * t / 60, 0)[0], "y": [float(v * t.min() + c), float(v * t.max() + c)]},
                               {"role": "expected", "x": flat(100 * t / 60, 0)[0], "y": [float(s[0]), float(s[0])]}]},
        }
    M = float(expected.get("large_mass_kg", 1.500))
    d = float(expected.get("arm_mm", 50.0)) / 1000
    r = float(expected.get("centre_distance_mm", 46.5)) / 1000
    eg = float(expected.get("geometry_precision_mm", 0.2)) / 1000
    t = col(header, data, "time_s") / 100                        # hundreds of seconds
    S = col(header, data, "beam_position_mm")
    post = t >= 0
    s_old0 = float(S[np.argmin(np.abs(t))])
    s_new0 = float(np.mean(S[t > 0.5 * t.max()]))
    # first guess at the period from where the motion crosses its final level
    z = np.where(np.diff(np.sign(S[post] - s_new0)) != 0)[0]
    per0 = float(2 * np.mean(np.diff(t[post][z]))) if z.size >= 3 else float(t.max() / 3)
    p0 = [s_old0, s_new0, 2 * per0, per0, 0.0]
    p, cov = curve_fit(cav_step, t, S, p0=p0, sigma=np.full(S.size, prec), absolute_sigma=True,
                       bounds=([-np.inf, -np.inf, 0.1 * per0, 0.3 * per0, -np.inf],
                               [np.inf, np.inf, 100 * per0, 3 * per0, np.inf]), maxfev=40000)
    s_old, s_new, tau, per, v = (float(a) for a in p)
    resid = S - cav_step(t, *p)
    mis = misfit_sigma(resid, prec, 5)
    cov = cov * birge(resid, prec, 5)
    e = np.sqrt(np.clip(np.diag(cov), 0, None))
    dS = s_new - s_old
    edS = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0)))
    T0 = 100 * float(natural_period(per, tau))
    h = 1e-6
    # cov[2:4, 2:4] is (tau, period)
    gT = np.array([natural_period(per, tau + h) - natural_period(per, tau - h),
                   natural_period(per + h, tau) - natural_period(per - h, tau)]) * 100 / (2 * h)
    eT0 = float(np.sqrt(max(gT @ cov[2:4, 2:4] @ gT, 0)))
    G = cavendish_G(dS, T0, M, d, r, D)
    # the geometry's share of the uncertainty: r and d (stated precision) and D (a tape, about 5 mm)
    rel_r = (cavendish_G(dS, T0, M, d, r + 1e-6, D) / G - 1) / 1e-6 * eg
    rel_d = (cavendish_G(dS, T0, M, d + 1e-6, r, D) / G - 1) / 1e-6 * eg
    eG = G * float(np.sqrt((edS / dS) ** 2 + (2 * eT0 / T0) ** 2 + (0.005 / D) ** 2 + rel_r ** 2 + rel_d ** 2))
    beta = _far_sphere(r, d)
    # the swing G = 6.674e-11 would give, for the plot
    dS_exp = np.sign(dS) * G_NEWTON / G * abs(dS)
    ts = np.linspace(float(t.min()), float(t.max()), 400)
    return {
        "values": {
            "G_fit": {"value": G, "uncertainty": eG},
            "G_accepted": {"value": G_NEWTON},
            "equilibrium_shift_mm": {"value": dS, "uncertainty": edS},
            "free_period_s": {"value": T0, "uncertainty": eT0},
            "damping_time_s": {"value": 100 * tau, "uncertainty": 100 * float(e[2])},
            "drift_mm_per_min": {"value": 0.6 * v, "uncertainty": 0.6 * float(e[4])},
            "beam_twist_mrad": {"value": abs(dS) / (4 * D)},
            "far_sphere_correction_pct": {"value": 100 * beta / (1 - beta)},
        },
        "evidence": {
            "G_error_pct": 100 * (G / G_NEWTON - 1),
            "G_shift_sigma": float((G - G_NEWTON) / eG),
            "drift_sigma": float(v / e[4]) if e[4] > 0 else 0.0,
            "misfit_sigma": mis,
        },
        "plot": {"x": (100 * t / 60).tolist(), "y": S.tolist(), "x_label": "time after swinging the large spheres (min)",
                 "y_label": "laser spot on the scale (mm)", "unit": "mm", "x_unit": "min", "connect": True,
                 "lines": [{"role": "fit", "x": (100 * ts / 60).tolist(), "y": cav_step(ts, *p).tolist()},
                           {"role": "expected", "x": (100 * ts / 60).tolist(),
                            "y": cav_step(ts, s_old, s_old + dS_exp, tau, per, v).tolist()}]},
    }


# --- Driven damped oscillator: A = a ω₀² / √((ω₀² − ω²)² + (γω)²) --------------------------
def osc_curve(f, a, f0, gam):
    """Steady amplitude against drive frequency (Hz); a is the low-frequency amplitude and
    gam = γ/2π in Hz, so Q = f0/gam."""
    return a * f0 ** 2 / np.sqrt((f0 ** 2 - f ** 2) ** 2 + (gam * f) ** 2)


def _osc_fit(f, A, prec, p0):
    p, cov = curve_fit(osc_curve, f, A, p0=p0, sigma=np.full(A.size, prec), absolute_sigma=True,
                       bounds=([0, 0.3 * p0[1], 1e-4], [np.inf, 3 * p0[1], 3 * p0[1]]), maxfev=20000)
    return p, cov


def osc_clipped(f, a, f0, gam, c):
    """The same curve with the motion stopped at c: no reading can exceed the stop."""
    return np.minimum(osc_curve(f, a, f0, gam), c)


def _clip_fit(f, A, prec, p):
    """Fit the curve with a ceiling; returns chi-square, parameters and covariance (or None)."""
    best = None
    for c0 in (float(np.mean(np.sort(A)[-3:])), float(A.max())):
        try:
            q, cq = curve_fit(osc_clipped, f, A, p0=[*p, c0], sigma=np.full(A.size, prec), absolute_sigma=True,
                              bounds=([0, 0.3 * p[1], 1e-4, 0], [np.inf, 3 * p[1], 3 * p[1], np.inf]), maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        c2 = float(np.sum(((A - osc_clipped(f, *q)) / prec) ** 2))
        if best is None or c2 < best[0]:
            best = (c2, q, cq)
    return best


def driven_oscillator(header, data, expected):
    f0_exp = float(expected.get("natural_frequency_Hz", 0.800))
    g_exp = float(expected.get("damping_per_s", 0.60))
    if "dial_frequency_Hz" in header:
        # follow-up: crank turns timed with a stopwatch at a few dial settings
        fd = col(header, data, "dial_frequency_Hz")
        n = col(header, data, "timed_cycles")
        ts = col(header, data, "stopwatch_s")
        fm = n / ts
        ratio = fd / fm
        er = ratio * 0.1 / ts                           # stopwatch start and stop, about 0.1 s
        w = 1 / er ** 2
        m = float(np.sum(w * ratio) / np.sum(w))
        e = max(float(1 / np.sqrt(np.sum(w))), float(np.std(ratio, ddof=1) / np.sqrt(ratio.size)) if ratio.size > 1 else 0.0)
        return {
            "values": {"dial_over_measured_frequency": {"value": m, "uncertainty": e},
                       "drive_frequency_measured_Hz": {"value": float(fm[np.argmax(fd)])},
                       "drive_frequency_dial_Hz": {"value": float(fd.max())}},
            "evidence": {"drive_dial_off": bool(abs(m - 1) > 0.01 and sig(m - 1, e) > 5)},
            "plot": {"x": fd.tolist(), "y": fm.tolist(), "x_label": "frequency on the motor dial (Hz)",
                     "y_label": "drive frequency timed with a stopwatch (Hz)", "unit": "Hz", "x_unit": "Hz",
                     "connect": False,
                     "lines": [{"role": "fit", "x": flat(fd, 0)[0], "y": [float(fd.min() / m), float(fd.max() / m)]},
                               {"role": "expected", "x": flat(fd, 0)[0], "y": flat(fd, 0)[0]}]},
        }
    if "free_cycles" in header:
        # follow-up: free oscillations (motor stopped, brake as set) timed with a stopwatch
        n = col(header, data, "free_cycles")
        ts = col(header, data, "stopwatch_s")
        fd = n / ts
        m, e = mean_err(fd)
        e = max(e, float(np.mean(fd * 0.1 / ts)) / np.sqrt(fd.size))
        # damped free frequency → natural frequency: ω₀² = ω_d² + γ²/4
        f0 = float(np.sqrt(m ** 2 + (g_exp / (4 * np.pi)) ** 2))
        return {
            "values": {"natural_frequency_measured_Hz": {"value": f0, "uncertainty": e},
                       "natural_frequency_entered_Hz": {"value": f0_exp}},
            "evidence": {"natural_frequency_differs": bool(abs(f0 / f0_exp - 1) > 0.01 and sig(f0 - f0_exp, e) > 5)},
            "plot": reading_plot(fd, 1, f0_exp, "timing of free oscillations", "free oscillation frequency (Hz)", "Hz"),
        }
    if "brake_current_A" in header:
        # follow-up: the electromagnet's current read on a separate ammeter in series
        i_set = float(expected.get("brake_current_A", 0.40))
        I = col(header, data, "brake_current_A")
        m, e = repeated(header, data, "brake_current_A", 0.002)
        return {
            "values": {"brake_current_measured_A": {"value": m, "uncertainty": e},
                       "brake_current_entered_A": {"value": i_set}},
            "evidence": {"brake_current_differs": bool(abs(m / i_set - 1) > 0.03 and sig(m - i_set, e) > 5)},
            "plot": reading_plot(I, 1, i_set, "reading on the separate ammeter", "brake magnet current (A)", "A"),
        }
    prec = float(expected.get("precision_mm", 0.4))
    f = col(header, data, "drive_frequency_Hz")
    if "settled_mm" in header:
        # follow-up: the points nearest the peak read straight away and again after five time constants
        early = col(header, data, "early_mm")
        late = col(header, data, "settled_mm")
        dm = float(np.mean(late - early))
        e = prec * np.sqrt(2 / early.size)
        return {
            "values": {"settled_minus_early_mm": {"value": dm, "uncertainty": float(e)}},
            "evidence": {"settled_readings_higher": bool(dm > 1 and dm / e > 5)},
            "plot": {"x": f.tolist(), "y": (late - early).tolist(), "x_label": "drive frequency (Hz)",
                     "y_label": "reading after five time constants − reading at once (mm)", "unit": "mm",
                     "x_unit": "Hz", "connect": False,
                     "lines": [{"role": "fit", "x": flat(f, dm)[0], "y": flat(f, dm)[1]},
                               {"role": "expected", "x": flat(f, 0)[0], "y": flat(f, 0)[1]}]},
        }
    A = col(header, data, "amplitude_mm")
    k = int(np.argmax(A))
    gam0 = g_exp / (2 * np.pi)
    p0 = np.array([float(A[k]) * gam0 / float(f[k]), float(f[k]), gam0])
    p, cov = _osc_fit(f, A, prec, p0)
    # An end stop: the same curve with a ceiling, at least three readings held at it.
    # Its significance is the chi-square the ceiling removes, √Δχ² for one more parameter.
    clip, model, stop = 0.0, osc_curve, None
    c2_free = float(np.sum(((A - osc_curve(f, *p)) / prec) ** 2))
    best = _clip_fit(f, A, prec, p)
    if best is not None:
        c2_clip, q, cq = best
        if int(np.sum(osc_curve(f, *q[:3]) > q[3])) >= 3:
            clip = float(np.sqrt(max(c2_free - c2_clip, 0.0)))
            if clip > 5:
                p, cov, stop = q[:3], cq[:3, :3], float(q[3])
                model = lambda f, *a: osc_clipped(f, *a, stop)
    resid = A - model(f, *p)
    mis = misfit_sigma(resid, prec, 3 if stop is None else 4)
    cov = cov * birge(resid, prec, 3 if stop is None else 4)
    a, f0, gam = (float(x) for x in p)
    ea, ef0, egam = (float(x) for x in np.sqrt(np.clip(np.diag(cov), 0, None)))
    gamma, egamma = 2 * np.pi * gam, 2 * np.pi * egam
    q, eq = f0 / gam, f0 / gam * float(np.hypot(ef0 / f0, egam / gam))
    fs = np.linspace(float(f.min()), float(f.max()), 300)
    values = {
        "natural_frequency_fit_Hz": {"value": f0, "uncertainty": ef0},
        "natural_frequency_entered_Hz": {"value": f0_exp},
        "damping_fit_per_s": {"value": gamma, "uncertainty": egamma},
        "damping_entered_per_s": {"value": g_exp},
        "oscillator_quality_factor": {"value": q, "uncertainty": eq},
        "low_frequency_amplitude_mm": {"value": a, "uncertainty": ea},
        "peak_amplitude_fit_mm": {"value": float(osc_curve(np.sqrt(max(f0 ** 2 - gam ** 2 / 2, 1e-9)), a, f0, gam))},
    }
    if stop is not None:
        values["end_stop_level_mm"] = {"value": stop, "uncertainty": float(np.sqrt(max(best[2][3, 3], 0)))}
    return {
        "values": values,
        "evidence": {
            "f0_error_pct": 100 * (f0 / f0_exp - 1),
            # the free-oscillation f0 is good to about 0.3% (20 swings on a stopwatch),
            # the decay rate entered to about 5% (a log decrement over a few swings)
            "f0_shift_sigma": float((f0 - f0_exp) / np.hypot(ef0, 0.003 * f0_exp)),
            "gamma_shift_sigma": float((gamma - g_exp) / np.hypot(egamma, 0.05 * g_exp)),
            "misfit_sigma": mis,
            "clip_sigma": clip,
        },
        "plot": {"x": f.tolist(), "y": A.tolist(), "x_label": "drive frequency (Hz)",
                 "y_label": "steady amplitude (mm)", "unit": "mm", "x_unit": "Hz", "connect": False,
                 "vlines": [{"x": f0_exp, "label": f"f₀ entered {f0_exp:g} Hz"}],
                 "lines": [{"role": "fit", "x": fs.tolist(), "y": model(fs, a, f0, gam).tolist()},
                           {"role": "expected", "x": fs.tolist(),
                            "y": osc_curve(fs, a, f0_exp, g_exp / (2 * np.pi)).tolist()}]},
    }


# --- Boyle's law with a dead volume: p (V + V₀) = nRT ------------------------------------------
def boyle_gas(header, data, expected):
    v0e = float(expected.get("dead_volume_ml", 0.5))
    patm = float(expected.get("atmospheric_kPa", 101.3))
    sp = float(expected.get("precision_kPa", 0.2))
    if "tube_bore_mm" in header:
        # follow-up: the connecting tube's length and bore measured with a rule and callipers
        ln, eln = repeated(header, data, "tube_length_mm", 1.0)
        bo, ebo = repeated(header, data, "tube_bore_mm", 0.05)
        vt = np.pi * bo ** 2 / 4 * ln / 1000                    # ml
        evt = vt * float(np.hypot(eln / ln, 2 * ebo / bo))
        return {
            "values": {"tubing_volume_ml": {"value": float(vt), "uncertainty": float(evt)},
                       "dead_volume_entered_ml": {"value": v0e}},
            "evidence": {"tubing_not_counted": bool(vt - v0e > 0.5 and sig(vt - v0e, evt) > 5)},
            "plot": reading_plot(np.pi * col(header, data, "tube_bore_mm") ** 2 / 4 * col(header, data, "tube_length_mm") / 1000,
                                 1, v0e, "measurement of the connecting tube (length and bore)",
                                 "tube volume from each measurement (ml)", "ml"),
        }
    if "held_pressure_kPa" in header:
        # follow-up: plunger held still at the smallest volume, pressure logged from 30 s after the last push
        t = col(header, data, "time_s")
        p = col(header, data, "held_pressure_kPa")
        tm = t / 60
        (v, c), cv = wfit(tm, p, sp)
        r = p - (v * tm + c)
        ev = float(np.sqrt(cv[0, 0] * birge(r, sp, 2)))
        fall = -v * float(np.ptp(tm))
        return {
            "values": {"held_pressure_change_kPa_per_min": {"value": float(v), "uncertainty": ev},
                       "held_pressure_start_kPa": {"value": float(p[0])}},
            "evidence": {"pressure_falls_when_held": bool(fall > 1 and -v / ev > 5)},
            "plot": {"x": t.tolist(), "y": p.tolist(), "x_label": "time with the plunger held still (s)",
                     "y_label": "pressure (kPa)", "unit": "kPa", "x_unit": "s", "connect": True,
                     "lines": [{"role": "fit", "x": flat(t, 0)[0], "y": [float(v * tm.min() + c), float(v * tm.max() + c)]},
                               {"role": "expected", "x": flat(t, 0)[0], "y": [float(p[0]), float(p[0])]}]},
        }
    if "pressure_after_30s_kPa" in header:
        # follow-up: each volume read at once and again after 30 s at rest
        V = col(header, data, "volume_ml")
        p1 = col(header, data, "pressure_at_once_kPa")
        p2 = col(header, data, "pressure_after_30s_kPa")
        dm = float(np.mean(p1 - p2))
        e = sp * np.sqrt(2 / p1.size)
        return {
            "values": {"pressure_drop_after_30s_kPa": {"value": dm, "uncertainty": float(e)}},
            "evidence": {"pressure_relaxes": bool(dm > 0.5 and dm / e > 5)},
            "plot": {"x": V.tolist(), "y": (p1 - p2).tolist(), "x_label": "syringe volume (ml)",
                     "y_label": "pressure read at once − pressure after 30 s (kPa)", "unit": "kPa",
                     "x_unit": "ml", "connect": False,
                     "lines": [{"role": "fit", "x": flat(V, dm)[0], "y": flat(V, dm)[1]},
                               {"role": "expected", "x": flat(V, 0)[0], "y": flat(V, 0)[1]}]},
        }
    if "open_port_kPa" in header:
        # follow-up: the sensor read with its port open to the room, beside the barometer
        po = col(header, data, "open_port_kPa")
        m, e = repeated(header, data, "open_port_kPa", 0.05)
        e = float(np.hypot(e, 0.1))                             # a lab barometer, about 0.1 kPa
        return {
            "values": {"open_port_reading_kPa": {"value": m, "uncertainty": e},
                       "atmospheric_entered_kPa": {"value": patm}},
            "evidence": {"sensor_offset_found": bool(abs(m - patm) > 1 and sig(m - patm, e) > 5)},
            "plot": reading_plot(po, 1, patm, "reading, port open to the room", "sensor reading (kPa)", "kPa"),
        }
    sv = float(expected.get("precision_ml", 0.3))
    V = col(header, data, "volume_ml")
    p = col(header, data, "pressure_kPa")
    # V = C/p − V₀ is linear in C and V₀; x = 100/p keeps both of order ten to a hundred.
    # A pressure error moves a point sideways by C·σp/p², so each point's sigma combines the two.
    x = 100 / p
    C = float(p[0] * (V[0] + v0e) / 100)
    A = np.vstack([x, -np.ones_like(x)]).T
    for _ in range(4):
        s = np.hypot(sv, 100 * C * sp / p ** 2)
        (C, V0), cov = wlstsq(A, V, s)
    resid = V - (C * x - V0)
    mis = misfit_sigma(resid, s, 2)
    cov = cov * birge(resid, s, 2)
    eC, eV0 = (float(np.sqrt(c)) for c in np.diag(cov))
    # The syringe is connected with its port open, so the first reading is at room pressure.
    off = float(p[0] - patm)
    e_off = float(np.hypot(sp, 0.1))
    pv = p * (V + v0e)
    vs = dense(V)
    fit_pv = 100 * C * (vs + v0e) / (vs + V0)
    start = 100 * C * (V[0] + v0e) / (V[0] + V0)
    return {
        "values": {
            "dead_volume_fit_ml": {"value": float(V0), "uncertainty": eV0},
            "dead_volume_entered_ml": {"value": v0e},
            "nRT_mJ": {"value": 100 * float(C), "uncertainty": 100 * eC},
            "pV_change_over_run_pct": {"value": float(100 * (pv[-1] / pv[0] - 1))},
            "first_reading_kPa": {"value": float(p[0])},
            "atmospheric_entered_kPa": {"value": patm},
        },
        "evidence": {
            "dead_volume_excess_ml": float(V0 - v0e),
            # signed: tubing left out or gas leaking out read as extra volume; gas warmed
            # by fast compression reads as too little
            "dead_volume_excess_sigma": float((V0 - v0e) / eV0),
            "start_offset_sigma": float(off / e_off),
            "misfit_sigma": mis,
        },
        "plot": {"x": V.tolist(), "y": pv.tolist(), "x_label": "volume read on the syringe (ml)",
                 "y_label": "p × (V + dead volume entered), proportional to nRT (kPa·ml)", "unit": "kPa·ml",
                 "x_unit": "ml", "connect": False,
                 "lines": [{"role": "fit", "x": vs.tolist(), "y": fit_pv.tolist()},
                           {"role": "expected", "x": flat(vs, start)[0], "y": flat(vs, start)[1]}]},
    }


ANALYSES = {
    "kater_pendulum": kater_pendulum,
    "cavendish": cavendish,
    "driven_oscillator": driven_oscillator,
    "boyle_gas": boyle_gas,
}

COLUMNS = {
    "kater_pendulum": (["knife_edge_mass_position_cm", "period_1_s", "period_2_s"],
                       ["reading", "amplitude_deg"], ["reading", "separation_mm"]),
    "cavendish": (["time_s", "beam_position_mm"], ["reading", "mirror_to_scale_m"],
                  ["reading", "sphere_gap_mm"], ["time_s", "spot_masses_removed_mm"]),
    "driven_oscillator": (["drive_frequency_Hz", "amplitude_mm"], ["dial_frequency_Hz", "timed_cycles", "stopwatch_s"],
                          ["free_cycles", "stopwatch_s"], ["reading", "brake_current_A"],
                          ["drive_frequency_Hz", "early_mm", "settled_mm"]),
    "boyle_gas": (["volume_ml", "pressure_kPa"], ["tube_length_mm", "tube_bore_mm"], ["time_s", "held_pressure_kPa"],
                  ["volume_ml", "pressure_at_once_kPa", "pressure_after_30s_kPa"], ["reading", "open_port_kPa"]),
}
