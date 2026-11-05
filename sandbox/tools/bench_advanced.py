"""Advanced-lab experiments. Runs INSIDE the sandbox; loaded by experiments.py.

  gamma_attenuation  thickness_mm, counts               (follow-up: count_time_s, counts_source_removed)
  muon_lifetime      time_us, counts                    (follow-up: covered, singles_rate_Hz)
  faraday_rotation   field_mT, rotation_deg             (follow-up: position_mm, field_mT, a field map)
  hysteresis         H_A_per_m, B_T, in the order taken (follow-up: coil, flux_mWb)

Counting data use Poisson errors (σ = √N). Where a run has few points or a meter
reading, thresholds use a stated reading precision (an `expected` parameter)
with absolute sigma, not the fit's own scatter.
"""
import numpy as np
from scipy.optimize import curve_fit

from experiments import col, dense, groups_result, mean_err, sig  # noqa: F401


def _wfit(f, x, y, p0, s, bounds=(-np.inf, np.inf)):
    """Least squares with absolute per-point errors s (a number or an array)."""
    s = np.broadcast_to(np.asarray(s, dtype=float), np.shape(y))
    p, cov = curve_fit(f, x, y, p0=p0, sigma=s, absolute_sigma=True, bounds=bounds, maxfev=20000)
    return p, np.sqrt(np.clip(np.diag(cov), 0, None)), cov


# --- Gamma-ray attenuation in lead: N = N₀ e^(−μx) (+ floor, + counting losses) -------
def gamma_f(x, a, mu, c):
    return a * np.exp(-mu * x) + c


def gamma_dead_f(x, a, mu, f, c):
    # f = N₀·τ_dead/t: the fraction of counts a non-paralysable counter loses is f e^(−μx)/(1 + f e^(−μx))
    e = np.exp(-mu * x)
    return a * e / (1 + f * e) + c


def gamma_attenuation(header, data, expected):
    mu_rho = float(expected.get("mass_attenuation_cm2_g", 0.110))
    rho = float(expected.get("density_g_cm3", 11.34))
    t = float(expected.get("count_time_s", 300.0))
    b = float(expected.get("background_cps", 0.5))
    mu_exp = mu_rho * rho / 10                         # per mm
    if "counts_source_removed" in header:
        # follow-up: repeat counts with the source back in its pot, everything else as in the run
        c = col(header, data, "counts_source_removed")
        ct = col(header, data, "count_time_s")
        n = c.size
        rate, erate = float(c.sum() / ct.sum()), float(np.sqrt(max(c.sum(), 1)) / ct.sum())
        excess = rate - b
        idx = np.arange(1, n + 1)
        return {
            "values": {"background_measured_cps": {"value": rate, "uncertainty": erate},
                       "background_subtracted_cps": {"value": b}},
            "evidence": {"background_excess_cps": excess,
                         "background_above_stated": bool(excess > 0.3 * b and sig(excess, erate) > 5)},
            "plot": {"x": idx.tolist(), "y": (c / ct).tolist(), "x_label": "count number, source removed",
                     "y_label": "background count rate (/s)", "unit": "/s", "x_unit": "", "connect": False,
                     "lines": [{"role": "fit", "x": [1, n], "y": [rate, rate]},
                               {"role": "expected", "x": [1, n], "y": [b, b]}]},
        }
    x = col(header, data, "thickness_mm")
    g = col(header, data, "counts")
    net = g - b * t
    s = np.sqrt(np.maximum(g, 1.0))                   # Poisson error on the gross count
    a0 = float(net[np.argmin(x)])
    (a, mu, c), (ea, emu, ec), _ = _wfit(gamma_f, x, net, [a0, mu_exp, 0.0], s)
    # the same with counting losses at high rate (dead time) allowed for
    (ad, mud, fd, cd), (ead, emud, efd, ecd), _ = _wfit(
        gamma_dead_f, x, net, [a, mu, 0.0, c], s, bounds=([0, 0, -0.9, -np.inf], [np.inf, np.inf, 20, np.inf]))
    dead = sig(fd, efd) > 4 and fd > 0
    if dead:
        a_use, mu_use, emu_use = ad, mud, emud
        model = lambda xx: gamma_dead_f(xx, ad, mud, fd, cd)
    else:
        a_use, mu_use, emu_use = a, mu, emu
        model = lambda xx: gamma_f(xx, a, mu, c)
    xs = dense(np.append(x, 0))
    lnr = np.log(np.maximum(net, 0.5) / a_use)
    values = {
        "linear_attenuation_per_cm": {"value": mu_use * 10, "uncertainty": emu_use * 10},
        "mass_attenuation_cm2_g": {"value": mu_use * 10 / rho, "uncertainty": emu_use * 10 / rho},
        "accepted_mass_attenuation_cm2_g": {"value": mu_rho},
        "half_value_layer_mm": {"value": np.log(2) / mu_use, "uncertainty": np.log(2) * emu_use / mu_use ** 2},
        "unattenuated_rate_cps": {"value": a_use / t, "uncertainty": (ead if dead else ea) / t},
        "count_floor_cps": {"value": cd / t, "uncertainty": ecd / t},
    }
    if dead:
        values["counting_loss_thinnest_pct"] = {"value": 100 * fd / (1 + fd), "uncertainty": 100 * efd / (1 + fd) ** 2}
    return {
        "values": values,
        "evidence": {
            "mu_error_pct": 100 * (mu_use / mu_exp - 1),
            "floor_sigma": float(cd / ecd) if ecd > 0 else 0.0,
            "floor_fraction": float(cd / max(ad, 1.0)),
            "deadtime_sigma": float(fd / efd) if efd > 0 else 0.0,
        },
        "plot": {"x": x.tolist(), "y": lnr.tolist(),
                 "x_label": "lead thickness (mm); vertical: ln(count rate ÷ unattenuated rate)",
                 "y_label": "ln(N/N₀), background subtracted", "unit": "", "x_unit": "mm",
                 "lines": [{"role": "fit", "x": xs.tolist(), "y": np.log(np.maximum(model(xs), 0.5) / a_use).tolist()},
                           {"role": "expected", "x": xs.tolist(), "y": (-mu_exp * xs).tolist()}]},
    }


# --- Muon lifetime: decay-time histogram N(t) = A e^(−t/τ) + B ------------------------
FREE_MUON_US = 2.1969811


def muon_f(t, a, tau, b):
    return a * np.exp(-t / tau) + b


def muon_lifetime(header, data, expected):
    tau_exp = float(expected.get("lifetime_us", 2.12))
    t_fit = float(expected.get("fit_start_us", 1.0))
    if "covered" in header:
        # follow-up: discriminator (singles) rate with the room lights on, then with the detector under a black cloth
        (mo, eo, mc, ec), plot = groups_result(header, data, "covered", "singles_rate_Hz", "/s",
                                               "singles rate (/s)", "reading (detector open, then under a black cloth)")
        plot["lines"] = plot["lines"][:2]
        ratio = mc / mo
        return {
            "values": {"singles_open_cps": {"value": mo, "uncertainty": eo},
                       "singles_covered_cps": {"value": mc, "uncertainty": ec},
                       "covered_over_open": {"value": ratio}},
            "evidence": {"singles_drop_when_covered": bool(ratio < 0.7 and sig(mo - mc, float(np.hypot(eo, ec))) > 5)},
            "plot": plot,
        }
    o = np.argsort(col(header, data, "time_us"))
    t = col(header, data, "time_us")[o]
    n = col(header, data, "counts")[o]
    s = np.sqrt(np.maximum(n, 1.0))
    width = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    m = t >= t_fit
    if m.sum() < 8:
        m = np.ones(t.size, bool)
    tail = n[t > np.percentile(t, 75)]
    b0 = float(np.median(tail)) if tail.size else 0.0
    a0 = float(max(n[m][0] - b0, 1.0) * np.exp(t[m][0] / tau_exp))
    (a, tau, b), (ea, etau, eb), cov = _wfit(muon_f, t[m], n[m], [a0, tau_exp, b0], s[m],
                                             bounds=([0, 0.05, -np.inf], [np.inf, 50, np.inf]))
    # counts before the fit window against the fit extrapolated back (PMT afterpulses pile up there)
    early = ~m
    excess_sigma = 0.0
    if early.any():
        pred = muon_f(t[early], a, tau, b)
        J = np.stack([np.exp(-t[early] / tau), a * t[early] / tau ** 2 * np.exp(-t[early] / tau),
                      np.ones(early.sum())], axis=1).sum(axis=0)
        var = float(pred.sum()) + float(J @ cov @ J)
        excess_sigma = float((n[early].sum() - pred.sum()) / np.sqrt(max(var, 1.0)))
    total = float(n.sum())
    xs = np.linspace(float(t.min()), float(t.max()), 200)
    return {
        "values": {
            "lifetime_us": {"value": tau, "uncertainty": etau},
            "expected_lifetime_us": {"value": tau_exp},
            "free_muon_lifetime_us": {"value": FREE_MUON_US},
            "decays_in_fit": {"value": float((n[m] - b).sum())},
            "accidental_background_per_bin": {"value": b, "uncertainty": eb},
            "accidental_fraction_pct": {"value": 100 * b * t.size / max(total, 1.0)},
        },
        "evidence": {
            "tau_error_pct": 100 * (tau / tau_exp - 1),
            "tau_error_sigma": sig(tau - tau_exp, etau),
            "background_fraction": float(b * t.size / max(total, 1.0)),
            "early_excess_sigma": excess_sigma,
        },
        "plot": {"x": t.tolist(), "y": n.tolist(),
                 "x_label": f"decay time (µs); vertical: decays per {width:g} µs bin",
                 "y_label": "decays per bin", "unit": "", "x_unit": "µs",
                 "lines": [{"role": "fit", "x": xs[xs >= t[m].min()].tolist(),
                            "y": muon_f(xs[xs >= t[m].min()], a, tau, b).tolist()},
                           {"role": "expected", "x": xs.tolist(),
                            "y": muon_f(xs, a * np.exp(t[m].min() / tau_exp - t[m].min() / tau), tau_exp, b).tolist()}],
                 "vlines": [{"x": float(t[m].min()), "label": "fit starts"}] if early.any() else []},
    }


# --- Faraday rotation: θ = V·B·L ---------------------------------------------------
def faraday_rotation(header, data, expected):
    v_exp = float(expected.get("verdet_rad_per_T_m", 23.0))
    L = float(expected.get("rod_length_mm", 100.0))
    prec = float(expected.get("precision_deg", 0.05))
    gprec = float(expected.get("gaussmeter_precision_mT", 0.2))
    if "position_mm" in header:
        # follow-up: the field mapped along the rod at one coil current, probe stepped through the bore
        z = col(header, data, "position_mm")
        bz = col(header, data, "field_mT")
        o = np.argsort(z)
        z, bz = z[o], bz[o]
        avg = float(np.trapezoid(bz, z) / (z.max() - z.min()))
        peak = float(bz.max())
        ratio = avg / peak
        eratio = gprec * np.sqrt(1 / z.size + 1) / peak
        return {
            "values": {"field_at_probe_position_mT": {"value": peak, "uncertainty": gprec},
                       "field_averaged_over_rod_mT": {"value": avg, "uncertainty": gprec / np.sqrt(z.size)},
                       "average_over_peak": {"value": ratio, "uncertainty": eratio}},
            "evidence": {"field_falls_off_along_rod": bool(1 - ratio > 0.03 and sig(1 - ratio, eratio) > 5)},
            "plot": {"x": z.tolist(), "y": bz.tolist(), "x_label": "position along the rod (mm)",
                     "y_label": "field on the axis (mT)", "unit": "mT", "x_unit": "mm", "connect": True,
                     "lines": [{"role": "fit", "x": [float(z.min()), float(z.max())], "y": [avg, avg]},
                               {"role": "expected", "x": [float(z.min()), float(z.max())], "y": [peak, peak]}],
                     "vlines": []},
        }
    B = col(header, data, "field_mT")
    th = col(header, data, "rotation_deg")
    s0 = np.polyfit(B, th, 1)[0]
    s_eff = float(np.hypot(prec, s0 * gprec))         # angle precision plus the field reading's, through the slope
    (k, c), (ek, ec), _ = _wfit(lambda b, k, c: k * b + c, B, th, [s0, 0.0], s_eff)
    to_v = np.pi / 180 * 1e3 / (L * 1e-3)              # °/mT -> rad/(T·m)
    V, eV = k * to_v, ek * to_v
    xs = dense(np.append(B, 0))
    return {
        "values": {
            "verdet_rad_per_T_m": {"value": V, "uncertainty": eV},
            "expected_verdet_rad_per_T_m": {"value": v_exp},
            "rotation_per_mT_deg": {"value": k, "uncertainty": ek},
            "rotation_at_zero_field_deg": {"value": c, "uncertainty": ec},
        },
        "evidence": {
            "verdet_error_pct": 100 * (V / v_exp - 1),
            "zero_offset_sigma": sig(c, ec),
        },
        "plot": {"x": B.tolist(), "y": th.tolist(), "x_label": "magnetic field at the coil centre (mT)",
                 "y_label": "polarisation rotation (degrees)", "unit": "°", "x_unit": "mT", "connect": False,
                 "lines": [{"role": "fit", "x": xs.tolist(), "y": (k * xs + c).tolist()},
                           {"role": "expected", "x": xs.tolist(), "y": (v_exp / to_v * xs).tolist()}]},
    }


# --- Magnetic hysteresis: B–H loop of a transformer core ---------------------------
def loop_branch(h, a, h0, w, chi, c):
    return a * np.tanh((h - h0) / w) + chi * h + c


def _local(x, y, at, k, sy, sx, deg):
    """Value of y at x = at from a local polynomial through the k points nearest it, with its error."""
    i = np.argsort(np.abs(x - at))[:k]
    xi, yi = x[i] - at, y[i]
    slope = np.polyfit(xi, yi, 1)[0]
    s = float(np.hypot(sy, slope * sx))
    p, cov = np.polyfit(xi, yi, deg, w=np.full(k, 1 / s), cov="unscaled")
    return float(p[-1]), float(np.sqrt(cov[-1, -1])), float(p[-2]), cov


def _zero_crossing(h, b, sb, sh):
    """H where B changes sign on one branch, from a straight line through the 4 points around it."""
    o = np.argsort(h)
    h, b = h[o], b[o]
    j = np.where(np.diff(np.sign(b)) != 0)[0]
    if not j.size:
        return float("nan"), float("nan")
    j = int(j[0])
    lo, hi = max(0, j - 1), min(h.size, j + 3)
    hi_, bi = h[lo:hi], b[lo:hi]
    slope = np.polyfit(hi_, bi, 1)[0]
    s = float(np.hypot(sb, slope * sh))
    (m, q), cov = np.polyfit(hi_, bi, 1, w=np.full(hi_.size, 1 / s), cov="unscaled")
    h0 = -q / m
    g = np.array([q / m ** 2, -1 / m])
    return float(h0), float(np.sqrt(max(g @ cov @ g, 0)))


def hysteresis(header, data, expected):
    hc_exp = float(expected.get("coercivity_A_per_m", 45.0))
    br_exp = float(expected.get("remanence_T", 0.83))
    bt_exp = float(expected.get("tip_field_T", 1.42))
    n2 = float(expected.get("search_turns", 100))
    sb = float(expected.get("precision_T", 0.005))
    sh = float(expected.get("precision_A_per_m", 2.0))
    if "coil" in header:
        # follow-up: fluxmeter change on reversing the magnetising current, search coil then a counted 10-turn coil
        n_ref = float(expected.get("reference_turns", 10))
        (ms, es, mr, er), plot = groups_result(header, data, "coil", "flux_mWb", "mWb",
                                               "flux-linkage change on reversal (mWb-turns)",
                                               "reversal (search coil, then the 10-turn reference coil)")
        plot["lines"] = plot["lines"][:2]
        turns = n_ref * ms / mr
        eturns = turns * float(np.hypot(es / ms, er / mr))
        frac = turns / n2
        return {
            "values": {"search_turns_measured": {"value": turns, "uncertainty": eturns},
                       "search_turns_stated": {"value": n2}},
            "evidence": {"search_turns_off": bool(abs(frac - 1) > 0.03 and sig(turns - n2, eturns) > 5)},
            "plot": plot,
        }
    H = col(header, data, "H_A_per_m")
    B = col(header, data, "B_T")
    # readings in the order taken, once round the loop, starting at either tip
    if H[0] >= 0:
        k = int(np.argmin(H))
        dh, db, ah, ab = H[:k + 1], B[:k + 1], H[k:], B[k:]     # +H_max down to −H_max, then back up
        top, bottom = (B[0] + B[-1]) / 2, B[k]                  # the start and end tip, and the far tip
    else:
        k = int(np.argmax(H))
        ah, ab, dh, db = H[:k + 1], B[:k + 1], H[k:], B[k:]
        top, bottom = B[k], (B[0] + B[-1]) / 2
    if min(dh.size, ah.size) < 6 or not (np.any(db > 0) and np.any(db < 0) and np.any(ab > 0) and np.any(ab < 0)):
        raise RuntimeError("the readings need to go once round the loop, with B crossing zero on both branches")
    hd, ehd = _zero_crossing(dh, db, sb, sh)
    ha, eha = _zero_crossing(ah, ab, sb, sh)
    bd, ebd = _local(dh, db, 0.0, 5, sb, sh, 2)[:2]
    ba, eba = _local(ah, ab, 0.0, 5, sb, sh, 2)[:2]
    hc, ehc = (ha - hd) / 2, float(np.hypot(eha, ehd) / 2)
    br, ebr = (bd - ba) / 2, float(np.hypot(ebd, eba) / 2)
    tip = (top - bottom) / 2
    etip = sb * np.sqrt(0.5 + 1) / 2
    offset = (top + bottom) / 2                        # loop's centre, vertically
    gap = float(B[-1] - B[0])
    values = {
        "coercivity_A_per_m": {"value": hc, "uncertainty": ehc},
        "expected_coercivity_A_per_m": {"value": hc_exp},
        "remanence_T": {"value": br, "uncertainty": ebr},
        "expected_remanence_T": {"value": br_exp},
        "tip_field_T": {"value": tip, "uncertainty": etip},
        "expected_tip_field_T": {"value": bt_exp},
        "loop_closure_gap_T": {"value": gap, "uncertainty": sb * np.sqrt(2)},
        "loop_vertical_offset_T": {"value": offset, "uncertainty": etip},
    }
    # loop area: energy lost per cycle (J/m³), shoelace formula over the closed loop
    values["loss_per_cycle_J_per_m3"] = {"value": float(abs(0.5 * np.sum(H * np.roll(B, -1) - np.roll(H, -1) * B)))}
    lines = []
    try:
        hmax = float(np.max(np.abs(H)))
        fits = []
        for hh, bb, sgn in ((dh, db, -1), (ah, ab, 1)):
            p, _, _ = _wfit(loop_branch, hh, bb, [tip, sgn * abs(hc), max(abs(hc), 1.0), 0.0, 0.0],
                            np.hypot(sb, 0.02 * sh), bounds=([0, -np.inf, 0.1, -np.inf, -np.inf], np.inf))
            fits.append(p)
            xs = np.linspace(float(hh.min()), float(hh.max()), 160)
            lines.append({"role": "fit", "x": xs.tolist(), "y": loop_branch(xs, *p).tolist()})
        # the loop the core should give: same shape, with the expected coercivity, remanence and tip
        chi = float(np.mean([p[3] for p in fits]))
        bs = bt_exp - chi * hmax
        w = hc_exp / np.arctanh(np.clip(br_exp / bs, 1e-6, 0.999999)) if bs > 0 else 0
        if w > 0:
            for sgn in (-1, 1):
                xs = np.linspace(-hmax, hmax, 160)
                lines.append({"role": "expected", "x": xs.tolist(),
                              "y": (bs * np.tanh((xs + sgn * hc_exp) / w) + chi * xs).tolist()})
    except (RuntimeError, ValueError):
        pass
    return {
        "values": values,
        "evidence": {
            "b_scale_error_pct": 100 * (tip / bt_exp - 1),
            "scale_mismatch_pct": 100 * ((br / br_exp) / (tip / bt_exp) - 1),
            "coercivity_error_pct": 100 * (hc / hc_exp - 1),
            "closure_gap_sigma": float(gap / (sb * np.sqrt(2))),
            "offset_sigma": float(offset / etip),
        },
        "plot": {"x": H.tolist(), "y": B.tolist(), "x_label": "magnetising field H (A/m), around the loop",
                 "y_label": "flux density B (T)", "unit": "T", "x_unit": "A/m", "connect": False,
                 "lines": lines},
    }


ANALYSES = {
    "gamma_attenuation": gamma_attenuation,
    "muon_lifetime": muon_lifetime,
    "faraday_rotation": faraday_rotation,
    "hysteresis": hysteresis,
}

COLUMNS = {
    "gamma_attenuation": (["thickness_mm", "counts"], ["count_time_s", "counts_source_removed"]),
    "muon_lifetime": (["time_us", "counts"], ["covered", "singles_rate_Hz"]),
    "faraday_rotation": (["field_mT", "rotation_deg"], ["position_mm", "field_mT"]),
    "hysteresis": (["H_A_per_m", "B_T"], ["coil", "flux_mWb"]),
}
