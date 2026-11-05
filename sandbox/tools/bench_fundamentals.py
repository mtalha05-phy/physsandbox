"""First- and second-year lab fundamentals: Young's double slit, series RLC
resonance, viscosity by Stokes' law (falling ball), thermal conductivity by
Lee's disc. Runs INSIDE the sandbox; loaded by experiments.py.

Same contract as experiments.py: (header, data, expected) -> values, evidence, plot.
Few points, so every fit uses the precision the researcher states (absolute
sigma), not the scatter of the points themselves.

  young_double_slit  order_m, position_mm                 (follow-up: reading, slit_to_screen_m)
  rlc_resonance      frequency_Hz, resistor_V             (follow-up: frequency_Hz, input_V)
  stokes_viscosity   ball_diameter_mm, fall_time_s        (follow-up: depth_cm, liquid_temperature_C)
  lees_disc          stage, time_s, steam_C, disc_C       (follow-up: reading, steam_C, disc_C)
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


def wlstsq(A, y, s):
    """Weighted linear least squares y ≈ A·p with absolute sigma s: p and covariance."""
    w = 1 / np.broadcast_to(s, y.shape)
    Aw = A * w[:, None]
    cov = np.linalg.inv(Aw.T @ Aw)
    return cov @ Aw.T @ (y * w), cov


def flat(x, v):
    return [float(np.min(x)), float(np.max(x))], [float(v), float(v)]


# --- Young's double slit: y_m = m λ L / d -------------------------------------------
def young_double_slit(header, data, expected):
    L = float(expected.get("screen_distance_m", 2.000))
    if "slit_to_screen_m" in header:
        # follow-up: slit-to-screen distance measured again with a steel tape, from the slide itself
        Lm = col(header, data, "slit_to_screen_m")
        m, e = mean_err(Lm)
        e = max(e, 0.001)                                   # a steel tape reads to about 1 mm
        idx = np.arange(1, Lm.size + 1)
        return {
            "values": {"slit_to_screen_measured_m": {"value": m, "uncertainty": e},
                       "slit_to_screen_entered_m": {"value": L}},
            "evidence": {"screen_distance_differs": bool(abs(m / L - 1) > 0.02 and sig(m - L, e) > 5)},
            "plot": {"x": idx.tolist(), "y": (100 * Lm).tolist(), "x_label": "tape reading, slide to screen",
                     "y_label": "slit-to-screen distance (cm)", "unit": "cm", "x_unit": "", "connect": False,
                     "lines": [{"role": "fit", "x": [1, float(idx[-1])], "y": [100 * m, 100 * m]},
                               {"role": "expected", "x": [1, float(idx[-1])], "y": [100 * L, 100 * L]}]},
        }
    lam_exp = float(expected.get("wavelength_nm", 632.8))
    d = float(expected.get("slit_separation_mm", 0.250))
    prec = float(expected.get("precision_mm", 0.3))
    m = col(header, data, "order_m")
    y = col(header, data, "position_mm")
    # Positions are ruler readings with an arbitrary zero, so the centre is a free parameter.
    # The angles are below 0.03 rad: tan and sin differ from θ by less than 0.03%.
    (s, c), cov = wfit(m, y, prec)
    es = float(np.sqrt(cov[0, 0]))
    s_exp = lam_exp * L / d * 1e-3                          # fringe spacing, mm
    lam, elam = s * d / L * 1e3, es * d / L * 1e3
    resid = y - (s * m + c)
    mis = misfit_sigma(resid, prec, 2)
    # Fringes numbered wrongly past some order (a fringe hidden in the single-slit envelope's
    # dark band and not counted) put every fringe beyond it one spacing too far out, on both
    # sides: a step of size j at |m| >= k, outward. Try each break k; keep the most significant.
    am = np.abs(m)
    step_z, step_j, step_ej, step_k = 0.0, 0.0, 0.0, 0.0
    for k in sorted(set(am[am >= 2].tolist())):
        out = am >= k
        if (~out).sum() < 3:
            continue
        A = np.vstack([m, np.ones_like(m), np.sign(m) * out]).T
        p, cv = wlstsq(A, y, prec)
        ej = float(np.sqrt(cv[2, 2]))
        z = float(p[2] / ej)
        if abs(z) > abs(step_z):
            step_z, step_j, step_ej, step_k = z, float(p[2]), ej, float(k)
    ms = np.array([float(m.min()), float(m.max())])
    return {
        "values": {
            "fringe_wavelength_fit_nm": {"value": lam, "uncertainty": elam},
            "fringe_wavelength_expected_nm": {"value": lam_exp},
            "fringe_spacing_mm": {"value": s, "uncertainty": es},
            "fringe_spacing_expected_mm": {"value": s_exp},
            "slit_separation_implied_mm": {"value": lam_exp * L / s * 1e-3, "uncertainty": lam_exp * L / s ** 2 * es * 1e-3},
            "outer_fringe_step_mm": {"value": step_j, "uncertainty": step_ej},
        },
        "evidence": {
            "lambda_error_pct": 100 * (lam / lam_exp - 1),
            "misfit_sigma": mis,
            "outer_step_sigma": abs(step_z),
            "outer_step_fringes": step_j / s if s else 0.0,
            "outer_step_from_order": step_k,
        },
        "plot": {"x": m.tolist(), "y": (y - c).tolist(), "x_label": "fringe order m (as numbered)",
                 "y_label": "fringe position from the centre (mm)", "unit": "mm", "x_unit": "", "connect": False,
                 "lines": [{"role": "fit", "x": ms.tolist(), "y": (s * ms).tolist()},
                           {"role": "expected", "x": ms.tolist(), "y": (s_exp * ms).tolist()}]},
    }


# --- Series RLC resonance: V_R = V R / |R_tot + j(ωL − 1/ωC)| ---------------------------
def resonance_curve(f_khz, a, f0, q):
    return a / np.sqrt(1 + q ** 2 * (f_khz / f0 - f0 / f_khz) ** 2)


def rlc_resonance(header, data, expected):
    Ld = float(expected.get("inductance_H", 0.010))
    Cd = float(expected.get("capacitance_F", 1.0e-7))
    R = float(expected.get("resistor_ohm", 100.0))
    RL = float(expected.get("coil_resistance_ohm", 18.0))
    Vd = float(expected.get("drive_V", 2.0))
    rel = float(expected.get("precision_pct", 1.0)) / 100
    f = col(header, data, "frequency_Hz") / 1e3            # kHz, so the fit is well conditioned
    if "input_V" in header:
        # follow-up: the circuit's input (generator terminals) watched on CH1 through the sweep
        vin = col(header, data, "input_V")
        lo, hi = float(vin.min()), float(vin.max())
        e = rel * hi
        o = np.argsort(f)
        return {
            "values": {"input_lowest_V": {"value": lo, "uncertainty": e},
                       "input_highest_V": {"value": hi, "uncertainty": e},
                       "input_set_V": {"value": Vd}},
            "evidence": {"input_sags": bool(lo / hi < 0.9 and (hi - lo) / (e * np.sqrt(2)) > 5)},
            "plot": {"x": f[o].tolist(), "y": vin[o].tolist(), "x_label": "frequency (kHz)",
                     "y_label": "voltage at the circuit's input (V)", "unit": "V", "x_unit": "kHz", "connect": True,
                     "lines": [{"role": "expected", "x": [float(f.min()), float(f.max())], "y": [Vd, Vd]}]},
        }
    v = col(header, data, "resistor_V")
    rt = R + RL
    f0_exp = 1 / (2 * np.pi * np.sqrt(Ld * Cd)) / 1e3
    q_exp = np.sqrt(Ld / Cd) / rt
    a_exp = Vd * R / rt
    sv = rel * v
    k = int(np.argmax(v))
    # start from the highest reading, the frequency it was taken at, and the components' Q
    p0 = [float(v[k]), float(f[k]), float(q_exp)]
    try:
        p, cv = curve_fit(resonance_curve, f, v, p0=p0, sigma=sv, absolute_sigma=True,
                          bounds=([0, f.min() / 3, 0.05], [np.inf, f.max() * 3, 500]), maxfev=20000)
    except RuntimeError:
        p, cv = np.array(p0), np.diag(np.array(p0) ** 2)
    a, f0, q = (float(x) for x in p)
    ea, ef0, eq = (float(x) for x in np.sqrt(np.clip(np.diag(cv), 0, None)))
    resid = v - resonance_curve(f, *p)
    mis = misfit_sigma(resid, sv, 3)
    fs = np.geomspace(float(f.min()), float(f.max()), 200)
    return {
        "values": {
            "resonant_frequency_fit_Hz": {"value": 1e3 * f0, "uncertainty": 1e3 * ef0},
            "resonant_frequency_expected_Hz": {"value": 1e3 * f0_exp},
            "quality_factor_fit": {"value": q, "uncertainty": eq},
            "quality_factor_expected": {"value": float(q_exp)},
            "peak_resistor_voltage_V": {"value": a, "uncertainty": ea},
            "peak_resistor_voltage_expected_V": {"value": float(a_exp)},
            # what the curve says the series resistance is, against R + coil
            "series_resistance_from_peak_ohm": {"value": Vd * R / a, "uncertainty": Vd * R / a ** 2 * ea},
            "series_resistance_entered_ohm": {"value": rt},
        },
        "evidence": {
            "f0_error_pct": 100 * (f0 / f0_exp - 1),
            "q_error_pct": 100 * (q / q_exp - 1),
            "peak_error_pct": 100 * (a / a_exp - 1),
            "misfit_sigma": mis,
        },
        "plot": {"x": f.tolist(), "y": v.tolist(), "x_label": "frequency (kHz)",
                 "y_label": "voltage across the resistor (V)", "unit": "V", "x_unit": "kHz", "connect": False,
                 "vlines": [{"x": float(f0_exp), "label": f"f₀ expected {1e3 * f0_exp:.0f} Hz"}],
                 "lines": [{"role": "fit", "x": fs.tolist(), "y": resonance_curve(fs, *p).tolist()},
                           {"role": "expected", "x": fs.tolist(),
                            "y": resonance_curve(fs, a_exp, f0_exp, q_exp).tolist()}]},
    }


# --- Viscosity by Stokes' law: v = 2 r² g (ρ_s − ρ_l) / 9η ------------------------------
G = 9.81
RHO_GLYCEROL = 1261.0          # kg/m³ at 20 °C, falling 0.65 kg/m³ per °C


def glycerol_viscosity(t_c):
    """Pure glycerol, Pa·s: Vogel–Fulcher fit to Segur & Oberstar (1951), 10–30 °C to ~1%."""
    return float(np.exp(-8.9478 + 941.4 / (t_c + 273.15 - 191.83)))


def faxen(lam):
    """Wall factor for a sphere falling on the axis of a tube, v/v∞ (Faxén), λ = d_ball/D_tube."""
    return 1 - 2.104 * lam + 2.09 * lam ** 3 - 0.95 * lam ** 5


def stokes_viscosity(header, data, expected):
    t_c = float(expected.get("liquid_temperature_C", 20.0))
    if "liquid_temperature_C" in header:
        # follow-up: a thermometer lowered into the glycerol at several depths
        z = col(header, data, "depth_cm")
        tt = col(header, data, "liquid_temperature_C")
        m, e = mean_err(tt)
        e = max(e, 0.1)                                    # a lab thermometer is good to about 0.1 °C
        return {
            "values": {"liquid_temperature_measured_C": {"value": m, "uncertainty": e},
                       "liquid_temperature_entered_C": {"value": t_c},
                       "viscosity_at_measured_temperature_Pa_s": {"value": glycerol_viscosity(m)}},
            "evidence": {"liquid_temperature_differs": bool(abs(m - t_c) > 0.8 and sig(m - t_c, e) > 5)},
            "plot": {"x": z.tolist(), "y": tt.tolist(), "x_label": "thermometer depth below the surface (cm)",
                     "y_label": "glycerol temperature (°C)", "unit": "°C", "x_unit": "cm", "connect": True,
                     "lines": [{"role": "fit", "x": flat(z, m)[0], "y": flat(z, m)[1]},
                               {"role": "expected", "x": flat(z, t_c)[0], "y": flat(z, t_c)[1]}]},
        }
    d = col(header, data, "ball_diameter_mm") / 1e3        # m
    t = col(header, data, "fall_time_s")
    h = float(expected.get("fall_distance_mm", 300.0)) / 1e3
    D = float(expected.get("tube_diameter_mm", 50.0)) / 1e3
    rho_s = float(expected.get("ball_density_kg_m3", 7800.0))
    st = float(expected.get("timing_precision_s", 0.2))
    sd = float(expected.get("diameter_precision_mm", 0.003))
    rho_l = RHO_GLYCEROL - 0.65 * (t_c - 20)
    eta_exp = glycerol_viscosity(t_c)
    drho = rho_s - rho_l
    v_obs = h / t
    v_free = v_obs / faxen(d / D)                          # take the tube wall out
    r2 = (d / 2) ** 2
    # the uncertainty of each point: timing, plus the diameter reading carried through v ∝ d²
    sv_rel = np.hypot(st / t, 2 * sd / 1e3 / d)

    def through_origin(vs):
        s = sv_rel * vs
        w = 1 / s ** 2
        K = float(np.sum(w * r2 * vs) / np.sum(w * r2 ** 2))
        return K, float(1 / np.sqrt(np.sum(w * r2 ** 2))), s

    # inertia (Oseen): drag × (1 + 3Re/16); Re needs η, so iterate (converges in a few steps)
    vs = v_free
    for _ in range(4):
        K, eK, s = through_origin(vs)
        eta = 2 * G * drho / (9 * K)
        re = rho_l * v_free * d / eta
        vs = v_free * (1 + 3 * re / 16)
    K, eK, s = through_origin(vs)
    eta, eeta = 2 * G * drho / (9 * K), 2 * G * drho / (9 * K ** 2) * eK
    mis = misfit_sigma(vs - K * r2, s, 1)
    # Does the line go through the origin? v ∝ r² means √v ∝ d, so fit √v against d (mm) and
    # find the diameter where it meets zero: a micrometer zero error moves every d by the
    # same amount, which this finds directly. Diameter precision enters as an x-error.
    dm, q = d * 1e3, np.sqrt(vs * 1e3)
    a1 = float(np.sqrt(K * 1e-3) / 2)                      # √(mm/s) per mm, from the fit through the origin
    for _ in range(2):
        sq = np.hypot(0.5 * st / t * q, a1 * sd)
        (a1, a0), cv = wfit(dm, q, sq)
    d0 = -a0 / a1
    g = np.array([a0 / a1 ** 2, -1 / a1])
    ed0 = float(np.sqrt(max(g @ cv @ g, 0)))
    # The chart shows the viscosity each ball gives on its own: a scale error (temperature,
    # water) moves every ball alike, a diameter offset or bubbles push the small balls up.
    # In % of the table value, so the axis reads as a deviation whatever the scale.
    dev = lambda e: 100 * (e / eta_exp - 1)
    eta_i = dev(2 * G * drho * r2 / (9 * vs))
    return {
        "values": {
            "viscosity_fit_Pa_s": {"value": eta, "uncertainty": eeta},
            "viscosity_expected_Pa_s": {"value": eta_exp},
            "diameter_offset_um": {"value": 1e3 * float(d0), "uncertainty": 1e3 * ed0},
            "wall_correction_largest_ball_pct": {"value": float(100 * (1 / faxen(d.max() / D) - 1))},
            "reynolds_number_largest_ball": {"value": float(np.max(re))},
        },
        "evidence": {
            "eta_error_pct": 100 * (eta / eta_exp - 1),
            # signed: above zero = the small balls fall too slowly for the diameters recorded
            "diameter_offset_sigma": float(d0 / ed0) if ed0 > 0 else 0.0,
            "misfit_sigma": mis,
        },
        "plot": {"x": dm.tolist(), "y": eta_i.tolist(), "x_label": "ball diameter (mm)",
                 "y_label": "viscosity from each ball (wall and inertia corrected), against the table value (%)",
                 "unit": "%", "x_unit": "mm", "connect": False,
                 "lines": [{"role": "fit", "x": flat(dm, dev(eta))[0], "y": flat(dm, dev(eta))[1]},
                           {"role": "expected", "x": flat(dm, 0)[0], "y": flat(dm, 0)[1]}]},
    }


# --- Thermal conductivity, Lee's disc: k = m c (dT/dt)|T₂ · x / (A (T₁ − T₂)) --------------
def lees_disc(header, data, expected):
    prec = float(expected.get("precision_K", 0.1))
    if "stage" not in header:
        # follow-up: both thermometers side by side in the same place, read together
        r = col(header, data, "reading")
        diff = col(header, data, "disc_C") - col(header, data, "steam_C")
        m, e = mean_err(diff)
        e = max(e, prec / 2)
        return {
            "values": {"thermometer_difference_K": {"value": m, "uncertainty": e},
                       "common_reading_C": {"value": float(np.mean(col(header, data, "steam_C")))}},
            "evidence": {"thermometers_disagree": bool(abs(m) > 0.5 and sig(m, e) > 5)},
            "plot": {"x": r.tolist(), "y": diff.tolist(), "x_label": "reading, both thermometers in the same place",
                     "y_label": "disc thermometer − steam-side thermometer (K)", "unit": "K", "x_unit": "", "connect": True,
                     "lines": [{"role": "fit", "x": flat(r, m)[0], "y": flat(r, m)[1]},
                               {"role": "expected", "x": flat(r, 0)[0], "y": flat(r, 0)[1]}]},
        }
    k_exp = float(expected.get("conductivity_W_mK", 0.19))
    x = float(expected.get("sample_thickness_mm", 3.0)) / 1e3
    A = np.pi * (float(expected.get("disc_diameter_mm", 110.0)) / 2e3) ** 2
    mc = float(expected.get("disc_heat_capacity_J_K", 335.0))
    stage = col(header, data, "stage")
    t = col(header, data, "time_s")
    up = col(header, data, "steam_C")
    lo = col(header, data, "disc_C")
    ss, cool = stage == 1, stage == 2
    if ss.sum() < 1 or cool.sum() < 4:
        raise ValueError("Lee's disc needs stage 1 rows (steady state) and at least 4 stage 2 rows (cooling run).")
    T1, e1 = mean_err(up[ss])
    T2, e2 = mean_err(lo[ss])
    e1, e2 = max(e1, prec / np.sqrt(ss.sum())), max(e2, prec / np.sqrt(ss.sum()))
    dT, edT = T1 - T2, float(np.hypot(e1, e2))
    # cooling run: smooth polynomial in time scaled to [−1, 1], slope where it passes T₂
    tc, tt = t[cool], lo[cool]
    mid, half = float((tc.max() + tc.min()) / 2), float((tc.max() - tc.min()) / 2) or 1.0
    u = (tc - mid) / half
    deg = 3 if cool.sum() >= 8 else 2
    p, cv = wfit(u, tt, prec, deg)
    roots = [float(r.real) for r in np.roots(np.r_[p[:-1], p[-1] - T2]) if abs(r.imag) < 1e-9]
    us = min(roots, key=abs) if roots else float(u[np.argmin(np.abs(tt - T2))])
    us = float(np.clip(us, -1.2, 1.2))
    dp = np.polyder(p)
    g = np.array([(deg - i) * us ** (deg - i - 1) for i in range(deg)] + [0.0])
    rate = -float(np.polyval(dp, us)) / half              # K/s, positive while cooling
    erate = float(np.sqrt(max(g @ cv @ g, 0))) / half
    k = mc * rate * x / (A * dT)
    ek = abs(k) * float(np.hypot(erate / rate, edT / dT)) if rate and dT else 0.0
    t_star = mid + us * half
    # the tangent at T₂ that the expected k needs
    rate_exp = k_exp * A * dT / (x * mc)
    tm = tc / 60
    ts = np.linspace(float(tm.min()), float(tm.max()), 160)
    return {
        "values": {
            "conductivity_fit_W_mK": {"value": k, "uncertainty": ek},
            "conductivity_expected_W_mK": {"value": k_exp},
            "steam_side_temperature_C": {"value": T1, "uncertainty": e1},
            "disc_steady_temperature_C": {"value": T2, "uncertainty": e2},
            "temperature_drop_K": {"value": dT, "uncertainty": edT},
            "cooling_rate_at_T2_K_min": {"value": 60 * rate, "uncertainty": 60 * erate},
            "cooling_rate_needed_K_min": {"value": 60 * rate_exp},
        },
        "evidence": {"k_error_pct": 100 * (k / k_exp - 1),
                     "t2_inside_cooling_run": bool(tt.min() <= T2 <= tt.max())},
        "plot": {"x": tm.tolist(), "y": tt.tolist(), "x_label": "time into the cooling run (min)",
                 "y_label": "brass disc temperature (°C)", "unit": "°C", "x_unit": "min", "connect": False,
                 "vlines": [{"x": t_star / 60, "label": f"steady T₂ {T2:.1f} °C"}],
                 "lines": [{"role": "fit", "x": ts.tolist(), "y": np.polyval(p, (60 * ts - mid) / half).tolist()},
                           {"role": "expected", "x": ts.tolist(), "y": (T2 - rate_exp * (60 * ts - t_star)).tolist()}]},
    }


ANALYSES = {"young_double_slit": young_double_slit, "rlc_resonance": rlc_resonance,
            "stokes_viscosity": stokes_viscosity, "lees_disc": lees_disc}
COLUMNS = {
    "young_double_slit": (["order_m", "position_mm"], ["reading", "slit_to_screen_m"]),
    "rlc_resonance": (["frequency_Hz", "resistor_V"], ["frequency_Hz", "input_V"]),
    "stokes_viscosity": (["ball_diameter_mm", "fall_time_s"], ["depth_cm", "liquid_temperature_C"]),
    "lees_disc": (["stage", "time_s", "steam_C", "disc_C"], ["reading", "steam_C", "disc_C"]),
}
