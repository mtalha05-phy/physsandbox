"""Atomic-physics bench: electron diffraction, the normal Zeeman effect and
Millikan's oil drop. Runs INSIDE the sandbox; loaded by experiments.py.

  electron_diffraction  voltage_kV, ring, diameter_mm      (follow-up: camera_length_mm)
  zeeman                field_mT, r1_inner_mm, r1_outer_mm, r2_inner_mm, r2_outer_mm
                                                           (follow-up: current_A, field_lab_mT, field_reference_mT)
  millikan              drop, charge_1e19C                 (follow-up: plate_spacing_mm)
"""
import numpy as np

from experiments import col, mean_err, sig, dense

H = 6.62607015e-34
E_CH = 1.602176634e-19
C = 299792458.0
ME = 9.1093837e-31
HBAR = H / (2 * np.pi)
MU_B = 9.2740100783e-24


# --- Electron diffraction in graphite: λ = h/√(2meV(1 + eV/2mc²)), 2d sin θ = λ ----
def de_broglie_nm(v_kv):
    v = np.asarray(v_kv, dtype=float) * 1e3
    return H / np.sqrt(2 * ME * E_CH * v * (1 + E_CH * v / (2 * ME * C ** 2))) * 1e9


def ring_diameter_mm(lam_nm, d_nm, L_mm):
    """Ring diameter on a screen a distance L beyond the foil: D = 2L tan 2θ, sin θ = λ/2d."""
    return 2 * L_mm * np.tan(2 * np.arcsin(lam_nm / (2 * d_nm)))


def electron_diffraction(header, data, expected):
    L = float(expected.get("camera_length_mm", 135.0))
    d_exp = np.array([float(expected.get("d_inner_nm", 0.213)), float(expected.get("d_outer_nm", 0.123))])
    prec = float(expected.get("precision_mm", 0.5))
    if "camera_length_mm" in header:
        # follow-up: the foil-to-screen distance measured with a ruler, repeated
        Lm = col(header, data, "camera_length_mm")
        m, e = mean_err(Lm)
        e = max(e, 0.1)                                  # a ruler reads no better than about 0.1 mm
        idx = np.arange(1, Lm.size + 1)
        return {
            "values": {"camera_length_measured_mm": {"value": m, "uncertainty": e},
                       "camera_length_entered_mm": {"value": L}},
            "evidence": {"camera_length_differs": bool(abs(m / L - 1) > 0.02 and sig(m - L, e) > 5)},
            "plot": {"x": idx.tolist(), "y": Lm.tolist(), "x_label": "ruler reading", "y_label": "foil-to-screen distance L (mm)",
                     "unit": "mm", "x_unit": "", "connect": False,
                     "lines": [{"role": "fit", "x": [1, float(idx[-1])], "y": [m, m]},
                               {"role": "expected", "x": [1, float(idx[-1])], "y": [L, L]}]},
        }
    v = col(header, data, "voltage_kV")
    ring = col(header, data, "ring").astype(int)
    D = col(header, data, "diameter_mm")
    lam = de_broglie_nm(v)
    # each ring: the lattice spacing that puts the model through its diameters, by
    # Gauss-Newton on d (in nm, so it is well conditioned), stated precision, absolute sigma
    fits = {}
    for k, de in ((1, d_exp[0]), (2, d_exp[1])):
        m = ring == k
        dk = de
        for _ in range(6):
            gk = ring_diameter_mm(lam[m], dk, L)
            dD = (ring_diameter_mm(lam[m], dk * 1.0001, L) - gk) / (dk * 1e-4)
            dk += float(np.sum(dD * (D[m] - gk)) / np.sum(dD * dD))
        dD = (ring_diameter_mm(lam[m], dk * 1.0001, L) - ring_diameter_mm(lam[m], dk, L)) / (dk * 1e-4)
        fits[k] = (dk, prec / np.sqrt(float(np.sum(dD * dD))))
    (d1, e1), (d2, e2) = fits[1], fits[2]
    # common scale of both spacings (what a wrong L or a wrong voltage does to every ring alike)
    r1, r2 = d1 / d_exp[0], d2 / d_exp[1]
    w1, w2 = (d_exp[0] / e1) ** 2, (d_exp[1] / e2) ** 2
    scale = (w1 * r1 + w2 * r2) / (w1 + w2)
    escale = 1 / np.sqrt(w1 + w2)
    ratio = d1 / d2
    eratio = ratio * np.hypot(e1 / d1, e2 / d2)
    ratio_exp = d_exp[0] / d_exp[1]
    x = 1 / np.sqrt(v)
    xs = dense(x)
    lam_s = de_broglie_nm(1 / xs ** 2)
    lines = []
    for k, dk, de in ((1, d1, d_exp[0]), (2, d2, d_exp[1])):
        lines.append({"role": "fit", "x": xs.tolist(), "y": ring_diameter_mm(lam_s, dk, L).tolist()})
        lines.append({"role": "expected", "x": xs.tolist(), "y": ring_diameter_mm(lam_s, de, L).tolist()})
    return {
        "values": {
            "d_inner_nm": {"value": d1, "uncertainty": e1},
            "d_outer_nm": {"value": d2, "uncertainty": e2},
            "accepted_d_inner_nm": {"value": float(d_exp[0])},
            "accepted_d_outer_nm": {"value": float(d_exp[1])},
            "spacing_ratio": {"value": ratio, "uncertainty": eratio},
            "accepted_spacing_ratio": {"value": ratio_exp},
            "wavelength_at_highest_kV_pm": {"value": float(de_broglie_nm(v.max()) * 1e3)},
        },
        "evidence": {
            "scale_error_pct": 100 * (scale - 1),
            "scale_sigma": sig(scale - 1, escale),
            "ring_ratio_error_pct": 100 * (ratio / ratio_exp - 1),
            "ring_ratio_sigma": sig(ratio - ratio_exp, eratio),
        },
        "plot": {"x": x.tolist(), "y": D.tolist(), "x_label": "1/√V  (V in kV; inner and outer ring)",
                 "y_label": "ring diameter on the screen (mm)", "unit": "mm", "x_unit": "", "connect": False,
                 "lines": lines},
    }


# --- Normal Zeeman effect: Fabry–Pérot rings, Δν̃(σ⁺ − σ⁻) = 2 μ_B B / hc -------------
def zeeman(header, data, expected):
    if "field_reference_mT" in header:
        # follow-up: the magnet stepped through several currents, field read on the lab
        # gaussmeter and on a calibrated one at the lamp position
        ref = col(header, data, "field_reference_mT")
        lab = col(header, data, "field_lab_mT")
        prec = float(expected.get("gaussmeter_precision_mT", 2.0))
        g = float(np.sum(ref * lab) / np.sum(ref * ref))      # lab = g·reference
        eg = prec / np.sqrt(float(np.sum(ref * ref)))
        xs = np.array([0.0, float(ref.max())])
        return {
            "values": {"lab_over_reference_field": {"value": g, "uncertainty": eg}},
            "evidence": {"field_disagrees": bool(abs(g - 1) > 0.03 and sig(g - 1, eg) > 5)},
            "plot": {"x": ref.tolist(), "y": lab.tolist(), "x_label": "field on the calibrated gaussmeter (mT)",
                     "y_label": "field on the lab gaussmeter (mT)", "unit": "mT", "x_unit": "mT", "connect": False,
                     "lines": [{"role": "fit", "x": xs.tolist(), "y": (g * xs).tolist()},
                               {"role": "expected", "x": xs.tolist(), "y": xs.tolist()}]},
        }
    B = col(header, data, "field_mT") / 1e3                     # T
    R = np.vstack([col(header, data, c) for c in ("r1_inner_mm", "r1_outer_mm", "r2_inner_mm", "r2_outer_mm")]).T
    mut = float(expected.get("etalon_thickness_mm", 3.0)) * 1e-3 * float(expected.get("etalon_index", 1.457))
    mu_exp = float(expected.get("bohr_magneton_JT", MU_B))
    prec = float(expected.get("precision_mm", 0.01))

    def split(r):
        """σ⁺ − σ⁻ wavenumber difference (/m) from one row of ring radii (ring-squared method)."""
        q = r ** 2
        delta = 0.5 * ((q[..., 1] - q[..., 0]) + (q[..., 3] - q[..., 2]))   # split, same order
        Delta = 0.5 * ((q[..., 2] - q[..., 0]) + (q[..., 3] - q[..., 1]))   # one order, same component
        return delta / Delta / (2 * mut)

    y = split(R)
    # uncertainty of each point from the stated radius precision (numerical gradient)
    sy = np.zeros_like(y)
    for j in range(4):
        dr = np.zeros_like(R)
        dr[:, j] = 1e-4
        sy += ((split(R + dr) - y) / 1e-4 * prec) ** 2
    sy = np.sqrt(sy)
    # splitting is proportional to B: fit through the origin with absolute sigma
    w = 1 / sy ** 2

    def prop(m):
        a = float(np.sum(w[m] * B[m] * y[m]) / np.sum(w[m] * B[m] ** 2))
        return a, float(1 / np.sqrt(np.sum(w[m] * B[m] ** 2)))

    allm = np.ones(B.size, bool)
    a, ea = prop(allm)
    worst = 0.0
    if B.size >= 4:
        for i in range(B.size):
            m = allm.copy()
            m[i] = False
            ai, eai = prop(m)
            worst = max(worst, abs(float(y[i] - ai * B[i])) / float(np.hypot(sy[i], eai * B[i])))
    a_exp = 2 * mu_exp / (H * C)
    muB = a * H * C / 2
    emuB = ea * H * C / 2
    xs = np.array([0.0, float(B.max()) * 1.05])
    return {
        "values": {
            "splitting_per_tesla_per_m": {"value": a, "uncertainty": ea},
            "expected_splitting_per_tesla_per_m": {"value": a_exp},
            "bohr_magneton_JT": {"value": muB, "uncertainty": emuB},
            "accepted_bohr_magneton_JT": {"value": mu_exp},
            "e_over_m_C_per_kg": {"value": 2 * muB / HBAR, "uncertainty": 2 * emuB / HBAR},
        },
        "evidence": {"muB_error_pct": 100 * (a / a_exp - 1), "outlier_sigma": worst},
        "plot": {"x": B.tolist(), "y": y.tolist(), "x_label": "magnetic field B (T)",
                 "y_label": "σ⁺ − σ⁻ splitting (wavenumber, /m)", "unit": "/m", "x_unit": "T", "connect": False,
                 "lines": [{"role": "fit", "x": xs.tolist(), "y": (a * xs).tolist()},
                           {"role": "expected", "x": xs.tolist(), "y": (a_exp * xs).tolist()}]},
    }


# --- Millikan oil drop: every charge a whole multiple of e -----------------------------
def millikan(header, data, expected):
    e_acc = float(expected.get("elementary_charge_C", E_CH)) * 1e19     # in 10^-19 C
    if "plate_spacing_mm" in header:
        # follow-up: calliper readings of the plate spacing (spacer ring) at several places
        s = col(header, data, "plate_spacing_mm")
        d_used = float(expected.get("plate_spacing_used_mm", 7.60))
        m, e = mean_err(s)
        e = max(e, 0.01)                                     # callipers read to about 0.01 mm
        idx = np.arange(1, s.size + 1)
        return {
            "values": {"plate_spacing_measured_mm": {"value": m, "uncertainty": e},
                       "plate_spacing_used_mm": {"value": d_used},
                       "charge_correction_factor": {"value": m / d_used}},
            "evidence": {"spacing_differs": bool(abs(m / d_used - 1) > 0.02 and sig(m - d_used, e) > 5)},
            "plot": {"x": idx.tolist(), "y": s.tolist(), "x_label": "calliper reading",
                     "y_label": "plate spacing (mm)", "unit": "mm", "x_unit": "", "connect": False,
                     "lines": [{"role": "fit", "x": [1, float(idx[-1])], "y": [m, m]},
                               {"role": "expected", "x": [1, float(idx[-1])], "y": [d_used, d_used]}]},
        }
    q = col(header, data, "charge_1e19C")
    rel = float(expected.get("precision_pct", 2.0)) / 100
    sq = rel * np.abs(q)
    dof = max(1, q.size - 1)

    def assign(e_try):
        n = np.maximum(1, np.round(q / e_try))
        w = n / sq ** 2
        e_hat = float(np.sum(w * q) / np.sum(w * n))
        chi2 = float(np.sum(((q - n * e_hat) / sq) ** 2))
        return n, e_hat, chi2, float(1 / np.sqrt(np.sum(n ** 2 / sq ** 2)))

    # Search for the charge unit between 0.6 and 1.5 times the accepted e: no realistic
    # calibration slip moves it further, and the window keeps e/2, e/3 (which every set of
    # charges "fits") out. Take the largest unit the charges are whole multiples of.
    cands = []
    for e_try in np.linspace(0.6 * e_acc, 1.5 * e_acc, 1800):
        n, e_hat, chi2, ee = assign(e_try)
        if 0.6 * e_acc <= e_hat <= 1.5 * e_acc:
            cands.append(((chi2 - dof) / np.sqrt(2 * dof), e_hat, ee, n, chi2))
    good = [c for c in cands if c[0] < 4]
    best = max(good, key=lambda c: c[1]) if good else min(cands, key=lambda c: c[0])
    qs, e_fit, e_err, n, chi2 = best
    ns = np.array([0.0, float(n.max()) + 0.5])
    return {
        "values": {
            "elementary_charge_C": {"value": e_fit * 1e-19, "uncertainty": e_err * 1e-19},
            "accepted_elementary_charge_C": {"value": e_acc * 1e-19},
            "quantisation_chi2_per_dof": {"value": chi2 / dof},
            "largest_multiple_n": {"value": float(n.max())},
        },
        "evidence": {"e_error_pct": 100 * (e_fit / e_acc - 1), "quantisation_sigma": float(max(qs, 0.0))},
        "plot": {"x": n.tolist(), "y": q.tolist(), "x_label": "number of electron charges n (assigned)",
                 "y_label": "drop charge (×10⁻¹⁹ C)", "unit": "×10⁻¹⁹ C", "x_unit": "", "connect": False,
                 "lines": [{"role": "fit", "x": ns.tolist(), "y": (e_fit * ns).tolist()},
                           {"role": "expected", "x": ns.tolist(), "y": (e_acc * ns).tolist()}]},
    }


ANALYSES = {"electron_diffraction": electron_diffraction, "zeeman": zeeman, "millikan": millikan}
COLUMNS = {
    "electron_diffraction": (["voltage_kV", "ring", "diameter_mm"], ["camera_length_mm"]),
    "zeeman": (["field_mT", "r1_inner_mm", "r1_outer_mm", "r2_inner_mm", "r2_outer_mm"],
               ["field_lab_mT", "field_reference_mT"]),
    "millikan": (["charge_1e19C"], ["plate_spacing_mm"]),
}
