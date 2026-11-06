"""Seeded datasets for the optics, semiconductor and thermocouple bench: thin-lens focal
length, Brewster's angle, band gap of germanium, type K thermocouple calibration.

Run by data/generate.py after the other generators. Ground truth is in
data/manifest.d/optics2.yaml.
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "seeded"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7303)


def write(name, header, *cols, fmt="{:.6g}"):
    fmts = fmt if isinstance(fmt, (list, tuple)) else [fmt] * len(cols)
    lines = [",".join(header)]
    for row in zip(*cols):
        lines.append(",".join(f.format(v) for f, v in zip(fmts, row)))
    (OUT / name).write_text("\n".join(lines) + "\n")


# --- Thin lens on a 1 m optical bench: +10 D lens, f = 10.0 cm ------------------------
# The lens holder is set so that u (read on the bench scale at the holder's index mark) takes
# these values; the screen is moved until the image looks sharpest and v is read. Bracketing the
# focus (blurred on each side, take the midpoint) finds it to about a tenth of the depth of focus.
U_SET = np.array([13, 14, 15, 16, 18, 20, 22, 25, 30, 35, 40, 50], dtype=float)


def lens(f=10.0, delta=0.0, u_set=U_SET, v_sd=lambda v: 0.1 + 0.005 * v):
    """delta: the index mark sits delta cm on the screen side of the lens, so every u is read
    delta long and every v delta short."""
    u = u_set - delta + rng.normal(0, 0.06, u_set.size)          # where the lens really is
    v = f * u / (u - f)
    return u_set, v - delta + rng.normal(0, 1, u.size) * v_sd(v)


hdr = ["object_distance_cm", "image_distance_cm"]
fm = ["{:.1f}", "{:.1f}"]
write("lens_clean.csv", hdr, *lens(), fmt=fm)
# the lens position read at the edge of its holder, 1.2 cm on the screen side of the glass
write("lens_holder_edge.csv", hdr, *lens(delta=1.2), fmt=fm)
# follow-up: the holder at five places along the bench, index mark and the lens's own mid-plane
# (sighted across the glass with a set square) both read on the bench scale
pos = np.array([20.0, 30.0, 40.0, 50.0, 60.0])
write("lens_holder_edge_index.csv", ["index_mark_cm", "lens_centre_cm"], pos + rng.normal(0, 0.03, 5),
      pos - 1.2 + rng.normal(0, 0.08, 5), fmt=fm)
# a +8 D lens (f = 12.5 cm) taken from the box instead of the +10 D one: no image fits on the bench
# below u = 16 cm, so the run starts there
write("lens_wrong_lens.csv", hdr, *lens(f=12.5, u_set=np.array([16, 17, 18, 20, 22, 25, 30, 35, 40, 50, 60.0])),
      fmt=fm)
# image judged sharp by eye in one go, no bracketing: the depth of focus (about ±8% of v: blur of
# 2.4 mm goes unnoticed on a dim ground-glass image through a 3 cm aperture) spreads the image distances
write("lens_focus_by_eye.csv", hdr, *lens(v_sd=lambda v: 0.08 * v), fmt=fm)


# --- Brewster's angle: 1 mW HeNe (632.8 nm), crown-glass plate n = 1.515 ----------------
# Polariser set for p, plate on a rotation stage read to 0.1°, photodiode power meter on the arm
# behind an iris closed round the front-face reflection; room lights off, meter zeroed.
N_GLASS, P_INC = 1.515, 1000.0
ANG = np.array([10, 20, 30, 40, 45, 50, 52, 54, 56, 58, 60, 62, 65, 70, 75, 80, 85], dtype=float)


def fresnel(th, n=N_GLASS):
    t = np.radians(np.abs(th))
    c, s2 = np.cos(t), np.sin(t) ** 2
    root = np.sqrt(n ** 2 - s2)
    return ((n ** 2 * c - root) / (n ** 2 * c + root)) ** 2, ((c - root) / (c + root)) ** 2


def reflected(th_true, phi=0.0, ext=1e-4, ghost=False):
    """Power reaching the detector: polariser axis phi degrees from p, extinction ratio ext;
    ghost adds the back face's reflection (incoherent, the two spots don't overlap)."""
    rp, rs = fresnel(th_true)
    if ghost:
        rp, rs = rp + (1 - rp) ** 2 * rp / (1 - rp ** 2), rs + (1 - rs) ** 2 * rs / (1 - rs ** 2)
    c2, s2 = np.cos(np.radians(phi)) ** 2, np.sin(np.radians(phi)) ** 2
    return P_INC * ((c2 + ext * s2) * rp + (s2 + ext * c2) * rs)


def meter(p):
    return p * (1 + rng.normal(0, 0.005, p.size)) + rng.normal(0, 0.05, p.size)


def brewster(zero=0.0, **kw):
    th_true = ANG - zero + rng.normal(0, 0.1, ANG.size)          # zero: stage reads `zero` too high
    return ANG, meter(reflected(th_true, **kw))


hdr = ["incidence_angle_deg", "reflected_power_uW"]
fm = ["{:.1f}", "{:.4g}"]
write("brewster_clean.csv", hdr, *brewster(), fmt=fm)
# polariser set by eye 5° off p: sin²5° = 0.8% of the light is s-polarised and reflects at Brewster's angle
write("brewster_polariser.csv", hdr, *brewster(phi=5.0), fmt=fm)
# follow-up: stage at the minimum (56.5°), polariser turned from −10° to +10° about its setting
PHI = np.arange(-10.0, 10.1, 2.0)
th_min = 56.5 + rng.normal(0, 0.1, PHI.size)
write("brewster_polariser_rotate.csv", ["polariser_angle_deg", "reflected_power_uW"], PHI,
      meter(np.array([reflected(np.array([t]), phi=ph - 5.0)[0] for t, ph in zip(th_min, PHI)])), fmt=fm)
# stage zeroed by eye against the plate's edge, 2.0° out: every angle reads 2.0° high
write("brewster_stage_zero.csv", hdr, *brewster(zero=2.0), fmt=fm)
# iris left wide open: the reflection from the plate's back face, 3–4 mm beside the first, is counted too
write("brewster_ghost.csv", hdr, *brewster(ghost=True), fmt=fm)


# --- Band gap of germanium: undoped bar 20 × 10 × 1 mm, 1 mA through it, in a small oven --------
# n_i = C T^{3/2} exp(−Eg(T)/2kT) with Varshni's Eg(T) for Ge, lattice-scattering mobilities, and
# 5×10¹² cm⁻³ residual donors (negligible above 340 K). A straight line through ln(R T^{3/2})
# against 1/T over 340–450 K gives about 0.66 eV.
KB = 8.617333262e-5


def r_ge(T, n_d=5e12):
    eg = 0.7437 - 4.774e-4 * T ** 2 / (T + 235.0)
    ni = 1.436e15 * T ** 1.5 * np.exp(-eg / (2 * KB * T))
    n = n_d / 2 + np.sqrt(n_d ** 2 / 4 + ni ** 2)
    p = ni ** 2 / n
    mu_n, mu_p = 3900 * (T / 300) ** -1.66, 1900 * (T / 300) ** -2.33
    sigma = 1.602176634e-19 * (n * mu_n + p * mu_p)          # S/cm
    return 20.0 / sigma                                       # L/A = 2.0 cm / 0.10 cm²


T_SET = np.arange(340.0, 450.1, 10.0)


def gap_run(t_sample, r_series=0.0):
    t_s = t_sample + rng.normal(0, 0.3, t_sample.size)       # sample and thermometer differ a little
    return (r_ge(t_s) + r_series) * (1 + rng.normal(0, 0.002, t_s.size))


def heating_lag(t_read, rate=10.0, tau=2.0, start=295.0):
    """Sample temperature while the oven thermometer ramps up at `rate` K/min from `start`:
    a first-order lag with time constant tau (min)."""
    t = (t_read - start) / rate
    return t_read - rate * tau * (1 - np.exp(-t / tau))


hdr = ["temperature_K", "resistance_ohm"]
fm = ["{:.1f}", "{:.4g}"]
write("bandgap_clean.csv", hdr, T_SET, gap_run(T_SET), fmt=fm)
# heated at 10 K/min with the sample on its board lagging the oven's thermometer by up to 20 K
write("bandgap_lag.csv", hdr, T_SET, gap_run(heating_lag(T_SET)), fmt=fm)
# follow-up: the same heating sweep, then the controller ramps the oven down at 5 K/min
# (readings from 430 K down); on cooling the sample is warmer than the thermometer
t_end = heating_lag(np.array([450.0]))[0]
T_COOL = np.arange(430.0, 339.9, -10.0)
tc = (450.0 - T_COOL) / 5.0
ts_cool = T_COOL + 5.0 * 2.0 + (t_end - 450.0 - 5.0 * 2.0) * np.exp(-tc / 2.0)
write("bandgap_lag_cooling.csv", ["temperature_K", "resistance_ohm", "cooling"],
      np.r_[T_SET, T_COOL], np.r_[gap_run(heating_lag(T_SET)), gap_run(ts_cool)],
      np.r_[np.zeros(T_SET.size), np.ones(T_COOL.size)], fmt=fm + ["{:.0f}"])
# solder contacts gone bad: 3.0 Ω in series with the bar in the two-terminal reading
write("bandgap_contacts.csv", hdr, T_SET, gap_run(T_SET, r_series=3.0), fmt=fm)


# --- Type K thermocouple: hot junction in a stirred oil bath, reference in ice water -----------
# emf read on a 4½-digit meter (0.001 mV shown), bath temperature on a calibrated thermometer.
K_C = [-0.176004136860e-01, 0.389212049750e-01, 0.185587700320e-04, -0.994575928740e-07,
       0.318409457190e-09, -0.560728448890e-12, 0.560750590590e-15, -0.320207200030e-18,
       0.971511471520e-22, -0.121047212750e-25]
J_C = [0.0, 5.0381187815e-02, 3.0475836930e-05, -8.5681065720e-08, 1.3228195295e-10,
       -1.7052958337e-13, 2.0948090697e-16, -1.2538395336e-19, 1.5631725697e-23]
emf_k = lambda t: np.polyval(K_C[::-1], t) + 0.1185976 * np.exp(-1.183432e-4 * (t - 126.9686) ** 2)
emf_j = lambda t: np.polyval(J_C[::-1], t)
T_BATH = np.array([25, 40, 60, 80, 100, 120, 140, 160, 180, 200], dtype=float)


def tc_run(table=emf_k, t_ref=0.0):
    t_read = np.round(T_BATH + rng.normal(0, 0.5, T_BATH.size), 1)   # the bath is never exactly on the mark
    t_junction = t_read + rng.normal(0, 0.1, T_BATH.size)
    return t_read, table(t_junction) - table(t_ref) + rng.normal(0, 0.004, T_BATH.size)


hdr = ["hot_junction_C", "emf_mV"]
fm = ["{:.1f}", "{:.3f}"]
write("thermocouple_clean.csv", hdr, *tc_run(), fmt=fm)
# the ice in the reference flask has all melted and the water has warmed to 8 °C
write("thermocouple_ice_melted.csv", hdr, *tc_run(t_ref=8.0), fmt=fm)
# follow-up: a thermometer in the reference flask, six readings a few seconds apart
write("thermocouple_ice_melted_bath.csv", ["reading", "ice_bath_C"], np.arange(1, 7),
      8.0 + rng.normal(0, 0.1, 6), fmt=["{:.0f}", "{:.1f}"])
# a type J (iron–constantan) thermocouple from the drawer, read as if it were type K
write("thermocouple_type_j.csv", hdr, *tc_run(table=emf_j), fmt=fm)

print(f"Wrote optics2 datasets to {OUT}")
