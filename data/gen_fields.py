"""Seeded datasets for the fields and scattering bench: e/m of the electron in a
fine-beam tube, electron spin resonance of DPPH, Rutherford scattering of alpha
particles from gold, X-ray Bragg diffraction from NaCl.

Run by data/generate.py after the other generators. Ground truth is in
data/manifest.d/fields.yaml.
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "seeded"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7301)


def write(name, header, *cols, fmt="{:.6g}"):
    fmts = fmt if isinstance(fmt, (list, tuple)) else [fmt] * len(cols)
    lines = [",".join(header)]
    for row in zip(*cols):
        lines.append(",".join(f.format(v) for f, v in zip(fmts, row)))
    (OUT / name).write_text("\n".join(lines) + "\n")


# --- e/m: fine-beam tube in Helmholtz coils, R = 150 mm, 130 turns each ----------------
# Coil axis east–west, so the Earth's horizontal field has no part along it. At each
# anode voltage the current is set (to 0.01 A) for a beam of about 3, 4 and 5 cm radius,
# and the radius read to about ±0.5 mm.
MU0, Q_M = 4e-7 * np.pi, 1.75882001076e11
K_COIL = (4 / 5) ** 1.5 * MU0 * 130 / 0.150         # T per A
V_SET = np.repeat([100.0, 150.0, 200.0, 250.0, 300.0], 3)
R_AIM = np.tile([3.0, 4.0, 5.0], 5)                  # cm


def em_run(scale=1.0, v_bias=0.0):
    """scale: radius read ÷ true radius (parallax); v_bias: volts the meter reads above the anode–cathode voltage."""
    v_true = V_SET - v_bias
    amps = np.round(np.sqrt(2 * v_true / Q_M) / (K_COIL * R_AIM / scale / 100), 2)
    r_true = 100 * np.sqrt(2 * v_true / Q_M) / (K_COIL * amps)
    return V_SET, amps, scale * r_true + rng.normal(0, 0.04, V_SET.size)


hdr = ["accelerating_voltage_V", "coil_current_A", "beam_radius_cm"]
fm = ["{:.0f}", "{:.2f}", "{:.2f}"]
write("em_clean.csv", hdr, *em_run(), fmt=fm)
# diameters read on a ruler held against the front of the bulb, eye not lined up: every
# diameter reads 5% small, so e/m comes out (1/0.95)² = 11% high at every setting
write("em_parallax.csv", hdr, *em_run(scale=0.95), fmt=fm)
# follow-up: the same settings, every diameter reread on the mirror scale behind the tube
write("em_parallax_mirror.csv", hdr + ["mirror_scale"], *em_run(), np.ones(V_SET.size), fmt=fm + ["{:.0f}"])
# voltmeter across the supply's outputs, anode to Wehnelt: it reads the 25 V Wehnelt bias
# as well, so the electrons get 25 V less than the reading
write("em_wehnelt.csv", hdr, *em_run(v_bias=25.0), fmt=fm)

# --- ESR of DPPH: Helmholtz coils, field modulated, read on a Hall-probe gaussmeter ----
# h f = g μ_B B with g = 2.0036; RF from three plug-in coils, 15 to 125 MHz.
G_DPPH, MUB_H = 2.0036, 13.99624493                  # μ_B/h in MHz per mT
F_RF = np.array([15.3, 24.8, 35.1, 44.6, 55.2, 64.8, 75.4, 84.9, 95.3, 104.7, 115.2, 124.6])
b_res = lambda f: f / (G_DPPH * MUB_H)
read_b = lambda b: b + rng.normal(0, 0.008, np.size(b))   # line centre read to about ±0.01 mT
hdr = ["rf_frequency_MHz", "resonance_field_mT"]
fm = ["{:.1f}", "{:.3f}"]
write("esr_clean.csv", hdr, F_RF, read_b(b_res(F_RF)), fmt=fm)
# Hall probe not zeroed in its zero-gauss chamber: it reads 0.10 mT with no field
write("esr_probe_offset.csv", hdr, F_RF, read_b(b_res(F_RF) + 0.10), fmt=fm)
# follow-up: six frequencies forward, then with the coil leads swapped; magnitudes recorded,
# so the probe's offset adds one way (B + 0.10) and subtracts the other (B − 0.10)
f6 = F_RF[1::2]
write("esr_probe_offset_reversed.csv", hdr + ["field_reversed"], np.r_[f6, f6],
      np.r_[read_b(b_res(f6) + 0.10), read_b(b_res(f6) - 0.10)], np.r_[np.zeros(6), np.ones(6)],
      fmt=fm + ["{:.0f}"])
# field from the coil current with a coil constant 4% too high: every field 4% high, g 4% low
write("esr_field_scale.csv", hdr, F_RF, read_b(1.04 * b_res(F_RF)), fmt=fm)
# the counter read 64.8 MHz, written down as 68.4 MHz; the field is the one for 64.8 MHz
f_noted = F_RF.copy()
f_noted[5] = 68.4
write("esr_counter_misread.csv", hdr, f_noted, read_b(b_res(F_RF)), fmt=fm)

# --- Rutherford scattering: sealed Am-241 source, 2 µm gold foil, surface-barrier detector ---
# Rate R(θ) = A sin⁴(10°)/sin⁴(θ/2) with A = 1.2 /s at 20°, Poisson counts; live times grow
# with angle so every point has about 200 counts or more.
A20 = 1.2
S20 = np.sin(np.radians(10.0)) ** 4
ANG = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0])
LIVE = np.array([60.0, 120.0, 300.0, 600.0, 900.0, 1500.0, 2400.0, 6000.0])
rate = lambda th: A20 * S20 / np.sin(np.radians(np.abs(th)) / 2) ** 4
count = lambda r, t: rng.poisson(r * t).astype(float)
hdr = ["angle_deg", "counts", "live_time_s"]
fm = ["{:.0f}", "{:.0f}", "{:.0f}"]
write("ruth_clean.csv", hdr, ANG, count(rate(ANG), LIVE), LIVE, fmt=fm)
# the detector arm's zero mark sits 1.5° off the beam axis: every angle read is 1.5° above the true one
write("ruth_zero_offset.csv", hdr, ANG, count(rate(ANG - 1.5), LIVE), LIVE, fmt=fm)
# follow-up: both sides of the beam, one minute at ±10° and ±15°, two at ±20° (angles as read)
a2 = np.array([-20.0, -15.0, -10.0, 10.0, 15.0, 20.0])
t2 = np.array([120.0, 60.0, 60.0, 60.0, 60.0, 120.0])
write("ruth_zero_offset_both_sides.csv", hdr, a2, count(rate(a2 - 1.5), t2), t2, fmt=fm)
# alphas leaking round the source's collimator reach the detector directly: 0.04 /s at every angle
write("ruth_leakage.csv", hdr, ANG, count(rate(ANG) + 0.04, LIVE), LIVE, fmt=fm)

# --- X-ray Bragg diffraction: Mo anode, NaCl (200), coupled θ–2θ scan ------------------
# Kα peak (the unresolved Kα₁/Kα₂ pair, 71.07 pm) of orders 1–4, centre read to about ±0.04°.
LA, LB, D_NACL = 71.07, 63.26, 282.0
NB = np.arange(1.0, 5.0)
bragg = lambda n, lam=LA, d=D_NACL: np.degrees(np.arcsin(n * lam / (2 * d)))
read_t = lambda t: t + rng.normal(0, 0.04, np.size(t))
hdr = ["order_n", "angle_deg"]
fm = ["{:.0f}", "{:.2f}"]
write("bragg_clean.csv", hdr, NB, read_t(bragg(NB)), fmt=fm)
# goniometer zero 0.55° off: every angle reads 0.55° high
write("bragg_zero_offset.csv", hdr, NB, read_t(bragg(NB) + 0.55), fmt=fm)
# follow-up: crystal out, detector alone through the attenuated direct beam; the zero error
# puts the beam at 2 × 0.55° on the detector scale. Counts in 2 s per step.
x2 = np.round(np.arange(-1.5, 3.01, 0.1), 2)
beam = 15 + 1800 * np.exp(-2 * ((x2 - 1.10) / 0.6) ** 2)
write("bragg_zero_offset_direct_beam.csv", ["detector_angle_deg", "count_rate"], x2,
      rng.poisson(beam * 2.0) / 2.0, fmt=["{:.1f}", "{:.1f}"])
# third order: the weaker Kβ peak (63.26 pm), 2.5° below Kα, was read as the Kα peak
tk = bragg(NB)
tk[2] = bragg(3.0, lam=LB)
write("bragg_kbeta.csv", hdr, NB, read_t(tk), fmt=fm)
# a LiF crystal (200), d = 201.4 pm, in place of the NaCl
write("bragg_lif.csv", hdr, NB, read_t(bragg(NB, d=201.4)), fmt=fm)

print(f"Wrote fields-and-scattering datasets to {OUT}")
