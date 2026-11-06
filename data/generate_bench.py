"""Seeded datasets for the extended experiment bench.

Each experiment gets a healthy run, two common faults, and the follow-up
measurement a researcher would record after running the recommended test.
Ground truth is in data/manifest.yaml.
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "seeded"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(2026)
MU0 = 4e-7 * np.pi


def write(name, header, *cols, fmt="{:.6g}"):
    lines = [",".join(header)]
    for row in zip(*cols):
        lines.append(",".join(fmt.format(v) for v in row))
    (OUT / name).write_text("\n".join(lines) + "\n")


def groups(name, header, a, b, n=10, noise=1.0):
    g = [0] * n + [1] * n
    v = np.concatenate([a + rng.normal(0, noise, n), b + rng.normal(0, noise, n)])
    write(name, header, g, v)


# --- Helmholtz coil: R = 0.15 m, N = 100 -> k = 599.5 µT/A -------------------
k = (4 / 5) ** 1.5 * MU0 * 100 / 0.15 * 1e6
I = np.linspace(0.1, 1.0, 10)
write("helmholtz_clean.csv", ["current_A", "measured_B_uT"], I, k * I + rng.normal(0, 1.5, I.size))
write("helmholtz_ambient.csv", ["current_A", "measured_B_uT"], I, k * I + 45 + rng.normal(0, 1.5, I.size))
write("helmholtz_gain.csv", ["current_A", "measured_B_uT"], I, 0.82 * k * I + rng.normal(0, 1.5, I.size))
# follow-up: probe at I = 0 in the open (Earth field ~45 µT), then in a zero-gauss chamber
groups("helmholtz_ambient_shield.csv", ["in_shield", "measured_B_uT"], 45.0, 0.0, noise=0.8)

# --- Laser and photodetector alignment: w = 1.0 mm, peak 600 mV ---------------
x = np.linspace(-3, 3, 25)
g = lambda x0, a, d=12: d + a * np.exp(-2 * ((x - x0) / 1.0) ** 2)
write("laser_clean.csv", ["position_mm", "detector_voltage_mV"], x, g(0.0, 600) + rng.normal(0, 3, x.size), fmt="{:.2f}")
write("laser_misaligned.csv", ["position_mm", "detector_voltage_mV"], x, g(0.85, 600) + rng.normal(0, 3, x.size), fmt="{:.2f}")
write("laser_low_power.csv", ["position_mm", "detector_voltage_mV"], x, g(0.0, 180) + rng.normal(0, 3, x.size), fmt="{:.2f}")
# follow-up: repeat scan 15 minutes later; the centre has not moved, so it is not warm-up drift
write("laser_misaligned_rescan.csv", ["position_mm", "detector_voltage_mV"], x, g(0.86, 600) + rng.normal(0, 3, x.size), fmt="{:.2f}")

# --- Faraday's law: N = 200, 10 Hz sinusoidal flux, 0.1 mWb peak --------------
t = np.linspace(0, 0.2, 101)
phi = 1e-4 * np.sin(2 * np.pi * 10 * t)
emf = -200 * np.gradient(phi, t)
hdr = ["time_s", "magnetic_flux_Wb", "induced_voltage_V"]
write("faraday_clean.csv", hdr, t, phi, emf + rng.normal(0, 0.02, t.size))
write("faraday_offset.csv", hdr, t, phi, emf + 0.15 + rng.normal(0, 0.02, t.size))
write("faraday_half_turns.csv", hdr, t, phi, 0.5 * emf + rng.normal(0, 0.02, t.size))
# follow-up: magnet held still; coil connected, then amplifier input shorted
groups("faraday_offset_shorted.csv", ["input_shorted", "voltage_V"], 0.15, 0.15, noise=0.01)

# --- Beer-Lambert: alpha = 0.5 /cm, I0 = 1 -------------------------------------
d = np.linspace(0, 5, 11)
bl = lambda a, bg=0.0: np.exp(-a * d) + bg
write("beer_clean.csv", ["thickness_cm", "transmitted_intensity"], d, bl(0.5) + rng.normal(0, 0.004, d.size))
write("beer_stray_light.csv", ["thickness_cm", "transmitted_intensity"], d, bl(0.5, 0.12) + rng.normal(0, 0.004, d.size))
write("beer_high_alpha.csv", ["thickness_cm", "transmitted_intensity"], d, bl(1.1) + rng.normal(0, 0.004, d.size))
# follow-up: same scan with the room lights off and the enclosure lid closed
write("beer_stray_light_enclosed.csv", ["thickness_cm", "transmitted_intensity"], d, bl(0.5, 0.0) + rng.normal(0, 0.004, d.size))

# --- Malus's law: I = I0 cos^2(theta) ------------------------------------------
th = np.arange(0, 181, 10)
ml = lambda off=0.0, bg=0.0: 1.0 * np.cos(np.radians(th - off)) ** 2 + bg
write("malus_clean.csv", ["angle_deg", "intensity"], th, ml() + rng.normal(0, 0.005, th.size))
write("malus_poor_extinction.csv", ["angle_deg", "intensity"], th, ml(bg=0.08) + rng.normal(0, 0.005, th.size))
write("malus_angle_offset.csv", ["angle_deg", "intensity"], th, ml(off=12) + rng.normal(0, 0.005, th.size))
# follow-up: polarisers crossed, then the laser beam blocked: the light is still there -> room light
groups("malus_poor_extinction_blocked.csv", ["beam_blocked", "intensity"], 0.08, 0.078, noise=0.003)

# --- Newton's law of cooling, K-type thermocouple, room at 21 °C ---------------
tc = np.arange(0, 1801, 60.0)
cool = lambda ta: ta + (80 - 21) * np.exp(-tc / 420)
write("cooling_clean.csv", ["time_s", "temperature_C"], tc, cool(21) + rng.normal(0, 0.15, tc.size), fmt="{:.2f}")
write("cooling_cjc_offset.csv", ["time_s", "temperature_C"], tc, cool(21) + 2.5 + rng.normal(0, 0.15, tc.size), fmt="{:.2f}")
write("cooling_slow_contact.csv", ["time_s", "temperature_C"], tc, 21 + 59 * np.exp(-tc / 900) + rng.normal(0, 0.15, tc.size), fmt="{:.2f}")
# follow-up: thermocouple in an ice-water bath, which should read 0.0 °C
ti = np.arange(0, 120, 10.0)
write("cooling_cjc_ice_point.csv", ["time_s", "temperature_C"], ti, 2.5 + rng.normal(0, 0.1, ti.size), fmt="{:.2f}")

# --- RC charging: R = 1 MΩ, C = 100 nF, 5 V supply, tau = 0.1 s ---------------
tr = np.linspace(0, 0.6, 61)
rc = lambda vf, tau: vf * (1 - np.exp(-tr / tau))
write("rc_clean.csv", ["time_s", "voltage_V"], tr, rc(5, 0.1) + rng.normal(0, 0.01, tr.size), fmt="{:.4f}")
# x1 scope probe (1 MΩ) in parallel with the 1 MΩ resistor: half the voltage, half the time constant
write("rc_probe_loading.csv", ["time_s", "voltage_V"], tr, rc(2.5, 0.05) + rng.normal(0, 0.01, tr.size), fmt="{:.4f}")
write("rc_cap_tolerance.csv", ["time_s", "voltage_V"], tr, rc(5, 0.068) + rng.normal(0, 0.01, tr.size), fmt="{:.4f}")
# follow-up: same circuit measured with a x10 (10 MΩ) probe: loading almost disappears
write("rc_probe_loading_x10.csv", ["time_s", "voltage_V"], tr, rc(5 * 10 / 11, 0.1 * 10 / 11) + rng.normal(0, 0.01, tr.size), fmt="{:.4f}")

# =============================================================================
# Second bench extension. A separate generator keeps the files above unchanged.
rng = np.random.default_rng(2027)

# --- Simple pendulum: g = 9.81 m/s², T timed over 10 swings -------------------
Lp = np.linspace(0.2, 1.2, 11)
per = lambda L, amp_deg=5, timer=1.0: timer * 2 * np.pi * np.sqrt(L / 9.81) * (1 + np.radians(amp_deg) ** 2 / 16)
hdr = ["length_m", "period_s"]
write("pendulum_clean.csv", hdr, Lp, per(Lp) + rng.normal(0, 0.003, Lp.size), fmt="{:.4f}")
# released from 30° instead of a few degrees: every period ~1.7% long, g reads ~3% low
write("pendulum_large_amp.csv", hdr, Lp, per(Lp, 30) + rng.normal(0, 0.003, Lp.size), fmt="{:.4f}")
# length measured to the top of the bob, not its centre (bob radius 2 cm)
write("pendulum_length_offset.csv", hdr, Lp, per(Lp + 0.02) + rng.normal(0, 0.003, Lp.size), fmt="{:.4f}")
# follow-up: repeated from 5°
write("pendulum_large_amp_small.csv", hdr, Lp, per(Lp, 5) + rng.normal(0, 0.003, Lp.size), fmt="{:.4f}")

# --- Hooke's law: k = 25 N/m, 50 g hanger ---------------------------------------
mk = np.linspace(0.05, 0.5, 10)
ext = lambda m, c=0.0: m * 9.81 / 25 + c * (m * 9.81) ** 2
hdr = ["mass_kg", "extension_m"]
write("hooke_clean.csv", hdr, mk, ext(mk + 0.05) + rng.normal(0, 0.0005, mk.size) - ext(0.05), fmt="{:.5f}")
# the 50 g hanger is on the spring but left out of the masses (zero read without it)
write("hooke_hanger.csv", hdr, mk, ext(mk + 0.05) + rng.normal(0, 0.0005, mk.size), fmt="{:.5f}")
# loaded past the elastic limit: the spring stretches more at high loads
write("hooke_overstretched.csv", hdr, mk, ext(mk, 0.002) + rng.normal(0, 0.0005, mk.size), fmt="{:.5f}")
# follow-up: hanger weighed (50 g) and added to every load
write("hooke_hanger_included.csv", hdr, mk + 0.05, ext(mk + 0.05) + rng.normal(0, 0.0005, mk.size), fmt="{:.5f}")

# --- Diode I-V: n = 1.9, I_s = 2 nA, 300 K, 0.6 Ω bulk resistance -------------
Id = np.geomspace(1e-6, 50e-3, 20)
vd = lambda I, rs=0.6: 1.9 * 0.025852 * np.log(I / 2e-9) + I * rs
hdr = ["voltage_V", "current_A"]
write("diode_clean.csv", hdr, vd(Id) + rng.normal(0, 5e-4, Id.size), Id)
# voltmeter across diode + ammeter: the ammeter's 10 Ω burden adds I·10 Ω
write("diode_burden.csv", hdr, vd(Id, 0.6 + 10) + rng.normal(0, 5e-4, Id.size), Id)
# diode in backwards: only nanoamps of leakage flow
Vr = np.linspace(0.1, 0.9, 20)
write("diode_reversed.csv", hdr, Vr, 2e-9 + rng.normal(0, 2e-10, Vr.size))
# follow-up: voltmeter moved to the diode's own leads
write("diode_burden_rewired.csv", hdr, vd(Id) + rng.normal(0, 5e-4, Id.size), Id)

# --- Diffraction grating: 600 lines/mm, 532 nm green laser ----------------------
mo = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
ang = lambda lam, lpmm=600, d0=0.0: np.degrees(np.arcsin(mo * lam * 1e-9 * lpmm * 1e3)) + d0
hdr = ["order_m", "angle_deg"]
write("grating_clean.csv", hdr, mo, ang(532) + rng.normal(0, 0.05, mo.size), fmt="{:.3f}")
# the grating in the holder is a 500 lines/mm one, not the 600 assumed
write("grating_wrong_spec.csv", hdr, mo, ang(532, 500) + rng.normal(0, 0.05, mo.size), fmt="{:.3f}")
# spectrometer vernier zero misread by 0.8°
write("grating_zero_offset.csv", hdr, mo, ang(532, 600, 0.8) + rng.normal(0, 0.05, mo.size), fmt="{:.3f}")
# follow-up: same grating with a HeNe laser of known wavelength (632.8 nm)
lines = ["order_m,angle_deg,reference_nm"] + [f"{m:.6g},{a:.3f},632.8" for m, a in
                                             zip(mo, ang(632.8, 500) + rng.normal(0, 0.05, mo.size))]
(OUT / "grating_wrong_spec_ref.csv").write_text("\n".join(lines) + "\n")

# --- Hall effect: B = 0.3 T, t = 1 mm, n = 1e21 /m³ -> 1.87 V/A ----------------
Ih = np.linspace(1e-3, 10e-3, 10)
kh = 0.3 / (1e21 * 1.602176634e-19 * 1e-3)
hdr = ["current_A", "hall_voltage_V"]
write("hall_clean.csv", hdr, Ih, kh * Ih + rng.normal(0, 2e-5, Ih.size))
# Hall contacts not exactly opposite: an IR drop of -20% of the Hall voltage rides on top
write("hall_misaligned.csv", hdr, Ih, 0.8 * kh * Ih + rng.normal(0, 2e-5, Ih.size))
# thermoelectric voltage at the contacts: 0.5 mV at zero current
write("hall_thermal.csv", hdr, Ih, kh * Ih + 5e-4 + rng.normal(0, 2e-5, Ih.size))
# follow-up: field forward, then reversed; the Hall part flips, the misalignment part doesn't
rev = np.r_[np.zeros(10), np.ones(10)]
Ir = np.r_[Ih, Ih]
Vr2 = np.where(rev == 0, kh, -kh) * Ir - 0.2 * kh * Ir + rng.normal(0, 2e-5, Ir.size)
write("hall_misaligned_reversed.csv", ["current_A", "hall_voltage_V", "field_reversed"], Ir, Vr2, rev)

# =============================================================================
# Quantum area. Its own generator, so the files above stay unchanged.
rng = np.random.default_rng(2028)
H, QE, CL, KB, ME = 6.62607015e-34, 1.602176634e-19, 299792458.0, 1.380649e-23, 9.1093837e-31
HBAR = H / (2 * np.pi)

# --- Planck's constant from LED turn-on voltages ------------------------------
lam_led = np.array([940, 850, 660, 590, 525, 470, 405], dtype=float)
von = lambda lam, gain=1.0: gain * (H * CL / (lam * 1e-9) / QE - 0.35)
hdr = ["wavelength_nm", "turn_on_voltage_V"]
write("planck_clean.csv", hdr, lam_led, von(lam_led) + rng.normal(0, 0.01, lam_led.size), fmt="{:.4g}")
# the lab voltmeter reads 8% low, so h comes out about 8% low
write("planck_gain.csv", hdr, lam_led, von(lam_led, 0.92) + rng.normal(0, 0.01, lam_led.size), fmt="{:.4g}")
# the LED labelled 590 nm (amber) is really a 640 nm red part
true_lam = lam_led.copy(); true_lam[3] = 640
write("planck_mislabelled.csv", hdr, lam_led, von(true_lam) + rng.normal(0, 0.01, lam_led.size), fmt="{:.4g}")
# follow-up: the 590 nm LED's turn-on read on the lab meter, then on a calibrated reference meter
groups("planck_gain_refmeter.csv", ["meter", "voltage_V"], float(von(590.0, 0.92)), float(von(590.0)), noise=0.003)

# --- Quantum tunnelling simulation: rectangular barrier, V0 = 1 eV, E = 0.5 eV ---
aw = np.round(np.arange(0.1, 1.21, 0.1), 2)
def t_exact(a_nm, E=0.5, V0=1.0, m=1.0):
    k = np.sqrt(2 * m * ME * (V0 - E) * QE) / HBAR
    return 1 / (1 + V0 ** 2 * np.sinh(k * a_nm * 1e-9) ** 2 / (4 * E * (V0 - E)))
noisy = lambda t: t * (1 + rng.normal(0, 0.01, t.size))
hdr = ["barrier_width_nm", "transmission"]
write("tunnel_clean.csv", hdr, aw, noisy(t_exact(aw)), fmt="{:.6g}")
# the simulation used half the electron mass (a units slip), so the decay is too slow
write("tunnel_mass.csv", hdr, aw, noisy(t_exact(aw, m=0.5)), fmt="{:.6g}")
# coarse grid: the solver can't resolve tiny transmissions and levels off near 2e-4
write("tunnel_floor.csv", hdr, aw, noisy(t_exact(aw) + 2e-4), fmt="{:.6g}")
# follow-up: the same (half-mass) simulation run at two energies
e2 = np.r_[np.full(aw.size, 0.5), np.full(aw.size, 0.25)]
a2 = np.r_[aw, aw]
write("tunnel_mass_two_energies.csv", ["barrier_width_nm", "transmission", "energy_eV"], a2,
      noisy(t_exact(a2, E=e2, m=0.5)), e2, fmt="{:.6g}")

# --- Blackbody radiation: tungsten lamp at 2800 K -----------------------------
lb = np.arange(400, 1701, 50, dtype=float)
def pl(lam_nm, T=2800.0):
    l = lam_nm * 1e-9
    return (1e-6 / l) ** 5 / np.expm1(H * CL / (l * KB * T))
norm = pl(np.linspace(400, 1700, 500)).max()
bb = lambda lam, T=2800.0, c=0.0: pl(lam, T) / norm + c
hdr = ["wavelength_nm", "spectral_radiance"]
write("blackbody_clean.csv", hdr, lb, bb(lb) + rng.normal(0, 0.005, lb.size), fmt="{:.5g}")
# spectrometer wavelength scale reads 30 nm long: every point is labelled 30 nm too far
write("blackbody_shift.csv", hdr, lb + 30, bb(lb) + rng.normal(0, 0.005, lb.size), fmt="{:.5g}")
# stray light / dark current adds a constant 4% background
write("blackbody_offset.csv", hdr, lb, bb(lb, c=0.04) + rng.normal(0, 0.005, lb.size), fmt="{:.5g}")
# follow-up: mercury calibration lamp lines, true against measured
hg = np.array([404.66, 435.83, 546.07, 576.96, 579.07])
write("blackbody_shift_hg.csv", ["reference_nm", "measured_nm"], hg, hg + 30 + rng.normal(0, 0.2, hg.size), fmt="{:.2f}")

# --- Atomic spectral lines: hydrogen Balmer series ------------------------------
RH = 1.0967758e7
nn = np.arange(3, 8, dtype=float)
balmer = lambda n, d=0.0: 1e9 / (RH * (0.25 - 1 / n ** 2)) + d
hdr = ["n_upper", "wavelength_nm"]
write("balmer_clean.csv", hdr, nn, balmer(nn) + rng.normal(0, 0.05, nn.size), fmt="{:.6g}")
# the line read as H-gamma (n = 5) is really the 435.8 nm mercury line from a contaminated tube
lines_imp = balmer(nn) + rng.normal(0, 0.05, nn.size); lines_imp[2] = 435.83
write("balmer_impurity.csv", hdr, nn, lines_imp, fmt="{:.6g}")
# spectrometer calibration off by +2 nm on every line
write("balmer_offset.csv", hdr, nn, balmer(nn, 2.0) + rng.normal(0, 0.05, nn.size), fmt="{:.6g}")
# follow-up: the same lines measured with a fresh hydrogen tube
write("balmer_fresh_tube.csv", hdr, nn, balmer(nn) + rng.normal(0, 0.05, nn.size), fmt="{:.6g}")

# =============================================================================
# Partner cases for the classical bench: the lower-prior fault of each ambiguous
# pair. In each demo above the first measurement can't tell two faults apart, and
# the truth is the more common one. Here the truth is the other fault, generated
# with the same physics so the first measurement looks the same; the follow-up is
# what that fault really gives in the recommended test. Own random stream, so
# every dataset above is unchanged.
rng = np.random.default_rng(4301)

# Helmholtz: the Hall probe was never zeroed and reads +30 µT on its own. The coil
# axis runs east-west, so Earth's field adds only 1.5 µT along it.
write("helmholtz_probe_zero.csv", ["current_A", "measured_B_uT"], I,
      k * I + 31.5 + rng.normal(0, 1.5, I.size))
# follow-up: at I = 0, in the open and then inside the zero-gauss chamber: the
# 30 µT stays in the chamber, so it is the probe, not the surroundings
groups("helmholtz_probe_zero_shield.csv", ["in_shield", "measured_B_uT"], 31.5, 30.0, noise=0.8)

# Laser: the beam's pointing drifts as the laser warms up. Scanned 5 minutes after
# switch-on, the centre is 0.40 mm out (200 µrad over the 2 m path).
write("laser_drift.csv", ["position_mm", "detector_voltage_mV"], x,
      g(0.40, 600) + rng.normal(0, 3, x.size), fmt="{:.2f}")
# follow-up: repeat scan 15 minutes later; the centre has moved on to 0.62 mm
write("laser_drift_rescan.csv", ["position_mm", "detector_voltage_mV"], x,
      g(0.62, 600) + rng.normal(0, 3, x.size), fmt="{:.2f}")

# Faraday: a 60 mV ground potential difference between the coil circuit and the
# recorder (their shared return also carries another rig's current)
hdr_f = ["time_s", "magnetic_flux_Wb", "induced_voltage_V"]
write("faraday_ground.csv", hdr_f, t, phi, emf + 0.06 + rng.normal(0, 0.02, t.size))
# follow-up: magnet still; the 60 mV is there with the coil connected and goes
# when the amplifier input is shorted to its own ground
groups("faraday_ground_shorted.csv", ["input_shorted", "voltage_V"], 0.06, 0.0, noise=0.01)

# Beer-Lambert: the photodiode amplifier reads 0.07 with no light on it at all
hdr_b = ["thickness_cm", "transmitted_intensity"]
write("beer_dark_offset.csv", hdr_b, d, bl(0.5, 0.07) + rng.normal(0, 0.004, d.size))
# follow-up: lid closed and room lights off; the dark level is still there
write("beer_dark_offset_enclosed.csv", hdr_b, d, bl(0.5, 0.07) + rng.normal(0, 0.004, d.size))

# Malus: a worn analyser with an extinction ratio of 14:1, I = (1 - 1/14) cos² + 1/14
er = 14.0
write("malus_leaky.csv", ["angle_deg", "intensity"], th,
      (1 - 1 / er) * ml() + 1 / er + rng.normal(0, 0.005, th.size))
# follow-up: crossed, the detector reads 0.071; with the laser blocked only the
# 0.001 dark level is left, so the light came through the polariser
groups("malus_leaky_blocked.csv", ["beam_blocked", "intensity"], 1 / er, 0.001, noise=0.003)

# Cooling: the thermocouple reads true, but the room is at 23.0 °C on a sunny
# afternoon, not the 21 °C assumed
hdr_c = ["time_s", "temperature_C"]
write("cooling_warm_room.csv", hdr_c, tc, 23.0 + (80 - 23.0) * np.exp(-tc / 420)
      + rng.normal(0, 0.15, tc.size), fmt="{:.2f}")
# follow-up: in ice water the thermocouple reads 0.05 °C, so the meter is right
write("cooling_warm_room_ice_point.csv", hdr_c, ti, 0.05 + rng.normal(0, 0.1, ti.size), fmt="{:.2f}")

# RC: the capacitor leaks through 3.3 MΩ (1 µA at 3.5 V, within an electrolytic's
# rating), measured with a x10 (10 MΩ) probe from the start. Leak and probe in
# parallel load the node: V_final and tau both scale by Rp / (R + Rp) = 0.71.
rp = 1 / (1 / 3.3e6 + 1 / 10e6)
fr = rp / (1e6 + rp)
write("rc_leaky_cap.csv", ["time_s", "voltage_V"], tr, rc(5 * fr, 0.1 * fr)
      + rng.normal(0, 0.01, tr.size), fmt="{:.4f}")
# follow-up: re-recorded with the probe and channel checked at x10: no change,
# because the load is inside the capacitor
write("rc_leaky_cap_x10.csv", ["time_s", "voltage_V"], tr, rc(5 * fr, 0.1 * fr)
      + rng.normal(0, 0.01, tr.size), fmt="{:.4f}")

# Pendulum: a worn mechanical stopwatch gaining 1.5% (4.5 s in five minutes);
# released from 5°, every period reads 1.5% long and g comes out 3% low
write("pendulum_timer.csv", ["length_m", "period_s"], Lp, per(Lp, 5, 1.015)
      + rng.normal(0, 0.003, Lp.size), fmt="{:.4f}")
# follow-up: three lengths from 4° with the same stopwatch: g is still 3% low
Ls = np.array([0.4, 0.8, 1.2])
write("pendulum_timer_small.csv", ["length_m", "period_s"], Ls, per(Ls, 4, 1.015)
      + rng.normal(0, 0.003, Ls.size), fmt="{:.4f}")

# Hooke: zero read with the hanger on (right), but off the hook rather than the
# pointer, 8 mm out: every extension is 8 mm long
hdr_h = ["mass_kg", "extension_m"]
write("hooke_ruler_zero.csv", hdr_h, mk, ext(mk + 0.05) - ext(0.05) + 0.008
      + rng.normal(0, 0.0005, mk.size), fmt="{:.5f}")
# follow-up: hanger weighed (50 g) and added to every load. The zero was taken
# with the hanger on, so this only moves the offset: the line still misses the origin
write("hooke_ruler_zero_hanger_included.csv", hdr_h, mk + 0.05, ext(mk + 0.05) - ext(0.05) + 0.008
      + rng.normal(0, 0.0005, mk.size), fmt="{:.5f}")

# Diode: an overheated diode with 5 Ω of bulk series resistance; the voltmeter is
# already on its own legs
hdr_d = ["voltage_V", "current_A"]
write("diode_bulk_rs.csv", hdr_d, vd(Id, 5.0) + rng.normal(0, 5e-4, Id.size), Id)
# follow-up: voltmeter leads checked on the diode's legs: the droop stays
write("diode_bulk_rs_rewired.csv", hdr_d, vd(Id, 5.0) + rng.normal(0, 5e-4, Id.size), Id)

# Grating: the green laser labelled 532 nm is a 515 nm direct diode, with the
# right 600 lines/mm grating: the wavelength comes out 3% short
write("grating_laser_label.csv", ["order_m", "angle_deg"], mo, ang(515)
      + rng.normal(0, 0.05, mo.size), fmt="{:.3f}")
# follow-up: HeNe (632.8 nm) on the same grating. Orders ±3 don't exist at
# 600 lines/mm (3 λ/d > 1), so orders −2 to +2 are read; the line comes out right.
m2 = np.arange(-2, 3, dtype=float)
th2 = np.degrees(np.arcsin(m2 * 632.8e-9 * 600e3)) + rng.normal(0, 0.05, m2.size)
(OUT / "grating_laser_label_ref.csv").write_text(
    "\n".join(["order_m,angle_deg,reference_nm"] + [f"{m:.6g},{a:.3f},632.8" for m, a in zip(m2, th2)]) + "\n")

# Hall: the electromagnet gives 0.255 T, not the 0.3 T set (its calibration chart
# is for a narrower pole gap), so the carrier density reads 16% high. A 1%
# misalignment voltage is always there and doesn't flip with the field.
kl = kh * 0.255 / 0.3
write("hall_field_low.csv", ["current_A", "hall_voltage_V"], Ih,
      kl * Ih + 0.01 * kh * Ih + rng.normal(0, 2e-5, Ih.size))
# follow-up: field forward, then reversed; the voltage flips almost exactly
Vf = np.where(rev == 0, kl, -kl) * Ir + 0.01 * kh * Ir + rng.normal(0, 2e-5, Ir.size)
write("hall_field_low_reversed.csv", ["current_A", "hall_voltage_V", "field_reversed"], Ir, Vf, rev)

# --- Quantum partner cases: the lower-prior fault of each ambiguous pair ---------
# Each demo's first measurement can't tell two faults apart, and its truth is the
# more common one. Here the truth is the other, at a realistic size, with the
# recommended test's result as that fault really produces it. Own random stream,
# so every dataset above is unchanged.
rng = np.random.default_rng(4401)

# Planck: the meter is fine, but the wavelengths are the datasheet's. Measured on a
# fibre spectrometer these LEDs peak 7-18 nm away from their labels, the blue and
# violet ones long and the infrared ones short, which flattens the line by ~9%.
q_nominal = np.array([940, 850, 660, 590, 525, 470, 405], dtype=float)
q_peak = np.array([922, 843, 670, 603, 541, 487, 423], dtype=float)
write("planck_partner.csv", ["wavelength_nm", "turn_on_voltage_V"], q_nominal,
      von(q_peak) + rng.normal(0, 0.01, q_nominal.size), fmt="{:.4g}")
# follow-up: the "590 nm" LED (peak 603 nm) on the lab meter, then a reference meter: they agree
groups("planck_partner_refmeter.csv", ["meter", "voltage_V"], float(von(603.0)), float(von(603.0)), noise=0.003)

# Tunnelling: the mass is right, but the barrier height was entered from the wrong zero
# (0.13 eV above the leads' band edge), so the simulated barrier is 0.87 eV, not 1 eV
q_aw = np.round(np.arange(0.1, 1.21, 0.1), 2)
write("tunnel_partner.csv", ["barrier_width_nm", "transmission"], q_aw,
      noisy(t_exact(q_aw, V0=0.87)), fmt="{:.6g}")
# follow-up: the same simulation at 0.5 and 0.25 eV; V0 - E changes by a different ratio
q_e = np.r_[np.full(q_aw.size, 0.5), np.full(q_aw.size, 0.25)]
q_a = np.r_[q_aw, q_aw]
write("tunnel_partner_two_energies.csv", ["barrier_width_nm", "transmission", "energy_eV"], q_a,
      noisy(t_exact(q_a, E=q_e, V0=0.87)), q_e, fmt="{:.6g}")

# Blackbody: the wavelength scale is right, but the 12 V lamp is run at 11.0 V; a
# tungsten filament's temperature goes roughly as V^0.42, so it sits near 2700 K
q_lb = np.arange(400, 1701, 50, dtype=float)
write("blackbody_partner.csv", ["wavelength_nm", "spectral_radiance"], q_lb,
      bb(q_lb, T=2700.0) + rng.normal(0, 0.005, q_lb.size), fmt="{:.5g}")
# follow-up: mercury calibration lines land where they should
q_hg = np.array([404.66, 435.83, 546.07, 576.96, 579.07])
write("blackbody_partner_hg.csv", ["reference_nm", "measured_nm"], q_hg,
      q_hg + rng.normal(0, 0.2, q_hg.size), fmt="{:.2f}")

# Balmer: the tube is clean, but the faint 397.0 nm line (n = 7) was missed and the next
# one, 388.9 nm (n = 8), was logged as n = 7
q_n = np.arange(3, 8, dtype=float)
q_true_n = np.array([3, 4, 5, 6, 8], dtype=float)
write("balmer_partner.csv", ["n_upper", "wavelength_nm"], q_n,
      balmer(q_true_n) + rng.normal(0, 0.05, q_n.size), fmt="{:.6g}")
# follow-up: a fresh hydrogen tube shows the same lines, so the odd one stays put
write("balmer_partner_fresh_tube.csv", ["n_upper", "wavelength_nm"], q_n,
      balmer(q_true_n) + rng.normal(0, 0.05, q_n.size), fmt="{:.6g}")

print(f"Wrote bench datasets to {OUT}")
