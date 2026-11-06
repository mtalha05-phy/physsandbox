"""Seeded datasets for the mechanics benches: Kater's reversible pendulum, the
Cavendish torsion balance, a driven damped oscillator, Boyle's law in a syringe.
Ground truth is in data/manifest.d/mechanics.yaml.
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "seeded"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7302)


def write(name, header, *cols, fmt="{:.6g}"):
    fmts = fmt if isinstance(fmt, (list, tuple)) else [fmt] * len(cols)
    lines = [",".join(header)]
    for row in zip(*cols):
        lines.append(",".join(f.format(v) for f, v in zip(fmts, row)))
    (OUT / name).write_text("\n".join(lines) + "\n")


# --- Kater's reversible pendulum --------------------------------------------------
# Steel bar 1.40 m long (1.5 kg), knife edges 995.0 mm apart; a 2.5 kg brass bob
# beyond edge 2, a light wooden bob beyond edge 1 to balance the air drag, and a
# 1.25 kg slider between the edges. Positions in metres from knife edge 1.
G_LOCAL = 9.8119
L_KATER = 0.9950
BODIES = [(1.50, 0.50, 1.40 ** 2 / 12),          # (mass kg, centre m, I about own centre / mass, m²)
          (2.50, 1.12, 0.0015),
          (0.10, -0.08, 0.0015)]


def kater_periods(slider, amplitude_deg):
    ms = np.array([b[0] for b in BODIES] + [1.25])
    xs = np.array([b[1] for b in BODIES] + [slider])
    k2 = np.array([b[2] for b in BODIES] + [0.0008])
    M, xc = ms.sum(), (ms * xs).sum() / ms.sum()
    th = np.radians(amplitude_deg)
    big = 1 + th ** 2 / 16 + 11 * th ** 4 / 3072
    out = []
    for pivot in (0.0, L_KATER):
        I = (ms * (k2 + (xs - pivot) ** 2)).sum()
        out.append(2 * np.pi * np.sqrt(I / (M * G_LOCAL * abs(xc - pivot))) * big)
    return out


slider_cm = np.arange(68.0, 80.01, 1.5)
hdr = ["knife_edge_mass_position_cm", "period_1_s", "period_2_s"]
fm = ["{:.1f}", "{:.4f}", "{:.4f}"]


def kater_run(amplitude=2.0, noise_2=0.001, miscount=None):
    """Each period is 50 swings on a stopwatch, good to about 1 ms."""
    T = np.array([kater_periods(x / 100, amplitude) for x in slider_cm])
    t1 = T[:, 0] + rng.normal(0, 0.001, slider_cm.size)
    t2 = T[:, 1] + rng.normal(0, noise_2, slider_cm.size)
    if miscount is not None:
        # 49 swings counted as 50 at one position: that period reads 49/50 of its true value
        t1[miscount] *= 49 / 50
    return slider_cm, t1, t2


write("kater_clean.csv", hdr, *kater_run(), fmt=fm)
# released from about 15° (the run was entered as 2°): every period is 0.42% long
write("kater_amplitude.csv", hdr, *kater_run(amplitude=15.0), fmt=fm)
# follow-up: the swing amplitude read on a protractor behind the bar's lower end
write("kater_amplitude_protractor.csv", ["reading", "amplitude_deg"], np.arange(1, 6),
      15.0 + rng.normal(0, 0.6, 5), fmt=["{:.0f}", "{:.1f}"])
# one position timed over 49 swings but divided by 50
write("kater_miscount.csv", hdr, *kater_run(miscount=3), fmt=fm)

# --- Cavendish torsion balance -------------------------------------------------------
# 1.5 kg lead spheres, 15 g small spheres on arms d = 50.0 mm, centre distance
# r = 46.5 mm with the spheres against the case; laser spot on a scale D = 4.20 m away.
G_TRUE = 6.6743e-11
M_BIG, ARM, R_CEN = 1.500, 0.050, 0.0465
T_FREE, TAU = 540.0, 1500.0                      # undamped period, amplitude decay time (s)


def spot_shift_mm(D, r=R_CEN):
    beta = r ** 3 / (r ** 2 + 4 * ARM ** 2) ** 1.5
    theta = G_TRUE * M_BIG * T_FREE ** 2 * (1 - beta) / (2 * np.pi ** 2 * (2 * ARM) * r ** 2)
    return 4 * theta * D * 1000


t_cav = np.arange(0, 3001, 20.0)


def cavendish_run(D=4.20, r=R_CEN, drift=0.0, air=0.0, start=312.0):
    wd = np.sqrt((2 * np.pi / T_FREE) ** 2 - 1 / TAU ** 2)
    dS = spot_shift_mm(D, r)
    S = start + dS - dS * np.exp(-t_cav / TAU) * (np.cos(wd * t_cav) + np.sin(wd * t_cav) / (wd * TAU))
    S = S + drift * t_cav + rng.normal(0, 0.9, t_cav.size)
    if air:
        # draughts: a jittering spot, correlated over a minute or so
        w = np.zeros(t_cav.size)
        for i in range(1, w.size):
            w[i] = 0.6 * w[i - 1] + rng.normal(0, air)
        S = S + w
    return t_cav, np.round(2 * S) / 2           # read to half a millimetre


hdr = ["time_s", "beam_position_mm"]
fm = ["{:.0f}", "{:.1f}"]
write("cavendish_clean.csv", hdr, *cavendish_run(), fmt=fm)
# the balance now stands 3.86 m from the scale; 4.20 m was copied from the lab sheet
write("cavendish_scale_distance.csv", hdr, *cavendish_run(D=3.86), fmt=fm)
# follow-up: a steel tape from the mirror to the scale, five times
write("cavendish_scale_distance_tape.csv", ["reading", "mirror_to_scale_m"], np.arange(1, 6),
      3.860 + rng.normal(0, 0.002, 5), fmt=["{:.0f}", "{:.3f}"])
# the room warming after the heating came on: the equilibrium creeps 0.5 mm a minute
write("cavendish_drift.csv", hdr, *cavendish_run(drift=0.5 / 60), fmt=fm)

# --- Driven damped oscillator ------------------------------------------------------
# 200 g mass on a spring, driven at the top by a motor and crank (3.0 mm throw); an
# aluminium vane between the poles of an electromagnet brakes it, γ = 0.60 /s at 0.40 A.
F0, GAMMA, THROW = 0.800, 0.60, 3.0
dial = np.array([0.50, 0.60, 0.66, 0.70, 0.73, 0.75, 0.77, 0.79, 0.81, 0.83, 0.85, 0.88, 0.92, 0.98, 1.05, 1.15])


def steady(f, f0=F0, gamma=GAMMA):
    return THROW * f0 ** 2 / np.sqrt((f0 ** 2 - f ** 2) ** 2 + (gamma / (2 * np.pi) * f) ** 2)


def osc_run(dial_ratio=1.0, f0=F0, gamma=GAMMA, stop=None):
    """Amplitude read on a mirror scale behind the mass, to half a millimetre."""
    A = steady(dial / dial_ratio, f0, gamma)
    if stop is not None:
        A = np.minimum(A, stop)
    return dial, np.round(2 * (A + rng.normal(0, 0.3, dial.size))) / 2


hdr = ["drive_frequency_Hz", "amplitude_mm"]
fm = ["{:.2f}", "{:.1f}"]
write("oscillator_clean.csv", hdr, *osc_run(), fmt=fm)
# frequencies taken from the motor's dial, which reads 4% low
write("oscillator_dial.csv", hdr, *osc_run(dial_ratio=0.96), fmt=fm)
# follow-up: 20 crank turns timed with a stopwatch at three dial settings
dset = np.array([0.60, 0.80, 1.00])
write("oscillator_dial_stopwatch.csv", ["dial_frequency_Hz", "timed_cycles", "stopwatch_s"], dset,
      np.full(3, 20), 20 * 0.96 / dset + rng.normal(0, 0.08, 3), fmt=["{:.2f}", "{:.0f}", "{:.2f}"])
# the vane hits the magnet's pole piece at 16 mm, so the top of the curve is cut off
write("oscillator_end_stop.csv", hdr, *osc_run(stop=16.0), fmt=fm)
# brake magnet at 0.52 A, not the 0.40 A its decay was measured at: γ = 0.05 + 3.44 I²
write("oscillator_brake.csv", hdr, *osc_run(gamma=0.05 + 0.55 * (0.52 / 0.40) ** 2), fmt=fm)

# --- Boyle's law ---------------------------------------------------------------------
# 60 ml plastic syringe on an absolute pressure sensor (0.5 ml inside the sensor),
# connected with the plunger at 60 ml, then pushed in 4 ml steps to 20 ml.
P_ATM = 101.3
vol = np.arange(60.0, 19.9, -4.0)


def boyle_run(dead=0.5, offset=0.0):
    nrt = P_ATM * (vol[0] + dead)
    v_true = vol + rng.normal(0, 0.25, vol.size)        # plunger set to each mark by eye
    p = nrt / (v_true + dead) + rng.normal(0, 0.15, vol.size) + offset
    p[0] = P_ATM + offset + rng.normal(0, 0.15)        # sealed at room pressure
    return vol, p


hdr = ["volume_ml", "pressure_kPa"]
fm = ["{:.0f}", "{:.1f}"]
write("boyle_clean.csv", hdr, *boyle_run(), fmt=fm)
# 40 cm of 3.0 mm bore tubing between syringe and sensor, not in the dead volume entered
TUBE = np.pi * 3.0 ** 2 / 4 * 400 / 1000
write("boyle_dead_volume.csv", hdr, *boyle_run(dead=0.5 + TUBE), fmt=fm)
# follow-up: the tube's length on a rule and its bore with callipers, four times
write("boyle_dead_volume_tube.csv", ["tube_length_mm", "tube_bore_mm"], 400 + rng.normal(0, 1.0, 4),
      3.00 + rng.normal(0, 0.02, 4), fmt=["{:.0f}", "{:.2f}"])
# the sensor reads 3.5 kPa high (zero drift since its last calibration)
write("boyle_sensor_offset.csv", hdr, *boyle_run(offset=3.5), fmt=fm)

print(f"Wrote mechanics datasets to {OUT}")
