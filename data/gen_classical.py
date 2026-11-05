"""Seeded datasets for the classic lab experiments: Michelson interferometer,
speed of sound in a resonance tube, Stefan-Boltzmann law with a lamp filament.
Ground truth is in data/manifest.d/classical.yaml.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

OUT = Path(__file__).parent / "seeded"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(2031)


def write(name, header, *cols, fmt="{:.6g}"):
    fmts = fmt if isinstance(fmt, (list, tuple)) else [fmt] * len(cols)
    lines = [",".join(header)]
    for row in zip(*cols):
        lines.append(",".join(f.format(v) for f, v in zip(fmts, row)))
    (OUT / name).write_text("\n".join(lines) + "\n")


# --- Michelson interferometer: HeNe laser, 632.8 nm ---------------------------
# Mirror travel read on a micrometer through a 10:1 lever (readable to ~0.2 µm),
# noted every 20 fringes counted past the reference mark, 300 fringes in all.
LAM = 0.6328                                       # µm
nf = np.arange(0, 301, 20, dtype=float)
read = lambda d: d + rng.normal(0, 0.1, d.size)
hdr = ["micrometer_um", "fringes"]
fm = ["{:.2f}", "{:.0f}"]
write("michelson_clean.csv", hdr, read(nf * LAM / 2), nf, fmt=fm)
# turning too fast: about one fringe in 30 slips past uncounted, so each reading
# belongs to more fringes than were written down
missed = np.r_[0, np.cumsum(rng.poisson(0.7, nf.size - 1))]
write("michelson_miscount.csv", hdr, read((nf + missed) * LAM / 2), nf, fmt=fm)
# follow-up: the same travel counted again, slowly, turning one way only
write("michelson_miscount_recount.csv", hdr + ["recount"], read(nf * LAM / 2), nf, np.ones(nf.size),
      fmt=fm + ["{:.0f}"])
# backlash: the knob was wound back to the start, so the first ~2.5 µm of turning
# took up slack in the screw and lever before the mirror moved at all
dead = np.where(nf > 0, 2.5, 0.0)
write("michelson_backlash.csv", hdr, read(nf * LAM / 2 + dead), nf, fmt=fm)

# --- Speed of sound: closed tube over water, signal generator and small speaker ---
# First resonance L1 = v/4f − e, end correction e ≈ 0.3 D, D = 35 mm.
D = 0.035
# frequencies chosen evenly spaced in period, 1 to 5 ms
fr = np.array([200, 222, 250, 286, 333, 400, 500, 571, 667, 800, 1000], dtype=float)
vs = lambda t_c: 331.3 * np.sqrt(1 + t_c / 273.15)
L1 = lambda t_c=20.0, zero=0.0: vs(t_c) / (4 * fr) - 0.3 * D + zero + rng.normal(0, 0.0007, fr.size)
hdr = ["frequency_Hz", "resonance_length_m"]
fm = ["{:.0f}", "{:.4f}"]
write("sound_clean.csv", hdr, fr, L1(), fmt=fm)
# the reservoir was filled from the hot tap: the air column is at about 34 °C, not the 20 °C entered
write("sound_warm_air.csv", hdr, fr, L1(34.0), fmt=fm)
# follow-up: a thermometer lowered down the tube, warmest just above the water
depth = np.arange(5, 66, 10, dtype=float)
write("sound_warm_air_thermometer.csv", ["depth_cm", "air_temperature_C"], depth,
      31.0 + 6.0 * depth / 65 + rng.normal(0, 0.2, depth.size), fmt=["{:.0f}", "{:.1f}"])
# lengths read from the metre rule's zero, which sits 2.5 cm above the tube's open end
write("sound_ruler_zero.csv", hdr, fr, L1(zero=0.025), fmt=fm)

# --- Stefan-Boltzmann: 12 V 6 W vacuum lamp, tungsten filament ------------------
# Electrical power in = radiation a·T⁴ + conduction G·(T − T₀) through the supports.
R0, T0, BETA = 1.60, 293.0, 1.209
Rt = lambda T: R0 * (T / T0) ** BETA
A_RAD = 6.0 / 2700.0 ** 4


def filament_T(v, g):
    return brentq(lambda T: v ** 2 / Rt(T) - A_RAD * T ** 4 - g * (T - T0), T0 + 1e-6, 5000)


volts = np.array([1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)


def lamp(g=4e-6, i_offset=0.0, r_series=0.0):
    T = np.array([filament_T(v, g) for v in volts])
    i = volts / Rt(T)
    v_read = volts + i * r_series                 # r_series: voltage read at the supply, not the lamp
    noisy = lambda a: a * (1 + rng.normal(0, 0.002, a.size))
    return noisy(v_read), noisy(i) + i_offset


hdr = ["voltage_V", "current_A"]
fm = ["{:.4f}", "{:.5f}"]
write("stefan_clean.csv", hdr, *lamp(), fmt=fm)
# the ammeter's zero is set wrong: it reads 12 mA with no current flowing
write("stefan_ammeter_zero.csv", hdr, *lamp(i_offset=0.012), fmt=fm)
# follow-up: ten readings of the same ammeter with the lamp circuit open
write("stefan_ammeter_open.csv", ["reading", "open_circuit_A"], np.arange(1, 11),
      0.012 + rng.normal(0, 0.0003, 10), fmt=["{:.0f}", "{:.4f}"])
# voltmeter across the supply terminals: the leads and ammeter (1.5 Ω) count as lamp
write("stefan_supply_voltage.csv", hdr, *lamp(r_series=1.5), fmt=fm)

print(f"Wrote classical datasets to {OUT}")
