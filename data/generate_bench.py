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

print(f"Wrote bench datasets to {OUT}")
