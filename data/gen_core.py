"""Seeded datasets for the core undergraduate experiments: photoelectric effect,
Franck–Hertz (mercury) and radioactive decay with a Geiger–Müller counter.

Each experiment gets a healthy run, the demo fault, a second fault and the
follow-up a researcher would record after the recommended test.
Ground truth is in data/manifest.d/core.yaml. Run via data/generate.py.
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "seeded"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(2031)
H, QE, CL = 6.62607015e-34, 1.602176634e-19, 299792458.0


def write(name, header, *cols, fmt="{:.6g}"):
    lines = [",".join(header)]
    for row in zip(*cols):
        lines.append(",".join(fmt.format(v) for v in row))
    (OUT / name).write_text("\n".join(lines) + "\n")


# --- Photoelectric effect: mercury lines through filters, phototube with φ = 1.80 eV ----
hg = np.array([365.02, 404.66, 435.83, 546.07, 576.96])     # UV, violet, blue, green, yellow
ephot = lambda lam: H * CL / (lam * 1e-9) / QE             # photon energy in eV
vs = lambda lam: ephot(lam) - 1.80
read = lambda v: v + rng.normal(0, 0.006, np.size(v))       # meter and reading scatter, 6 mV
hdr = ["wavelength_nm", "stopping_voltage_V"]
write("photo_clean.csv", hdr, hg, read(vs(hg)), fmt="{:.4f}")
# Light reaching the anode (alkali from the cathode has coated it): a reverse
# photocurrent that grows with photon energy pulls every stopping voltage down,
# the highest frequencies most, so the line stays straight but comes out ~12% too shallow.
anode = lambda lam, beta=0.12: vs(lam) - beta * (ephot(lam) - 2.0)
write("photo_anode.csv", hdr, hg, read(anode(hg)), fmt="{:.4f}")
# The yellow filter leaks the green 546 nm line; the stopping voltage is set by the
# most energetic photons present, so the "yellow" point reads at the green line's value.
leak = vs(hg)
leak[4] = vs(546.07) - 0.01
write("photo_filter_leak.csv", hdr, hg, read(leak), fmt="{:.4f}")
# Follow-up: every line read again as it was, then with a mask so only the cathode is lit.
g = np.r_[np.zeros(5), np.ones(5)]
write("photo_anode_masked.csv", hdr + ["anode_masked"], np.r_[hg, hg],
      np.r_[read(anode(hg)), read(anode(hg, beta=0.01))], g, fmt="{:.4f}")

# --- Franck–Hertz, mercury at 180 °C: peaks every 4.9 V, contact potential 2.1 V ---
V = np.round(np.arange(0, 30.01, 0.1), 2)
E1, VC = 4.9, 2.1


def fh_current(m, gain=1.0, clip=None):
    """Anode current: a rising envelope with a dip after every 4.9 eV of energy gained."""
    fine = np.arange(-2, 32, 0.01)
    ph = ((fine - VC) / E1) % 1.0
    h = np.where(ph < 0.18, 0.5 * (1 + np.cos(np.pi * ph / 0.18)), np.clip((ph - 0.18) / 0.82, 0, None) ** 1.4)
    h[fine < VC + 0.18 * E1] = 0.0
    k = np.exp(-0.5 * (np.arange(-60, 61) * 0.01 / 0.25) ** 2)   # 0.25 V thermal energy spread
    h = np.convolve(h, k / k.sum(), mode="same")
    env = 40 * gain * np.clip((fine - VC) / 25, 0, None) ** 1.5
    i_true = np.interp(V, fine, env * (1 - m + m * h))
    i = i_true * (1 + rng.normal(0, 0.004, V.size)) + rng.normal(0, 0.02, V.size) + 0.05
    return np.minimum(i, clip) if clip else i


hdr = ["accelerating_voltage_V", "anode_current_nA"]
write("fh_clean.csv", hdr, V, fh_current(0.62), fmt="{:.4f}")
# Oven at about 145 °C instead of 180 °C: too little mercury vapour, so fewer electrons
# lose their energy in collisions; the dips are shallow and the current is higher.
write("fh_cool_oven.csv", hdr, V, fh_current(0.30, gain=1.5), fmt="{:.4f}")
# Filament turned up too far: the current amplifier hits its 50 nA limit on the upper peaks.
write("fh_saturated.csv", hdr, V, fh_current(0.62, gain=2.2, clip=50.0), fmt="{:.4f}")
# Follow-up: the oven thermometer read once a minute for six minutes.
mins = np.arange(0, 6, 1.0)
write("fh_oven_thermometer.csv", ["minute", "oven_temperature_C"], mins,
      145 + 0.6 * mins + rng.normal(0, 0.4, mins.size), fmt="{:.1f}")

# --- Radioactive decay: Ba-137m from a Cs-137 generator, GM tube with 150 µs dead time ---
T_HALF, TAU, BKG, DT = 153.1, 150e-6, 0.8, 10.0
t = np.arange(0, 1500, DT)


def gm_counts(n0, t=t):
    """Counts per 10 s interval: source plus room background, through the counter's dead time."""
    n = n0 * np.exp(-np.log(2) / T_HALF * (t + DT / 2)) + BKG
    return rng.poisson(n / (1 + n * TAU) * DT).astype(float)


bkg_sub = round(BKG * DT)                                 # background measured beforehand: 8 per interval
hdr = ["time_s", "counts"]
write("decay_clean.csv", hdr, t, gm_counts(120) - bkg_sub, fmt="{:.0f}")
# Background never subtracted: the tail levels off at the room background.
raw = gm_counts(120)
write("decay_background.csv", hdr, t, raw, fmt="{:.0f}")
# Source right against the tube window: 2500 counts per second at the start, and the
# counter's dead time loses about a quarter of them, so the early decay looks too slow.
write("decay_dead_time.csv", hdr, t, gm_counts(2500) - bkg_sub, fmt="{:.0f}")
# Follow-up: the same log with the eluate then removed and the room background counted for 3 minutes.
tb = np.arange(1500, 1680, DT)
write("decay_background_removed.csv", hdr + ["source_removed"], np.r_[t, tb],
      np.r_[raw, rng.poisson(BKG * DT, tb.size).astype(float)], np.r_[np.zeros(t.size), np.ones(tb.size)],
      fmt="{:.0f}")

print(f"Wrote core-lab datasets to {OUT}")
