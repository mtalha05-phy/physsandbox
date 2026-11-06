"""Seeded datasets for the advanced-lab experiments (gamma attenuation, muon lifetime,
Faraday rotation, magnetic hysteresis).

Run by data/generate.py after the other generators. Ground truth is in
data/manifest.d/advanced.yaml.
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "seeded"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(6620)


def write(name, header, *cols, fmt="{:.6g}"):
    lines = [",".join(header)]
    for row in zip(*cols):
        lines.append(",".join(fmt.format(v) for v in row))
    (OUT / name).write_text("\n".join(lines) + "\n")


# --- Gamma-ray attenuation in lead: Cs-137 662 keV, end-window GM tube with a 100 µs dead time -------
# μ/ρ = 0.110 cm²/g (narrow beam), ρ = 11.34 g/cm³, so μ = 0.1247 /mm and the half-value layer is 5.6 mm.
MU = 0.110 * 11.34 / 10
X = np.arange(0.0, 51.0, 5.0)                     # lead added in 5 mm sheets
T, R0, BG, TAU_GM = 300.0, 200.0, 0.5, 100e-6     # 300 s counts; 200 /s unattenuated; background 0.5 /s


def gamma(r0=R0, bg=BG, extra=0.0):
    n = r0 * np.exp(-MU * X) + bg + extra
    return rng.poisson(n / (1 + n * TAU_GM) * T)  # non-paralysable dead time: 2% lost at 200 /s


hdr = ["thickness_mm", "counts"]
write("gamma_clean.csv", hdr, X, gamma(), fmt="{:.0f}")
# background counted before the next bench got its Cs-137 source out: during the run it is 2.0 /s,
# but only the 0.5 /s measured earlier is subtracted, so 1.5 /s is left under every count
write("gamma_background.csv", hdr, X, gamma(bg=2.0), fmt="{:.0f}")
# source moved up against the tube's window for more counts: at 3000 /s the 100 µs dead time loses 23%
# of the counts with no lead in, falling below 1% behind 30 mm
write("gamma_dead_time.csv", hdr, X, gamma(r0=3000.0), fmt="{:.0f}")
# follow-up: five 60 s counts with the source back in its lead pot, the bench otherwise as in the run
write("gamma_background_source_out.csv", ["count_time_s", "counts_source_removed"], np.full(5, 60.0),
      rng.poisson(2.0 * 60, 5), fmt="{:.0f}")

# --- Muon lifetime: stopping cosmic muons in a plastic scintillator, start-stop timer, 0–20 µs ---------
# μ⁺ decay with the free lifetime; μ⁻ in carbon live 2.026 µs (nuclear capture), and capture takes
# 7.8% of them with no decay electron. With μ⁺/μ⁻ = 1.27 a single exponential fitted from 1 µs gives
# τ = 2.12 µs. 20 000 decays is about a day and a half of counting.
EDGES = np.arange(0.0, 20.001, 0.5)
TC = (EDGES[:-1] + EDGES[1:]) / 2
W_PLUS = 1.27 / (1.27 + 0.922)
N_DECAY = 20000


def muon(n=N_DECAY, bg=12.0, clock=1.0):
    plus = rng.random(n) < W_PLUS
    t = rng.exponential(np.where(plus, 2.1969811, 2.026))
    h = np.histogram(t * clock, EDGES)[0]
    return h + rng.poisson(bg / clock, TC.size)


hdr = ["time_us", "counts"]
write("muon_clean.csv", hdr, TC, muon(), fmt="{:.4g}")
# light leak in the scintillator's wrapping: room light adds random pulses, the singles rate goes from
# 15 /s to about 48 /s, and random stops (accidentals, which go as the square of it) rise tenfold
write("muon_light_leak.csv", hdr, TC, muon(bg=120.0), fmt="{:.4g}")
# the timer board's clock runs 10% faster than the frequency its software assumes (a replacement
# oscillator), so it counts 10% more ticks in every interval and logs each one 10% long
write("muon_tdc_clock.csv", hdr, TC, muon(clock=1.10), fmt="{:.4g}")
# follow-up: ten 10 s singles readings with the room lights on, then ten with the detector under a black cloth
cov = np.r_[np.zeros(10), np.ones(10)]
rate = np.r_[rng.poisson(48.0 * 10, 10), rng.poisson(15.0 * 10, 10)] / 10
write("muon_light_leak_covered.csv", ["covered", "singles_rate_Hz"], cov, rate, fmt="{:.4g}")

# --- Faraday rotation: SF-57 rod, 100 mm long, in a 200 mm solenoid of 20 mm bore radius ---------------
# V = 23 rad/(T·m); field read on a Hall probe at the coil centre, −60 to +60 mT
V, L_ROD, L_COIL, A_COIL = 23.0, 100.0, 200.0, 20.0


def bz(z):
    """On-axis field of the solenoid, relative to its centre."""
    f = lambda z: 0.5 * ((z + L_COIL / 2) / np.hypot(z + L_COIL / 2, A_COIL)
                         - (z - L_COIL / 2) / np.hypot(z - L_COIL / 2, A_COIL))
    return f(z) / f(0.0)


def rod_average(shift):
    z = np.linspace(-L_ROD / 2 + shift, L_ROD / 2 + shift, 2001)
    return float(np.trapezoid(bz(z), z) / L_ROD)


BF = np.arange(-60.0, 60.1, 10.0)


def faraday_rot(shift=0.0, zero=0.0, probe_gain=1.0):
    true_b = BF + rng.normal(0, 0.3, BF.size)        # the supply sets the field only roughly
    th = np.degrees(V * L_ROD * 1e-3 * true_b * 1e-3 * rod_average(shift)) + zero
    return true_b * probe_gain + rng.normal(0, 0.1, BF.size), th + rng.normal(0, 0.04, BF.size)


hdr = ["field_mT", "rotation_deg"]
write("faraday_rot_clean.csv", hdr, *faraday_rot(), fmt="{:.4g}")
# rod slid 55 mm along the bore when the laser mount was adjusted: its far end now sits just outside the
# coil, so the field it sees averages 11% below the field at the probe
write("faraday_rot_nonuniform.csv", hdr, *faraday_rot(shift=55.0), fmt="{:.4g}")
# analyser zero read 0.5° off at the start (extinction taken on the wrong side of the minimum)
write("faraday_rot_zero_offset.csv", hdr, *faraday_rot(zero=0.5), fmt="{:.4g}")
# follow-up: field mapped along the rod at the run's top current (60 mT at the probe), probe in 10 mm steps
pos = np.arange(0.0, 100.1, 10.0)
write("faraday_rot_nonuniform_map.csv", ["position_mm", "field_mT"], pos,
      60.0 * bz(pos - L_ROD / 2 + 55.0) + rng.normal(0, 0.1, pos.size), fmt="{:.4g}")

# --- Magnetic hysteresis: silicon-steel toroid, quasi-static loop, fluxmeter on a search coil ---------
# B = Bs·tanh((H ± Hc)/w) + χH with Bs = 1.30 T, Hc = 45 A/m, w = 60 A/m, χ = 1.5e-4 T/(A/m):
# remanence 0.83 T, B = 1.42 T at the loop tip (800 A/m). Search coil 100 turns, core 1.20 cm².
BS, HC, WH, CHI = 1.30, 45.0, 60.0, 1.5e-4
MAG = np.array([800, 650, 500, 400, 300, 220, 160, 120, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0.0])
H_DOWN = np.r_[MAG, -MAG[::-1][1:]]
H_LOOP = np.r_[H_DOWN, -H_DOWN[1:]]               # +800 → −800 → +800 A/m
DOWN = np.r_[np.ones(H_DOWN.size), np.zeros(H_DOWN.size - 1)].astype(bool)


def loop(scale=1.0, drift=0.0):
    h = H_LOOP + rng.normal(0, 1.0, H_LOOP.size)     # the supply sets the current only roughly
    b = np.where(DOWN, BS * np.tanh((h + HC) / WH), BS * np.tanh((h - HC) / WH)) + CHI * h
    k = np.arange(H_LOOP.size) / (H_LOOP.size - 1)  # fraction of the run's time
    return (h + rng.normal(0, 1.5, H_LOOP.size),
            scale * b + drift * k + rng.normal(0, 0.004, H_LOOP.size))


hdr = ["H_A_per_m", "B_T"]
write("hysteresis_clean.csv", hdr, *loop(), fmt="{:.4g}")
# search coil has 88 turns, not the 100 on its label (it was rewound), so every B comes out 12% low
write("hysteresis_turns.csv", hdr, *loop(scale=0.88), fmt="{:.4g}")
# fluxmeter integrator drifting: 0.06 T of creep over the eight minutes the loop took
write("hysteresis_drift.csv", hdr, *loop(drift=0.06), fmt="{:.4g}")
# follow-up: five current reversals read on the search coil, then five on a freshly wound 10-turn coil
# (flux-linkage change 2·N·B·A at the 1.42 T tip, 1.20 cm² core)
dphi = lambda n: 2 * n * 1.42 * 1.20e-4 * 1e3     # mWb-turns
g = np.r_[np.zeros(5), np.ones(5)]
fl = np.r_[dphi(88) + rng.normal(0, 0.02, 5), dphi(10) + rng.normal(0, 0.02, 5)]
write("hysteresis_turns_reference.csv", ["coil", "flux_mWb"], g, fl, fmt="{:.4g}")

# --- Partner cases: the lower-prior fault of each ambiguous pair ----------------------
# Each first measurement above that has a follow-up can't tell its fault from a second one
# that moves the result the same way, so the data alone favour the more common one. Here
# the truth is that other fault, and the recommended test has to find it.
# Own random stream, so every dataset above is unchanged.
rng = np.random.default_rng(4504)
# Gamma: the background stays at the 0.5 /s subtracted, but with no collimator or tube shield
# gamma rays scattered off the bench and source holder reach the tube round the lead: 2.0 /s,
# 1% of the beam, behind every thickness
write("gamma_partner.csv", ["thickness_mm", "counts"], X, gamma(extra=2.0), fmt="{:.0f}")
# follow-up: five 60 s counts with the source back in its lead pot, the bench otherwise as in the run
write("gamma_partner_gamma_source_out_count.csv", ["count_time_s", "counts_source_removed"], np.full(5, 60.0),
      rng.poisson(BG * 60, 5), fmt="{:.0f}")
# Muon: the wrapping is light-tight, but the discriminator threshold sits in the photomultiplier's
# noise: singles rise from 15 /s to 45 /s whether the room is lit or not, and accidentals ninefold
write("muon_partner.csv", ["time_us", "counts"], TC, muon(bg=108.0), fmt="{:.4g}")
# follow-up: ten 10 s singles readings with the room lights on, then ten under a black cloth
write("muon_partner_muon_dark_cloth_singles.csv", ["covered", "singles_rate_Hz"], np.r_[np.zeros(10), np.ones(10)],
      rng.poisson(45.0 * 10, 20) / 10, fmt="{:.4g}")
# Faraday rotation: the rod is centred in the coil, but the Hall probe reads 9% high (never
# calibrated), so every field is overstated and V comes out 8% low
write("faraday_rot_partner.csv", ["field_mT", "rotation_deg"], *faraday_rot(probe_gain=1.09), fmt="{:.4g}")
# follow-up: field mapped along the centred rod at the run's top current, probe in 10 mm steps
write("faraday_rot_partner_faraday_rot_field_map.csv", ["position_mm", "field_mT"], pos,
      1.09 * 60.0 * bz(pos - L_ROD / 2) + rng.normal(0, 0.1, pos.size), fmt="{:.4g}")
# Hysteresis: the search coil has its 100 turns, but B was worked out with the taped core's
# outside cross-section, 1.20 cm², while the iron itself is 1.05 cm², so every B comes out 12.5% low
write("hysteresis_partner.csv", ["H_A_per_m", "B_T"], *loop(scale=1.05 / 1.20), fmt="{:.4g}")
# follow-up: five current reversals read on the search coil, then five on a freshly wound
# 10-turn coil (flux-linkage change 2·N·B·A at the 1.42 T tip, through 1.05 cm² of iron)
dphi_iron = lambda n: 2 * n * 1.42 * 1.05e-4 * 1e3     # mWb-turns
write("hysteresis_partner_hysteresis_reference_coil.csv", ["coil", "flux_mWb"], np.r_[np.zeros(5), np.ones(5)],
      np.r_[dphi_iron(100) + rng.normal(0, 0.02, 5), dphi_iron(10) + rng.normal(0, 0.02, 5)], fmt="{:.4g}")

print(f"Wrote advanced-lab datasets to {OUT}")
