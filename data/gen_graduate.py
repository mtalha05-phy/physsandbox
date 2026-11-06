"""Seeded datasets for the graduate advanced-lab experiments.

Run by data/generate.py after the other generators. Ground truth is in
data/manifest.d/graduate.yaml.
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "seeded"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(5150)


def write(name, header, *cols, fmt="{:.6g}"):
    lines = [",".join(header)]
    for row in zip(*cols):
        lines.append(",".join(fmt.format(v) for v in row))
    (OUT / name).write_text("\n".join(lines) + "\n")


# --- Pulsed NMR spin echo: doped water, T2 = 20 ms, 500 mV echo at 2τ → 0 ------
t = np.arange(2.0, 60.0, 4.0)                     # echo times 2τ (ms)
A0, T2, NOISE = 500.0, 20.0, 3.0
echo = lambda a=A0, t2=T2, td=None: a * np.exp(-t / t2 - ((t / td) ** 3 if td else 0))
hdr = ["echo_time_ms", "echo_amplitude_mV"]
write("nmr_clean.csv", hdr, t, echo() + rng.normal(0, NOISE, t.size), fmt="{:.4g}")
# 90° pulse set at 10 µs but the RF amplifier now delivers less power: the true 90° is 16 µs,
# so the pulses are 56° and 112°, and the echo is sin θ · sin²θ = 0.57 of full size; T2 is unchanged.
th = np.pi / 2 * 10 / 16
write("nmr_pulse_misset.csv", hdr, t, echo(A0 * np.sin(th) * np.sin(th) ** 2) + rng.normal(0, NOISE, t.size),
      fmt="{:.4g}")
# sample in a field gradient (poor shim, sample off the sweet spot): diffusion adds exp(−(2τ/T_D)³), T_D = 40 ms
write("nmr_diffusion.csv", hdr, t, echo(td=40.0) + rng.normal(0, NOISE, t.size), fmt="{:.4g}")
# follow-up: nutation sweep, FID amplitude after a single pulse of each length; first maximum at 16 µs
pl = np.arange(2.0, 35.0, 2.0)
write("nmr_pulse_misset_nutation.csv", ["pulse_length_us", "fid_amplitude_mV"], pl,
      A0 * np.abs(np.sin(np.pi / 2 * pl / 16.0)) + rng.normal(0, NOISE, pl.size), fmt="{:.4g}")

# --- Superconducting transition: YBCO pellet, Tc = 92 K, four-probe at 100 mA -------
TT = np.arange(80.0, 110.01, 0.5)
def r_sc(T, tc=92.0, w=0.2, r0=0.0):
    return (5.0 + 0.04 * (T - 92.0)) / (1 + np.exp(-np.clip((T - tc) / w, -60, 60))) + r0
NR = 0.008                                         # mΩ (0.8 µV at 100 mA)
hdr = ["temperature_K", "resistance_mOhm"]
write("sc_clean.csv", hdr, TT, r_sc(TT) + rng.normal(0, NR, TT.size), fmt="{:.5g}")
# cooling sweep, thermometer on the cold stage ahead of the sample: the sample is warmer than the
# reading, so the transition appears 1.5 K low and is smeared by the gradient across the pellet
write("sc_lag.csv", hdr, TT, r_sc(TT, tc=90.5, w=0.6) + rng.normal(0, NR, TT.size), fmt="{:.5g}")
# thermal EMF at the voltage contacts: 6 µV at 100 mA looks like 0.06 mΩ that never goes to zero
write("sc_thermal_emf.csv", hdr, TT, r_sc(TT, r0=0.06) + rng.normal(0, NR, TT.size), fmt="{:.5g}")
# follow-up: the same cooling sweep, then a warming sweep; the lag now runs the other way
TW = np.r_[TT, TT]
wm = np.r_[np.zeros(TT.size), np.ones(TT.size)]
RW = np.where(wm == 0, r_sc(TW, tc=90.5, w=0.6), r_sc(TW, tc=93.5, w=0.6)) + rng.normal(0, NR, TW.size)
write("sc_lag_warming.csv", ["temperature_K", "resistance_mOhm", "warming"], TW, RW, wm, fmt="{:.5g}")

# --- Compton scattering: Cs-137 662 keV into an aluminium rod, NaI(Tl) detector -------
ang = np.arange(15.0, 136.0, 15.0)              # 15° to 135°
E_CS, MEC2, EN = 661.66, 510.999, 1.5
ep = lambda th: E_CS / (1 + E_CS / MEC2 * (1 - np.cos(np.radians(th))))
hdr = ["angle_deg", "energy_keV"]
write("compton_clean.csv", hdr, ang, ep(ang) + rng.normal(0, EN, ang.size), fmt="{:.1f}")
# MCA gain drifted 5% low since it was calibrated (PMT high voltage / temperature)
write("compton_gain.csv", hdr, ang, 0.95 * ep(ang) + rng.normal(0, EN, ang.size), fmt="{:.1f}")
# at 135° the backscatter peak (≈184 keV, from the shielding) was taken for the scattered photopeak
eb = ep(ang) + rng.normal(0, EN, ang.size)
eb[-1] = ep(180.0) + rng.normal(0, EN)
write("compton_backscatter.csv", hdr, ang, eb, fmt="{:.1f}")
# follow-up: calibration lines read with the same MCA settings (Am-241, Na-22, Cs-137, Na-22)
ref = np.array([59.54, 511.0, 661.66, 1274.5])
write("compton_gain_calibration.csv", ["reference_keV", "measured_keV"], ref,
      0.95 * ref + rng.normal(0, EN, ref.size), fmt="{:.1f}")

# --- Optical pumping of rubidium: sweep coil 60 µT/A, field nulled -------------------
MUB_H = 13.99624                                   # kHz per µT
K_COIL = 60.0
Ic = np.round(np.arange(0.1, 1.01, 0.1), 2)
Ii = np.r_[Ic, Ic]
iso = np.r_[np.full(Ic.size, 87), np.full(Ic.size, 85)]
gf = np.where(iso == 87, 0.5, 1 / 3)
fr = lambda i_true, b0=0.0, g=gf: g * MUB_H * np.abs(K_COIL * i_true + b0)
FN = 0.8
hdr = ["coil_current_A", "frequency_kHz", "isotope"]
write("rb_clean.csv", hdr, Ii, fr(Ii) + rng.normal(0, FN, Ii.size), iso, fmt="{:.5g}")
# the current-sense reading is 8% high, so every current is logged 8% above the real one
write("rb_current_scale.csv", hdr, Ii, fr(Ii / 1.08) + rng.normal(0, FN, Ii.size), iso, fmt="{:.5g}")
# horizontal field coil left off: 12 µT of the lab field lies along the sweep axis
write("rb_residual_field.csv", hdr, Ii, fr(Ii, b0=12.0) + rng.normal(0, FN, Ii.size), iso, fmt="{:.5g}")
# follow-up: coil current set at 0.5 A on the lab reading, read with a calibrated ammeter in series
g = [0] * 10 + [1] * 10
cur = np.r_[0.5 + rng.normal(0, 0.0005, 10), 0.5 / 1.08 + rng.normal(0, 0.0005, 10)]
write("rb_current_scale_refmeter.csv", ["ammeter", "coil_current_A"], g, cur, fmt="{:.5g}")

# --- Partner cases: the lower-prior fault of each ambiguous pair -------------------
# Each demo's first measurement can't tell two faults apart, and its truth is the
# more common one. Here the truth is the other, at a realistic size, with the
# recommended test's result as that fault really produces it. Own random stream,
# so every dataset above is unchanged.
rng = np.random.default_rng(4403)

# NMR: the pulses are right, but the tube sits a centimetre high, so only about 60% of
# the liquid is inside the RF coil; T2 is the sample's own
hdr = ["echo_time_ms", "echo_amplitude_mV"]
write("nmr_partner.csv", hdr, t, echo(0.60 * A0) + rng.normal(0, NOISE, t.size), fmt="{:.4g}")
# follow-up: nutation sweep; the liquid inside the coil turns 90° at the set 10 µs, the signal is
# just smaller. The RF field varies by about 12% across the liquid, so the 270° maximum at
# 30 µs comes out lower than the 90° one (a Gaussian spread s damps sin θ by exp(−(θs)²/2)).
pl_p = np.arange(2.0, 35.0, 2.0)
th_p = np.pi / 2 * pl_p / 10.0
write("nmr_partner_nutation.csv", ["pulse_length_us", "fid_amplitude_mV"], pl_p,
      0.60 * A0 * np.abs(np.sin(th_p)) * np.exp(-(0.12 * th_p) ** 2 / 2) + rng.normal(0, NOISE, pl_p.size),
      fmt="{:.4g}")

# Superconducting transition: the thermometer is on the sample block, but the supply was
# left at 500 mA instead of 100 mA. Near the critical current the transition moves 1.2 K
# down and broadens. Five times the current also cuts the noise in mΩ fivefold.
NR_HI = NR * 0.1 / 0.5
hdr = ["temperature_K", "resistance_mOhm"]
write("sc_partner.csv", hdr, TT, r_sc(TT, tc=90.8, w=0.5) + rng.normal(0, NR_HI, TT.size), fmt="{:.5g}")
# follow-up: cooling then warming at the same current; no lag, so the two sweeps agree
RW_P = r_sc(TW, tc=90.8, w=0.5) + rng.normal(0, NR_HI, TW.size)
write("sc_partner_warming.csv", ["temperature_K", "resistance_mOhm", "warming"], TW, RW_P, wm, fmt="{:.5g}")

# Compton: the MCA gain is right, but its zero has moved: every peak reads 20 keV low
hdr = ["angle_deg", "energy_keV"]
write("compton_partner.csv", hdr, ang, ep(ang) - 20.0 + rng.normal(0, EN, ang.size), fmt="{:.1f}")
# follow-up: calibration lines on the same settings; the slope is 1, the intercept is -20 keV
write("compton_partner_calibration.csv", ["reference_keV", "measured_keV"], ref,
      ref - 20.0 + rng.normal(0, EN, ref.size), fmt="{:.1f}")

# Optical pumping: the current reading is right, but the coils sit 10% further apart than
# their radius, so the centre field is (1.25 / (1 + 0.55²))^1.5 = 0.940 of k·I
spacing = ((1 + 0.25) / (1 + 0.55 ** 2)) ** 1.5
hdr = ["coil_current_A", "frequency_kHz", "isotope"]
write("rb_partner.csv", hdr, Ii, fr(Ii * spacing) + rng.normal(0, FN, Ii.size), iso, fmt="{:.5g}")
# follow-up: 0.5 A on the lab reading, and the calibrated ammeter in series agrees
write("rb_partner_refmeter.csv", ["ammeter", "coil_current_A"], [0] * 10 + [1] * 10,
      0.5 + rng.normal(0, 0.0005, 20), fmt="{:.5g}")

print(f"Wrote graduate datasets to {OUT}")
