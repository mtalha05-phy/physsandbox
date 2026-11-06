"""Seeded datasets for the first- and second-year lab fundamentals: Young's double
slit, series RLC resonance, viscosity by Stokes' law (falling ball), thermal
conductivity by Lee's disc. Ground truth is in data/manifest.d/fundamentals.yaml.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

OUT = Path(__file__).parent / "seeded"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(1801)


def write(name, header, *cols, fmt="{:.6g}"):
    fmts = fmt if isinstance(fmt, (list, tuple)) else [fmt] * len(cols)
    lines = [",".join(header)]
    for row in zip(*cols):
        lines.append(",".join(f.format(v) for f, v in zip(fmts, row)))
    (OUT / name).write_text("\n".join(lines) + "\n")


# --- Young's double slit: HeNe laser, slide with d = 0.250 mm, a = 0.040 mm --------------
# Fringe centres marked on paper on the screen and read off a ruler (zero near the left
# edge), stated to 0.3 mm. A fringe's centre, judged between its two dark edges, sits at
# m λ L / d whatever the single-slit envelope does, because the dark lines are exact zeros.
LAM, DS = 632.8e-9, 0.250e-3


def screen(orders, L=2.000):
    y = L * np.tan(np.arcsin(orders * LAM / DS)) * 1e3     # mm from the central fringe
    return 147.3 + y + rng.normal(0, 0.15, y.size)


hdr = ["order_m", "position_mm"]
fm = ["{:.0f}", "{:.1f}"]
orders = np.arange(-5, 6, dtype=float)
write("young_clean.csv", hdr, orders, screen(orders), fmt=fm)
# slit-to-screen distance measured from the front of the laser, 15 cm behind the slide:
# 2.000 m entered, 1.850 m really
write("young_screen_distance.csv", hdr, orders, screen(orders, L=1.850), fmt=fm)
# follow-up: five steel-tape readings from the slide itself to the screen
write("young_screen_distance_tape.csv", ["reading", "slit_to_screen_m"], np.arange(1, 6),
      1.850 + rng.normal(0, 0.0008, 5), fmt=["{:.0f}", "{:.3f}"])
# a/d = 0.16, so the 6th-order fringe falls in the envelope's first dark band (0.2% of the
# central intensity) and isn't seen; the next visible fringes, true orders 7 and 8, are
# numbered 6 and 7 on both sides
labels = np.arange(-7, 8, dtype=float)
true_m = np.where(np.abs(labels) >= 6, labels + np.sign(labels), labels)
write("young_missing_order.csv", hdr, labels, screen(true_m), fmt=fm)

# --- Series RLC resonance: 10 mH coil (18 Ω), 100 nF, 100 Ω sense resistor ---------------
# Function generator with 50 Ω output impedance; V_R read as amplitude on a scope (CH2).
L_, C_, R_, RL_, RG = 0.010, 100e-9, 100.0, 18.0, 50.0
fr = np.array([1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000,
               10000, 12000, 15000], dtype=float)


def z_circuit(f):
    w = 2 * np.pi * f
    return R_ + RL_ + 1j * (w * L_ - 1 / (w * C_))


def sweep(set_once=False, meter_fc=None):
    Z = z_circuit(fr)
    if set_once:
        # 2.00 V set at the circuit's input at the first frequency, then left alone: the
        # generator's 50 Ω output impedance pulls the input down as the circuit's impedance falls
        emf = 2.0 * abs(Z[0] + RG) / abs(Z[0])
        i = emf / np.abs(Z + RG)
    else:
        i = 2.0 / np.abs(Z)                                # input held at 2.00 V on CH1 at every step
    v = i * R_
    if meter_fc:
        # V_R read on a handheld multimeter whose AC range rolls off above a few kHz
        v = v / np.sqrt(1 + (fr / meter_fc) ** 2)
    return v * (1 + rng.normal(0, 0.004, fr.size))


hdr = ["frequency_Hz", "resistor_V"]
fm = ["{:.0f}", "{:.4f}"]
write("rlc_clean.csv", hdr, fr, sweep(), fmt=fm)
write("rlc_generator_loading.csv", hdr, fr, sweep(set_once=True), fmt=fm)
# follow-up: CH1 moved to the circuit's input and the same sweep repeated, generator untouched
Z = z_circuit(fr)
emf = 2.0 * abs(Z[0] + RG) / abs(Z[0])
write("rlc_generator_loading_ch1.csv", ["frequency_Hz", "input_V"], fr,
      emf * np.abs(Z) / np.abs(Z + RG) * (1 + rng.normal(0, 0.004, fr.size)), fmt=fm)
write("rlc_meter_bandwidth.csv", hdr, fr, sweep(meter_fc=7000.0), fmt=fm)

# --- Stokes' law: steel balls in glycerol, 50 mm tube, timed over 300 mm -------------------
G, RHO_S, D_TUBE, H_FALL = 9.81, 7800.0, 0.050, 0.300
balls = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0])            # mm, nominal bearing sizes


def eta_glycerol(t_c, water_pct=0.0):
    # Vogel–Fulcher fit to Segur & Oberstar (1951); about 15% thinner per 1% of water near 20 °C
    return np.exp(-8.9478 + 941.4 / (t_c + 273.15 - 191.83)) * np.exp(-0.165 * water_pct)


def faxen(lam):
    return 1 - 2.104 * lam + 2.09 * lam ** 3 - 0.95 * lam ** 5


def drops(t_c=20.0, zero_mm=0.0):
    eta = eta_glycerol(t_c)
    rho_l = 1261.0 - 0.65 * (t_c - 20)
    d = (balls + rng.normal(0, 0.001, balls.size)) / 1e3     # true diameters, m
    v_st = 2 * (d / 2) ** 2 * G * (RHO_S - rho_l) / (9 * eta)
    a = 3 * rho_l * d / (16 * eta)                          # Oseen: v (1 + 3Re/16) = v_Stokes
    v_free = (np.sqrt(1 + 4 * a * v_st) - 1) / (2 * a)
    v = v_free * faxen(d / D_TUBE)
    t = H_FALL / v + rng.normal(0, 0.08, d.size)            # stopwatch, by hand
    d_read = 1e3 * d + zero_mm + rng.normal(0, 0.002, d.size)
    return d_read, t


hdr = ["ball_diameter_mm", "fall_time_s"]
fm = ["{:.3f}", "{:.2f}"]
write("stokes_clean.csv", hdr, *drops(), fmt=fm)
# the lab (and the glycerol) is at 22 °C, but 20 °C was entered
write("stokes_warm_liquid.csv", hdr, *drops(t_c=22.0), fmt=fm)
# follow-up: a thermometer lowered into the tube at seven depths
depth = np.arange(5, 66, 10, dtype=float)
write("stokes_warm_liquid_thermometer.csv", ["depth_cm", "liquid_temperature_C"], depth,
      22.1 - 0.3 * depth / 65 + rng.normal(0, 0.05, depth.size), fmt=["{:.0f}", "{:.1f}"])
# micrometer not zeroed: it reads +0.050 mm with the jaws closed, and every ball was read on it
write("stokes_micrometer_zero.csv", hdr, *drops(zero_mm=0.050), fmt=fm)

# --- Lee's disc: Perspex sample 3.0 mm thick between the steam chamber and a brass disc -------
# Brass disc 110 mm across, 11 mm thick, heat capacity m c = 335 J/K. Heat lost from the disc
# by natural convection (faces and rim, standard air correlations) and radiation (ε = 0.5).
K_PMMA, X_S, D_D, H_D, MC, T_AIR, T_STEAM = 0.19, 0.0030, 0.110, 0.011, 335.0, 20.0, 99.0
A_D = np.pi * (D_D / 2) ** 2
A_RIM = np.pi * D_D * H_D


def disc_loss(T, top_bare=False):
    dT = np.maximum(T - T_AIR, 1e-9)
    rad = 0.5 * 5.670e-8 * ((T + 273.15) ** 4 - (T_AIR + 273.15) ** 4)
    lh = D_D / 4
    down = 0.59 * (dT / lh) ** 0.25 * dT + rad
    rim = 1.42 * (dT / H_D) ** 0.25 * dT + rad
    up = 1.32 * (dT / lh) ** 0.25 * dT + rad
    return A_D * down + A_RIM * rim + (A_D * up if top_bare else 0.0)


T2 = brentq(lambda T: K_PMMA * A_D / X_S * (T_STEAM - T) - disc_loss(T), T_AIR + 1, T_STEAM - 0.01)


def cooling_run(top_bare=False, span=8.0, every=30.0):
    """Disc heated to T₂ + span, sample put back on top (unless top_bare), read every 30 s."""
    Tg = np.linspace(T2 + span + 0.5, T2 - span - 3, 4000)
    rate = disc_loss(Tg, top_bare) / MC
    tg = np.concatenate([[0], np.cumsum(-np.diff(Tg) / (0.5 * (rate[1:] + rate[:-1])))])
    t0 = np.interp(T2 + span, Tg[::-1], tg[::-1])
    t1 = np.interp(T2 - span, Tg[::-1], tg[::-1])
    ts = np.arange(0, t1 - t0 + 1e-6, every)
    return ts, np.interp(ts + t0, tg, Tg)


def lee(offset=0.0, top_bare=False):
    """offset: disc thermometer reads this much off (both stages, same thermometer)."""
    # stage 1: the readings taken once the student judged the temperatures steady
    n_ss = 1
    t_ss = 2700 + 30 * np.arange(n_ss)
    tc, Tc = cooling_run(top_bare)
    stage = np.r_[np.ones(n_ss), 2 * np.ones(tc.size)]
    t = np.r_[t_ss, tc]
    steam = T_STEAM + rng.normal(0, 0.05, t.size)
    disc = np.r_[T2 * np.ones(n_ss), Tc] + offset + rng.normal(0, 0.05, t.size)
    return stage, t, np.round(steam, 1), np.round(disc, 1)


hdr = ["stage", "time_s", "steam_C", "disc_C"]
fm = ["{:.0f}", "{:.0f}", "{:.1f}", "{:.1f}"]
write("leesdisc_clean.csv", hdr, *lee(), fmt=fm)
# the disc thermocouple reads 1.8 K low (against the steam-side one)
write("leesdisc_thermometer_offset.csv", hdr, *lee(offset=-1.8), fmt=fm)
# follow-up: both thermocouples side by side in the steam chamber, eight readings
write("leesdisc_thermometer_offset_sidebyside.csv", ["reading", "steam_C", "disc_C"], np.arange(1, 9),
      np.round(T_STEAM + rng.normal(0, 0.05, 8), 1), np.round(T_STEAM - 1.8 + rng.normal(0, 0.05, 8), 1),
      fmt=["{:.0f}", "{:.1f}", "{:.1f}"])
# cooling run done with the brass disc's top face bare (sample not put back on top)
write("leesdisc_top_bare.csv", hdr, *lee(top_bare=True), fmt=fm)

# --- Partner cases: the lower-prior fault of each ambiguous pair ----------------------
# Each first measurement above that has a follow-up can't tell its fault from a second one
# that moves the result the same way, so the data alone favour the more common one. Here
# the truth is that other fault, and the recommended test has to find it.
# Own random stream, so every dataset above is unchanged.
rng = np.random.default_rng(4503)


# Young's double slit: L = 2.000 m is right, but this slide's slits are 0.270 mm apart, not the
# 0.250 mm on its label, so the fringes are 7.4% closer together than λL/d predicts
def slide(orders, d):
    y = 2.000 * np.tan(np.arcsin(orders * LAM / d)) * 1e3
    return 147.3 + y + rng.normal(0, 0.15, y.size)


write("young_partner.csv", ["order_m", "position_mm"], orders, slide(orders, 0.270e-3), fmt=["{:.0f}", "{:.1f}"])
# follow-up: five steel-tape readings from the slide itself to the screen
write("young_partner_young_tape_measure.csv", ["reading", "slit_to_screen_m"], np.arange(1, 6),
      2.000 + rng.normal(0, 0.0008, 5), fmt=["{:.0f}", "{:.3f}"])
# RLC: the input is reset to 2.00 V at every frequency, but the 10 mH coil on this bench is
# wound with finer wire than the one on the bench sheet: 52 Ω in series, not the 18 Ω entered,
# so Q and the peak both come out 22% low
w = 2 * np.pi * fr
Z = R_ + 52.0 + 1j * (w * L_ - 1 / (w * C_))
write("rlc_partner.csv", ["frequency_Hz", "resistor_V"], fr,
      2.0 / np.abs(Z) * R_ * (1 + rng.normal(0, 0.004, fr.size)), fmt=["{:.0f}", "{:.4f}"])
# follow-up: CH1 on the circuit's input through the same sweep, reset to 2.00 V at each step as before
write("rlc_partner_rlc_watch_input.csv", ["frequency_Hz", "input_V"], fr,
      2.0 * (1 + rng.normal(0, 0.004, fr.size)), fmt=["{:.0f}", "{:.4f}"])


# Stokes: the glycerol is at the 20 °C entered, but the stock bottle was left unstoppered and
# has taken up 1.1% water, which thins it by 17% (and lowers its density by 0.3%)
def drops_wet(water_pct, t_c=20.0):
    eta = eta_glycerol(t_c, water_pct)
    rho_l = 1 / ((1 - water_pct / 100) / (1261.0 - 0.65 * (t_c - 20)) + (water_pct / 100) / 998.2)
    d = (balls + rng.normal(0, 0.001, balls.size)) / 1e3
    v_st = 2 * (d / 2) ** 2 * G * (RHO_S - rho_l) / (9 * eta)
    a = 3 * rho_l * d / (16 * eta)
    v = (np.sqrt(1 + 4 * a * v_st) - 1) / (2 * a) * faxen(d / D_TUBE)
    t = H_FALL / v + rng.normal(0, 0.08, d.size)
    return 1e3 * d + rng.normal(0, 0.002, d.size), t


write("stokes_partner.csv", ["ball_diameter_mm", "fall_time_s"], *drops_wet(1.1), fmt=["{:.3f}", "{:.2f}"])
# follow-up: a thermometer lowered into the tube at seven depths
write("stokes_partner_stokes_thermometer.csv", ["depth_cm", "liquid_temperature_C"], depth,
      20.1 - 0.2 * depth / 65 + rng.normal(0, 0.05, depth.size), fmt=["{:.0f}", "{:.1f}"])


# Lee's disc: the thermocouples agree, but T₂ was read 27 minutes after the steam went on, while
# the disc was still 1.5 K short of steady and rising by 0.2 K a minute (time constant about 7 min);
# the cooling run was then taken through that reading
def warm_up_time(T_read, dt=1.0):
    """Seconds for the disc to warm from room temperature to T_read once the steam is on."""
    T, t = T_AIR, 0.0
    while T < T_read:
        T += dt * (K_PMMA * A_D / X_S * (T_STEAM - T) - disc_loss(T)) / MC
        t += dt
    return t


def cooling_about(Tc, span=8.0, every=30.0):
    """As cooling_run, through Tc rather than the steady T₂."""
    Tg = np.linspace(Tc + span + 0.5, Tc - span - 3, 4000)
    rate = disc_loss(Tg) / MC
    tg = np.concatenate([[0], np.cumsum(-np.diff(Tg) / (0.5 * (rate[1:] + rate[:-1])))])
    t0 = np.interp(Tc + span, Tg[::-1], tg[::-1])
    t1 = np.interp(Tc - span, Tg[::-1], tg[::-1])
    ts = np.arange(0, t1 - t0 + 1e-6, every)
    return ts, np.interp(ts + t0, tg, Tg)


T2_read = T2 - 1.5
tc, Tc = cooling_about(T2_read)
t = np.r_[30 * round(warm_up_time(T2_read) / 30), tc]
write("leesdisc_partner.csv", ["stage", "time_s", "steam_C", "disc_C"], np.r_[1, 2 * np.ones(tc.size)], t,
      np.round(T_STEAM + rng.normal(0, 0.05, t.size), 1), np.round(np.r_[T2_read, Tc] + rng.normal(0, 0.05, t.size), 1),
      fmt=["{:.0f}", "{:.0f}", "{:.1f}", "{:.1f}"])
# follow-up: both thermocouples side by side in the steam chamber, eight readings
write("leesdisc_partner_leesdisc_side_by_side.csv", ["reading", "steam_C", "disc_C"], np.arange(1, 9),
      np.round(T_STEAM + rng.normal(0, 0.05, 8), 1), np.round(T_STEAM + rng.normal(0, 0.05, 8), 1),
      fmt=["{:.0f}", "{:.1f}", "{:.1f}"])

print(f"Wrote fundamentals datasets to {OUT}")
