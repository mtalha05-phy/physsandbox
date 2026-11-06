"""Seeded datasets for the atomic-physics bench (electron diffraction, Zeeman, Millikan).

Run by data/generate.py after generate_bench.py. Ground truth: data/manifest.d/atomic.yaml.
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "seeded"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(3141)
H, QE, CL, ME = 6.62607015e-34, 1.602176634e-19, 299792458.0, 9.1093837e-31
MU_B = 9.2740100783e-24


def write(name, header, *cols, fmt="{:.6g}"):
    lines = [",".join(header)]
    for row in zip(*cols):
        lines.append(",".join(fmt.format(v) for v in row))
    (OUT / name).write_text("\n".join(lines) + "\n")


# --- Electron diffraction (graphite foil, foil-to-screen L = 135 mm) ---------------
def lam_nm(v_kv):
    v = np.asarray(v_kv, dtype=float) * 1e3
    return H / np.sqrt(2 * ME * QE * v * (1 + QE * v / (2 * ME * CL ** 2))) * 1e9


def diam(v_kv, d_nm, L_mm=135.0):
    return 2 * L_mm * np.tan(2 * np.arcsin(lam_nm(v_kv) / (2 * d_nm)))


kv = np.repeat(np.arange(2.5, 5.01, 0.5), 2)
ring = np.tile([1, 2], kv.size // 2)
d_true = np.where(ring == 1, 0.213, 0.123)
hdr = ["voltage_kV", "ring", "diameter_mm"]
noise = lambda: rng.normal(0, 0.3, kv.size)          # vernier on a glowing ring, ±0.3 mm
write("ediff_clean.csv", hdr, kv, ring, diam(kv, d_true) + noise(), fmt="{:.4g}")
# the tube's real foil-to-screen distance is 125 mm, but 135 mm (another tube's data sheet) was entered
write("ediff_camera_length.csv", hdr, kv, ring, diam(kv, d_true, 125.0) + noise(), fmt="{:.4g}")
# the faint third ring (d = 0.107 nm) was measured as the outer ring
d_third = np.where(ring == 1, 0.213, 0.107)
write("ediff_third_ring.csv", hdr, kv, ring, diam(kv, d_third) + noise(), fmt="{:.4g}")
# follow-up: foil-to-screen distance measured with a steel rule, eight readings
write("ediff_camera_length_ruler.csv", ["camera_length_mm"], 125.0 + rng.normal(0, 0.4, 8), fmt="{:.4g}")

# --- Normal Zeeman effect: Cd 643.847 nm, fused-silica étalon t = 3 mm, n = 1.457 ----
NU0 = 1 / 643.847e-9                 # /m
MUT = 3.0e-3 * 1.457
F_LENS = 150.0                       # mm, camera lens behind the étalon
fields = np.arange(200, 551, 50, dtype=float)      # mT, true field at the lamp


def radius(nu, N):
    cos_in = N / (2 * MUT * nu)
    s_in = np.sqrt(1 - cos_in ** 2)
    return F_LENS * np.tan(np.arcsin(1.457 * s_in))


def zeeman_rows(B_T, misassign_at=None):
    rows = []
    for i, b in enumerate(B_T):
        dnu = MU_B * b / (H * CL)
        lo, hi = NU0 - dnu, NU0 + dnu                # σ⁻ (inner ring of an order), σ⁺ (outer)
        N1 = np.floor(2 * MUT * lo)                  # innermost order in which both components show
        r = [radius(lo, N1), radius(hi, N1), radius(lo, N1 - 1), radius(hi, N1 - 1)]
        if i == misassign_at:
            # near half a free spectral range the components of neighbouring orders are
            # almost evenly spaced: σ⁺ of one order is paired with σ⁻ of the next
            r = [radius(hi, N1), radius(lo, N1 - 1), radius(hi, N1 - 1), radius(lo, N1 - 2)]
        rows.append(np.array(r) + rng.normal(0, 0.006, 4))
    return np.array(rows)


hdr = ["field_mT", "r1_inner_mm", "r1_outer_mm", "r2_inner_mm", "r2_outer_mm"]
read_B = lambda b, gain=1.0: gain * b + rng.normal(0, 1.0, b.size)   # gaussmeter, ±1 mT
z = zeeman_rows(fields / 1e3)
write("zeeman_clean.csv", hdr, read_B(fields), *z.T, fmt="{:.5g}")
# the lab gaussmeter reads 12% high (wrong range factor after a probe swap)
z = zeeman_rows(fields / 1e3)
write("zeeman_field_scale.csv", hdr, read_B(fields, 1.12), *z.T, fmt="{:.5g}")
# at the highest field the rings of neighbouring orders were paired wrongly
z = zeeman_rows(fields / 1e3, misassign_at=fields.size - 1)
write("zeeman_ring_order.csv", hdr, read_B(fields), *z.T, fmt="{:.5g}")
# follow-up: magnet current stepped, lab gaussmeter against a calibrated one at the lamp position
cur = np.arange(1.0, 6.01, 1.0)
b_ref = 100.0 * cur                                  # mT, about 0.1 T per amp
write("zeeman_field_scale_gaussmeter.csv", ["current_A", "field_lab_mT", "field_reference_mT"],
      cur, 1.12 * b_ref + rng.normal(0, 1.0, cur.size), b_ref + rng.normal(0, 0.5, cur.size), fmt="{:.5g}")

# --- Millikan oil drop: charges in units of 1e-19 C ---------------------------------
E19 = QE * 1e19
n_drops = np.array([1, 2, 3, 2, 4, 5, 1, 3, 6, 2, 4, 3, 7, 1, 5, 2, 3, 4, 6, 2])
drop = np.arange(1, n_drops.size + 1)
charge = lambda scale=1.0: scale * n_drops * E19 * (1 + rng.normal(0, 0.012, n_drops.size))
hdr = ["drop", "charge_1e19C"]
write("millikan_clean.csv", hdr, drop, charge(), fmt="{:.4g}")
# the analysis used 7.60 mm (the manual's spacing) but this chamber's spacer is 6.70 mm,
# so every charge (q ∝ d) comes out 13% high, still in whole steps
write("millikan_spacing.csv", hdr, drop, charge(7.60 / 6.70), fmt="{:.4g}")
# drops evaporate while being timed: each comes out 3-20% low, by a different amount
q_evap = n_drops * E19 * (1 - rng.uniform(0.03, 0.20, n_drops.size)) * (1 + rng.normal(0, 0.012, n_drops.size))
write("millikan_evaporation.csv", hdr, drop, q_evap, fmt="{:.4g}")
# follow-up: calliper readings of the spacer ring at eight places
write("millikan_spacing_callipers.csv", ["plate_spacing_mm"], 6.70 + rng.normal(0, 0.01, 8), fmt="{:.4g}")

# --- Partner cases: the lower-prior fault of each ambiguous pair ----------------------
# Each first measurement above that has a follow-up can't tell its fault from a second one
# that scales the result the same way, so the data alone favour the more common one. Here
# the truth is that other fault, and the recommended test has to find it.
# Own random stream, so every dataset above is unchanged.
rng = np.random.default_rng(4501)
# Electron diffraction: L = 135 mm is right, but the supply meter reads 15% high, so the
# tube really runs at V/1.15 and λ, and both spacings, come out about 7% small
write("ediff_partner.csv", ["voltage_kV", "ring", "diameter_mm"], kv, ring,
      diam(kv / 1.15, d_true) + noise(), fmt="{:.4g}")
# follow-up: foil-to-screen distance measured with a steel rule, eight readings
write("ediff_partner_measure_camera_length.csv", ["camera_length_mm"], 135.0 + rng.normal(0, 0.4, 8),
      fmt="{:.4g}")


# Zeeman: the gaussmeter is right, but this bench's étalon is air-spaced (4 mm spacers,
# optical thickness 4.00 mm) while the solid étalon's figures were entered (3 mm, n = 1.457,
# 4.37 mm), so every splitting comes out 8.5% small
def air_etalon_rows(B_T, gap=4.0e-3):
    rows = []
    for b in B_T:
        dnu = MU_B * b / (H * CL)
        lo, hi = NU0 - dnu, NU0 + dnu
        N1 = np.floor(2 * gap * lo)
        r = [F_LENS * np.tan(np.arccos(n / (2 * gap * nu))) for nu, n in ((lo, N1), (hi, N1), (lo, N1 - 1), (hi, N1 - 1))]
        rows.append(np.array(r) + rng.normal(0, 0.006, 4))
    return np.array(rows)


z = air_etalon_rows(fields / 1e3)
write("zeeman_partner.csv", ["field_mT", "r1_inner_mm", "r1_outer_mm", "r2_inner_mm", "r2_outer_mm"],
      read_B(fields), *z.T, fmt="{:.5g}")
# follow-up: magnet current stepped, lab gaussmeter against a calibrated one at the lamp position
write("zeeman_partner_calibrated_gaussmeter.csv", ["current_A", "field_lab_mT", "field_reference_mT"],
      cur, b_ref + rng.normal(0, 1.0, cur.size), b_ref + rng.normal(0, 0.5, cur.size), fmt="{:.5g}")
# Millikan: the 7.60 mm spacing is right, but the plate-voltage meter reads 10% high, so every
# charge (q ∝ 1/V) comes out 9% low, still in whole steps
write("millikan_partner.csv", ["drop", "charge_1e19C"], drop, charge(1 / 1.10), fmt="{:.4g}")
# follow-up: calliper readings of the spacer ring at eight places
write("millikan_partner_calliper_plate_spacing.csv", ["plate_spacing_mm"], 7.60 + rng.normal(0, 0.01, 8),
      fmt="{:.4g}")

print(f"Wrote atomic-physics datasets to {OUT}")
