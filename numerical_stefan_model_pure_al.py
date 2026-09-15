"""Numerical 1-D Stefan model for the pure-Al horizontal-solidification experiment.

The finite-h_i model uses two-domain ALE front tracking, with h_amb applied only at the
chill and h_i applied only at the liquid side of the moving interface.  An independent
cell-centred finite-volume enthalpy solver is also available for the conduction limit.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ANALYTICAL_PATH = HERE / "analytical_model_2024_Al_melt_convection_corrected.py"
OUT_DIR = HERE / "outputs" / "numerical_comparison"
FIG_KINETICS = OUT_DIR / "Figure_analytical_vs_numerical_kinetics.png"
FIG_COOLING = OUT_DIR / "Figure_analytical_vs_numerical_cooling_curves.png"
FIG_POSITION = OUT_DIR / "Figure_position_vs_time_analytical_vs_numerical.png"
FIG_VELOCITY = OUT_DIR / "Figure_velocity_vs_position_analytical_vs_numerical.png"
FIG_GRADIENT = OUT_DIR / "Figure_thermal_gradient_vs_position_analytical_vs_numerical.png"
FIG_COOLING_RATE = OUT_DIR / "Figure_cooling_rate_TF_vs_position_analytical_vs_numerical.png"
CSV_COMPARISON = OUT_DIR / "analytical_vs_numerical_comparison.csv"
CSV_ARRIVALS = OUT_DIR / "analytical_vs_numerical_arrival_times.csv"
CSV_METRICS = OUT_DIR / "analytical_vs_numerical_metrics.csv"


def load_analytical_module():
    spec = importlib.util.spec_from_file_location("analytical_corrected", ANALYTICAL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def temperature_from_enthalpy(H, mat):
    """Invert specific enthalpy [J/kg] using an isothermal pure-metal phase change."""
    Hs = mat.c_S * mat.T_F
    Hl = Hs + mat.L
    return np.where(
        H <= Hs,
        H / mat.c_S,
        np.where(H >= Hl, mat.T_F + (H - Hl) / mat.c_L, mat.T_F),
    )


def solid_fraction(H, mat):
    Hs = mat.c_S * mat.T_F
    Hl = Hs + mat.L
    return np.clip((Hl - H) / mat.L, 0.0, 1.0)


def numerical_enthalpy_model(prob, t_end=200.0, length=0.30, nx=600,
                             cfl=0.35, output_dt=0.25, finite_film=True,
                             probe_positions=None):
    """Explicit finite-volume enthalpy solution on a semi-infinite-domain surrogate.

    The solid-liquid face resistance is dx/(2*k_left) + 1/h_i + dx/(2*k_right).
    With ``finite_film=False`` the 1/h_i term is removed, giving the conduction limit.
    """
    mat = prob.mat
    probe_positions = np.asarray(probe_positions, dtype=float)
    dx = float(length) / int(nx)
    x = (np.arange(nx, dtype=float) + 0.5) * dx
    rho = mat.rho_S
    Hs = mat.c_S * mat.T_F
    Hl = Hs + mat.L
    H = np.full(nx, Hl + mat.c_L * (prob.T_P - mat.T_F), dtype=float)
    T = np.full(nx, prob.T_P, dtype=float)

    dt_stable = cfl * rho * min(mat.c_S, mat.c_L) * dx * dx / (2.0 * max(mat.k_S, mat.k_L))
    nt = int(math.ceil(t_end / dt_stable))
    dt = t_end / nt
    out_t = np.arange(output_dt, t_end + 0.5 * output_dt, output_dt)
    rec_t, rec_s, rec_vfilm, rec_gliq, rec_gsol, rec_q0, rec_tw, rec_T, rec_energy_res = ([] for _ in range(9))
    oi = 0
    time = 0.0
    energy_initial = float(rho * np.sum(H) * dx)
    extracted_energy = 0.0

    for _ in range(nt):
        fs = solid_fraction(H, mat)
        k_cell = mat.k_L + fs * (mat.k_S - mat.k_L)
        resistance = 0.5 * dx / k_cell[:-1] + 0.5 * dx / k_cell[1:]

        # Distribute the total film resistance 1/h_i continuously across the enthalpy
        # interface.  For a monotone front, sum(abs(Delta f_s)) = 1, so the distributed
        # resistances add to exactly 1/h_i and do not jump when the front changes cells.
        if finite_film:
            resistance += np.abs(np.diff(fs)) / prob.h_i

        flux = -(T[1:] - T[:-1]) / resistance       # +x-directed Fourier flux [W/m2]
        h_wall = float(prob.h(max(time + 0.5 * dt, 1e-12)))
        wall_U = 1.0 / (1.0 / h_wall + 0.5 * dx / k_cell[0])
        q_wall = wall_U * (T[0] - prob.T_inf)        # positive heat extraction

        dH = np.zeros_like(H)
        dH[0] = (-q_wall - flux[0]) * dt / (rho * dx)
        dH[1:-1] = (flux[:-1] - flux[1:]) * dt / (rho * dx)
        dH[-1] = flux[-1] * dt / (rho * dx)         # distant adiabatic boundary
        H += dH
        extracted_energy += q_wall * dt
        T = temperature_from_enthalpy(H, mat)
        time += dt

        while oi < out_t.size and time >= out_t[oi] - 0.5 * dt:
            fs_now = solid_fraction(H, mat)
            s_now = float(np.sum(fs_now) * dx)
            # Voller sub-cell front interpolation.  The enthalpy fraction places the
            # isothermal front inside its phase-change cell at
            # s = sum(f_s)*dx = x_pc + (f_s-1/2)*dx.  The first fully liquid cell is
            # therefore at the physical distance x_L-s (not at dx) from T(s)=T_F.
            # This keeps the smeared phase-change cell out of the liquid gradient.
            liquid_nodes = np.flatnonzero((x > s_now) & (T > mat.T_F + 1e-8))
            if liquid_nodes.size:
                first_liquid = int(liquid_nodes[0])
                distance = x[first_liquid] - s_now
                G_liquid = max(0.0, float((T[first_liquid] - mat.T_F) / distance))
                q_liquid = mat.k_L * G_liquid
            else:
                G_liquid = 0.0
                q_liquid = 0.0
            # Independent solid-side reconstruction at the same energy-equivalent
            # interface location.  It is later combined with the computed front speed
            # through the Stefan balance to audit the liquid-side reconstruction.
            solid_nodes = np.flatnonzero((x < s_now) & (T < mat.T_F - 1e-8))
            solid_stencil = solid_nodes[-max(4, min(6, solid_nodes.size)):]
            if solid_stencil.size >= 3:
                distance_s = s_now - x[solid_stencil]
                design_s = np.column_stack((distance_s, distance_s ** 2))
                coefficients_s = np.linalg.lstsq(
                    design_s, mat.T_F - T[solid_stencil], rcond=None)[0]
                G_solid = max(0.0, float(coefficients_s[0]))
            else:
                G_solid = 0.0
            rec_t.append(time)
            rec_s.append(s_now)
            rec_vfilm.append(q_liquid)
            rec_gliq.append(G_liquid / 1e3)             # K/mm
            rec_gsol.append(G_solid / 1e3)              # K/mm
            rec_q0.append(q_wall)
            rec_tw.append(float(T[0]))
            rec_T.append(np.interp(probe_positions, x, T))
            rec_energy_res.append(float(rho * np.sum(H) * dx - energy_initial
                                        + extracted_energy))
            oi += 1

    t = np.asarray(rec_t)
    s = np.asarray(rec_s)
    # The enthalpy front is continuous but has small grid-scale ripples; a local polynomial
    # filter differentiates it without imposing a global power law.
    from scipy.signal import savgol_filter
    window = min(21, len(s) - (1 - len(s) % 2))
    window = max(5, window if window % 2 else window - 1)
    s_smooth = savgol_filter(s, window_length=window, polyorder=3, mode="interp")
    v = np.maximum(np.gradient(s_smooth, t), 0.0)
    G_solid = np.asarray(rec_gsol)
    G_liquid_balance = (mat.k_S * G_solid * 1e3 - rho * mat.L * v) / mat.k_L / 1e3
    return {
        "t": t,
        "x": x,
        "s": s,
        "s_smooth": s_smooth,
        "v": v,
        "G_L": np.asarray(rec_gliq),
        "G_S": G_solid,
        "G_L_balance": G_liquid_balance,
        "q_liquid": np.asarray(rec_vfilm),
        "q_wall": np.asarray(rec_q0),
        "T_wall": np.asarray(rec_tw),
        "T": np.asarray(rec_T),
        "energy_residual_J_m2": np.asarray(rec_energy_res),
        "extracted_energy_J_m2": extracted_energy,
        "dx": dx,
        "dt": dt,
        "nx": nx,
        "finite_film": finite_film,
        "label": "numerical FV-enthalpy (Voller)",
    }


def numerical_front_tracking_model(mod, prob, analytical_history, t_end=200.0,
                                   length=0.30, ns=45, nl=100, output_dt=0.25,
                                   start_time=1.0, max_step=0.5, probe_positions=None):
    """Two-domain sharp-interface ALE/front-tracking Stefan solution.

    The domains are mapped to xi=x/s(t) and z=(x-s)/(Lx-s).  The numerical PDE march is
    independent of the analytical fields after ``start_time``; analytical profiles are used
    only to provide a smooth, boundary-compatible initial state after the incubation singularity.
    """
    from scipy.integrate import solve_ivp

    mat = prob.mat
    probe_positions = np.asarray(probe_positions, dtype=float)
    xi = np.linspace(0.0, 1.0, ns)
    z = np.linspace(0.0, 1.0, nl)
    dxi = 1.0 / (ns - 1)
    dz = 1.0 / (nl - 1)
    s0 = float(analytical_history.s_of(start_time))
    xs0 = xi * s0
    xl0 = s0 + z * (length - s0)
    Ts0 = np.asarray(mod.T_solid(prob, xs0, s0, start_time), dtype=float)
    Tl0 = np.asarray(mod.T_liquid(prob, xl0, s0, start_time), dtype=float)
    y0 = np.concatenate(([s0], Ts0[1:-1], Tl0[1:-1]))

    def reconstruct(y):
        s = float(y[0])
        dxs = s * dxi
        dxl = (length - s) * dz
        Ts = np.empty(ns)
        Tl = np.empty(nl)
        Ts[1:-1] = y[1:ns - 1]
        Tl[1:-1] = y[ns - 1:]
        Ts[-1] = mat.T_F
        Tl[-1] = prob.T_P
        h_amb = float(prob.h(1.0))
        Ts[0] = ((4.0 * mat.k_S * Ts[1] - mat.k_S * Ts[2]
                  + 2.0 * dxs * h_amb * prob.T_inf)
                 / (3.0 * mat.k_S + 2.0 * dxs * h_amb))
        if prob.liquid_model == "conduction":
            # Local equilibrium: no artificial temperature jump at the front.
            Tl[0] = mat.T_F
        else:
            Tl[0] = ((4.0 * mat.k_L * Tl[1] - mat.k_L * Tl[2]
                      + 2.0 * dxl * prob.h_i * mat.T_F)
                     / (3.0 * mat.k_L + 2.0 * dxl * prob.h_i))
        return s, dxs, dxl, Ts, Tl

    def rhs(t, y):
        s, dxs, dxl, Ts, Tl = reconstruct(y)
        Gs = (3.0 * Ts[-1] - 4.0 * Ts[-2] + Ts[-3]) / (2.0 * dxs)
        Gl = (-3.0 * Tl[0] + 4.0 * Tl[1] - Tl[2]) / (2.0 * dxl)
        sdot = (mat.k_S * Gs - mat.k_L * Gl) / (mat.rho_S * mat.L)

        Ts_xixi = (Ts[2:] - 2.0 * Ts[1:-1] + Ts[:-2]) / dxi ** 2
        Ts_xi = (Ts[2:] - Ts[:-2]) / (2.0 * dxi)
        Tl_zz = (Tl[2:] - 2.0 * Tl[1:-1] + Tl[:-2]) / dz ** 2
        Tl_z = (Tl[2:] - Tl[:-2]) / (2.0 * dz)
        dTs = mat.alpha_S / s ** 2 * Ts_xixi + xi[1:-1] * sdot / s * Ts_xi
        dTl = (mat.alpha_L / (length - s) ** 2 * Tl_zz
               + (1.0 - z[1:-1]) * sdot / (length - s) * Tl_z)
        return np.concatenate(([sdot], dTs, dTl))

    t_eval = np.arange(start_time, t_end + 0.5 * output_dt, output_dt)
    sol = solve_ivp(rhs, (start_time, t_end), y0, method="BDF", t_eval=t_eval,
                    rtol=2e-6, atol=1e-5, max_step=max_step)
    if not sol.success:
        raise RuntimeError("ALE front-tracking integration failed: " + sol.message)

    s_hist, v_hist, g_hist, gs_hist, tli_hist, q0_hist, tw_hist, fields = ([] for _ in range(8))
    for t, y in zip(sol.t, sol.y.T):
        s, dxs, dxl, Ts, Tl = reconstruct(y)
        Gs = (3.0 * Ts[-1] - 4.0 * Ts[-2] + Ts[-3]) / (2.0 * dxs)
        Gl = (-3.0 * Tl[0] + 4.0 * Tl[1] - Tl[2]) / (2.0 * dxl)
        velocity = (mat.k_S * Gs - mat.k_L * Gl) / (mat.rho_S * mat.L)
        xs = xi * s
        xl = s + z * (length - s)
        Tprobe = np.empty(probe_positions.size)
        for j, xp in enumerate(probe_positions):
            Tprobe[j] = (np.interp(xp, xs, Ts) if xp <= s
                         else np.interp(xp, xl, Tl))
        s_hist.append(s)
        v_hist.append(max(0.0, velocity))
        g_hist.append(max(0.0, Gl) / 1e3)
        gs_hist.append(max(0.0, Gs) / 1e3)
        tli_hist.append(float(Tl[0]))
        q0_hist.append(prob.h(t) * (Ts[0] - prob.T_inf))
        tw_hist.append(Ts[0])
        fields.append(Tprobe)
    s_hist = np.asarray(s_hist)
    return {
        "t": sol.t,
        "s": s_hist,
        "s_smooth": s_hist,
        "v": np.asarray(v_hist),
        "G_L": np.asarray(g_hist),
        "G_S": np.asarray(gs_hist),
        "T_L_interface": np.asarray(tli_hist),
        "q_liquid": mat.k_L * np.asarray(g_hist) * 1e3,
        "q_wall": np.asarray(q0_hist),
        "T_wall": np.asarray(tw_hist),
        "T": np.asarray(fields),
        "nx": ns + nl,
        "ns": ns,
        "nl": nl,
        "dt": float("nan"),
        "finite_film": prob.liquid_model != "conduction",
        "label": "numerical ALE sharp-interface",
    }


def first_crossing_time(t, y, level):
    y = np.asarray(y)
    idx = np.flatnonzero(y >= level)
    if not idx.size:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    return float(np.interp(level, [y[i - 1], y[i]], [t[i - 1], t[i]]))


def write_outputs(mod, prob, hist, numerical, h_bar, h_i, liquid_model):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t = numerical["t"]
    s_an = np.asarray(hist.s_of(t), dtype=float)
    v_an = np.array([mod.dsdt(prob, float(s), float(tt)) for s, tt in zip(s_an, t)])
    G_an = np.array([mod.grad_liquid_interface(prob, float(s), float(tt)) / 1e3
                     for s, tt in zip(s_an, t)])
    T_an = mod.predict_thermocouples(prob, hist, mod.TC_POSITIONS_M, t)
    position_labels = [f"{x*1e3:g}mm" for x in mod.TC_POSITIONS_M]
    G_s_num = numerical.get("G_S", np.full_like(t, np.nan))
    G_l_balance = numerical.get("G_L_balance", np.full_like(t, np.nan))

    with CSV_COMPARISON.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "s_analytical_mm", "s_numerical_mm",
                    "v_analytical_mm_s", "v_numerical_mm_s",
                    "G_L_analytical_K_mm", "G_L_numerical_K_mm",
                    "G_S_numerical_K_mm", "G_L_from_Stefan_balance_K_mm",
                    "h_bar_W_m2K", "h_i_W_m2K", "liquid_model"]
                   + [f"T_analytical_{lab}_C" for lab in position_labels]
                   + [f"T_numerical_{lab}_C" for lab in position_labels])
        for i, tt in enumerate(t):
            w.writerow([f"{tt:.6f}", f"{s_an[i]*1e3:.8f}",
                        f"{numerical['s'][i]*1e3:.8f}", f"{v_an[i]*1e3:.8f}",
                        f"{numerical['v'][i]*1e3:.8f}", f"{G_an[i]:.8f}",
                        f"{numerical['G_L'][i]:.8f}", f"{G_s_num[i]:.8f}",
                        f"{G_l_balance[i]:.8f}", f"{h_bar:.6f}",
                        f"{h_i:.6f}", liquid_model]
                       + [f"{T_an[j, i]-mod.KELVIN:.6f}" for j in range(len(mod.TC_LABELS))]
                       + [f"{numerical['T'][i, j]-mod.KELVIN:.6f}" for j in range(len(mod.TC_LABELS))])

    with CSV_ARRIVALS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["thermocouple", "position_mm", "analytical_arrival_s",
                    "numerical_arrival_s", "numerical_minus_analytical_s"])
        for lab, x in zip(position_labels, mod.TC_POSITIONS_M):
            ta = hist.arrival_time(x)
            tn = first_crossing_time(t, numerical["s"], x)
            w.writerow([lab, f"{x*1e3:.3f}", f"{ta:.6f}", f"{tn:.6f}", f"{tn-ta:.6f}"])

    make_kinetics_figure(mod, prob, hist, numerical, None, h_bar, h_i, liquid_model)
    make_cooling_figure(mod, prob, hist, numerical, None, h_bar, h_i, liquid_model)

    valid = (t >= 1.0) & np.isfinite(s_an)
    ds_mm = (numerical["s"] - s_an) * 1e3
    metrics = {
        "rmse_s_mm": float(np.sqrt(np.mean(ds_mm[valid] ** 2))),
        "max_abs_s_mm": float(np.max(np.abs(ds_mm[valid]))),
        "final_analytical_mm": float(s_an[-1] * 1e3),
        "final_numerical_mm": float(numerical["s"][-1] * 1e3),
    }
    for name, indices in (("all_probes", list(range(len(mod.TC_POSITIONS_M)))),
                          ("third_to_sixth_probe", [2, 3, 4, 5])):
        delta_T = T_an[indices].T - numerical["T"][:, indices]
        metrics[f"rmse_T_analytical_vs_numerical_{name}_C"] = float(
            np.sqrt(np.mean(delta_T ** 2)))
    with CSV_METRICS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value", "unit"])
        for key, value in metrics.items():
            unit = "degC" if "rmse_T" in key else "mm"
            w.writerow([key, f"{value:.8f}", unit])
    return metrics


def experimental_kinetics(mod, data):
    fit = mod.fit_cooling_curve_position_law(data, level=660.0)
    exponent = (fit["b"] - 1.0) / fit["b"]
    coefficient = fit["b"] * fit["A"] ** (1.0 / fit["b"])
    v = coefficient * fit["P"] ** exponent
    points = mod.interface_from_cooling_curves(data, level=660.0)
    rates = np.array([mod.cooling_rate_at_crossing(*data[row[4]], level=660.0)
                      for row in points])
    return fit, v, rates / v


def make_kinetics_figure(mod, prob, hist, num, data, h_bar, h_i, liquid_model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = num["t"]
    s_an = np.asarray(hist.s_of(t)) * 1e3
    v_an = np.array([mod.dsdt(prob, float(s/1e3), float(tt)) * 1e3
                     for s, tt in zip(s_an, t)])
    G_an = np.array([mod.grad_liquid_interface(prob, float(s/1e3), float(tt)) / 1e3
                     for s, tt in zip(s_an, t)])
    p_min = float(min(mod.TC_POSITIONS_M) * 1e3)
    p_max = max(100.0, float(max(mod.TC_POSITIONS_M) * 1e3))
    good_an = (s_an >= p_min) & (s_an <= p_max) & np.isfinite(v_an) & np.isfinite(G_an)
    good_num = ((num["s_smooth"] * 1e3 >= p_min) & (num["s_smooth"] * 1e3 <= p_max)
                & np.isfinite(num["v"]) & np.isfinite(num["G_L"]))
    v_num = num["v"] * 1e3
    from scipy.signal import savgol_filter
    g_window = min(51, len(num["G_L"]) - (1 - len(num["G_L"]) % 2))
    g_window = max(7, g_window if g_window % 2 else g_window - 1)
    if "ALE" in num["label"]:
        G_num_plot = np.maximum(num["G_L"], 0.0)
    else:
        G_num_plot = np.maximum(savgol_filter(num["G_L"], g_window, 3, mode="interp"), 0.0)
    cooling_an = G_an * v_an
    cooling_num = G_num_plot * v_num
    film_label = (f"h_i={h_i:.0f} W/m2K" if liquid_model == "interface-film"
                  else "h_i -> infinity (conduction limit)")
    subtitle = f"h_bar={h_bar:.0f} W/m2K, {film_label}"

    def two_curve_figure(path, x_an, y_an, x_num, y_num, xlabel, ylabel, title,
                         xlim=None, ylim=None):
        fig, ax = plt.subplots(figsize=(9.4, 7.0), dpi=170)
        ax.plot(x_an, y_an, color="#1a6fde", lw=2.5, label="analytical")
        ax.plot(x_num, y_num, color="#ed8b00", lw=2.1, ls="--", label=num["label"])
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title + "\n" + subtitle, fontsize=12.5)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(fontsize=10, framealpha=0.94)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    two_curve_figure(FIG_POSITION, t, s_an, t, num["s"] * 1e3,
                     "Time, t [s]", "Interface position, P [mm]",
                     "Pure Al: interface position - analytical versus numerical",
                     xlim=(0, 200), ylim=(0, 150))
    two_curve_figure(FIG_VELOCITY, s_an[good_an], v_an[good_an],
                     num["s_smooth"][good_num] * 1e3, v_num[good_num],
                     "Interface position, P [mm]", "Interface velocity, V [mm/s]",
                     "Pure Al: interface velocity - analytical versus numerical",
                     xlim=(p_min, p_max))
    two_curve_figure(FIG_GRADIENT, s_an[good_an], G_an[good_an],
                     num["s_smooth"][good_num] * 1e3, G_num_plot[good_num],
                     "Interface position, P [mm]", "Liquid-side thermal gradient, G_L [K/mm]",
                     "Pure Al: thermal gradient - analytical versus numerical",
                     xlim=(p_min, p_max))
    two_curve_figure(FIG_COOLING_RATE, s_an[good_an], cooling_an[good_an],
                     num["s_smooth"][good_num] * 1e3, cooling_num[good_num],
                     "Interface position, P [mm]", "Cooling rate at T_F, |dT_F/dt| [K/s]",
                     "Pure Al: cooling rate at the solidification front",
                     xlim=(p_min, p_max))

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), dpi=170)
    ax = axes[0]
    ax.plot(t, s_an, color="#1a6fde", lw=2.3, label="analytical")
    ax.plot(t, num["s"] * 1e3, color="#ed8b00", lw=2.0, label=num["label"])
    ax.set(xlabel="Time, t [s]", ylabel="Interface position, P [mm]", xlim=(0, 200), ylim=(0, 150))

    ax = axes[1]
    ax.plot(s_an[good_an], v_an[good_an], color="#1a6fde", lw=2.3, label="analytical")
    ax.plot(num["s_smooth"][good_num] * 1e3, num["v"][good_num] * 1e3,
            color="#ed8b00", lw=2.0, label=num["label"])
    ax.set(xlabel="Interface position, P [mm]", ylabel="Interface velocity, V [mm/s]",
           xlim=(p_min, p_max), ylim=(0, 3.0))

    ax = axes[2]
    ax.plot(s_an[good_an], G_an[good_an], color="#1a6fde", lw=2.3, label="analytical")
    ax.plot(num["s_smooth"][good_num] * 1e3, G_num_plot[good_num],
            color="#ed8b00", lw=1.8, label=num["label"])
    ax.set(xlabel="Interface position, P [mm]", ylabel="Liquid-side gradient, G_L [K/mm]",
           xlim=(p_min, p_max))

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(fontsize=8.1, framealpha=0.94)
    fig.suptitle("Pure Al: analytical versus independent numerical Stefan solution\n"
                 f"h_bar={h_bar:.0f} W/m2K, {film_label}, closure={liquid_model}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIG_KINETICS, bbox_inches="tight")
    plt.close(fig)


def make_cooling_figure(mod, prob, hist, num, data, h_bar, h_i, liquid_model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    t = num["t"]
    T_an = mod.predict_thermocouples(prob, hist, mod.TC_POSITIONS_M, t) - mod.KELVIN
    T_num = num["T"].T - mod.KELVIN
    fig, ax = plt.subplots(figsize=(11.0, 8.0), dpi=170)
    for j, lab in enumerate(mod.TC_LABELS):
        c = mod.TC_COLOURS[lab]
        ax.plot(t, T_an[j], color=c, lw=2.0)
        ax.plot(t, T_num[j], color=c, lw=1.5, ls="--")
    ax.axhline(660.0, color="crimson", lw=1.2, alpha=0.8)
    probe_handles = [Line2D([], [], color=mod.TC_COLOURS[lab], lw=2,
                            label=f"{x*1e3:g}mm")
                     for lab, x in zip(mod.TC_LABELS, mod.TC_POSITIONS_M)]
    style_handles = [Line2D([], [], color="0.25", lw=2, label="analytical"),
                     Line2D([], [], color="0.25", lw=1.5, ls="--", label=num["label"])]
    first = ax.legend(handles=probe_handles, loc="lower left", ncol=3, fontsize=9,
                      title="Thermocouple")
    ax.add_artist(first)
    ax.legend(handles=style_handles, loc="upper right", fontsize=9)
    ax.set(xlabel="Time, t [s]", ylabel="Temperature, T [C]", xlim=(0, 200), ylim=(100, 710))
    film_label = (f"h_i={h_i:.0f} W/m2K" if liquid_model == "interface-film"
                  else "h_i -> infinity (conduction limit)")
    ax.set_title("Pure Al cooling curves: analytical versus numerical solution\n"
                 f"h_bar={h_bar:.0f} W/m2K, {film_label}, closure={liquid_model}")
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG_COOLING, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h-bar", type=float, default=4264.0)
    ap.add_argument("--h-i", type=float, default=1000.0)
    ap.add_argument("--t-end", type=float, default=200.0)
    ap.add_argument("--length", type=float, default=0.30)
    ap.add_argument("--nx", type=int, default=600)
    ap.add_argument("--ns", type=int, default=45,
                    help="number of ALE nodes in the solid domain")
    ap.add_argument("--nl", type=int, default=100,
                    help="number of ALE nodes in the liquid domain")
    ap.add_argument("--output-dt", type=float, default=0.25)
    ap.add_argument("--tc-positions", type=float, nargs=6, metavar="X_M",
                    help="six strictly increasing thermocouple positions in metres; "
                         "example: --tc-positions 0.003 0.010 0.015 0.030 0.050 0.090")
    ap.add_argument("--finite-film", action="store_true",
                    help="allow a finite h_i and the associated jump T_L(s+) > T_F; the default enforces local equilibrium with h_i -> infinity")
    ap.add_argument("--sharp-interface", action="store_true",
                    help="use the ALE sharp-interface solver instead of the default "
                         "FV-enthalpy solver with Voller sub-cell interpolation")
    args = ap.parse_args(argv)

    mod = load_analytical_module()
    if args.tc_positions is not None:
        try:
            mod.set_thermocouple_positions(args.tc_positions)
        except AssertionError as exc:
            ap.error(str(exc))
    liquid_model = "interface-film" if args.finite_film else "conduction"
    effective_h_i = args.h_i if args.finite_film else float("inf")
    prob = mod.experiment_problem(h_i=args.h_i, T_P_C=690.9,
                                  liquid_model=liquid_model, h_bar=args.h_bar)
    print("Numerical 1-D Stefan model")
    hi_text = (f"{effective_h_i:.1f} W/m2K" if np.isfinite(effective_h_i)
               else "infinity (local-equilibrium limit)")
    print(f"  h_bar={args.h_bar:.1f} W/m2K; h_i={hi_text}; closure={liquid_model}")
    if args.sharp_interface or args.finite_film:
        print(f"  length={args.length:.3f} m; two-domain sharp-interface ALE solver")
    else:
        print(f"  length={args.length:.3f} m; nx={args.nx}; "
              f"dx={args.length/args.nx*1e3:.3f} mm; Voller sub-cell front")
    hist = mod.march_interface(prob, t_end=max(args.t_end, 230.0))
    if args.sharp_interface or args.finite_film:
        numerical = numerical_front_tracking_model(
            mod, prob, hist, t_end=args.t_end, length=args.length,
            ns=args.ns, nl=args.nl, output_dt=args.output_dt,
            probe_positions=mod.TC_POSITIONS_M,
        )
    else:
        numerical = numerical_enthalpy_model(
            prob, t_end=args.t_end, length=args.length, nx=args.nx,
            output_dt=args.output_dt, finite_film=False,
            probe_positions=mod.TC_POSITIONS_M,
        )
    metrics = write_outputs(mod, prob, hist, numerical, args.h_bar,
                            effective_h_i, liquid_model)
    if np.isfinite(numerical["dt"]):
        print(f"  explicit dt={numerical['dt']:.6g} s")
    else:
        print("  time integration=BDF; ALE grids=%d solid + %d liquid nodes" %
              (numerical["ns"], numerical["nl"]))
    print(f"  final numerical front={metrics['final_numerical_mm']:.3f} mm")
    print(f"  final analytical front={metrics['final_analytical_mm']:.3f} mm")
    print(f"  position RMSE (t>=1 s)={metrics['rmse_s_mm']:.3f} mm; max error={metrics['max_abs_s_mm']:.3f} mm")
    print("  cooling-curve analytical/numerical RMSE, all probes=%.3f C" %
          metrics["rmse_T_analytical_vs_numerical_all_probes_C"])
    print("  cooling-curve analytical/numerical RMSE, probes 3-6=%.3f C" %
          metrics["rmse_T_analytical_vs_numerical_third_to_sixth_probe_C"])
    print(f"  outputs: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
