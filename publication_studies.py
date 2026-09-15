"""Generate reproducible data and figures for the corrected two-coefficient paper."""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import numerical_stefan_model_pure_al as numod


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "publication"
DATA = OUT / "data"
FIG = OUT / "figures"
HI_VALUES = (500.0, 800.0, 1400.0, 1600.0, 1800.0)
H_AMB = 400.0
TP_C = 970.48 - 273.15
PROBES = np.array((0.003, 0.010, 0.015, 0.030, 0.050, 0.090))


def error_metrics(reference, candidate, percentage=True):
    r = np.asarray(reference, dtype=float)
    c = np.asarray(candidate, dtype=float)
    d = c - r
    out = {
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d*d))),
        "maxae": float(np.max(np.abs(d))),
        "r2": float(1.0 - np.sum(d*d) / np.sum((r - np.mean(r))**2))
              if np.sum((r - np.mean(r))**2) > 0 else float("nan"),
    }
    threshold = max(1e-12, 0.01 * float(np.max(np.abs(r))))
    valid = np.abs(r) > threshold
    out["mape_pct"] = (float(100.0*np.mean(np.abs(d[valid]/r[valid])))
                       if percentage and np.any(valid) else float("nan"))
    return out


def analytical_arrays(mod, prob, hist, t):
    s = np.asarray(hist.s_of(t), dtype=float)
    v = np.array([mod.dsdt(prob, float(x), float(tt)) for x, tt in zip(s, t)])
    gl = np.array([mod.grad_liquid_interface(prob, float(x), float(tt))/1e3
                   for x, tt in zip(s, t)])
    gs = np.array([mod.grad_solid_interface(prob, float(x), float(tt))/1e3
                   for x, tt in zip(s, t)])
    tint = np.array([mod.T_liquid_drive(prob, float(x), float(tt))
                     for x, tt in zip(s, t)])
    temp = mod.predict_thermocouples(prob, hist, PROBES, t).T
    return {"s": s, "v": v, "G_L": gl, "G_S": gs, "T_int": tint, "T": temp}


def run_case(mod, h_i, ns=45, nl=100, length=0.30, max_step=0.5, t_end=200.0):
    prob = mod.experiment_problem(h_i=h_i, T_P_C=TP_C,
                                  liquid_model="interface-film", h_bar=H_AMB)
    hist = mod.march_interface(prob, t_end=max(t_end, 205.0))
    numerical = numod.numerical_front_tracking_model(
        mod, prob, hist, t_end=t_end, length=length, ns=ns, nl=nl,
        output_dt=0.5, start_time=1.0, max_step=max_step, probe_positions=PROBES)
    analytical = analytical_arrays(mod, prob, hist, numerical["t"])
    return prob, hist, analytical, numerical


def write_rows(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def parametric_study(mod):
    summaries, cases = [], {}
    for hi in HI_VALUES:
        prob, hist, ana, num = run_case(mod, hi)
        cases[hi] = (prob, hist, ana, num)
        raw_rows = []
        for j, tt in enumerate(num["t"]):
            row = {
                "time_s": f"{tt:.9g}",
                "analytical_position_m": f"{ana['s'][j]:.9g}",
                "numerical_position_m": f"{num['s'][j]:.9g}",
                "analytical_velocity_m_s": f"{ana['v'][j]:.9g}",
                "numerical_velocity_m_s": f"{num['v'][j]:.9g}",
                "analytical_G_L_K_m": f"{ana['G_L'][j]*1e3:.9g}",
                "numerical_G_L_K_m": f"{num['G_L'][j]*1e3:.9g}",
                "analytical_G_S_K_m": f"{ana['G_S'][j]*1e3:.9g}",
                "numerical_G_S_K_m": f"{num['G_S'][j]*1e3:.9g}",
                "analytical_T_interface_K": f"{ana['T_int'][j]:.9g}",
                "numerical_T_interface_K": f"{num['T_L_interface'][j]:.9g}",
            }
            for k, xp in enumerate(PROBES):
                tag = f"{xp*1e3:g}mm".replace(".", "p")
                row[f"analytical_T_{tag}_K"] = f"{ana['T'][j, k]:.9g}"
                row[f"numerical_T_{tag}_K"] = f"{num['T'][j, k]:.9g}"
            raw_rows.append(row)
        write_rows(DATA / f"timeseries_h_i_{hi:.0f}_W_m2K.csv", raw_rows)
        mask = num["t"] >= 5.0
        variables = {
            "position_mm": (ana["s"][mask]*1e3, num["s"][mask]*1e3),
            "velocity_mm_s": (ana["v"][mask]*1e3, num["v"][mask]*1e3),
            "liquid_gradient_K_mm": (ana["G_L"][mask], num["G_L"][mask]),
            "solid_gradient_K_mm": (ana["G_S"][mask], num["G_S"][mask]),
            "cooling_rate_K_s": (ana["G_L"][mask]*ana["v"][mask]*1e3,
                                  num["G_L"][mask]*num["v"][mask]*1e3),
            "temperature_C": ((ana["T"][mask]-273.15).ravel(),
                               (num["T"][mask]-273.15).ravel()),
        }
        for variable, (ref, val) in variables.items():
            m = error_metrics(ref, val)
            summaries.append({"h_amb_W_m2K": H_AMB, "h_i_W_m2K": hi,
                              "variable": variable, **{k: f"{v:.9g}" for k, v in m.items()}})
    write_rows(DATA / "parametric_hi_error_metrics.csv", summaries)
    return cases


def convergence_studies(mod):
    rows_grid = []
    for ns, nl in ((23, 50), (33, 75), (45, 100), (61, 140)):
        _, _, _, n = run_case(mod, 1400.0, ns=ns, nl=nl, t_end=100.0)
        rows_grid.append({"ns": ns, "nl": nl, "total_nodes": ns+nl,
                          "s_100s_mm": f"{n['s'][-1]*1e3:.9f}",
                          "V_100s_mm_s": f"{n['v'][-1]*1e3:.9f}",
                          "G_L_100s_K_mm": f"{n['G_L'][-1]:.9f}",
                          "G_S_100s_K_mm": f"{n['G_S'][-1]:.9f}"})
    write_rows(DATA / "grid_independence_ALE.csv", rows_grid)

    rows_dt = []
    for step in (2.0, 1.0, 0.5, 0.25):
        _, _, _, n = run_case(mod, 1400.0, max_step=step, t_end=100.0)
        rows_dt.append({"max_step_s": step, "s_100s_mm": f"{n['s'][-1]*1e3:.9f}",
                        "V_100s_mm_s": f"{n['v'][-1]*1e3:.9f}",
                        "G_L_100s_K_mm": f"{n['G_L'][-1]:.9f}"})
    write_rows(DATA / "time_step_independence_ALE.csv", rows_dt)

    rows_length = []
    for length in (0.20, 0.25, 0.30, 0.40):
        _, _, _, n = run_case(mod, 1400.0, length=length, t_end=100.0)
        rows_length.append({"domain_length_m": length,
                            "s_100s_mm": f"{n['s'][-1]*1e3:.9f}",
                            "V_100s_mm_s": f"{n['v'][-1]*1e3:.9f}",
                            "G_L_100s_K_mm": f"{n['G_L'][-1]:.9f}"})
    write_rows(DATA / "domain_independence_ALE.csv", rows_length)


def enthalpy_conservation(mod):
    prob = mod.experiment_problem(h_i=1e9, T_P_C=TP_C,
                                  liquid_model="conduction", h_bar=H_AMB)
    hist = mod.march_interface(prob, t_end=105.0)
    rows = []
    for nx, cfl in ((300, 0.20), (450, 0.25), (600, 0.35), (750, 0.40)):
        n = numod.numerical_enthalpy_model(prob, t_end=100.0, length=0.30, nx=nx,
                                           cfl=cfl, output_dt=0.5, finite_film=False,
                                           probe_positions=PROBES)
        energy_scale = max(abs(float(n["extracted_energy_J_m2"])), 1.0)
        rows.append({"nx": nx, "dx_mm": f"{n['dx']*1e3:.9f}", "cfl": cfl,
                     "dt_s": f"{n['dt']:.9g}", "s_100s_mm": f"{n['s'][-1]*1e3:.9f}",
                     "analytical_s_100s_mm": f"{hist.s_of(100.0)*1e3:.9f}",
                     "max_energy_residual_J_m2": f"{np.max(np.abs(n['energy_residual_J_m2'])):.9g}",
                     "max_relative_energy_residual":
                         f"{np.max(np.abs(n['energy_residual_J_m2']))/energy_scale:.9g}"})
    write_rows(DATA / "enthalpy_grid_and_energy_conservation.csv", rows)


def controlled_sensitivity(mod):
    """One-at-a-time quasi-similar screening about the h_amb=400, h_i=1400 case."""
    base = mod.experiment_problem(h_i=1400.0, T_P_C=TP_C,
                                  liquid_model="interface-film", h_bar=H_AMB)
    cases = []
    for value in (200.0, 400.0, 800.0):
        cases.append(("h_amb_W_m2K", value,
                      mod.Problem(mat=base.mat, T_inf=base.T_inf, T_P=base.T_P,
                                  h_env=lambda t, h=value: h, h_i=base.h_i,
                                  liquid_model=base.liquid_model)))
    for superheat in (0.0, base.T_P-base.mat.T_F, 67.33):
        cases.append(("superheat_K", superheat,
                      mod.Problem(mat=base.mat, T_inf=base.T_inf,
                                  T_P=base.mat.T_F+superheat, h_env=base.h_env,
                                  h_i=base.h_i, liquid_model=base.liquid_model)))
    for scale in (0.8, 1.0, 1.2):
        mat = replace(base.mat, L=base.mat.L*scale)
        cases.append(("latent_heat_scale", scale,
                      mod.Problem(mat=mat, T_inf=base.T_inf, T_P=base.T_P,
                                  h_env=base.h_env, h_i=base.h_i,
                                  liquid_model=base.liquid_model)))
    for scale in (0.8, 1.0, 1.2):
        mat = replace(base.mat, k_L=base.mat.k_L*scale)
        cases.append(("liquid_conductivity_scale", scale,
                      mod.Problem(mat=mat, T_inf=base.T_inf, T_P=base.T_P,
                                  h_env=base.h_env, h_i=base.h_i,
                                  liquid_model=base.liquid_model)))
    rows = []
    for parameter, value, prob in cases:
        hist = mod.march_interface(prob, t_end=100.0)
        s = float(hist.s_of(100.0))
        v = float(mod.dsdt(prob, s, 100.0))
        rows.append({
            "parameter": parameter, "value": f"{value:.9g}",
            "s_100s_mm": f"{s*1e3:.9f}", "V_100s_mm_s": f"{v*1e3:.9f}",
            "Ste": f"{prob.mat.c_S*(prob.mat.T_F-prob.T_inf)/prob.mat.L:.9f}",
            "N_kS_over_kL": f"{prob.mat.N:.9f}",
        })
    write_rows(DATA / "controlled_parameter_sensitivity.csv", rows)


def figures(mod, cases):
    prob, hist, a, n = cases[1400.0]
    t, p_a, p_n = n["t"], a["s"]*1e3, n["s"]*1e3
    mask = (t >= 2.0) & (p_a <= 100.0) & (p_n <= 100.0)
    specs = [
        ("position", t, p_a, t, p_n, "Time, t [s]", "Interface position, s [mm]"),
        ("velocity", p_a, a["v"]*1e3, p_n, n["v"]*1e3, "Interface position, s [mm]", "Interface velocity, V [mm/s]"),
        ("cooling_rate", p_a, a["G_L"]*a["v"]*1e3, p_n, n["G_L"]*n["v"]*1e3, "Interface position, s [mm]", "Interface cooling-rate magnitude [K/s]"),
    ]
    for name, xa, ya, xn, yn, xlabel, ylabel in specs:
        fig, ax = plt.subplots(figsize=(7.2, 5.1), dpi=220)
        use = mask
        ax.plot(np.asarray(xa)[use], np.asarray(ya)[use], lw=2.3, color="#1469c9", label="corrected quasi-similar")
        ax.plot(np.asarray(xn)[use], np.asarray(yn)[use], lw=1.9, ls="--", color="#e78600", label="numerical ALE")
        ax.set(xlabel=xlabel, ylabel=ylabel,
               title=rf"Pure Al, $h_{{amb}}=400$, $h_i=1400$ W m$^{{-2}}$ K$^{{-1}}$")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / f"Figure_baseline_{name}.png", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=220)
    for ax, key, ylabel in ((axes[0], "G_S", "Solid-side gradient, G_S [K/mm]"),
                            (axes[1], "G_L", "Liquid-side gradient, G_L [K/mm]")):
        ax.plot(p_a[mask], a[key][mask], lw=2.3, color="#1469c9",
                label="corrected quasi-similar")
        ax.plot(p_n[mask], n[key][mask], lw=1.9, ls="--", color="#e78600",
                label="numerical ALE")
        ax.set(xlabel="Interface position, s [mm]", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5)
    fig.suptitle(r"Pure Al, $h_{amb}=400$, $h_i=1400$ W m$^{-2}$ K$^{-1}$")
    fig.tight_layout()
    fig.savefig(FIG / "Figure_baseline_thermal_gradients.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=220)
    for j, xp in enumerate(PROBES):
        ax.plot(t, a["T"][:, j]-273.15, lw=1.8, label=f"{xp*1e3:g} mm analytical")
        ax.plot(t, n["T"][:, j]-273.15, lw=1.2, ls="--", color=ax.lines[-1].get_color())
    ax.set(xlabel="Time, t [s]", ylabel="Temperature [degC]",
           title=r"Cooling curves for pure Al, $h_{amb}=400$, $h_i=1400$ W m$^{-2}$ K$^{-1}$")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=7.2)
    fig.tight_layout()
    fig.savefig(FIG / "Figure_baseline_cooling_curves.png", bbox_inches="tight")
    plt.close(fig)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    mod = numod.load_analytical_module()
    mod.set_thermocouple_positions(PROBES)
    cases = parametric_study(mod)
    convergence_studies(mod)
    enthalpy_conservation(mod)
    controlled_sensitivity(mod)
    figures(mod, cases)
    print(f"Publication studies written to {OUT}")


if __name__ == "__main__":
    main()
