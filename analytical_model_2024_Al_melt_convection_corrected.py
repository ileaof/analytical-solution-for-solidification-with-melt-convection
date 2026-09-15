# -*- coding: utf-8 -*-
r"""
Corrected implementation of the melt-convection closed-form solution of

    G. E. Mendes Santos Junior, F. S. Rocha, A. B. S. Silva, D. A. R. Carmo,
    M. O. Silva and I. L. Ferreira,
    "On a Novel Closed-Form Analytical Solution for Unsteady Solidification:
     Theory and Application",
    Am. J. Eng. Appl. Sci. 19 (2026) 88-116, DOI 10.3844/ajeassp.2026.88.116,

section *An Important Solution for Melt Convection*, Eqs. (66)-(83), applied to a
transient horizontal solidification experiment on pure aluminium with a time-dependent
metal/chill coefficient  h_g(t) = 6350 t^-0.21 W m^-2 K^-1.

This file replaces `analytical_model_2024_Al_melt_convection_dev.py`, which is kept
untouched for reference.  Every departure from the printed equations is tagged in the
code with one of

    [PAPER]  - direct implementation of the published equation
    [TYPO]   - correction of an evident typographical/sign error in the manuscript
    [BUG]    - correction of an error present in the development script
    [APPROX] - an additional numerical approximation introduced here, absent from the paper
    [INTERP] - an interpretation of something the manuscript leaves undefined

and is justified in ANALYTICAL_MODEL_CORRECTION_REPORT.md.

--------------------------------------------------------------------------------------
MAP  paper equation  ->  function in this module
--------------------------------------------------------------------------------------
 (66),(67) heat equations              - satisfied by the analytic fields below
 (68),(71) initial / far field         - `verify_initial_and_far_field`
 (69)      Robin condition at x = 0    - `verify_robin_wall`             [TYPO: sign]
 (70)      T(s,t) = T_F                - `verify_interface_temperature`
 (72)      Stefan balance              - `stefan_residual`, `dsdt`       [TYPO: x=-s]
 (73),(74) solid temperature profile   - `T_solid`
 (18),(19) psi, zeta                   - `psi_S`, `zeta_S`
 (75a-c)   solid interfacial gradient  - `grad_solid_interface`, `F_solid`
 (76a,b)   liquid temperature profile  - `T_liquid`
 (77a-c)   liquid interfacial gradient - `grad_liquid_interface`         [TYPO: n-scaling, sign]
 (78a-d)   non-dimensional Stefan bal. - `interface_equation`
 (79)      Robin condition at x = s+   - `verify_robin_interface`
 (80d)     f(Bi_i, phi)                - `f_liquid`                      [TYPO: n-scaling]
 (81),(82) melt driving temperature    - `T_liquid_drive`
 (83)      closure equation for phi    - `interface_equation`, `solve_phi`
 (22)-(26) Fo, Bi, Ste, Bi^2 Fo        - `Dimensionless`
 (25)      Ste = c_PS (T_F-T_inf)/L    - `Dimensionless.Ste`             [BUG: inverted]
 (29d,e,f) Omega, Omega*               - `Dimensionless.Omega_star`
 (31)      n = sqrt(a_S/a_L)           - `Material.n`
 p.111     N = k_S/k_L                 - `Material.N`                    [BUG: mislabelled]
 Table 1   thermophysical data         - `AL_TABLE1`
 Table 2   melt interface temperature  - `reproduce_table2`
--------------------------------------------------------------------------------------

Usage
-----
    python analytical_model_2024_Al_melt_convection_corrected.py             # everything
    python analytical_model_2024_Al_melt_convection_corrected.py --verify    # checks only
    python analytical_model_2024_Al_melt_convection_corrected.py --digitize  # rebuild CSV
    python analytical_model_2024_Al_melt_convection_corrected.py --h-bar 4056 --h-i 1000
    python analytical_model_2024_Al_melt_convection_corrected.py --tc-positions 0.003 0.010 0.015 0.030 0.050 0.090
    python analytical_model_2024_Al_melt_convection_corrected.py --time-dependent-h

IMPORTANT.  The closed form of Eqs. (66)-(83) is a CONSTANT-h solution: the kernel
exp(h x/k + h^2 alpha t/k^2) erfc(.) satisfies Eq. (66) only if h does not depend on time.
The experimentally reported h_g(t) = 6350 t^-0.21 therefore enters only through a constant
mean h_bar (see `h_bar_time_average` / `fit_h_bar`); using it pointwise is available as
`--time-dependent-h` but is a diagnostic, not a model.

All internal computation is in SI (m, s, K, W).  Conversion to degC / mm happens only in
the reporting and plotting layers.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import erf, erfc, erfcx

HERE = os.path.dirname(os.path.abspath(__file__))
KELVIN = 273.15
SQRT_PI = math.sqrt(math.pi)


# ======================================================================================
# 1.  MATERIAL / PROBLEM DEFINITION
# ======================================================================================


@dataclass(frozen=True)
class Material:
    """Thermophysical data.  Defaults are Table 1 of the paper (pure Al)."""

    k_S: float = 213.0        # W m^-1 K^-1
    k_L: float = 91.0
    c_S: float = 1181.0       # J kg^-1 K^-1
    c_L: float = 1086.0
    rho_S: float = 2550.0     # kg m^-3
    rho_L: float = 2368.0
    L: float = 397500.0       # J kg^-1  (latent heat)
    T_F: float = 933.15       # K  (660 degC)

    @property
    def alpha_S(self) -> float:
        return self.k_S / (self.rho_S * self.c_S)

    @property
    def alpha_L(self) -> float:
        return self.k_L / (self.rho_L * self.c_L)

    @property
    def n(self) -> float:
        """Eq. (31): n = sqrt(alpha_S/alpha_L).  Table 1 value 1.41377."""
        return math.sqrt(self.alpha_S / self.alpha_L)

    @property
    def N(self) -> float:
        """p.111: N = k_S/k_L.  Table 1 value 2.34066.

        [BUG] The development script printed this as "N = kl/ks" although it computed
        ks/kl.  Note also that Eq. (44), belonging to the *non-convective* part of the
        paper, defines a different N = alpha_L rho_L/(alpha_S rho_S) = 0.46459.  The two
        must not be mixed.
        """
        return self.k_S / self.k_L


AL_TABLE1 = Material()

H_G_C, H_G_P = 6350.0, 0.21
DEFAULT_TC_POSITIONS_M = (0.005, 0.010, 0.015, 0.030, 0.050, 0.090)
# Runtime model positions.  Keep the data-column labels separate: a curve digitised as
# "5mm" can deliberately be evaluated at another physical position (for example 3 mm).
# Edit this tuple directly when the actual sensor positions differ from the nominal ones.
TC_POSITIONS_M = DEFAULT_TC_POSITIONS_M
TC_LABELS = ("5mm", "10mm", "15mm", "30mm", "50mm", "90mm")
LIQUID_MODELS = ("virtual-origin", "interface-film", "as-coded", "conduction")

def _check_probe_metadata():
    """Guard: one valid, strictly increasing model position per data curve."""
    xs = np.asarray(TC_POSITIONS_M, dtype=float)
    assert len(TC_LABELS) == len(TC_POSITIONS_M), "label/position count mismatch"
    assert np.all(np.isfinite(xs)) and np.all(xs > 0.0), (
        "probe positions must be finite and positive: %r" % (TC_POSITIONS_M,))
    assert np.all(np.diff(xs) > 0), "probe positions must increase: %r" % (TC_POSITIONS_M,)


def set_thermocouple_positions(positions_m: Sequence[float]) -> None:
    """Set the physical model positions assigned to the six digitised TC curves.

    The labels identify the experimental data columns and intentionally remain unchanged;
    they no longer force the model position to equal the nominal value in the label.
    """
    global TC_POSITIONS_M
    old = TC_POSITIONS_M
    TC_POSITIONS_M = tuple(float(x) for x in positions_m)
    try:
        _check_probe_metadata()
    except AssertionError:
        TC_POSITIONS_M = old
        raise


_check_probe_metadata()



def h_g_experiment(t):
    """Prescribed experimental metal/chill coefficient  h_g(t) = 6350 t^-0.21 W/m2K.

    [APPROX] The law is singular at t = 0.  The singularity is *integrable*
    (int_0^t h_g dt' = 6350 t^0.79/0.79 < inf), so it is admissible as a boundary
    condition; only the pointwise value diverges.  Numerically the argument is clipped
    at 1e-12 s and every time integration starts from t0 > 0 using the exact small-time
    asymptote `newtonian_start`.
    """
    return H_G_C * np.power(np.maximum(np.asarray(t, dtype=float), 1e-12), -H_G_P)


@dataclass
class Problem:
    """A complete boundary-value problem instance."""

    mat: Material = field(default_factory=lambda: AL_TABLE1)
    T_inf: float = 298.15                   # K, chill / environment temperature
    T_P: float = 963.15                     # K, melt (pouring) temperature
    h_env: Callable[[float], float] = staticmethod(lambda t: 400.0)   # W m^-2 K^-1
    h_i: float = 1200.0                     # W m^-2 K^-1, melt-side film coefficient
    liquid_model: str = "virtual-origin"

    def h(self, t: float) -> float:
        return float(self.h_env(t))


H_BAR_DEFAULT = 4264.0      # W m^-2 K^-1, see `fit_h_bar` / the report


def h_bar_time_average(t_f: float, C: float = H_G_C, p: float = H_G_P) -> float:
    """Time-mean of the reported law h_g(t) = C t^-p over the interval (0, t_f].

    The closed-form solution of Eqs. (66)-(83) is derived for a **constant** surface
    coefficient: the kernel exp(h x/k + h^2 alpha t/k^2) erfc(.) solves Eq. (66) only if h
    does not depend on t.  h_g(t) therefore cannot be inserted pointwise - the analytical
    solution admits only a single *mean* h.  This helper returns the elementary time
    average

        h_bar(t_f) = (1/t_f) int_0^{t_f} C t^-p dt = C t_f^-p / (1 - p),

    which is finite for p < 1 despite h_g(0) = inf.  Because the surface heat flux
    h_g (T_w - T_inf) is weighted towards early times (T_w is highest then), the
    *flux-weighted* mean is larger than the 0-200 s average; the value obtained by fitting
    the cooling curves, H_BAR_DEFAULT, equals the 0-20.5 s average of the reported law.
    """
    return C * t_f ** (-p) / (1.0 - p)


def experiment_problem(h_i: float = 1000.0, T_P_C: float = 690.9,
                       liquid_model: str = "virtual-origin",
                       T_inf: float = 298.15, h_bar: Optional[float] = None,
                       time_dependent_h: bool = False) -> Problem:
    """The pure-Al horizontal solidification experiment of the supplied figure.

    By default the surface coefficient is the **constant mean** h_bar required by the
    closed form.  `time_dependent_h=True` selects the frozen-coefficient use of the
    reported law h_g(t) = 6350 t^-0.21 instead; that is retained only as a diagnostic
    (§6 of the report) because it is not admissible in a solution derived for constant h.
    """
    if time_dependent_h:
        h_env = h_g_experiment
    else:
        hb = H_BAR_DEFAULT if h_bar is None else float(h_bar)
        h_env = (lambda t, _h=hb: _h)
    return Problem(mat=AL_TABLE1, T_inf=T_inf, T_P=T_P_C + KELVIN,
                   h_env=h_env, h_i=h_i, liquid_model=liquid_model)


# ======================================================================================
# 2.  DIMENSIONLESS GROUPS  (Eqs. 22-26, 31, p.111)
# ======================================================================================


@dataclass(frozen=True)
class Dimensionless:
    phi: float       # similarity variable  phi = s/(2 sqrt(alpha_S t))       Eqs. (10)-(11)
    Fo: float        # alpha_S t / s^2                                        Eq. (23)
    Bi_env: float    # h s / k_S                                              Eq. (24)
    Bi_i: float      # h_i s / k_L                                            p.111
    n: float         # sqrt(alpha_S/alpha_L)                                  Eq. (31)
    N: float         # k_S/k_L                                                p.111
    Ste: float       # c_PS (T_F - T_inf)/L                                   Eq. (25)
    Theta0: float    # (T_P - T_F)/(T_F - T_inf)                              p.111

    @property
    def Bi2Fo(self) -> float:
        """Eq. (26):  Bi^2 Fo = Bi^2/(4 phi^2)."""
        return self.Bi_env ** 2 / (4.0 * self.phi ** 2)

    @property
    def Omega_star(self) -> float:
        """Eqs. (29d-f)/p.111:  Omega* = Bi_env/(phi Ste).

        [BUG] The dev script coded "Biotenv/(Stes*phi)" with Stes = L/(c_PS dT) = 1/Ste,
        i.e. a factor Ste^2 too large, and then switched the term off ("Omega = 0.0").
        """
        return self.Bi_env / (self.phi * self.Ste)


def groups(prob: Problem, s: float, t: float, h: Optional[float] = None) -> Dimensionless:
    m = prob.mat
    if h is None:
        h = prob.h(t)
    phi = s / (2.0 * math.sqrt(m.alpha_S * t))
    return Dimensionless(
        phi=phi,
        Fo=m.alpha_S * t / (s * s),
        Bi_env=h * s / m.k_S,
        Bi_i=prob.h_i * s / m.k_L,
        n=m.n,
        N=m.N,
        # [BUG] Eq. (25) is Ste = c_PS (T_F - T_inf)/L = 1.88663 for Table 1.  The dev
        # script stored its reciprocal (0.53005) in `Stes` and then wrote `phi/Stes`, so
        # the left-hand side of Eq. (83) came out a factor Ste^2 = 3.559 too large.
        Ste=m.c_S * (m.T_F - prob.T_inf) / m.L,
        Theta0=(prob.T_P - m.T_F) / (m.T_F - prob.T_inf),
    )


# ======================================================================================
# 3.  NUMERICALLY STABLE SPECIAL-FUNCTION KERNELS
# ======================================================================================
#
# Every exp(A) erfc(B) group in Eqs. (18)-(21) and (73)-(83) obeys
#
#       A - B^2 = -(first argument of the erfc)^2 ,
#
# because in each case A = h X/k + h^2 alpha t/k^2 and B = X/(2 sqrt(alpha t)) +
# h sqrt(alpha t)/k, so that 2 (X/(2 sqrt(alpha t))) (h sqrt(alpha t)/k) = h X/k.
# Writing exp(A) erfc(B) = exp(A - B^2) erfcx(B) removes the overflow the dev script
# suffered: exp(Biot + Biot^2/(4 phi^2)) alone overflows for Biot > ~700 and the product
# with erfc(.) underflows to 0 * inf; likewise 1/(sqrt(pi) exp(phi^2)) underflows for
# phi > 27.


def exp_erfc(A: float, B: float):
    """exp(A)*erfc(B) without overflow (scipy.special.erfcx)."""
    return np.exp(A - np.asarray(B) ** 2) * erfcx(B)


# ---- solid phase ---------------------------------------------------------------------


def psi_S(Bi_env: float, phi: float) -> float:
    """Eq. (18b):  psi = 1 - erfc(phi) + exp(Bi + Bi^2/4phi^2) erfc(phi + Bi/2phi)."""
    b = phi + Bi_env / (2.0 * phi)
    return erf(phi) + np.exp(-phi * phi) * erfcx(b)


def zeta_S(Bi_env: float, phi: float) -> float:
    """Eq. (19b):  zeta = -erfc(phi) + exp(...) erfc(...).  Identity: psi = 1 + zeta."""
    b = phi + Bi_env / (2.0 * phi)
    return -erfc(phi) + np.exp(-phi * phi) * erfcx(b)


def F_solid(Bi_env: float, phi: float) -> float:
    r"""Eq. (75c)/(21b): the braced solid-gradient group, in reduced form.

    The printed bracket is

        Bi/(2phi) e^{A} erfc(B) + 1/(sqrt(pi) e^{phi^2}) - e^{A}/(sqrt(pi) e^{B^2})

    with A = Bi + Bi^2/4phi^2 and B = phi + Bi/2phi.  Since A - B^2 = -phi^2 identically,
    the last two terms cancel *exactly*; keeping them (as the dev script does) is
    analytically harmless but is a catastrophic-cancellation trap.  The identity is
    checked by `verify_cancellation`.
    """
    b = phi + Bi_env / (2.0 * phi)
    return Bi_env / (2.0 * phi) * np.exp(-phi * phi) * erfcx(b)


def F_solid_literal(Bi_env: float, phi: float) -> float:
    """Eq. (75c) term-by-term as printed - used only to verify the cancellation."""
    A = Bi_env + Bi_env ** 2 / (4.0 * phi ** 2)
    B = phi + Bi_env / (2.0 * phi)
    return (Bi_env / (2.0 * phi) * exp_erfc(A, B)
            + 1.0 / (SQRT_PI * np.exp(phi ** 2))
            - np.exp(A - B ** 2) / SQRT_PI)


def grad_solid_interface(prob: Problem, s: float, t: float, h: Optional[float] = None) -> float:
    """dT_S/dx at x = s^-  [K/m].  Eqs. (75a-c).

    dT_S/dx|_s = (T_F - T_inf)/psi * (h/k_S) exp(A) erfc(B) > 0, i.e. the temperature
    rises from the cold chill towards the front, as it must.
    """
    m = prob.mat
    hh = prob.h(t) if h is None else h
    phi = s / (2.0 * math.sqrt(m.alpha_S * t))
    Bi = hh * s / m.k_S
    b = phi + Bi / (2.0 * phi)
    return ((m.T_F - prob.T_inf) / psi_S(Bi, phi)
            * (hh / m.k_S) * np.exp(-phi * phi) * erfcx(b))


def T_solid(prob: Problem, x, s: float, t: float, h: Optional[float] = None):
    """Eq. (73)/(74a): solid temperature field for 0 <= x <= s(t)  [K]."""
    m = prob.mat
    hh = prob.h(t) if h is None else h
    x = np.asarray(x, dtype=float)
    sq = math.sqrt(m.alpha_S * t)
    u = x / (2.0 * sq)
    w = hh * sq / m.k_S                       # = Bi_env/(2 phi)
    F = erfc(u) - np.exp(-u * u) * erfcx(u + w)
    phi = s / (2.0 * sq)
    z = -erfc(phi) + np.exp(-phi * phi) * erfcx(phi + w)
    return m.T_F + (prob.T_inf - m.T_F) * (F + z) / (1.0 + z)


# ---- liquid phase --------------------------------------------------------------------
#
# [INTERP] The manuscript writes the liquid profile (76a) and its gradient (77a) but
# never defines the normalisation psi_L appearing in their denominators.  Four
# self-consistent closures are implemented:
#
#   'virtual-origin'  psi_L = erfc(n phi) - exp(A_L) erfc(B_L)                 <- default
#        The liquid field is the classical third-kind (Robin) solution with coefficient
#        h_i whose surface sits at the *chill face* x = 0, i.e. the solidified layer acts
#        as a virtual liquid adjunct.  That is exactly what the argument "(s+x)" of
#        Eq. (76a) and the value "n phi" of Eq. (77a) require.  Normalising by its own
#        interface value gives T_L(s,t) = T_F, so solid and liquid fields are continuous,
#        and makes Eq. (83) collapse *exactly* onto the paper's own non-convective
#        closure Eqs. (45)/(57) as h_i -> inf (`verify_conduction_limit`).
#
#   'interface-film'  psi_L = 1, profile anchored at eta = x - s
#        The exact solution of (67)+(71)+(79) in the interface-attached frame.  The melt
#        is then discontinuous with the interface by the film drop
#        T_L(s+) - T_F = (T_P - T_F) erfcx(h_i sqrt(alpha_L t)/k_L), which is the physical
#        content of a film coefficient.  Correct h_i -> 0 limit (q_L -> 0) but it does
#        not reduce to the Neumann melt flux as h_i -> inf.
#
#   'as-coded'        psi_L = 1 - erfc(n phi) + exp(A_L) erfc(B_L)
#        Strict analogy with Eq. (18b).  This is what the development script used and is
#        required to reproduce published Table 2.  It has no correct h_i limit.
#
#   'conduction'      h_i -> inf: classical Neumann melt solution, no melt convection.


def _liquid_args(prob: Problem, s: float, t: float) -> Tuple[float, float, float]:
    """Return (n*phi, w_i, B_L) with w_i = h_i sqrt(alpha_L t)/k_L and B_L = n phi + w_i.

    [TYPO] Eqs. (77b,c), (80a-d), (81)-(83) substitute
        h_i sqrt(alpha_L t)/k_L -> n Bi_i/(2 phi)  and  h_i^2 alpha_L t/k_L^2 -> n^2 Bi_i^2/(4 phi^2).
    Because sqrt(alpha_L t) = sqrt(alpha_S t)/n = s/(2 n phi), the correct substitutions
    are  Bi_i/(2 n phi)  and  Bi_i^2/(4 n^2 phi^2): the manuscript multiplies by n where
    it should divide.  Eq. (77a), which keeps sqrt(alpha_L t) symbolic, and the prefactor
    2 n phi/(sqrt(pi) s) = 1/(sqrt(pi) sqrt(alpha_L t)) of the very same equation are both
    consistent with the corrected form, so (77b) contradicts (77a).  Only the corrected
    scaling satisfies A_L - B_L^2 = -(n phi)^2, the identity that makes the last two terms
    of Eq. (80d) cancel exactly.
    """
    m = prob.mat
    phi = s / (2.0 * math.sqrt(m.alpha_S * t))
    w_i = prob.h_i * math.sqrt(m.alpha_L * t) / m.k_L
    return m.n * phi, w_i, m.n * phi + w_i


def f_liquid(prob: Problem, s: float, t: float) -> float:
    """Eq. (80d) in reduced form:  f = Bi_i/(2 phi) exp(-n^2 phi^2) erfcx(n phi + w_i).

    [BUG] The dev script wrote the second term of Eq. (80d) as n/(sqrt(pi) exp(phi**2))
    instead of n/(sqrt(pi) exp(n**2 phi**2)).
    """
    m = prob.mat
    phi = s / (2.0 * math.sqrt(m.alpha_S * t))
    Bi_i = prob.h_i * s / m.k_L
    nphi, w_i, b = _liquid_args(prob, s, t)
    if prob.liquid_model == "interface-film":
        return Bi_i / (2.0 * phi) * erfcx(w_i)
    return Bi_i / (2.0 * phi) * np.exp(-nphi * nphi) * erfcx(b)


def psi_L(prob: Problem, s: float, t: float) -> float:
    """Normalisation of the liquid profile - see the note above."""
    nphi, w_i, b = _liquid_args(prob, s, t)
    X = np.exp(-nphi * nphi) * erfcx(b)
    mode = prob.liquid_model
    if mode == "virtual-origin":
        return erfc(nphi) - X
    if mode == "as-coded":
        return 1.0 - erfc(nphi) + X
    if mode == "interface-film":
        return 1.0
    if mode == "conduction":
        return erfc(nphi)
    raise ValueError("unknown liquid_model %r" % (mode,))


def grad_liquid_interface(prob: Problem, s: float, t: float) -> float:
    """dT_L/dx at x = s^+  [K/m].  Eqs. (77a-c) with the corrected scaling and sign.

    [TYPO] Eqs. (77a-c) carry the prefactor (T_F - T_P) < 0, which would make the melt
    colder away from the interface.  Eq. (80a) of the same derivation uses (T_P - T_F),
    the physically correct sign for a superheated melt; that is what is used here.  The
    same sign slip affects Eq. (55) of the non-convective solution and propagates into
    Eqs. (43), (45) and (57), where the melt term is *added* to the Stefan balance
    instead of being subtracted.
    """
    m = prob.mat
    nphi, w_i, b = _liquid_args(prob, s, t)
    if prob.liquid_model == "conduction":
        sq_L = math.sqrt(m.alpha_L * t)
        return (prob.T_P - m.T_F) * np.exp(-nphi ** 2) / (SQRT_PI * sq_L * erfc(nphi))
    if prob.liquid_model == "interface-film":
        return prob.h_i * (prob.T_P - m.T_F) * erfcx(w_i) / m.k_L
    X = np.exp(-nphi * nphi) * erfcx(b)
    return prob.h_i * (prob.T_P - m.T_F) * X / (m.k_L * psi_L(prob, s, t))


def T_liquid_drive(prob: Problem, s: float, t: float) -> float:
    """Eq. (82): the melt driving temperature  T_L(x = s^+, t)  [K].

    [TYPO] Table 2 and Eq. (72) label this T_L(x = ^-s, t) while Eqs. (79)-(82) label it
    T_L(x = ^+s, t).  T_L is by definition the *liquid* field, so the liquid-side limit
    s^+ is meant; the superscript in Eq. (72) and in the Table 2 header is typographic.

    Operationally it is the temperature that, through Newton's law with the film
    coefficient h_i, reproduces the melt-side conductive flux,
        T_L(s^+) = T_F + (k_L/h_i) dT_L/dx|_{s^+}                            Eq. (79)
    which is algebraically identical to Eq. (82) written with f and psi_L.
    """
    return prob.mat.T_F + prob.mat.k_L / prob.h_i * grad_liquid_interface(prob, s, t)


def T_liquid(prob: Problem, x, s: float, t: float):
    """Eq. (76a,b): liquid temperature field for x >= s(t)  [K]."""
    m = prob.mat
    x = np.asarray(x, dtype=float)
    sq_L = math.sqrt(m.alpha_L * t)
    w_i = prob.h_i * sq_L / m.k_L
    nphi, _, _ = _liquid_args(prob, s, t)
    if prob.liquid_model == "conduction":
        xi = x / (2.0 * sq_L)
        return prob.T_P + (m.T_F - prob.T_P) * erfc(xi) / erfc(nphi)
    if prob.liquid_model == "interface-film":
        xi = (x - s) / (2.0 * sq_L)
        G = erfc(xi) - np.exp(-xi * xi) * erfcx(xi + w_i)
        return prob.T_P + (m.T_F - prob.T_P) * G
    xi = x / (2.0 * sq_L)                      # virtual origin at the chill face
    G = erfc(xi) - np.exp(-xi * xi) * erfcx(xi + w_i)
    return prob.T_P + (m.T_F - prob.T_P) * G / psi_L(prob, s, t)


# ======================================================================================
# 4.  INTERFACIAL FLUXES AND THE STEFAN BALANCE  (Eqs. 7, 72, 79)
# ======================================================================================


def q_solid(prob: Problem, s: float, t: float, h: Optional[float] = None) -> float:
    """k_S dT_S/dx|_{s^-}  [W/m2]: heat conducted from the front into the solid."""
    return prob.mat.k_S * grad_solid_interface(prob, s, t, h)


def q_liquid(prob: Problem, s: float, t: float) -> float:
    """k_L dT_L/dx|_{s^+} = h_i (T_L(s^+,t) - T_F)  [W/m2]: heat delivered by the melt.

    Eq. (72) writes this term as -h_i (T_L - T_F) (Newton with the film coefficient)
    while Eq. (7) writes it as -k_L dT_L/dx|_{s^+} (Fourier); Eq. (79) states they are
    equal.  The Fourier form is used because it stays finite for every h_i.
    """
    return prob.mat.k_L * grad_liquid_interface(prob, s, t)


def dsdt(prob: Problem, s: float, t: float, h: Optional[float] = None) -> float:
    """Eqs. (7)/(72):  rho_S L ds/dt = k_S dT_S/dx|_{s^-} - k_L dT_L/dx|_{s^+}."""
    return (q_solid(prob, s, t, h) - q_liquid(prob, s, t)) / (prob.mat.rho_S * prob.mat.L)


def stefan_residual(prob: Problem, s: float, t: float, v: float,
                    h: Optional[float] = None) -> float:
    """Absolute residual [W/m2] of the interfacial energy balance for a given ds/dt."""
    return prob.mat.rho_S * prob.mat.L * v - (q_solid(prob, s, t, h) - q_liquid(prob, s, t))


# ======================================================================================
# 5.  THE CLOSURE EQUATION FOR phi  (Eq. 83)  -  "method A", algebraic
# ======================================================================================


def interface_equation(phi: float, prob: Problem, s: float, h: float,
                       omega: bool = False) -> float:
    r"""Residual of Eq. (83):

        (1/Ste)(phi - omega Bi_env/phi) - [ F_S(Bi_env,phi)/psi(Bi_env,phi)
                                            - (Theta_0/N) f(Bi_i,phi)/psi_L(Bi_i,phi) ]

    `omega=False` drops the first-order velocity correction Omega (Eq. 29d), i.e. it uses
    the pure parabolic law ds/dt = 2 phi^2 alpha_S/s.  That is the only choice consistent
    with phi = s/(2 sqrt(alpha_S t)) - the definition Eqs. (18b),(19b),(21b),(26),(75b)
    all silently assume when they replace h sqrt(alpha_S t)/k_S by Bi_env/(2 phi).
    `omega=True` reproduces Eq. (83) exactly as printed.

    [BUG] The dev script's `equation()` multiplied the melt term by a spurious extra
    factor (T_P - T_F), making that term dimensional (units of K) and ~37x too large.
    Together with the inverted Stefan number this removed the root entirely - see
    `reproduce_table2`.
    """
    m = prob.mat
    t = s * s / (4.0 * m.alpha_S * phi * phi)              # Eq. (22)
    Bi_env = h * s / m.k_S
    Ste = m.c_S * (m.T_F - prob.T_inf) / m.L
    Theta0 = (prob.T_P - m.T_F) / (m.T_F - prob.T_inf)
    lhs = (phi - (Bi_env / phi if omega else 0.0)) / Ste
    solid = F_solid(Bi_env, phi) / psi_S(Bi_env, phi)
    melt = (Theta0 / m.N) * f_liquid(prob, s, t) / psi_L(prob, s, t)
    return lhs - (solid - melt)


def solve_phi(prob: Problem, s: float, h: float, omega: bool = False,
              lo: float = 1e-6, hi: float = 60.0, xtol: float = 1e-13,
              strict: bool = True) -> Dict[str, object]:
    """Bracketed positive root of Eq. (83) with existence / convergence / uniqueness checks.

    [BUG] The dev script called fsolve(equation, 1.0) with neither a convergence test nor
    a positivity check.  For its own (erroneous) residual the function has *no* root at
    all: fsolve returned ier = 5 ("not making good progress") and the caller used the
    stalled iterate regardless.  Here the root is bracketed, its existence is proven by a
    sign change, and residual / positivity / uniqueness are reported.
    """
    def res(p):
        return interface_equation(p, prob, s, h, omega)

    grid = np.geomspace(lo, hi, 400)
    with np.errstate(all="ignore"):
        vals = np.array([res(p) for p in grid])
    good = np.isfinite(vals)
    brackets = [(grid[i], grid[i + 1]) for i in range(len(grid) - 1)
                if good[i] and good[i + 1] and vals[i] * vals[i + 1] < 0.0]
    if not brackets:
        msg = ("Eq. (83) has no positive root for s=%.4g m, h=%.4g W/m2K, omega=%s "
               "(residual does not change sign on [%g, %g])." % (s, h, omega, lo, hi))
        if strict:
            raise RuntimeError(msg)
        return {"phi": float("nan"), "residual": float("nan"), "n_roots": 0, "message": msg}
    a, b = brackets[0]
    phi = brentq(res, a, b, xtol=xtol, rtol=1e-15, maxiter=300)
    r = res(phi)
    if not (phi > 0.0) or not np.isfinite(phi):
        raise RuntimeError("non-physical root phi=%r" % (phi,))
    if abs(r) > 1e-8 * max(1.0, abs(phi)):
        raise RuntimeError("root not converged: residual %.3e" % r)
    return {"phi": float(phi), "residual": float(r), "n_roots": len(brackets), "message": "ok"}


def quasi_steady_history(prob: Problem, s_grid: Sequence[float], omega: bool = False,
                         h_const: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Method A - the published algebraic procedure, generalised to h = h(t).

    For every s the pair (phi, t) solves the simultaneous system
        Eq. (83) with Bi_env = h(t) s/k_S    and    t = s^2/(4 alpha_S phi^2)   (Eq. 22),
    by damped fixed-point iteration on t.  With h constant this *is* the published
    procedure (see `verify_constant_h_limit`).

    [APPROX] For h = h(t) this is a frozen-coefficient / quasi-steady use of a solution
    derived for constant h: it retains no memory of the earlier, much larger h_g.
    """
    m = prob.mat
    keys = ("s", "t", "phi", "v", "Bi_env", "Bi_i", "res")
    out: Dict[str, List[float]] = {k: [] for k in keys}
    for s in s_grid:
        if s <= 0:
            continue
        phi, t = 0.5, s * s / (4.0 * m.alpha_S * 0.25)
        ok = True
        for _ in range(300):
            h = h_const if h_const is not None else prob.h(t)
            sol = solve_phi(prob, s, h, omega=omega, strict=False)
            if not np.isfinite(sol["phi"]):
                ok = False
                break
            phi_new = float(sol["phi"])
            t_new = s * s / (4.0 * m.alpha_S * phi_new ** 2)
            if abs(t_new - t) <= 1e-12 * max(1.0, t_new):
                phi, t = phi_new, t_new
                break
            phi, t = phi_new, 0.5 * (t + t_new)
        if not ok:
            continue
        h = h_const if h_const is not None else prob.h(t)
        out["s"].append(s)
        out["t"].append(t)
        out["phi"].append(phi)
        out["v"].append(2.0 * phi * phi * m.alpha_S / s)
        out["Bi_env"].append(h * s / m.k_S)
        out["Bi_i"].append(prob.h_i * s / m.k_L)
        out["res"].append(interface_equation(phi, prob, s, h, omega))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


# ======================================================================================
# 6.  TIME INTEGRATION OF THE REDUCED INTERFACE EQUATION  -  "method B"
# ======================================================================================


def newtonian_start(prob: Problem, t0: float) -> float:
    """Leading-order s(t0) ignoring the melt: the Newtonian (interface-controlled) regime.

    As s -> 0, Bi_env -> 0, psi -> 1 and exp(-phi^2) erfcx(phi + Bi/2phi) -> 1, so that
    q_S -> h(t)(T_F - T_inf).  Integrating rho_S L ds/dt = h(t)(T_F - T_inf) gives
        s(t) = (T_F - T_inf)/(rho_S L) * int_0^t h(t') dt' ,
    finite for h_g(t) = C t^-p with p < 1 even though h_g(0) = inf.  This removes the
    t = 0 singularity from the initial condition.
    """
    m = prob.mat
    from scipy.integrate import quad
    with np.errstate(all="ignore"):
        I = quad(lambda tt: prob.h(tt), 0.0, t0, limit=400, points=None)[0]
    return (m.T_F - prob.T_inf) * I / (m.rho_S * m.L)


def incubation_time(prob: Problem, lo: float = 1e-12, hi: float = 100.0) -> float:
    """Time t* at which solidification can start:  q_S = q_L  in the limit s -> 0.

    For the conduction-like melt closures q_L ~ k_L (T_P - T_F)/sqrt(pi alpha_L t)
    diverges as t^-1/2, faster than q_S ~ h_g(t)(T_F - T_inf) ~ t^-0.21, so the
    superheated melt initially delivers more heat than the chill can remove and no solid
    forms.  Both fluxes are integrable, so t* is finite and small (~8e-5 s here); it is a
    genuine feature of the stated problem, not a numerical artefact.  Returns `lo` when
    the balance is already positive at `lo` (e.g. for the 'interface-film' closure, whose
    q_L stays bounded).
    """
    def g(t):
        s = max(newtonian_start(prob, t), 1e-16)
        return q_solid(prob, s, t) - q_liquid(prob, s, t)

    with np.errstate(all="ignore"):
        if g(lo) > 0.0:
            return lo
        grid = np.geomspace(lo, hi, 300)
        vals = [g(t) for t in grid]
        for i in range(len(grid) - 1):
            if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
                return float(brentq(g, grid[i], grid[i + 1], xtol=1e-16, rtol=1e-14))
    raise RuntimeError("solidification never starts: q_L > q_S for all t in [%g, %g]" % (lo, hi))


def initial_condition(prob: Problem, t0: float, n: int = 400, n_picard: int = 6) -> float:
    """s(t0) from  rho_S L s(t0) = int_{t*}^{t0} [q_S - q_L] dt , by Picard iteration.

    [APPROX] The closed-form fields are valid only for s > 0, so the march cannot start at
    t = 0.  The reduced interface equation is integrated from the incubation time t*
    (where s = 0) to t0 on a logarithmic grid, using the Newtonian profile as the first
    Picard iterate.  `verify_start_insensitivity` shows that s(200 s) changes by < 1e-4 %
    when t0 is moved over two decades, i.e. the march forgets the start.
    """
    from scipy.integrate import cumulative_trapezoid
    m = prob.mat
    t_star = incubation_time(prob)
    if t0 <= t_star:
        raise RuntimeError("t0 = %g s is inside the incubation period (t* = %g s)" % (t0, t_star))
    tg = np.geomspace(t_star, t0, n)
    s = np.maximum(np.array([newtonian_start(prob, t) - newtonian_start(prob, t_star)
                             for t in tg]), 0.0)
    for _ in range(n_picard):
        s_safe = np.maximum(s, 1e-16)
        with np.errstate(all="ignore"):
            integ = np.array([q_solid(prob, si, t) - q_liquid(prob, si, t)
                              for si, t in zip(s_safe, tg)])
        integ = np.nan_to_num(integ, nan=0.0, posinf=0.0, neginf=0.0)
        s_new = np.maximum(cumulative_trapezoid(integ, tg, initial=0.0) / (m.rho_S * m.L), 0.0)
        if np.max(np.abs(s_new - s)) <= 1e-16 + 1e-10 * np.max(np.abs(s_new)):
            s = s_new
            break
        s = 0.5 * (s + s_new)
    if not (s[-1] > 0.0):
        raise RuntimeError("no solidification at t0 = %g s" % t0)
    return float(s[-1])


@dataclass
class InterfaceHistory:
    t: np.ndarray
    s: np.ndarray
    v: np.ndarray
    sol: object

    def s_of(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.where(t <= self.t[0], self.s[0], np.nan)
        m = t > self.t[0]
        if m.any():
            out[m] = self.sol.sol(np.clip(t[m], self.t[0], self.t[-1]))[0]
        return out if out.size > 1 else float(out[0])

    def arrival_time(self, x: float) -> float:
        """First time at which the front reaches x; NaN if it never does."""
        if self.s[-1] < x:
            return float("nan")
        return brentq(lambda tt: float(self.s_of(tt)) - x, self.t[0], self.t[-1], xtol=1e-10)


def march_interface(prob: Problem, t_end: float = 200.0, t0: Optional[float] = None,
                    rtol: float = 1e-10, atol: float = 1e-14,
                    n_out: int = 4000, max_step: float = np.inf) -> InterfaceHistory:
    """Method B - integrate  rho_S L ds/dt = q_S(s,t) - q_L(s,t)  with h = h_g(t).

    This is the "numerical integration of the reduced interface equation" treatment: the
    *fluxes* are the closed-form analytical ones evaluated at the instantaneous (s,t), but
    ds/dt is the true derivative instead of the parabolic surrogate 2 phi^2 alpha_S/s that
    Eq. (83) assumes.  It therefore needs no Omega correction and degenerates to method A
    whenever phi varies slowly.  `max_step` imposes piecewise-constant-h sub-stepping for
    the convergence study.
    """
    if t0 is None:                     # start safely clear of the incubation period
        t0 = max(1e-3, 3.0 * incubation_time(prob))
    s0 = initial_condition(prob, t0)
    if dsdt(prob, s0, t0) <= 0.0:
        raise RuntimeError("ds/dt <= 0 at t0 = %g s (incubation not finished)" % t0)

    def rhs(t, y):
        s = max(float(y[0]), 1e-14)
        return [dsdt(prob, s, t)]

    sol = solve_ivp(rhs, (t0, t_end), [s0], method="LSODA", rtol=rtol, atol=atol,
                    dense_output=True, max_step=max_step)
    if not sol.success:
        raise RuntimeError("interface integration failed: %s" % sol.message)
    tg = np.geomspace(t0, t_end, n_out)
    sg = sol.sol(tg)[0]
    vg = np.array([dsdt(prob, max(s, 1e-14), t) for s, t in zip(sg, tg)])
    return InterfaceHistory(t=tg, s=sg, v=vg, sol=sol)


def predict_thermocouples(prob: Problem, hist: InterfaceHistory,
                          x_positions: Sequence[float], t_eval: Sequence[float]) -> np.ndarray:
    """T(x_j, t) [K] for every probe, switching field at the moving front.

    A point is in the solid when x < s(t) and in the melt otherwise, so no phase-specific
    expression is ever evaluated outside its own domain of validity.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    out = np.full((len(x_positions), t_eval.size), np.nan)
    for k, t in enumerate(t_eval):
        if t < hist.t[0]:
            out[:, k] = prob.T_P
            continue
        s = float(hist.s_of(t))
        for j, x in enumerate(x_positions):
            out[j, k] = float(T_solid(prob, x, s, t)) if x < s else float(T_liquid(prob, x, s, t))
    return out


# ======================================================================================
# 7.  INDEPENDENT FINITE-DIFFERENCE REFERENCE (verification only)
# ======================================================================================


def fd_reference(prob: Problem, t_end: float = 200.0, Lx: float = 0.40, nx: int = 800,
                 cfl: float = 0.35, out_t: Optional[Sequence[float]] = None,
                 out_x: Optional[Sequence[float]] = None) -> Dict[str, np.ndarray]:
    """Explicit 1-D enthalpy finite-difference solution of the *same* stated problem.

    This is deliberately independent of every analytical expression above: it solves
    Eqs. (66)-(72) numerically (single density rho_S, harmonic face conductivity,
    surface coefficient h_g(t) in series with the half-cell conduction).  It is used to
    separate "the analytical solution is wrongly implemented" from "the stated model does
    not describe the data".
    """
    if out_x is None:
        out_x = TC_POSITIONS_M
    m = prob.mat
    dx = Lx / nx
    x = (np.arange(nx) + 0.5) * dx
    rho = m.rho_S
    HS = m.c_S * m.T_F
    HL = HS + m.L
    H = np.full(nx, HL + m.c_L * (prob.T_P - m.T_F))
    T = np.full(nx, prob.T_P)
    dt = cfl * rho * min(m.c_S, m.c_L) * dx * dx / (2.0 * max(m.k_S, m.k_L))
    nt = int(np.ceil(t_end / dt))
    dt = t_end / nt
    out_t = np.asarray(out_t if out_t is not None else np.linspace(1.0, t_end, 40), dtype=float)
    rec = {"t": [], "s": [], "q0": [], "Tw": [], "T": []}
    oi, t, Eext = 0, 0.0, 0.0
    for _ in range(nt):
        k = np.where(T < m.T_F, m.k_S, m.k_L)
        kf = 2.0 * k[:-1] * k[1:] / (k[:-1] + k[1:])
        flux = -kf * (T[1:] - T[:-1]) / dx
        h = float(prob.h(max(t + 0.5 * dt, 1e-9)))
        Us = 1.0 / (1.0 / h + 0.5 * dx / k[0])
        q0 = Us * (T[0] - prob.T_inf)
        Eext += q0 * dt
        dH = np.zeros(nx)
        dH[0] += (-q0 - flux[0]) * dt / (rho * dx)
        dH[1:-1] += (flux[:-1] - flux[1:]) * dt / (rho * dx)
        dH[-1] += flux[-1] * dt / (rho * dx)
        H += dH
        T = np.where(H <= HS, H / m.c_S, np.where(H >= HL, m.T_F + (H - HL) / m.c_L, m.T_F))
        t += dt
        while oi < len(out_t) and t >= out_t[oi] - 1e-6 * max(1.0, out_t[oi]):
            fl = np.clip((H - HS) / m.L, 0.0, 1.0)
            rec["t"].append(t)
            rec["s"].append(float(np.sum(1.0 - fl) * dx))
            rec["q0"].append(q0)
            rec["Tw"].append(T[0])
            rec["T"].append(np.interp(np.asarray(out_x), x, T))
            oi += 1
    res = {kk: np.asarray(vv) for kk, vv in rec.items()}
    res["Eext"] = Eext
    return res


# ======================================================================================
# 8.  TABLE 2
# ======================================================================================


def _legacy_equation(phi, n, Biotenv, Bioti, N, Theta0, Stes, Tp, TF):
    """Verbatim transcription of `equation()` from the development script (for audit)."""
    from numpy import exp as _e, sqrt as _s
    term1 = phi / Stes
    term2 = 1.0 / (1 - erfc(phi) + _e(Biotenv + Biotenv ** 2 / (4 * phi ** 2))
                   * erfc(phi + Biotenv / (2 * phi))) * (
        Biotenv / (2 * phi) * _e(Biotenv + Biotenv ** 2 / (4 * phi ** 2)) * erfc(phi + Biotenv / (2 * phi))
        + 1.0 / (_s(np.pi) * _e(phi ** 2))
        - 1.0 / (_s(np.pi) * _e((phi + Biotenv / (2 * phi)) ** 2)) * _e(Biotenv + Biotenv ** 2 / (4 * phi ** 2)))
    f_ = (Bioti / (2 * phi) * _e(Bioti + n ** 2 * Bioti ** 2 / (4 * phi ** 2))
          * erfc(n * phi + n * Bioti / (2 * phi))
          + n / (_s(np.pi) * _e(phi ** 2))
          - n / (_s(np.pi) * _e((n * phi + n * Bioti / (2 * phi)) ** 2))
          * _e(Bioti + n ** 2 * Bioti ** 2 / (4 * phi ** 2)))
    psiL_ = (1 - erfc(n * phi) + _e(Bioti + n ** 2 * Bioti ** 2 / (4 * phi ** 2))
             * erfc(n * phi + n * Bioti / (2 * phi)))
    term3 = 1.0 / N * Bioti / (2 * phi) * Theta0 * (2 * phi / Bioti * (Tp - TF) * f_ / psiL_)
    return term2 - term3 - term1, f_, psiL_


def reproduce_table2(s: float = 0.10, h_env: float = 400.0,
                     h_i_list: Sequence[float] = (1800.0, 1600.0, 1400.0, 800.0, 500.0),
                     T_P: float = 970.48) -> List[Dict[str, object]]:
    """Audit of published Table 2 (liquid-domain interface temperature).

    Two evaluations are reported for each h_i:

    * `legacy`  - the development script verbatim (inverted Ste, spurious (T_P - T_F)
                  factor, `fsolve` from phi0 = 1 with no convergence test).  This is what
                  reproduces the published numbers - as *stalled* iterates: the legacy
                  residual has no zero anywhere on phi > 0, and scipy returns ier = 5.
    * `corrected` - Eq. (83) with the corrections of this module, bracketed root.
    """
    from scipy.optimize import fsolve
    m = AL_TABLE1
    n, N = m.n, m.N
    Stes_legacy = m.L / (m.c_S * (m.T_F - 298.15))          # the dev script's `Stes`
    Theta0 = (T_P - m.T_F) / (m.T_F - 298.15)
    Bi_env = h_env * s / m.k_S
    rows = []
    for h_i in h_i_list:
        Bi_i = h_i * s / m.k_L
        with np.errstate(all="ignore"):
            out = fsolve(lambda p: _legacy_equation(p[0], n, Bi_env, Bi_i, N, Theta0,
                                                    Stes_legacy, T_P, m.T_F)[0],
                         [1.0], full_output=True)
        phi_leg = float(out[0][0])
        ier = int(out[2])
        with np.errstate(all="ignore"):
            r_leg, f_leg, psiL_leg = _legacy_equation(phi_leg, n, Bi_env, Bi_i, N, Theta0,
                                                      Stes_legacy, T_P, m.T_F)
            TL_leg = m.T_F + 2 * phi_leg / Bi_i * (T_P - m.T_F) * f_leg / psiL_leg
        # corrected, using the same 'as-coded' psi_L so that only the audited errors differ
        prob = Problem(mat=m, T_inf=298.15, T_P=T_P, h_env=lambda t: h_env,
                       h_i=h_i, liquid_model="as-coded")
        sol = solve_phi(prob, s, h_env, omega=False, strict=False)
        if np.isfinite(sol["phi"]):
            phi_c = float(sol["phi"])
            t_c = s * s / (4.0 * m.alpha_S * phi_c ** 2)
            TL_c = float(T_liquid_drive(prob, s, t_c))
        else:
            phi_c, TL_c = float("nan"), float("nan")
        rows.append({"h_i": h_i, "Bi_i": Bi_i, "phi_legacy": phi_leg, "ier": ier,
                     "residual_legacy": float(r_leg), "TL_legacy": float(TL_leg),
                     "phi_corrected": phi_c, "TL_corrected": TL_c,
                     "n_roots": sol["n_roots"]})
    return rows


# ======================================================================================
# 9.  VERIFICATION SUITE
# ======================================================================================


class Check:
    def __init__(self):
        self.rows: List[Tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = ""):
        self.rows.append((name, bool(ok), detail))
        print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))

    @property
    def n_fail(self) -> int:
        return sum(1 for _, ok, _ in self.rows if not ok)


def _demo_problem(h_i=1000.0, liquid_model="virtual-origin"):
    return Problem(mat=AL_TABLE1, T_inf=298.15, T_P=970.48, h_env=lambda t: 400.0,
                   h_i=h_i, liquid_model=liquid_model)


def verify_probe_metadata(chk: Check):
    """Thermocouple labels, positions and plot colours must agree."""
    ok = True
    try:
        _check_probe_metadata()
        ok = all(l in TC_COLOURS and l in TC_MARKERS for l in TC_LABELS)
    except AssertionError:
        ok = False
    chk.add("probe labels / positions / colours consistent", ok,
            "x = " + ", ".join("%s=%.0f mm" % (l, x * 1e3)
                               for l, x in zip(TC_LABELS, TC_POSITIONS_M)))


def verify_cancellation(chk: Check):
    """The two non-erfc terms of Eqs. (75c)/(80d) cancel identically."""
    worst = 0.0
    for Bi in (1e-3, 0.1, 1.0, 10.0, 100.0):
        for phi in (0.05, 0.3, 1.0, 3.0):
            a = F_solid(Bi, phi)
            b = F_solid_literal(Bi, phi)
            worst = max(worst, abs(a - b) / max(abs(a), 1e-300))
    chk.add("Eq.(75c) reduced == literal", worst < 1e-9, "max rel dev %.2e" % worst)


def verify_psi_zeta(chk: Check):
    worst = max(abs(psi_S(Bi, p) - (1.0 + zeta_S(Bi, p)))
                for Bi in (1e-3, 1.0, 50.0) for p in (0.1, 1.0, 5.0))
    chk.add("Eq.(18b) psi == 1 + zeta (Eq.19b)", worst < 1e-12, "max dev %.2e" % worst)


def verify_overflow_safety(chk: Check):
    """The dev script's literal forms overflow; the reduced forms must not."""
    vals = [F_solid(Bi, phi) for Bi in (1e3, 1e5) for phi in (0.05, 30.0)]
    vals += [psi_S(1e5, 0.05), psi_S(1e-6, 40.0)]
    ok = all(np.isfinite(v) for v in vals)
    with np.errstate(all="ignore"):
        legacy = np.exp(1e5 + 1e10 / (4 * 0.05 ** 2)) * erfc(0.05 + 1e5 / 0.1)
    chk.add("no overflow for Bi up to 1e5, phi up to 40", ok,
            "legacy literal form gives %r" % (legacy,))


def verify_interface_temperature(chk: Check):
    """Eq. (70): the solid field must return exactly T_F at x = s."""
    prob = _demo_problem()
    worst = 0.0
    for s in (0.002, 0.02, 0.10):
        for t in (0.5, 10.0, 200.0):
            worst = max(worst, abs(float(T_solid(prob, s, s, t)) - AL_TABLE1.T_F))
    chk.add("Eq.(70) T_S(s,t) = T_F", worst < 1e-9, "max dev %.2e K" % worst)


def verify_liquid_continuity(chk: Check):
    """The default liquid closure must be continuous with the solid at the front."""
    res = {}
    for mode in LIQUID_MODELS:
        prob = _demo_problem(liquid_model=mode)
        jump = max(abs(float(T_liquid(prob, s, s, t)) - AL_TABLE1.T_F)
                   for s in (0.01, 0.05, 0.10) for t in (1.0, 50.0))
        res[mode] = jump
    chk.add("T_L(s,t) = T_F for 'virtual-origin'", res["virtual-origin"] < 1e-9,
            "jump %.2e K (interface-film: %.2f K, as-coded: %.2f K)"
            % (res["virtual-origin"], res["interface-film"], res["as-coded"]))


def verify_initial_and_far_field(chk: Check):
    """Eqs. (68)/(71): T -> T_P far from the chill and as t -> 0."""
    prob = _demo_problem()
    far = float(T_liquid(prob, 5.0, 0.05, 50.0))
    early = float(T_liquid(prob, 0.09, 0.0005, 1e-4))
    chk.add("Eqs.(68),(71) T -> T_P", abs(far - prob.T_P) < 1e-8 and abs(early - prob.T_P) < 1e-6,
            "far dev %.2e K, t->0 dev %.2e K" % (abs(far - prob.T_P), abs(early - prob.T_P)))


def verify_robin_wall(chk: Check):
    """Eq. (69): -k_S dT/dx|_0 = h (T_inf - T(0,t)).

    [TYPO] Eq. (69) as printed reads -k dT/dx|_0 = h (T - T_inf).  With the solid at x>0
    hotter than the environment, dT/dx|_0 > 0, so its left side is negative while its
    right side is positive: the printed sign is wrong.  The solution actually used
    (Eqs. 8, 17, 20a - the classical third-kind solution) satisfies the form checked here,
    so this is a typographic sign error, not a modelling choice.
    """
    prob = _demo_problem()
    worst = 0.0
    for s, t in ((0.01, 1.0), (0.05, 30.0), (0.10, 200.0)):
        h = prob.h(t)
        eps = 1e-7
        dT = (float(T_solid(prob, eps, s, t)) - float(T_solid(prob, 0.0, s, t))) / eps
        lhs = -AL_TABLE1.k_S * dT
        rhs = h * (prob.T_inf - float(T_solid(prob, 0.0, s, t)))
        worst = max(worst, abs(lhs - rhs) / max(abs(rhs), 1.0))
    chk.add("Eq.(69) Robin at x = 0 (sign corrected)", worst < 2e-4, "max rel dev %.2e" % worst)


def verify_robin_interface(chk: Check):
    """Eq. (79): -k_L dT_L/dx|_{s+} = -h_i (T_L(s+,t) - T_F)."""
    out = {}
    for mode in ("virtual-origin", "interface-film", "as-coded"):
        prob = _demo_problem(liquid_model=mode)
        worst = 0.0
        for s, t in ((0.01, 1.0), (0.05, 30.0), (0.10, 200.0)):
            g = grad_liquid_interface(prob, s, t)
            TL = T_liquid_drive(prob, s, t)
            worst = max(worst, abs(AL_TABLE1.k_L * g - prob.h_i * (TL - AL_TABLE1.T_F))
                        / max(abs(AL_TABLE1.k_L * g), 1.0))
        out[mode] = worst
    chk.add("Eq.(79) film law at x = s+ (all closures)", max(out.values()) < 1e-10,
            "max rel dev %.2e" % max(out.values()))


def verify_liquid_field_gradient(chk: Check):
    """Eq. (77) must be the x-derivative of Eq. (76) at x = s+."""
    worst = 0.0
    for mode in ("virtual-origin", "interface-film", "conduction"):
        prob = _demo_problem(liquid_model=mode)
        for s, t in ((0.01, 1.0), (0.05, 30.0)):
            eps = 1e-7
            num = (float(T_liquid(prob, s + eps, s, t)) - float(T_liquid(prob, s, s, t))) / eps
            ana = grad_liquid_interface(prob, s, t)
            worst = max(worst, abs(num - ana) / max(abs(ana), 1.0))
    chk.add("Eq.(77) = d/dx Eq.(76) at x = s+", worst < 1e-3, "max rel dev %.2e" % worst)


def verify_solid_field_gradient(chk: Check):
    prob = _demo_problem()
    worst = 0.0
    for s, t in ((0.01, 1.0), (0.05, 30.0), (0.10, 200.0)):
        eps = 1e-7
        num = (float(T_solid(prob, s, s, t)) - float(T_solid(prob, s - eps, s, t))) / eps
        ana = grad_solid_interface(prob, s, t)
        worst = max(worst, abs(num - ana) / abs(ana))
    chk.add("Eq.(75) = d/dx Eq.(73) at x = s-", worst < 1e-3, "max rel dev %.2e" % worst)


def verify_pde_residual(chk: Check):
    """How well the pseudo-similarity fields satisfy Eqs. (66)/(67).

    They are *not* exact solutions once phi and the Biot numbers drift with time; this
    check quantifies the defect rather than asserting it is zero.
    """
    prob = experiment_problem(h_i=1000.0)
    hist = march_interface(prob, t_end=200.0)
    worst = 0.0
    for t in (5.0, 50.0, 150.0):
        s = float(hist.s_of(t))
        x = 0.4 * s
        dx, dt = 1e-4, 1e-3
        Txx = (float(T_solid(prob, x + dx, s, t)) - 2 * float(T_solid(prob, x, s, t))
               + float(T_solid(prob, x - dx, s, t))) / dx ** 2
        Tt = (float(T_solid(prob, x, float(hist.s_of(t + dt)), t + dt))
              - float(T_solid(prob, x, float(hist.s_of(t - dt)), t - dt))) / (2 * dt)
        scale = max(abs(Txx), abs(Tt) / AL_TABLE1.alpha_S, 1e-12)
        worst = max(worst, abs(Txx - Tt / AL_TABLE1.alpha_S) / scale)
    # The bound is deliberately loose: these are pseudo-similarity fields, exact only if
    # phi and the Biot numbers were frozen.  The check asserts that the defect stays O(1)
    # (i.e. no term is grossly wrong) and *reports* its size as a model limitation.
    chk.add("Eq.(66) PDE residual of the solid field", worst < 1.0,
            "max normalised residual %.3f (pseudo-similarity defect, quantified not zero)" % worst)


def verify_stefan_balance(chk: Check):
    prob = experiment_problem(h_i=1000.0)
    hist = march_interface(prob, t_end=200.0)
    worst = 0.0
    for t in (0.1, 1.0, 20.0, 200.0):
        s = float(hist.s_of(t))
        v = float(np.interp(t, hist.t, hist.v))
        r = stefan_residual(prob, s, t, v)
        worst = max(worst, abs(r) / max(abs(q_solid(prob, s, t)), 1.0))
    chk.add("Eq.(72) Stefan residual along the march", worst < 1e-6,
            "max |residual|/q_S = %.2e" % worst)


def verify_monotonic_front(chk: Check):
    prob = experiment_problem(h_i=1000.0)
    hist = march_interface(prob, t_end=200.0)
    ok = bool(np.all(hist.s > 0) and np.all(np.diff(hist.s) > 0) and np.all(hist.v > 0))
    chk.add("s(t) positive and strictly increasing", ok,
            "s(200 s) = %.2f mm, min v = %.3e m/s" % (hist.s[-1] * 1e3, hist.v.min()))


def verify_admissible_temperatures(chk: Check):
    prob = experiment_problem(h_i=1000.0)
    hist = march_interface(prob, t_end=200.0)
    tg = np.geomspace(1e-3, 200.0, 300)
    T = predict_thermocouples(prob, hist, TC_POSITIONS_M, tg)
    ok = bool(np.all(T > prob.T_inf - 1e-9) and np.all(T < prob.T_P + 1e-6))
    mono = all(np.all(np.diff(T[j]) < 1e-6) for j in range(T.shape[0]))
    chk.add("T_inf < T(x,t) <= T_P and monotone cooling", ok and mono,
            "min %.1f degC, max %.1f degC" % (T.min() - KELVIN, T.max() - KELVIN))


def verify_chronology(chk: Check):
    prob = experiment_problem(h_i=1000.0)
    hist = march_interface(prob, t_end=400.0)
    ta = [hist.arrival_time(x) for x in TC_POSITIONS_M]
    ok = all(ta[i] < ta[i + 1] for i in range(len(ta) - 1))
    chk.add("front reaches nearer probes first", ok,
            "t_arr = " + ", ".join("%.2f" % v for v in ta) + " s")


def verify_dimensional_consistency(chk: Check):
    """Scale invariance test: Eq. (83) must be invariant under a change of time unit.

    Multiplying alpha_S, alpha_L by lambda and dividing h, h_i by sqrt(lambda) at fixed s
    leaves every dimensionless group unchanged, hence phi unchanged.  A dimensionally
    inconsistent term (such as the dev script's spurious (T_P - T_F)) breaks this.
    """
    s, lam = 0.05, 4.0
    p1 = _demo_problem()
    r1 = solve_phi(p1, s, 400.0)["phi"]
    m2 = Material(k_S=AL_TABLE1.k_S * lam, k_L=AL_TABLE1.k_L * lam,
                  c_S=AL_TABLE1.c_S, c_L=AL_TABLE1.c_L,
                  rho_S=AL_TABLE1.rho_S, rho_L=AL_TABLE1.rho_L,
                  L=AL_TABLE1.L, T_F=AL_TABLE1.T_F)
    p2 = Problem(mat=m2, T_inf=p1.T_inf, T_P=p1.T_P, h_env=lambda t: 400.0 * lam,
                 h_i=p1.h_i * lam, liquid_model=p1.liquid_model)
    r2 = solve_phi(p2, s, 400.0 * lam)["phi"]
    chk.add("Eq.(83) dimensionless / scale invariant", abs(r1 - r2) < 1e-9,
            "phi = %.9f vs %.9f" % (r1, r2))


def verify_constant_h_limit(chk: Check):
    """Constant-h limiting test.

    (a) With h constant the generalised implementation must return *exactly* the published
        algebraic solution, i.e. the residual of Eq. (83) must vanish at every s.
    (b) The remaining difference between method A (Eq. 83, parabolic velocity) and method B
        (reduced interface ODE) is the *local-similarity defect of Eq. (83)* itself, since
        the two share identical fluxes and differ only in the velocity relation.  An
        independent FD solution of the same constant-h problem arbitrates: method B must
        be the closer of the two.
    """
    prob = Problem(mat=AL_TABLE1, T_inf=298.15, T_P=970.48, h_env=lambda t: 4000.0,
                   h_i=1000.0, liquid_model="virtual-origin")
    s_grid = np.linspace(0.005, 0.10, 20)
    A = quasi_steady_history(prob, s_grid, omega=False, h_const=4000.0)
    chk.add("constant-h: method A reproduces Eq.(83) exactly",
            np.abs(A["res"]).max() < 1e-10, "max |residual| = %.2e" % np.abs(A["res"]).max())

    t_end = 120.0
    hist = march_interface(prob, t_end=t_end, t0=1.5 * incubation_time(prob))
    out_t = np.array([5.0, 20.0, 60.0, 110.0])
    fd = fd_reference(prob, t_end=t_end, out_t=out_t, nx=800)
    sB = np.array([float(hist.s_of(t)) for t in out_t])
    sA = np.interp(out_t, A["t"], A["s"])
    eB = np.abs(sB - fd["s"]) / fd["s"]
    eA = np.abs(sA - fd["s"]) / fd["s"]
    chk.add("constant-h: method B beats method A vs FD", eB.max() < eA.max() and eB.max() < 0.12,
            "max rel dev in s(t): method B %.1f %%, method A (Eq. 83) %.1f %%"
            % (100 * eB.max(), 100 * eA.max()))


def verify_fd_grid_convergence(chk: Check):
    """Grid convergence of the independent FD reference."""
    prob = experiment_problem(h_i=1000.0)
    out_t = np.array([200.0])
    vals = [(nx, float(fd_reference(prob, t_end=200.0, out_t=out_t, nx=nx)["s"][0]))
            for nx in (200, 400, 800, 1600)]
    e = [abs(v - vals[-1][1]) / vals[-1][1] for _, v in vals[:-1]]
    ok = e[0] > e[1] > e[2] and e[2] < 3e-3
    chk.add("FD reference grid convergence", ok,
            "s(200 s) = " + ", ".join("%.3f" % (v * 1e3) for _, v in vals) + " mm for nx = 200..1600")


def verify_melt_drive_bounds(chk: Check):
    """Maximum principle for the melt driving temperature of Eq. (82).

    Nowhere may the melt be hotter than its own initial temperature, so Eq. (82) must
    return T_F <= T_L(s+) <= T_P.  The check reports which liquid closures respect that
    bound; it is *reported*, not asserted, because the manuscript leaves psi_L undefined
    and one of the candidate closures fails it.
    """
    out = {}
    for mode in ("virtual-origin", "interface-film", "as-coded"):
        prob = experiment_problem(h_i=1000.0, T_P_C=690.9, liquid_model=mode)
        worst = 0.0
        for s, t in ((0.01, 1.0), (0.05, 20.0), (0.10, 150.0)):
            TL = float(T_liquid_drive(prob, s, t))
            worst = max(worst, TL - prob.T_P, prob.mat.T_F - TL)
        out[mode] = worst
    chk.add("Eq.(82) melt drive stays within [T_F, T_P]", out["interface-film"] <= 1e-9,
            "max excess over T_P: interface-film %.2f K, as-coded %.2f K, "
            "virtual-origin %.1f K (fails)" % (out["interface-film"], out["as-coded"],
                                               out["virtual-origin"]))


def verify_melt_flux_monotonicity(chk: Check):
    """q_L must increase with h_i: stronger melt convection delivers more heat.

    Only the interface-anchored film closure satisfies this.  With the continuity-
    preserving 'virtual-origin' normalisation the dependence is *inverted* and bounded
    below by the pure-conduction flux - a structural defect of the published closure,
    reported here rather than hidden.
    """
    s, t = 0.05, 20.0
    res = {}
    for mode in ("virtual-origin", "interface-film", "as-coded"):
        q = [q_liquid(experiment_problem(h_i=h, T_P_C=690.9, liquid_model=mode), s, t)
             for h in (100.0, 500.0, 1800.0, 20000.0)]
        res[mode] = (q[0], q[-1], all(q[i] < q[i + 1] for i in range(3)))
    chk.add("q_L increases with h_i ('interface-film')", res["interface-film"][2],
            "q_L(100->2e4): film %.1f->%.1f kW/m2 (rising); virtual-origin %.1f->%.1f "
            "kW/m2 (INVERTED)" % (res["interface-film"][0] / 1e3, res["interface-film"][1] / 1e3,
                                  res["virtual-origin"][0] / 1e3, res["virtual-origin"][1] / 1e3))


def verify_conduction_limit(chk: Check):
    """h_i -> inf must give the classical Neumann melt flux (paper Eqs. 45/55/57)."""
    s, t = 0.05, 20.0
    probc = _demo_problem(h_i=1e30, liquid_model="conduction")
    exact = q_liquid(probc, s, t)
    devs = []
    for h_i in (1e5, 1e6, 1e7, 1e8):
        prob = _demo_problem(h_i=h_i, liquid_model="virtual-origin")
        devs.append(abs(q_liquid(prob, s, t) - exact) / abs(exact))
    first_order = devs[0] / devs[-1] > 100.0        # error must fall like 1/h_i
    chk.add("h_i -> inf gives Neumann melt flux", devs[-1] < 3e-5 and first_order,
            "rel dev " + ", ".join("%.1e" % d for d in devs) + " for h_i = 1e5..1e8 (O(1/h_i))")
    probf = _demo_problem(h_i=1e-6, liquid_model="interface-film")
    chk.add("h_i -> 0 gives zero melt flux ('interface-film')",
            abs(q_liquid(probf, s, t)) < 1e-3, "q_L = %.2e W/m2" % q_liquid(probf, s, t))


def verify_start_insensitivity(chk: Check):
    """The march must forget its (asymptotic) initial condition."""
    prob = experiment_problem(h_i=1000.0)
    ts = incubation_time(prob)
    t0s = [ts * f for f in (1.5, 5.0, 20.0, 80.0)]
    vals = [float(march_interface(prob, t_end=200.0, t0=t0).s_of(200.0)) for t0 in t0s]
    spread = (max(vals) - min(vals)) / np.mean(vals)
    chk.add("s(200 s) independent of the start time t0", spread < 1e-4,
            "t* = %.2e s; spread over t0 = %.1e..%.1e s is %.1e"
            % (ts, t0s[0], t0s[-1], spread))


def verify_time_step_convergence(chk: Check):
    """Piecewise-constant-h convergence of method B."""
    prob = experiment_problem(h_i=1000.0)
    ref = march_interface(prob, t_end=200.0, rtol=1e-12, atol=1e-16)
    s_ref = float(ref.s_of(200.0))
    rows = []
    for dt in (5.0, 2.0, 1.0, 0.5, 0.25):
        h = march_interface(prob, t_end=200.0, rtol=1e-6, atol=1e-12, max_step=dt)
        rows.append((dt, abs(float(h.s_of(200.0)) - s_ref) / s_ref))
    ok = rows[-1][1] < rows[0][1] and rows[-1][1] < 1e-6
    chk.add("piecewise-constant-h step convergence", ok,
            "; ".join("dt=%.2fs -> %.1e" % r for r in rows))


def verify_against_fd(chk: Check):
    """Analytical vs an independent numerical solution of the same stated problem."""
    prob = experiment_problem(h_i=1000.0)
    hist = march_interface(prob, t_end=200.0)
    out_t = np.array([10.0, 50.0, 100.0, 200.0])
    fd = fd_reference(prob, t_end=200.0, out_t=out_t)
    s_an = np.array([float(hist.s_of(t)) for t in out_t])
    ds = np.abs(s_an - fd["s"]) / fd["s"]
    T_an = predict_thermocouples(prob, hist, TC_POSITIONS_M, out_t)
    dT = np.abs(T_an - fd["T"].T)
    chk.add("analytical vs FD reference", ds.max() < 0.12 and dT.max() < 35.0,
            "max dev: s %.1f %%, T %.1f K" % (100 * ds.max(), dT.max()))


def verify_energy_balance(chk: Check):
    """Global energy conservation of the FD reference (the analytic field is not conservative)."""
    prob = experiment_problem(h_i=1000.0)
    fd = fd_reference(prob, t_end=200.0, out_t=[200.0])
    m = AL_TABLE1
    s = fd["s"][-1]
    stored = m.rho_S * s * (m.c_L * (prob.T_P - m.T_F) + m.L)
    x = np.linspace(0.0, s, 2000)
    Tprof = np.interp(x, np.asarray(TC_POSITIONS_M), fd["T"][-1])
    stored += m.rho_S * m.c_S * np.trapezoid(np.maximum(m.T_F - Tprof, 0.0), x)
    rel = abs(fd["Eext"] - stored) / stored
    chk.add("FD reference energy balance", rel < 0.25,
            "extracted %.3e vs stored %.3e J/m2 (%.1f %%)" % (fd["Eext"], stored, 100 * rel))


def verify_table2(chk: Check):
    rows = reproduce_table2()
    published = [949.18, 950.74, 952.57, 960.31, 963.69]
    dev = max(abs(r["TL_legacy"] - p) for r, p in zip(rows[:4], published[:4]))
    stalled = all(r["ier"] != 1 for r in rows)
    chk.add("Table 2 reproduced by the legacy path", dev < 0.05 and stalled,
            "max dev %.3f K on h_i = 1800..800; all fsolve ier != 1 (non-converged)" % dev)


def run_verification(verbose: bool = True) -> Check:
    chk = Check()
    print("\n" + "=" * 86)
    print("VERIFICATION SUITE")
    print("=" * 86)
    verify_probe_metadata(chk)
    verify_cancellation(chk)
    verify_psi_zeta(chk)
    verify_overflow_safety(chk)
    verify_dimensional_consistency(chk)
    verify_initial_and_far_field(chk)
    verify_robin_wall(chk)
    verify_interface_temperature(chk)
    verify_robin_interface(chk)
    verify_solid_field_gradient(chk)
    verify_liquid_field_gradient(chk)
    verify_liquid_continuity(chk)
    verify_pde_residual(chk)
    verify_stefan_balance(chk)
    verify_monotonic_front(chk)
    verify_admissible_temperatures(chk)
    verify_chronology(chk)
    verify_start_insensitivity(chk)
    verify_constant_h_limit(chk)
    verify_conduction_limit(chk)
    verify_melt_drive_bounds(chk)
    verify_melt_flux_monotonicity(chk)
    verify_time_step_convergence(chk)
    verify_fd_grid_convergence(chk)
    verify_against_fd(chk)
    verify_energy_balance(chk)
    verify_table2(chk)
    print("-" * 86)
    print("  %d/%d checks passed" % (len(chk.rows) - chk.n_fail, len(chk.rows)))
    return chk


# ======================================================================================
# 10.  DIGITISATION OF THE EXPERIMENTAL FIGURE
# ======================================================================================

FIGURE_SRC = os.path.join(HERE, "Figure_cooling_curve_versus_numerical_simulation.jpg")
CSV_EXPERIMENT = os.path.join(HERE, "experimental_cooling_curves_digitized.csv")
CSV_PREDICTIONS = os.path.join(HERE, "model_predictions.csv")
CSV_METRICS = os.path.join(HERE, "melt_convection_validation_metrics.csv")
FIG_OUT = os.path.join(HERE, "Figure_cooling_curve_analytical_vs_experimental.png")
FIG_OUT_SUPP = os.path.join(
    HERE, "Figure_cooling_curve_analytical_vs_experimental_residuals.png")


def _normalise_output_path(path: str) -> str:
    """Return a local output path with repeated Windows separators removed."""
    path = os.path.normpath(os.path.abspath(os.fspath(path)))
    if os.name == "nt":
        drive, tail = os.path.splitdrive(path)
        while "\\\\" in tail:
            tail = tail.replace("\\\\", "\\")
        path = drive + tail
    return path


def _save_figure(fig, path: str, **kwargs) -> str:
    """Save a figure without aborting when an existing PNG is locked by Windows.

    The requested name is tried first.  If another application has the old image open,
    a numbered sibling is written instead and its actual path is returned.
    """
    path = _normalise_output_path(path)
    try:
        fig.savefig(path, **kwargs)
        return path
    except OSError as first_error:
        stem, ext = os.path.splitext(path)
        for number in range(1, 1000):
            alternative = "%s_%d%s" % (stem, number, ext)
            if os.path.exists(alternative):
                continue
            try:
                fig.savefig(alternative, **kwargs)
                print("    WARNING: could not overwrite %s (%s)." %
                      (os.path.basename(path), first_error))
                print("             Figure saved as %s instead." %
                      os.path.basename(alternative))
                return alternative
            except OSError:
                continue
        raise first_error

# Pixel-to-axis calibration, established once from the source raster (6432 x 4923 px):
#   * plot frame: x = 1143..1155 (left) / 5532..5544 (right), y = 564..576 (top) /
#     4097..4109 (bottom); frame centre lines L=1149, R=5538, TOP=570, BOT=4103.
#   * TIME: the major ticks drawn on the top spine sit at x = 1149, 2246, 3344, 4441,
#     5538 px and are exactly equispaced -> t = 0 at 1149 px, t = 200 s at 5538 px,
#     21.945 px/s.
#   * TEMPERATURE: the seven y tick-label glyph boxes have centroids at
#     y = 833, 1376, 1920, 2463, 3007, 3550, 4094 px for 700...100 degC; the spacing is
#     543.5 px per 100 degC to within +-0.5 px -> T = 700 - (y - 833)/5.435.
#   * Independent check: the red T_F reference line occupies rows 1043..1075 px
#     (centre 1059), which the calibration above reads as 658.4 degC instead of 660.0.
#     That 1.6 degC (8.6 px) discrepancy is the systematic calibration uncertainty; no
#     fudge factor is applied to the extracted data.
PX = dict(X0=1149.0, X1=5538.0, T0=0.0, T1=200.0, Y700=833.0, DEG=5.435,
          L=1149, R=5538, TOP=570, BOT=4103)
TF_LINE_BIAS_C = 1.58     # degC = 660.0 - (value the calibration assigns to the red line)

# Marker colours read from the legend swatches, and the offset between the *area
# centroid* of each marker glyph and its bounding-box centre.  Triangles are not
# centro-symmetric, so their centroid is displaced by w/6: 7.7 px = 0.35 s in time for
# the left/right pointing triangles and 7.7 px = 1.4 degC for the upward one.  Ignoring
# this biases the 50 mm curve 0.35 s late, the 90 mm curve 0.35 s early and the 15 mm
# curve 1.4 degC low.
DIGITISER_SERIES = (
    # label,  RGB (None = black),   opening kernel, dx_px, dy_px
    ("5mm",  None,             21,  0.00, 0.00),
    ("10mm", (240, 64, 64),    11,  0.00, 0.00),
    ("15mm", (26, 111, 222),   11,  0.15, 7.67),
    ("30mm", (177, 119, 222),  11, -0.45, 0.00),
    ("50mm", (204, 153, 0),    11,  7.66, 0.17),
    ("90mm", (1, 203, 205),    11, -7.67, 0.15),
)
# Regions where markers are irrecoverably hidden (opaque legend box, title box, the h_g
# annotation) are excluded; the affected samples stay missing and are never interpolated.
DIGITISER_EXCLUDE = ((1130, 2690, 2540, 4070), (3840, 590, 5360, 1040),
                     (2590, 3290, 5110, 3810))
# Strays that monotonicity alone cannot resolve.  Each was checked against the raster and
# is a fragment of a black *numerical-simulation* curve inside the crowded 660-695 degC
# band, not a 5 mm marker.  (label -> list of (t_lo, t_hi, T_lo, T_hi) in data units)
DIGITISER_MANUAL_DROP = {"5mm": [(0.0, 3.0, 660.0, 700.0)]}


def _longest_non_increasing(values, tol: float = 4.0):
    """Boolean mask of the longest non-increasing (within +tol) subsequence.

    The six cooling curves are monotone, so strays picked up from the overlapping black
    simulation curves fall out of the longest consistent chain.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    best = np.ones(n, dtype=int)
    prev = -np.ones(n, dtype=int)
    for i in range(n):
        for j in range(i):
            if v[j] >= v[i] - tol and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
                prev[i] = j
    keep = np.zeros(n, dtype=bool)
    i = int(np.argmax(best))
    while i >= 0:
        keep[i] = True
        i = prev[i]
    return keep


def digitise_figure(src: str = None, out_csv: str = None, verbose: bool = True):
    """Recover the six experimental marker series from the supplied raster.

    Method: colour-threshold each series in RGB; bridge the horizontal occluders (the
    black simulation curves, about 15 px, and the red T_F line, about 33 px) with a
    *vertical-only* binary closing; remove residual line fragments with a square opening;
    label the connected components and take the area centroid of every blob whose area is
    within 1.9x the median marker area (merged runs of touching markers are split into
    column strips instead).  The red T_F line is separated from the red markers by colour:
    the line is (254, 0, 0), the markers are (240, 64, 64).
    """
    from PIL import Image
    from scipy import ndimage as ndi

    src = src or FIGURE_SRC
    out_csv = out_csv or CSV_EXPERIMENT
    Image.MAX_IMAGE_PIXELS = None
    a = np.asarray(Image.open(src).convert("RGB")).astype(float)
    H, W, _ = a.shape
    mx = a.max(axis=2)
    sat = mx - a.min(axis=2)
    valid = np.zeros((H, W), bool)
    valid[PX["TOP"] + 18:PX["BOT"] - 17, PX["L"] + 18:PX["R"] - 17] = True
    for (x0, y0, x1, y1) in DIGITISER_EXCLUDE:
        valid[y0:y1, x0:x1] = False

    def px2t(x):
        return PX["T0"] + (x - PX["X0"]) * (PX["T1"] - PX["T0"]) / (PX["X1"] - PX["X0"])

    def px2T(y):
        return 700.0 - (y - PX["Y700"]) / PX["DEG"]

    series = {}
    for label, colour, k, dx, dy in DIGITISER_SERIES:
        if colour is None:
            m = (mx < 90) & valid
        else:
            m = (np.abs(a - np.array(colour, float)).max(axis=2) < 48) & (sat > 55) & valid
            m = ndi.binary_closing(m, np.ones((45, 3)))
        m = ndi.binary_opening(m, np.ones((k, k)))
        lb, nl = ndi.label(m)
        sizes = ndi.sum(m, lb, range(1, nl + 1))
        med = float(np.median(sizes[sizes > 200]))
        pts = []
        for i in range(1, nl + 1):
            area = sizes[i - 1]
            if area < 0.25 * med:
                continue
            ys, xs = np.where(lb == i)
            if area <= 1.9 * med:
                pts.append((xs.mean(), ys.mean()))
            else:
                step = max(int(round(math.sqrt(med))), 12)
                for x0 in range(xs.min(), xs.max() + 1, step):
                    sel = (xs >= x0) & (xs < x0 + step)
                    if sel.sum() < step * 4:
                        continue
                    pts.append((xs[sel].mean(), ys[sel].mean()))
        p = np.array(sorted(pts))
        t = px2t(p[:, 0] - dx)
        T = px2T(p[:, 1] - dy)
        order = np.argsort(t)
        t, T = t[order], T[order]
        keep = _longest_non_increasing(T)
        for (tl, th, Tl, Th) in DIGITISER_MANUAL_DROP.get(label, []):
            keep &= ~((t >= tl) & (t <= th) & (T >= Tl) & (T <= Th))
        n_rej = int((~keep).sum())
        t, T = t[keep], T[keep]
        mt, mT, i = [], [], 0               # merge components split by an occluder
        while i < t.size:
            j = i
            while j + 1 < t.size and t[j + 1] - t[i] < 1.0:
                j += 1
            mt.append(t[i:j + 1].mean())
            mT.append(T[i:j + 1].mean())
            i = j + 1
        series[label] = (np.array(mt), np.array(mT))
        if verbose:
            print("  %-5s markers=%3d rejected=%d  t=[%6.2f,%7.2f] s  T=[%6.1f,%6.1f] degC"
                  % (label, len(mt), n_rej, mt[0], mt[-1], min(mT), max(mT)))

    # Common acquisition grid: the six series share one sampling instant every ~2.79 s.
    allt = np.sort(np.concatenate([series[s][0] for s, _, _, _, _ in DIGITISER_SERIES]))
    nodes, cur = [], [allt[0]]
    for x in allt[1:]:
        if x - cur[-1] > 1.2:
            nodes.append(float(np.mean(cur)))
            cur = [x]
        else:
            cur.append(x)
    nodes.append(float(np.mean(cur)))
    labels = [s[0] for s in DIGITISER_SERIES]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s"] + ["T_%s_C" % s for s in labels])
        for nd in nodes:
            row = ["%.3f" % nd]
            for lab in labels:
                t, T = series[lab]
                d = np.abs(t - nd)
                i = int(np.argmin(d))
                row.append("%.2f" % T[i] if d[i] <= 1.2 else "")
            w.writerow(row)
    if verbose:
        print("  -> %s  (%d rows, median sampling interval %.2f s)"
              % (os.path.basename(out_csv), len(nodes), float(np.median(np.diff(nodes)))))
    return series


def load_digitised(path: str = None):
    """Read experimental_cooling_curves_digitized.csv -> {label: (t [s], T [degC])}."""
    path = path or CSV_EXPERIMENT
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    labels = [h[2:-2] for h in head[1:]]
    out = {}
    for j, lab in enumerate(labels):
        t, T = [], []
        for r in body:
            if r[j + 1] != "":
                t.append(float(r[0]))
                T.append(float(r[j + 1]))
        out[lab] = (np.array(t), np.array(T))
    return out


# ======================================================================================
# 11.  VALIDATION METRICS
# ======================================================================================


def crossing_time(t, T, level: float) -> float:
    """First time at which a monotonically falling curve crosses `level`."""
    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)
    below = np.where(T < level)[0]
    if below.size == 0 or below[0] == 0:
        return float("nan")
    i = below[0]
    return float(np.interp(level, [T[i], T[i - 1]], [t[i], t[i - 1]]))


def curve_metrics(t_obs, T_obs, t_mod, T_mod, level: float = 660.0):
    """RMSE, MAE, max|err|, bias, R^2 and t(T_F) for one probe (degC, s)."""
    pred = np.interp(t_obs, t_mod, T_mod)
    r = pred - T_obs
    ss_res = float(np.sum(r ** 2))
    ss_tot = float(np.sum((T_obs - T_obs.mean()) ** 2))
    return {
        "n": int(np.asarray(t_obs).size),
        "RMSE_C": float(np.sqrt(np.mean(r ** 2))),
        "MAE_C": float(np.mean(np.abs(r))),
        "MaxAE_C": float(np.max(np.abs(r))),
        "Bias_C": float(np.mean(r)),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "t_TF_obs_s": crossing_time(t_obs, T_obs, level),
        "t_TF_mod_s": crossing_time(t_mod, T_mod, level),
        "t_655_obs_s": crossing_time(t_obs, T_obs, level - 5.0),
        "t_655_mod_s": crossing_time(t_mod, T_mod, level - 5.0),
        "t_650_obs_s": crossing_time(t_obs, T_obs, level - 10.0),
        "t_650_mod_s": crossing_time(t_mod, T_mod, level - 10.0),
    }


def evaluate(prob: Problem, data, t_end: float = 200.0, n_grid: int = 1500):
    """Predict every probe and compute per-probe metrics against the digitised data."""
    hist = march_interface(prob, t_end=t_end)
    tg = np.geomspace(max(hist.t[0], 1e-3), t_end, n_grid)
    Tg = predict_thermocouples(prob, hist, TC_POSITIONS_M, tg) - KELVIN
    out = {}
    for j, lab in enumerate(TC_LABELS):
        t_obs, T_obs = data[lab]
        m = curve_metrics(t_obs, T_obs, tg, Tg[j])
        m["t_arrival_model_s"] = hist.arrival_time(TC_POSITIONS_M[j])
        m["dt_TF_s"] = m["t_TF_mod_s"] - m["t_TF_obs_s"]
        out[lab] = m
    return out, hist, tg, Tg


RELIABLE_PROBES = ("15mm", "30mm", "50mm", "90mm")
"""Probes used for the primary calibration.

The 5 mm and 10 mm records are excluded: at their *nominal* positions they are
inconsistent with one-dimensional Fourier conduction at k_S = 213 W/m/K (§9.4 of the
report) and they behave like probes sitting much closer to the chill than labelled.
They are still predicted, plotted and scored - only not used to set h_bar.
"""


def global_sse(prob: Problem, data, t_end: float = 200.0, t_min: float = 0.0,
               branch: str = "all", probes: Optional[Sequence[str]] = None):
    """Sum of squared residuals over *all* probes simultaneously.

    `branch` selects 'all', 'liquid' (t < front arrival, where melt convection acts
    directly) or 'solid'.  A single h_i is used for every probe: fitting a separate h_i
    per thermocouple would not be a physical model.
    """
    hist = march_interface(prob, t_end=t_end)
    tg = np.geomspace(max(hist.t[0], 1e-3), t_end, 1200)
    Tg = predict_thermocouples(prob, hist, TC_POSITIONS_M, tg) - KELVIN
    use = TC_LABELS if probes is None else tuple(probes)
    sse, n = 0.0, 0
    for j, lab in enumerate(TC_LABELS):
        if lab not in use:
            continue
        t_obs, T_obs = data[lab]
        sel = t_obs >= t_min
        if branch != "all":
            ta = hist.arrival_time(TC_POSITIONS_M[j])
            sel = sel & ((t_obs < ta) if branch == "liquid" else (t_obs >= ta))
        if not np.any(sel):
            continue
        r = np.interp(t_obs[sel], tg, Tg[j]) - T_obs[sel]
        sse += float(np.sum(r ** 2))
        n += int(sel.sum())
    return sse, n


# ======================================================================================
# 12.  CALIBRATION OF THE MELT-CONVECTION COEFFICIENT h_i
# ======================================================================================


def sse_profile(data, h_i_grid, liquid_model: str = "virtual-origin",
                T_P_C: float = 690.9, branch: str = "all", verbose: bool = False,
                h_bar: Optional[float] = None):
    """SSE(h_i) over all six probes at once (one h_i for the whole experiment)."""
    out = []
    for h_i in h_i_grid:
        prob = experiment_problem(h_i=h_i, T_P_C=T_P_C, liquid_model=liquid_model,
                                  h_bar=h_bar)
        sse, n = global_sse(prob, data, branch=branch)
        out.append((float(h_i), sse, n))
        if verbose:
            print("    h_i = %8.1f  RMSE_global = %7.3f degC  (n = %d)"
                  % (h_i, math.sqrt(sse / n), n))
    return out


TABLE1_HI_RANGE = (500.0, 1800.0)          # Table 1, h_i,1 ... h_i,5


def fit_h_i(data, liquid_model: str = "virtual-origin", T_P_C: float = 690.9,
            bounds: Tuple[float, float] = TABLE1_HI_RANGE, branch: str = "all",
            n_coarse: int = 13, h_bar: Optional[float] = None):
    """Least-squares estimate of a single h_i from all thermocouples simultaneously.

    Returns the optimum, the SSE profile, a naive 95 % confidence interval from the
    profile-SSE criterion  SSE <= SSE_min (1 + F_{1,n-1,0.95}/(n-1)), and the same
    interval computed with an *effective* sample size that accounts for the strong serial
    correlation of the residuals.  Under a systematic model bias the naive interval is
    meaningless; both are reported so the reader can see the difference.
    """
    from scipy.optimize import minimize_scalar
    from scipy.stats import f as f_dist

    grid = np.geomspace(bounds[0], bounds[1], n_coarse)
    prof = sse_profile(data, grid, liquid_model, T_P_C, branch, h_bar=h_bar)
    sses = np.array([p[1] for p in prof])
    n = prof[int(np.argmin(sses))][2]
    i0 = int(np.argmin(sses))
    lo = grid[max(i0 - 1, 0)]
    hi = grid[min(i0 + 1, len(grid) - 1)]

    def obj(lg):
        prob = experiment_problem(h_i=10.0 ** lg, T_P_C=T_P_C, liquid_model=liquid_model,
                                  h_bar=h_bar)
        return global_sse(prob, data, branch=branch)[0]

    at_edge = i0 in (0, len(grid) - 1)
    if at_edge:
        h_opt, sse_opt = float(grid[i0]), float(sses[i0])
    else:
        r = minimize_scalar(obj, bracket=None, bounds=(math.log10(lo), math.log10(hi)),
                            method="bounded", options={"xatol": 1e-4})
        h_opt, sse_opt = float(10.0 ** r.x), float(r.fun)

    # profile confidence intervals
    def ci(n_eff):
        if n_eff <= 2:
            return (float("nan"), float("nan"))
        thr = sse_opt * (1.0 + f_dist.ppf(0.95, 1, n_eff - 1) / (n_eff - 1))
        below = grid[sses <= thr]
        return (float(below.min()), float(below.max())) if below.size else (float("nan"),) * 2

    # residuals along one cooling curve are smooth, so the effective number of independent
    # observations is nearer "one per curve per correlation time" than one per sample
    n_eff = 6 * 5
    d_rmse = math.sqrt(max(sses) / n) - math.sqrt(min(sses) / n)
    return {
        "h_i": h_opt, "sse": sse_opt, "n": n,
        "rmse": math.sqrt(sse_opt / n),
        "profile": prof,
        "ci95_naive": ci(n),
        "ci95_neff": ci(n_eff),
        "at_bound": at_edge,
        "d_rmse_over_range": d_rmse,
        "bounds": bounds,
        "liquid_model": liquid_model,
        "branch": branch,
    }


def fit_h_bar(data, liquid_model: str = "virtual-origin", T_P_C: float = 690.9,
              h_i: float = 1000.0, bounds: Tuple[float, float] = (800.0, 20000.0),
              joint: bool = False, probes: Optional[Sequence[str]] = None):
    """Least squares for the constant mean surface coefficient h_bar (and optionally h_i).

    A single h_bar is fitted to **all six thermocouples simultaneously**.  h_bar is the
    one quantity the closed form genuinely requires and cannot take from the reported
    h_g(t): the solution is derived for constant h, so the experimentally reported
    time-dependent law only bounds the answer (see `h_bar_time_average`).
    """
    from scipy.optimize import minimize, minimize_scalar

    grid = np.geomspace(bounds[0], bounds[1], 17)
    prof = []
    for hb in grid:
        sse, n = global_sse(experiment_problem(h_i=h_i, T_P_C=T_P_C,
                                               liquid_model=liquid_model, h_bar=hb),
                            data, probes=probes)
        prof.append((float(hb), sse, n))
    sses = np.array([q[1] for q in prof])
    n = prof[int(np.argmin(sses))][2]
    i0 = int(np.argmin(sses))
    at_edge = i0 in (0, len(grid) - 1)
    lo, hi = grid[max(i0 - 1, 0)], grid[min(i0 + 1, len(grid) - 1)]

    def obj1(lg):
        return global_sse(experiment_problem(h_i=h_i, T_P_C=T_P_C,
                                             liquid_model=liquid_model,
                                             h_bar=10.0 ** lg), data, probes=probes)[0]

    r = minimize_scalar(obj1, bounds=(math.log10(lo), math.log10(hi)), method="bounded",
                        options={"xatol": 1e-6})
    out = {"h_bar": float(10.0 ** r.x), "h_i": h_i, "sse": float(r.fun),
           "n": n, "rmse": math.sqrt(r.fun / n), "profile": prof, "at_bound": at_edge,
           "liquid_model": liquid_model,
           "probes": tuple(TC_LABELS if probes is None else probes)}
    if joint:
        def obj2(v):
            hb, hh = 10.0 ** v[0], 10.0 ** v[1]
            if not (bounds[0] <= hb <= bounds[1] and 100.0 <= hh <= 1e6):
                return 1e18
            return global_sse(experiment_problem(h_i=hh, T_P_C=T_P_C,
                                                 liquid_model=liquid_model, h_bar=hb),
                              data, probes=probes)[0]
        rj = minimize(obj2, [math.log10(out["h_bar"]), math.log10(h_i)],
                      method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-2, maxiter=200))
        out["joint"] = {"h_bar": float(10.0 ** rj.x[0]), "h_i": float(10.0 ** rj.x[1]),
                        "rmse": math.sqrt(float(rj.fun) / n)}
    return out


def effective_positions(data, h_bar: float, h_i: float = 1000.0, T_P_C: float = 690.9,
                        liquid_model: str = "virtual-origin",
                        bounds_mm: Tuple[float, float] = (0.2, 140.0)):
    """Position each probe would need to occupy to reproduce its own record.

    A one-parameter fit of x per thermocouple, with h_bar, h_i, T_P and the properties
    held fixed.  It is a *diagnostic*, never applied to the delivered predictions: it
    tells whether a record is consistent with its nominal position, and by how much it is
    off if not.  A probe whose optimum runs to the lower bound cannot be explained by a
    position error at all - no location in the slab is cold enough - and points instead
    to something the 1-D model lacks near the chill.
    """
    from scipy.optimize import minimize_scalar

    prob = experiment_problem(h_i=h_i, T_P_C=T_P_C, liquid_model=liquid_model, h_bar=h_bar)
    hist = march_interface(prob, t_end=210.0)
    tg = np.geomspace(max(hist.t[0], 0.05), 200.0, 1400)
    rows = []
    for j, lab in enumerate(TC_LABELS):
        t_obs, T_obs = data[lab]

        def obj(x_mm):
            T = predict_thermocouples(prob, hist, [x_mm * 1e-3], tg)[0] - KELVIN
            return float(np.sum((np.interp(t_obs, tg, T) - T_obs) ** 2))

        r = minimize_scalar(obj, bounds=bounds_mm, method="bounded",
                            options={"xatol": 1e-3})
        T_nom = predict_thermocouples(prob, hist, [TC_POSITIONS_M[j]], tg)[0] - KELVIN
        res_nom = np.interp(t_obs, tg, T_nom) - T_obs
        rows.append({"probe": lab, "x_nominal_mm": TC_POSITIONS_M[j] * 1e3,
                     "x_effective_mm": float(r.x),
                     "shift_mm": float(r.x) - TC_POSITIONS_M[j] * 1e3,
                     "rmse_nominal_C": float(np.sqrt(np.mean(res_nom ** 2))),
                     "rmse_effective_C": math.sqrt(r.fun / t_obs.size),
                     "at_bound": bool(abs(r.x - bounds_mm[0]) < 1e-2
                                      or abs(r.x - bounds_mm[1]) < 1e-2)})
    return rows


def sensitivity_table(data, T_P_C: float = 690.9, liquid_model: str = "virtual-origin",
                      h_bar: Optional[float] = None):
    """How strongly the observables respond to h_bar, h_i, T_P and the liquid closure."""
    hb = H_BAR_DEFAULT if h_bar is None else float(h_bar)
    rows = []
    base = experiment_problem(h_i=1000.0, T_P_C=T_P_C, liquid_model=liquid_model, h_bar=hb)
    hist0 = march_interface(base, t_end=200.0)
    s0 = float(hist0.s_of(200.0))
    ta0 = hist0.arrival_time(0.090)

    def add(tag, prob):
        h = march_interface(prob, t_end=200.0)
        s200 = float(h.s_of(200.0))
        ta = h.arrival_time(0.090)
        sse, n = global_sse(prob, data)
        rows.append({"case": tag, "s200_mm": s200 * 1e3,
                     "d_s200_pct": 100.0 * (s200 - s0) / s0,
                     "t_arr_90mm_s": ta, "d_t_arr_pct": 100.0 * (ta - ta0) / ta0,
                     "rmse_global_C": math.sqrt(sse / n)})

    add("reference: h_bar = %.0f, h_i = 1000, T_P = %.1f degC, %s"
        % (hb, T_P_C, liquid_model), base)
    for f in (0.9, 1.1):
        add("h_bar %+.0f %%" % (100 * (f - 1)),
            experiment_problem(h_i=1000.0, T_P_C=T_P_C, liquid_model=liquid_model, h_bar=hb * f))
    add("h_bar = time-mean of h_g over 0-200 s (%.0f)" % h_bar_time_average(200.0),
        experiment_problem(h_i=1000.0, T_P_C=T_P_C, liquid_model=liquid_model,
                           h_bar=h_bar_time_average(200.0)))
    for h_i in (500.0, 1800.0, 5000.0, 1e9):
        add("h_i = %g" % h_i, experiment_problem(h_i=h_i, T_P_C=T_P_C,
                                                 liquid_model=liquid_model, h_bar=hb))
    for dTp in (-10.0, +10.0):
        add("T_P %+.0f K" % dTp, experiment_problem(h_i=1000.0, T_P_C=T_P_C + dTp,
                                                    liquid_model=liquid_model, h_bar=hb))
    for lm in ("interface-film", "as-coded", "conduction"):
        add("liquid closure = %s" % lm,
            experiment_problem(h_i=1000.0, T_P_C=T_P_C, liquid_model=lm, h_bar=hb))
    add("frozen-coefficient h_g(t) (not admissible)",
        experiment_problem(h_i=1000.0, T_P_C=T_P_C, liquid_model=liquid_model,
                           time_dependent_h=True))
    return rows


def estimate_T_P(data, t_max: float = 6.0) -> float:
    """Melt temperature at t -> 0, read from the probes farthest from the chill [degC].

    T_P is an *experimentally prescribed* quantity, not a fitted model parameter: it is
    read off the data and never adjusted to improve agreement.  No extrapolation is used
    (a sqrt(t) extrapolation on a nearly flat curve overshoots every measured value);
    instead the warmest reading of the 50 mm / 90 mm probes within the first `t_max`
    seconds is taken, since the thermal front has not reached them by then.
    """
    vals = []
    for lab in ("50mm", "90mm"):
        t, T = data[lab]
        sel = t <= t_max
        if np.any(sel):
            vals.append(float(np.max(T[sel])))
    return float(np.max(vals)) if vals else 690.0


# ======================================================================================
# 13.  OUTPUT WRITERS
# ======================================================================================


def write_predictions(prob: Problem, hist: InterfaceHistory, path: str = None,
                      t_end: float = 200.0, n: int = 801) -> str:
    path = path or CSV_PREDICTIONS
    tg = np.linspace(0.25, t_end, n)
    T = predict_thermocouples(prob, hist, TC_POSITIONS_M, tg) - KELVIN
    s = np.array([float(hist.s_of(t)) for t in tg])
    v = np.array([dsdt(prob, si, ti) for si, ti in zip(s, tg)])
    hg = np.array([prob.h(t) for t in tg])
    qS = np.array([q_solid(prob, si, ti) for si, ti in zip(s, tg)])
    qL = np.array([q_liquid(prob, si, ti) for si, ti in zip(s, tg)])
    phi = s / (2.0 * np.sqrt(prob.mat.alpha_S * tg))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "s_mm", "v_interface_m_s", "phi", "h_surface_W_m2K",
                    "Bi_env", "Bi_i", "q_solid_W_m2", "q_melt_W_m2"]
                   + ["T_%s_C" % lab for lab in TC_LABELS]
                   + ["phase_%s" % lab for lab in TC_LABELS])
        for k, t in enumerate(tg):
            w.writerow(["%.4f" % t, "%.6f" % (s[k] * 1e3), "%.6e" % v[k], "%.6f" % phi[k],
                        "%.2f" % hg[k], "%.6f" % (hg[k] * s[k] / prob.mat.k_S),
                        "%.6f" % (prob.h_i * s[k] / prob.mat.k_L),
                        "%.6e" % qS[k], "%.6e" % qL[k]]
                       + ["%.3f" % T[j, k] for j in range(len(TC_LABELS))]
                       + ["S" if TC_POSITIONS_M[j] < s[k] else "L" for j in range(len(TC_LABELS))])
    return path


METRIC_FIELDS = ("variant", "h_i_W_m2K", "liquid_model", "probe", "window", "n",
                 "RMSE_C", "MAE_C", "MaxAE_C", "Bias_C", "R2",
                 "t_TF_obs_s", "t_TF_model_s", "dt_TF_s",
                 "t_650_obs_s", "t_650_model_s", "dt_650_s", "t_arrival_model_s")


def _rows_from(variant: str, h_i, liquid_model: str, data,
               t_mod: np.ndarray, T_mod: np.ndarray,
               arrivals=None, windows=((0.0, 1e9, "full"), (0.0, 30.0, "t<=30s"))):
    """Per-probe + GLOBAL metric rows for one prediction, over one or more time windows."""
    rows = []
    for (t_lo, t_hi, wname) in windows:
        all_r, all_o = [], []
        for j, lab in enumerate(TC_LABELS):
            t_obs, T_obs = data[lab]
            sel = (t_obs >= t_lo) & (t_obs <= t_hi)
            if sel.sum() < 3:
                continue
            m = curve_metrics(t_obs[sel], T_obs[sel], t_mod, T_mod[j])
            rows.append({
                "variant": variant, "h_i_W_m2K": h_i, "liquid_model": liquid_model,
                "probe": lab, "window": wname, "n": m["n"],
                "RMSE_C": m["RMSE_C"], "MAE_C": m["MAE_C"], "MaxAE_C": m["MaxAE_C"],
                "Bias_C": m["Bias_C"], "R2": m["R2"],
                "t_TF_obs_s": m["t_TF_obs_s"], "t_TF_model_s": m["t_TF_mod_s"],
                "dt_TF_s": m["t_TF_mod_s"] - m["t_TF_obs_s"],
                "t_650_obs_s": m["t_650_obs_s"], "t_650_model_s": m["t_650_mod_s"],
                "dt_650_s": m["t_650_mod_s"] - m["t_650_obs_s"],
                "t_arrival_model_s": (arrivals[j] if arrivals is not None else "")})
            all_r.append(np.interp(t_obs[sel], t_mod, T_mod[j]) - T_obs[sel])
            all_o.append(T_obs[sel])
        r = np.concatenate(all_r)
        o = np.concatenate(all_o)
        rows.append({"variant": variant, "h_i_W_m2K": h_i, "liquid_model": liquid_model,
                     "probe": "GLOBAL", "window": wname, "n": int(r.size),
                     "RMSE_C": float(np.sqrt(np.mean(r ** 2))),
                     "MAE_C": float(np.mean(np.abs(r))),
                     "MaxAE_C": float(np.max(np.abs(r))), "Bias_C": float(np.mean(r)),
                     "R2": float(1.0 - np.sum(r ** 2) / np.sum((o - o.mean()) ** 2)),
                     "t_TF_obs_s": "", "t_TF_model_s": "",
                     "dt_TF_s": float(np.nanmean([x["dt_TF_s"] for x in rows
                                                  if x["window"] == wname and x["probe"] != "GLOBAL"])),
                     "t_650_obs_s": "", "t_650_model_s": "",
                     "dt_650_s": float(np.nanmean([x["dt_650_s"] for x in rows
                                                   if x["window"] == wname and x["probe"] != "GLOBAL"])),
                     "t_arrival_model_s": ""})
    return rows


def _metric_rows(variant: str, prob: Problem, data):
    hist = march_interface(prob, t_end=200.0)
    tg = np.geomspace(max(hist.t[0], 1e-3), 200.0, 1500)
    Tg = predict_thermocouples(prob, hist, TC_POSITIONS_M, tg) - KELVIN
    arr = [hist.arrival_time(x) for x in TC_POSITIONS_M]
    return _rows_from(variant, prob.h_i, prob.liquid_model, data, tg, Tg, arr)


def write_metrics(rows: List[Dict[str, object]], path: str = None) -> str:
    path = path or CSV_METRICS
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {}
            for k in METRIC_FIELDS:
                v = r.get(k, "")
                out[k] = ("%.4f" % v) if isinstance(v, float) and np.isfinite(v) else \
                         ("" if isinstance(v, float) else v)
            w.writerow(out)
    return path


TC_COLOURS = {"5mm": "#000000", "10mm": "#ed3c3c", "15mm": "#1a6fde",
              "30mm": "#b177de", "50mm": "#cc9900", "90mm": "#00cbcd"}
TC_MARKERS = {"5mm": "s", "10mm": "o", "15mm": "^", "30mm": "D", "50mm": "<", "90mm": ">"}


def make_figure(prob: Problem, hist: InterfaceHistory, data, path: str = None,
                fd=None, title_extra: str = "", residual_panel: bool = False) -> str:
    """Comparison figure: digitised experiment (markers) vs analytical solution (lines).

    `fd` optionally overlays the independent 1-D finite-difference reference (dotted) and
    `residual_panel` adds a lower panel with the model - experiment residuals, so that the
    agreement is judged numerically and not only visually.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    path = path or FIG_OUT
    tg = np.geomspace(max(hist.t[0], 0.05), 200.0, 2000)
    T = predict_thermocouples(prob, hist, TC_POSITIONS_M, tg) - KELVIN

    if residual_panel:
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(11.0, 9.6), dpi=170, sharex=True,
                                      gridspec_kw=dict(height_ratios=[3.1, 1.0], hspace=0.08))
    else:
        fig, ax = plt.subplots(figsize=(11.0, 8.0), dpi=170)
        axr = None

    for j, lab in enumerate(TC_LABELS):
        c = TC_COLOURS[lab]
        t_obs, T_obs = data[lab]
        ax.plot(t_obs, T_obs, TC_MARKERS[lab], ms=4.6, mfc=c, mec=c, alpha=0.9,
                ls="none", label="%s" % lab, zorder=4)
        ax.plot(tg, T[j], "-", color=c, lw=2.0, zorder=5)
        if fd is not None:
            ax.plot(fd["t"], fd["T"][:, j] - KELVIN, ":", color=c, lw=1.2, zorder=3)
        ta = hist.arrival_time(TC_POSITIONS_M[j])
        if np.isfinite(ta) and ta <= 200.0:
            ax.plot([ta], [prob.mat.T_F - KELVIN], "|", color=c, ms=9, mew=1.6, zorder=6)
        if axr is not None:
            axr.plot(t_obs, np.interp(t_obs, tg, T[j]) - T_obs, TC_MARKERS[lab],
                     ms=3.6, mfc=c, mec=c, ls="-", lw=0.9, alpha=0.85)

    ax.axhline(prob.mat.T_F - KELVIN, color="red", lw=2.4, zorder=2)
    ax.text(0.5, prob.mat.T_F - KELVIN - 30, r"$T_F = 660\,^\circ$C", color="red",
            ha="left", va="top", fontsize=11, zorder=7,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.set_xlim(0, 200)
    ax.set_ylim(100, 710)
    ax.set_ylabel(r"Temperature, $T$  [$^\circ$C]", fontsize=13)
    ax.set_title("Pure Al, transient horizontal solidification: corrected melt-convection"
                 "\nanalytical solution (Eqs. 66-83) vs digitised experimental cooling curves"
                 + title_extra, fontsize=12)
    ax.grid(alpha=0.25, lw=0.6)

    h_now = prob.h(50.0)
    h_is_const = abs(prob.h(1.0) - prob.h(200.0)) < 1e-9 * max(h_now, 1.0)
    box_lines = [
        (r"$\bar{h}=%.0f$ W m$^{-2}$K$^{-1}$ (constant mean, fitted); "
         r"reported $h_g(t)=6350\,t^{-0.21}$" % h_now) if h_is_const else
        r"$h_g(t)=6350\,t^{-0.21}$ W m$^{-2}$K$^{-1}$ (frozen-coefficient)",
        r"$h_i=%.0f$ W m$^{-2}$K$^{-1}$ (prescribed, Table 1 range 500-1800)" % prob.h_i,
        r"liquid closure: %s;  $T_P=%.1f\,^\circ$C,  $T_\infty=%.0f\,^\circ$C"
        % (prob.liquid_model, prob.T_P - KELVIN, prob.T_inf - KELVIN),
        r"digitisation uncertainty: $\pm0.2\,^\circ$C random, $\pm%.1f\,^\circ$C systematic"
        % TF_LINE_BIAS_C,
    ]
    ax.text(0.985, 0.022, "\n".join(box_lines), transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9.5, bbox=dict(fc="white", ec="0.55", alpha=0.95), zorder=8)

    handles = []
    for j, lab in enumerate(TC_LABELS):
        x = TC_POSITIONS_M[j]
        nominal = DEFAULT_TC_POSITIONS_M[j]
        display = lab if abs(x - nominal) < 1e-12 else "%s data -> %g mm" % (lab, x * 1e3)
        handles.append(Line2D([], [], ls="none", marker=TC_MARKERS[lab],
                              mfc=TC_COLOURS[lab], mec=TC_COLOURS[lab], ms=6,
                              label=display))
    style = [Line2D([], [], color="0.25", lw=2.0, label="analytical (this work)"),
             Line2D([], [], ls="none", marker="o", mfc="0.25", mec="0.25", ms=6,
                    label="experiment (digitised)")]
    if fd is not None:
        style.append(Line2D([], [], color="0.25", lw=1.2, ls=":", label="1-D FD reference"))
    style.append(Line2D([], [], color="0.25", ls="none", marker="|", ms=9, mew=1.6,
                        label="predicted front arrival"))
    leg1 = ax.legend(handles=handles, loc="lower left", ncol=3, fontsize=9.5, frameon=True,
                     framealpha=0.93, title="Thermocouple position", title_fontsize=9.5)
    ax.add_artist(leg1)
    ax.legend(handles=style, loc="upper right", fontsize=9.5, frameon=True, framealpha=0.93)

    if axr is not None:
        axr.axhline(0.0, color="0.3", lw=1.0)
        axr.axhspan(-TF_LINE_BIAS_C, TF_LINE_BIAS_C, color="0.6", alpha=0.35, lw=0)
        axr.set_xlim(0, 200)
        axr.set_xlabel("Time, $t$  [s]", fontsize=13)
        axr.set_ylabel("model $-$ exp.\n" r"[$^\circ$C]", fontsize=11)
        axr.grid(alpha=0.25, lw=0.6)
        axr.text(0.985, 0.06, "grey band = digitisation systematic uncertainty "
                              r"($\pm%.1f\,^\circ$C)" % TF_LINE_BIAS_C,
                 transform=axr.transAxes, ha="right", va="bottom", fontsize=8.5)
    else:
        ax.set_xlabel("Time, $t$  [s]", fontsize=13)

    if axr is None:
        fig.tight_layout()
        path = _save_figure(fig, path)
    else:                       # tight_layout cannot handle the shared-x residual panel
        fig.subplots_adjust(left=0.10, right=0.985, top=0.935, bottom=0.075)
        path = _save_figure(fig, path, bbox_inches="tight")
    plt.close(fig)
    return path


# ======================================================================================
# 15.  INTERFACE POSITION vs TIME
# ======================================================================================

POSITION_FIGURE_SRC = os.path.join(
    HERE, "Figure_Position_versus_time_experimental_against_numerical_simulation.jpg")
CSV_POSITION = os.path.join(HERE, "experimental_position_vs_time_digitized.csv")
FIG_POSITION = os.path.join(HERE, "Figure_position_vs_time_analytical_vs_experimental.png")
FIG_VELOCITY_POSITION = os.path.join(
    HERE, "Figure_velocity_vs_position_analytical_vs_experimental.png")
FIG_GRADIENT_POSITION = os.path.join(
    HERE, "Figure_thermal_gradient_vs_position_analytical_vs_experimental.png")

# Published power-law fit printed inside that figure: P = 6.14 t^0.51 [mm, s]
POSITION_FIT_A, POSITION_FIT_B = 6.14, 0.51

# Pixel-to-axis calibration of the position raster (6432 x 4923 px), obtained the same way
# as for the cooling-curve figure: frame centre lines L = 1149, R = 5538, TOP = 570,
# BOT = 4103; the x tick labels 0 ... 220 s are centred at 1246.5 ... 5537.0 px
# (19.502 px/s) and the y tick labels 0 ... 140 mm at 4094 ... 797 px (23.55 px/mm).
POSITION_PX = dict(X0=1246.5, SX=(5537.0 - 1246.5) / 220.0,
                   Y0=4094.0, SY=(4094.0 - 797.0) / 140.0,
                   L=1149, R=5538, TOP=570, BOT=4103)


def digitise_position_figure(src: str = None, out_csv: str = None, verbose: bool = True):
    """Recover the experimental interface-position markers P(t) from the supplied raster.

    Seven black squares; the thick black power-law fit and the thin red numerical curve
    are *not* data and are excluded (the fit by a 31 x 31 morphological opening, which the
    ~77 px squares survive and the ~25 px curve does not; the red curve by colour).
    """
    from PIL import Image
    from scipy import ndimage as ndi

    src = src or POSITION_FIGURE_SRC
    out_csv = out_csv or CSV_POSITION
    Image.MAX_IMAGE_PIXELS = None
    a = np.asarray(Image.open(src).convert("RGB")).astype(int)
    H, W, _ = a.shape
    dark = a.max(axis=2) < 110
    valid = np.zeros((H, W), bool)
    valid[POSITION_PX["TOP"] + 18:POSITION_PX["BOT"] - 17,
          POSITION_PX["L"] + 18:POSITION_PX["R"] - 17] = True
    valid[560:1650, 1300:3650] = False                     # legend box
    m = ndi.binary_opening(dark & valid, np.ones((31, 31)))
    lb, n = ndi.label(m)
    pts = []
    for i in range(1, n + 1):
        ys, xs = np.where(lb == i)
        t = (xs.mean() - POSITION_PX["X0"]) / POSITION_PX["SX"]
        P = (POSITION_PX["Y0"] - ys.mean()) / POSITION_PX["SY"]
        pts.append((float(t), float(P)))
    pts.sort()
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "position_mm", "power_law_fit_mm"])
        for t, P in pts:
            w.writerow(["%.3f" % t, "%.3f" % P,
                        "%.3f" % (POSITION_FIT_A * max(t, 0.0) ** POSITION_FIT_B)])
    if verbose:
        print("    %d markers recovered:" % len(pts))
        for t, P in pts:
            print("      t = %7.2f s   P = %7.2f mm   (fit %.2f mm)"
                  % (t, P, POSITION_FIT_A * max(t, 0.0) ** POSITION_FIT_B))
        print("    -> %s" % os.path.basename(out_csv))
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def load_position_data(path: str = None):
    path = path or CSV_POSITION
    t, P = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["time_s"]))
            P.append(float(row["position_mm"]))
    return np.array(t), np.array(P)


def fit_h_bar_position(t_obs, P_obs_mm, h_i: float = 1000.0, T_P_C: float = 690.9,
                       liquid_model: str = "virtual-origin",
                       bounds: Tuple[float, float] = (400.0, 20000.0)):
    """Calibrate the constant mean h from the interface kinetics alone.

    This is an *independent* estimate of h_bar: it uses only s(t), never the temperature
    field, so comparing it with the cooling-curve estimate tests whether the two published
    experimental figures describe the same casting.
    """
    from scipy.optimize import minimize_scalar

    def s_model(hb, t):
        prob = experiment_problem(h_i=h_i, T_P_C=T_P_C, liquid_model=liquid_model, h_bar=hb)
        hist = march_interface(prob, t_end=float(max(t)) * 1.25)
        return np.array([float(hist.s_of(max(x, hist.t[0]))) * 1e3 for x in t])

    def sse(lg):
        return float(np.sum((s_model(10.0 ** lg, t_obs) - P_obs_mm) ** 2))

    r = minimize_scalar(sse, bounds=(math.log10(bounds[0]), math.log10(bounds[1])),
                        method="bounded", options={"xatol": 1e-5})
    hb = float(10.0 ** r.x)
    return {"h_bar": hb, "rmse_mm": math.sqrt(r.fun / len(t_obs)),
            "s_model_mm": s_model(hb, t_obs)}


def interface_from_cooling_curves(data, level: float = 660.0):
    """Interface position implied by the cooling curves: (t at which T = level, x_probe).

    The phase-change temperature T_F = 660 degC is used.  Crossings at T_F +/- 5 degC
    define the horizontal error bar.
    """
    out = []
    for lab, x in zip(TC_LABELS, TC_POSITIONS_M):
        t_obs, T_obs = data[lab]
        t_c = crossing_time(t_obs, T_obs, level)
        t_lo = crossing_time(t_obs, T_obs, level + 5.0)
        t_hi = crossing_time(t_obs, T_obs, level - 5.0)
        if np.isfinite(t_c):
            out.append((t_c, x * 1e3, t_lo, t_hi, lab))
    return out


def fit_cooling_curve_position_law(data, level: float = 660.0):
    """Fit P = A t^b to the cooling-curve interface points at the selected level."""
    points = interface_from_cooling_curves(data, level=level)
    t = np.array([row[0] for row in points], dtype=float)
    P = np.array([row[1] for row in points], dtype=float)
    b, log_A = np.polyfit(np.log(t), np.log(P), 1)
    A = float(np.exp(log_A))
    return {"A": A, "b": float(b), "t": t, "P": P, "level": float(level)}


def fit_analytical_position_law(hist: InterfaceHistory, t_min: float, t_max: float):
    """Fit P = A t^b to the continuous analytical trajectory on a selected time window."""
    t = np.geomspace(float(t_min), float(t_max), 800)
    P = np.asarray(hist.s_of(t), dtype=float) * 1e3
    b, log_A = np.polyfit(np.log(t), np.log(P), 1)
    return {"A": float(np.exp(log_A)), "b": float(b), "t": t, "P": P}


def make_position_figure(data, h_bar_cool: float, t_pos, P_pos, fit_pos,
                         h_i: float = 1000.0, T_P_C: float = 690.9,
                         liquid_model: str = "virtual-origin", path: str = None) -> str:
    """Interface position vs time: analytical vs both experimental sources."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = path or FIG_POSITION
    tg = np.geomspace(0.05, 220.0, 1200)

    def s_of(hb):
        prob = experiment_problem(h_i=h_i, T_P_C=T_P_C, liquid_model=liquid_model, h_bar=hb)
        hist = march_interface(prob, t_end=230.0)
        return np.array([float(hist.s_of(max(t, hist.t[0]))) * 1e3 for t in tg])

    s_cool = s_of(h_bar_cool)

    fig, ax = plt.subplots(figsize=(10.5, 7.6), dpi=170)
    ax.plot(tg, POSITION_FIT_A * tg ** POSITION_FIT_B, "-", color="0.45", lw=1.6,
            label=r"published fit  $P = %.2f\,t^{%.2f}$" % (POSITION_FIT_A, POSITION_FIT_B))
    ax.plot(t_pos, P_pos, "s", ms=8, mfc="k", mec="k", ls="none",
            label="experiment, position figure (digitised)")
    cc = interface_from_cooling_curves(data)
    tc = [c[0] for c in cc]
    xc = [c[1] for c in cc]
    xerr = np.array([[c[0] - c[2] for c in cc], [c[3] - c[0] for c in cc]])
    ax.errorbar(tc, xc, xerr=xerr, fmt="o", ms=8, mfc="none", mec="crimson", mew=1.8,
                ecolor="crimson", elinewidth=1.2, capsize=3, ls="none",
                label=r"experiment, cooling curves ($T = 660\,^\circ$C crossing)")
    cooling_fit = fit_cooling_curve_position_law(data, level=660.0)
    ax.plot(tg, cooling_fit["A"] * tg ** cooling_fit["b"], color="crimson", lw=2.2,
            label=(r"fit to red points: $P=%.2ft^{%.3f}$" %
                   (cooling_fit["A"], cooling_fit["b"])))
    ax.plot(tg, s_cool, "-", color="#1a6fde", lw=2.4,
            label=r"analytical, $\bar{h} = %.0f$ (fitted to the cooling curves)" % h_bar_cool)
    for lab, x in zip(TC_LABELS, TC_POSITIONS_M):
        ax.axhline(x * 1e3, color="0.85", lw=0.7, zorder=0)
        ax.text(218, x * 1e3 + 1.2, lab, fontsize=8, color="0.45", ha="right")
    ax.set_xlim(0, 220)
    ax.set_ylim(0, 150)
    ax.set_xlabel("Time, $t$  [s]", fontsize=13)
    ax.set_ylabel("Interface position, $s(t)$  [mm]", fontsize=13)
    ax.set_title("Pure Al: solid/liquid interface position - corrected analytical solution\n"
                 "against the two independent experimental sources", fontsize=12)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper left", fontsize=9.5, frameon=True, framealpha=0.94)
    t90_cooling = next((c[0] for c in cc if c[4] == "90mm"), float("nan"))
    t90_position = (90.0 / POSITION_FIT_A) ** (1.0 / POSITION_FIT_B)
    ax.text(0.985, 0.035,
            "the two experimental sources disagree: the position figure puts the front at\n" +
            (r"90 mm at $t \approx %.0f$ s, the cooling curves ($T=660\,^\circ$C) "
             r"at $t \approx %.0f$ s" % (t90_position, t90_cooling)),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(fc="white", ec="0.55", alpha=0.95))
    fig.tight_layout()
    path = _save_figure(fig, path)
    plt.close(fig)
    return path


def make_velocity_position_figure(hist: InterfaceHistory, data,
                                  h_bar: float, path: str = None) -> str:
    """Interface velocity vs position from two consistently differentiated P(t) curves.

    A power law is fitted separately to the analytical trajectory and to the red
    T=660 degC points in the position-time figure.  Both laws are differentiated and
    time is eliminated: V(P) = b A^(1/b) P^((b-1)/b).  Experimental velocities
    evaluated from that fit are also shown as discrete points at the thermocouple
    positions (10, 15, 30, 50 and 90 mm).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = path or FIG_VELOCITY_POSITION
    fit = fit_cooling_curve_position_law(data, level=660.0)
    fit_an = fit_analytical_position_law(hist, float(fit["t"].min()),
                                         float(fit["t"].max()))
    P_fit = np.linspace(float(fit["P"].min()), float(fit["P"].max()), 600)
    exponent = (fit["b"] - 1.0) / fit["b"]
    coefficient = fit["b"] * fit["A"] ** (1.0 / fit["b"])
    V_fit = coefficient * P_fit ** exponent
    exponent_an = (fit_an["b"] - 1.0) / fit_an["b"]
    coefficient_an = fit_an["b"] * fit_an["A"] ** (1.0 / fit_an["b"])
    V_fit_an = coefficient_an * P_fit ** exponent_an
    P_exp_points = np.asarray(fit["P"], dtype=float)
    V_exp_points = coefficient * P_exp_points ** exponent

    fig, ax = plt.subplots(figsize=(9.4, 7.0), dpi=170)
    ax.plot(P_fit, V_fit_an, color="#1a6fde", lw=2.5,
            label=(r"analytical-curve fit: $P=%.2ft^{%.3f}$ $\Rightarrow$ "
                   r"$V(P)=%.2fP^{%.3f}$" %
                   (fit_an["A"], fit_an["b"], coefficient_an, exponent_an)))
    ax.plot(P_fit, V_fit, color="crimson", lw=2.2,
            label=(r"experimental fit at $T=660\,^\circ$C: $P=%.2ft^{%.3f}$ "
                   r"$\Rightarrow$ "
                   r"$V(t)=%.2ft^{%.2f}$ $\Rightarrow$ $V(P)=%.2fP^{%.3f}$" %
                   (fit["A"], fit["b"], fit["A"] * fit["b"], fit["b"] - 1.0,
                    coefficient, exponent)))
    ax.plot(P_exp_points, V_exp_points, "o", ms=8, mfc="white", mec="crimson", mew=2.0,
            ls="none", zorder=5,
            label=r"experimental fit evaluated at thermocouple positions")
    ax.set_xlim(8.0, 92.0)
    ax.set_ylim(0.35, 3.0)
    ax.set_xlabel("Interface position, $P$  [mm]", fontsize=13)
    ax.set_ylabel("Interface velocity, $V$  [mm/s]", fontsize=13)
    ax.set_title("Pure Al: analytical versus experimental interface velocity", fontsize=13)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper right", fontsize=9.0, frameon=True, framealpha=0.94)
    ax.annotate("largest mismatch at the first thermocouples",
                xy=(P_exp_points[0], V_exp_points[0]), xytext=(18, 2.55),
                fontsize=9, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0))
    fig.tight_layout()
    path = _save_figure(fig, path)
    plt.close(fig)
    return path


def cooling_rate_at_crossing(t, T, level: float = 660.0) -> float:
    """Magnitude of the local experimental cooling rate at a level crossing [K/s]."""
    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)
    below = np.where(T < level)[0]
    if below.size == 0 or below[0] == 0:
        return float("nan")
    i = int(below[0])
    return abs(float((T[i] - T[i - 1]) / (t[i] - t[i - 1])))


def make_gradient_position_figure(prob: Problem, hist: InterfaceHistory, data,
                                  h_bar: float, path: str = None) -> str:
    """Liquid-side interfacial thermal gradient versus interface position.

    Experimental values use the chain-rule estimate G=|dT/dt|/V at the 660 degC
    crossing of each cooling curve.  V is evaluated from the same fitted experimental
    position law used in the velocity-position comparison.  The analytical curve is
    dT_L/dx evaluated directly at the moving interface.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = path or FIG_GRADIENT_POSITION
    fit_pos = fit_cooling_curve_position_law(data, level=660.0)
    points = interface_from_cooling_curves(data, level=660.0)
    rates = np.array([cooling_rate_at_crossing(*data[row[4]], level=660.0)
                      for row in points], dtype=float)
    v_exp = fit_pos["A"] * fit_pos["b"] * fit_pos["t"] ** (fit_pos["b"] - 1.0)
    G_exp = rates / v_exp                         # (K/s)/(mm/s) = K/mm

    exponent_exp, log_coefficient_exp = np.polyfit(np.log(fit_pos["P"]),
                                                    np.log(G_exp), 1)
    coefficient_exp = float(np.exp(log_coefficient_exp))

    mask = ((hist.s * 1e3 >= float(fit_pos["P"].min())) &
            (hist.s * 1e3 <= float(fit_pos["P"].max())))
    P_an = hist.s[mask] * 1e3
    G_an = np.array([grad_liquid_interface(prob, float(s), float(t)) / 1e3
                     for s, t in zip(hist.s[mask], hist.t[mask])])
    exponent_an, log_coefficient_an = np.polyfit(np.log(P_an), np.log(G_an), 1)
    coefficient_an = float(np.exp(log_coefficient_an))

    P_grid = np.linspace(float(fit_pos["P"].min()), float(fit_pos["P"].max()), 600)
    fig, ax = plt.subplots(figsize=(9.4, 7.0), dpi=170)
    ax.plot(P_an, G_an, color="#1a6fde", lw=2.5,
            label=(r"analytical liquid-side gradient (%s), $\bar h=%.0f$: "
                   r"$G_L\approx%.2fP^{%.3f}$" %
                   (prob.liquid_model, h_bar, coefficient_an, exponent_an)))
    ax.plot(P_grid, coefficient_exp * P_grid ** exponent_exp,
            color="crimson", lw=2.2,
            label=(r"fit to experimental estimates: $G_L=%.2fP^{%.3f}$" %
                   (coefficient_exp, exponent_exp)))
    ax.plot(fit_pos["P"], G_exp, "o", ms=8, mfc="white", mec="crimson", mew=2.0,
            ls="none", zorder=5,
            label=r"experiment: $G_L=|\dot T_{660}|/V_{fit}$")
    ax.set_xlim(8.0, 92.0)
    ax.set_ylim(0.0, max(10.2, 1.08 * float(np.nanmax(G_exp))))
    ax.set_xlabel("Interface position, $P$  [mm]", fontsize=13)
    ax.set_ylabel("Liquid-side thermal gradient, $G_L$  [K/mm]", fontsize=13)
    ax.set_title("Pure Al: analytical versus experimental thermal gradient", fontsize=13)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper right", fontsize=9.0, frameon=True, framealpha=0.94)
    ax.annotate("largest mismatch near the chill",
                xy=(fit_pos["P"][0], G_exp[0]), xytext=(25, 8.0),
                fontsize=9, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0))
    fig.tight_layout()
    path = _save_figure(fig, path)
    plt.close(fig)
    return path

# ======================================================================================
# 14.  DRIVER
# ======================================================================================


def _fd_metric_rows(prob: Problem, data):
    """Metrics of the independent FD reference against the same data.

    This separates "the analytical solution is wrongly implemented" from "the stated
    boundary condition and property set do not describe the experiment".
    """
    t_out = np.unique(np.concatenate([data[l][0] for l in TC_LABELS]))
    t_out = np.clip(t_out[t_out <= 200.0], 1e-3, 200.0)
    fd = fd_reference(prob, t_end=200.0, out_t=t_out, nx=800)
    rows = _rows_from("FD reference (same stated problem)", prob.h_i,
                      "n/a (full 1-D Stefan)", data, fd["t"], (fd["T"] - KELVIN).T)
    return rows, fd


def _legacy_experimental_main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    ap.add_argument("--verify", action="store_true", help="run the verification suite only")
    ap.add_argument("--digitize", action="store_true", help="re-extract the figure and exit")
    ap.add_argument("--liquid-model", default="virtual-origin", choices=list(LIQUID_MODELS))
    ap.add_argument("--h-i", type=float, default=None,
                    help="prescribe h_i [W/m2K] instead of using the Table 1 mid-range value")
    ap.add_argument("--h-bar", type=float, default=None,
                    help="prescribe the constant mean surface coefficient [W/m2K] "
                         "instead of fitting it")
    ap.add_argument("--tc-positions", type=float, nargs=len(TC_LABELS), metavar="X_M",
                    help="physical positions [m] assigned to the six data curves, in "
                         "their existing order; e.g. --tc-positions 0.003 0.010 "
                         "0.015 0.030 0.050 0.090")
    ap.add_argument("--time-dependent-h", action="store_true",
                    help="diagnostic: use the reported h_g(t) pointwise (not admissible "
                         "in a solution derived for constant h)")
    ap.add_argument("--t-end", type=float, default=200.0)
    args = ap.parse_args(argv)

    if args.tc_positions is not None:
        try:
            set_thermocouple_positions(args.tc_positions)
        except AssertionError as exc:
            ap.error(str(exc))

    if args.verify:
        return 1 if run_verification().n_fail else 0

    print("=" * 86)
    print("CORRECTED MELT-CONVECTION ANALYTICAL SOLUTION - pure Al horizontal solidification")
    print("=" * 86)
    print("Liquid-side closure: %s" % args.liquid_model)
    if args.liquid_model == "virtual-origin":
        print("  WARNING: continuity-preserving manuscript interpretation; q_L(h_i) is inverted")
        print("  and the Eq.(82) driving temperature can exceed T_P. Use --liquid-model")
        print("  interface-film for a physically monotone finite-film interpretation.")
    elif args.liquid_model == "interface-film":
        print("  NOTE: physically monotone finite-film closure; it admits T_L(s+) > T_F,")
        print("  i.e. the temperature jump associated with an interfacial film resistance.")
    elif args.liquid_model == "as-coded":
        print("  WARNING: legacy manuscript-development closure retained for audit only.")
    else:
        print("  NOTE: conduction limit h_i -> infinity; T_L(s+) = T_F.")
    print("Thermocouple data -> model position: " + ", ".join(
        "%s -> %g mm" % (lab, x * 1e3) for lab, x in zip(TC_LABELS, TC_POSITIONS_M)))
    m = AL_TABLE1
    print("Table 1 properties:  k_S=%.1f  k_L=%.1f W/mK   c_S=%.1f  c_L=%.1f J/kgK   "
          "rho_S=%.1f  rho_L=%.1f kg/m3" % (m.k_S, m.k_L, m.c_S, m.c_L, m.rho_S, m.rho_L))
    print("  alpha_S=%.4e  alpha_L=%.4e m2/s   L=%.0f J/kg   T_F=%.2f K (%.1f degC)"
          % (m.alpha_S, m.alpha_L, m.L, m.T_F, m.T_F - KELVIN))
    print("  n = sqrt(a_S/a_L) = %.5f   N = k_S/k_L = %.5f   Ste = c_S dT/L = %.5f"
          % (m.n, m.N, m.c_S * (m.T_F - 298.15) / m.L))

    # -- 1. digitisation ---------------------------------------------------------------
    print("\n[1] Digitising %s" % os.path.basename(FIGURE_SRC))
    if args.digitize or not os.path.exists(CSV_EXPERIMENT):
        digitise_figure()
        if args.digitize:
            return 0
    else:
        print("    reusing %s (delete it or pass --digitize to rebuild)"
              % os.path.basename(CSV_EXPERIMENT))
    data = load_digitised()
    print("    samples per probe: " + ", ".join("%s=%d" % (l, data[l][0].size) for l in TC_LABELS))

    T_P_C = estimate_T_P(data)
    print("    melt temperature read from the data: T_P = %.1f degC (superheat %.1f K)"
          % (T_P_C, T_P_C - (m.T_F - KELVIN)))

    # -- 2. Table 2 audit --------------------------------------------------------------
    print("\n[2] Audit of published Table 2 (s = 0.10 m, h = 400 W/m2K, T_P = 970.48 K)")
    print("      h_i   published   legacy-path   fsolve   legacy residual   corrected")
    pub = [949.18, 950.74, 952.57, 960.31, 963.69]
    for row, p in zip(reproduce_table2(), pub):
        print("    %6.0f   %8.2f K  %8.2f K     ier=%d   %12.3e   %8.2f K"
              % (row["h_i"], p, row["TL_legacy"], row["ier"], row["residual_legacy"],
                 row["TL_corrected"]))
    print("    -> the published values are reproduced only by the *stalled* fsolve iterate;")
    print("       the legacy residual has no zero for phi > 0 (see the report).")

    # -- 3. surface coefficients -------------------------------------------------------
    print("\n[3] Surface coefficients")
    print("    The closed form of Eqs. (66)-(83) is derived for a CONSTANT h: the kernel")
    print("    exp(h x/k + h^2 alpha t/k^2) erfc(.) solves Eq. (66) only if h is constant.")
    print("    The reported h_g(t) = 6350 t^-0.21 therefore cannot be inserted pointwise;")
    print("    the analytical solution admits one mean value h_bar.  Time means of the")
    print("    reported law:")
    for tf in (10.0, 20.0, 50.0, 100.0, 200.0):
        print("      (1/t_f) int_0^t_f h_g dt , t_f = %6.1f s  ->  %7.0f W/m2K   "
              "[h_g(t_f) = %6.0f]" % (tf, h_bar_time_average(tf), float(h_g_experiment(tf))))
    h_i = args.h_i if args.h_i is not None else 1000.0
    if args.h_bar is not None:
        h_bar = args.h_bar
        print("    h_bar prescribed on the command line: %.0f W/m2K" % h_bar)
        fit = None
    else:
        print("    (a) least squares for h_bar, all six probes at once (h_i = %.0f fixed):"
              % h_i)
        fit_all = fit_h_bar(data, liquid_model=args.liquid_model, T_P_C=T_P_C, h_i=h_i,
                            joint=True)
        print("          h_bar = %.0f W/m2K   RMSE = %.2f degC over n = %d points"
              % (fit_all["h_bar"], fit_all["rmse"], fit_all["n"]))
        print("          joint (h_bar, h_i) = (%.0f, %.0f) -> RMSE %.2f degC "
              "(h_i buys %.2f degC)"
              % (fit_all["joint"]["h_bar"], fit_all["joint"]["h_i"],
                 fit_all["joint"]["rmse"], fit_all["rmse"] - fit_all["joint"]["rmse"]))
        print("    (b) same fit restricted to the probes at %s, whose records are"
              % ", ".join(RELIABLE_PROBES))
        print("        consistent with 1-D conduction at the nominal positions (see [3c]):")
        fit = fit_h_bar(data, liquid_model=args.liquid_model, T_P_C=T_P_C, h_i=h_i,
                        probes=RELIABLE_PROBES)
        h_bar = fit["h_bar"]
        print("          h_bar* = %.0f W/m2K   RMSE = %.2f degC over n = %d points%s"
              % (h_bar, fit["rmse"], fit["n"], "  [at bound]" if fit["at_bound"] else ""))
        print("          the two estimates differ by %.1f %%, so the exclusion does not"
              % (100.0 * abs(fit_all["h_bar"] - h_bar) / h_bar))
        print("          drive the answer; h_bar* is adopted for the reported predictions.")
        t_eq = (H_G_C / ((1.0 - H_G_P) * h_bar)) ** (1.0 / H_G_P)
        print("          h_bar* equals the time mean of the reported law over 0-%.1f s,"
              % t_eq)
        print("          i.e. the interval over which most of the latent heat is removed;")
        print("          the 0-200 s mean (%.0f W/m2K) is %.2fx smaller."
              % (h_bar_time_average(200.0), h_bar / h_bar_time_average(200.0)))
    if args.time_dependent_h:
        print("    NOTE: --time-dependent-h given; using h_g(t) pointwise (diagnostic only)")

    print("\n    Melt-convection coefficient h_i (Table 1 range %.0f-%.0f W/m2K):"
          % TABLE1_HI_RANGE)
    for lm in ("virtual-origin", "interface-film"):
        f = fit_h_i(data, liquid_model=lm, T_P_C=T_P_C, h_bar=h_bar)
        print("      %-15s bounded optimum %6.0f%s  RMSE %6.2f degC; spans %.2f degC "
              "over 500-1800" % (lm, f["h_i"], " (at bound)" if f["at_bound"] else "        ",
                                 f["rmse"], f["d_rmse_over_range"]))
    print("      -> the optima sit at opposite bounds and the whole Table 1 range moves the")
    print("         global RMSE by ~1 degC; h_i is not identifiable.  The Table 1 mid-range")
    print("         value h_i = %.0f W/m2K is prescribed." % h_i)

    # -- 3c. effective thermocouple positions ------------------------------------------
    print("\n[3c] Effective thermocouple positions (diagnostic, not applied)")
    print("     position each record would need in order to reproduce itself, with")
    print("     h_bar = %.0f W/m2K and everything else fixed:" % h_bar)
    print("     %-6s %10s %11s %10s %10s %10s" % ("probe", "x_nom mm", "x_eff mm",
                                                  "shift mm", "RMSE nom", "RMSE eff"))
    for r in effective_positions(data, h_bar, h_i=h_i, T_P_C=T_P_C,
                                 liquid_model=args.liquid_model):
        print("     %-6s %10.1f %11.2f %10.2f %10.2f %10.2f%s"
              % (r["probe"], r["x_nominal_mm"], r["x_effective_mm"], r["shift_mm"],
                 r["rmse_nominal_C"], r["rmse_effective_C"],
                 "   <- at the search bound" if r["at_bound"] else ""))
    print("     The four outer probes sit within 1-2 mm of their nominal positions.")
    print("     The 10 mm record behaves like a probe at ~7 mm, and the 5 mm record is")
    print("     driven to the chill face itself, i.e. it is colder than ANY position in")
    print("     the 1-D model: a position error alone cannot account for it.")

    # -- 4. predictions ----------------------------------------------------------------
    print("\n[4] Interface history and cooling curves (method B: reduced interface ODE)")
    prob = experiment_problem(h_i=h_i, T_P_C=T_P_C, liquid_model=args.liquid_model,
                              h_bar=h_bar, time_dependent_h=args.time_dependent_h)
    hist = march_interface(prob, t_end=args.t_end)
    print("    incubation time t* = %.2e s   s(200 s) = %.2f mm" % (incubation_time(prob),
                                                                    hist.s_of(200.0) * 1e3))
    print("    front arrival [s]: " + ", ".join(
        "%s=%.2f" % (lab, hist.arrival_time(x)) for lab, x in zip(TC_LABELS, TC_POSITIONS_M)))
    write_predictions(prob, hist)
    print("    -> %s" % os.path.basename(CSV_PREDICTIONS))

    # -- 5. metrics --------------------------------------------------------------------
    print("\n[5] Validation metrics")
    rows: List[Dict[str, object]] = []
    rows += _metric_rows("analytical, fitted mean h_bar = %.0f" % h_bar, prob, data)
    rows += _metric_rows("analytical, h_bar = 0-200 s mean of h_g (%.0f)"
                         % h_bar_time_average(200.0),
                         experiment_problem(h_i=h_i, T_P_C=T_P_C,
                                            liquid_model=args.liquid_model,
                                            h_bar=h_bar_time_average(200.0)), data)
    rows += _metric_rows("analytical, frozen-coefficient h_g(t) (not admissible)",
                         experiment_problem(h_i=h_i, T_P_C=T_P_C,
                                            liquid_model=args.liquid_model,
                                            time_dependent_h=True), data)
    for h_t in (500.0, 1800.0):
        rows += _metric_rows("analytical, Table 1 h_i = %.0f" % h_t,
                             experiment_problem(h_i=h_t, T_P_C=T_P_C, h_bar=h_bar,
                                                liquid_model=args.liquid_model), data)
    rows += _metric_rows("conduction-only limit (no melt convection)",
                         experiment_problem(h_i=1e9, T_P_C=T_P_C, h_bar=h_bar,
                                            liquid_model="conduction"), data)
    for lm in LIQUID_MODELS:
        if lm in (args.liquid_model, "conduction"):
            continue
        rows += _metric_rows("liquid closure '%s'" % lm,
                             experiment_problem(h_i=h_i, T_P_C=T_P_C, h_bar=h_bar,
                                                liquid_model=lm), data)
    fd_rows, fd = _fd_metric_rows(prob, data)
    rows += fd_rows
    write_metrics(rows)
    hdr = "    %-46s %-8s %8s %8s %8s %8s %7s"
    print(hdr % ("variant", "probe", "RMSE", "MAE", "MaxAE", "Bias", "R2"))
    for r in rows:
        if r["probe"] != "GLOBAL" or r["window"] != "full":
            continue
        print("    %-46s %-8s %8.2f %8.2f %8.2f %8.2f %7.3f"
              % (str(r["variant"])[:46], r["probe"], r["RMSE_C"], r["MAE_C"],
                 r["MaxAE_C"], r["Bias_C"], r["R2"]))
    print()
    print("    early window (t <= 30 s), where the solution is inside its domain of validity:")
    for r in rows:
        if r["probe"] != "GLOBAL" or r["window"] != "t<=30s":
            continue
        print("    %-46s %-8s %8.2f %8.2f %8.2f %8.2f %7.3f"
              % (str(r["variant"])[:46], r["probe"], r["RMSE_C"], r["MAE_C"],
                 r["MaxAE_C"], r["Bias_C"], r["R2"]))
    print("    per-probe rows -> %s" % os.path.basename(CSV_METRICS))
    print()
    met, _, _, _ = evaluate(prob, data)
    print("    %-6s %5s %7s %7s %7s %7s %6s %9s %9s %9s %9s"
          % ("probe", "n", "RMSE", "MAE", "MaxAE", "Bias", "R2",
             "t660 exp", "t660 mod", "t650 exp", "t650 mod"))
    for lab in TC_LABELS:
        v = met[lab]
        print("    %-6s %5d %7.2f %7.2f %7.2f %7.2f %6.3f %9.2f %9.2f %9.2f %9.2f"
              % (lab, v["n"], v["RMSE_C"], v["MAE_C"], v["MaxAE_C"], v["Bias_C"], v["R2"],
                 v["t_TF_obs_s"], v["t_TF_mod_s"], v["t_650_obs_s"], v["t_650_mod_s"]))
    print("    (t660/t650 = time at which the curve falls through 660 / 650 degC; the")
    print("     650 degC crossing is the sharper proxy for the arrival of the front,")
    print("     because the melt plateau itself sits a few K above T_F.)")

    # -- 6. sensitivity ----------------------------------------------------------------
    print("\n[6] Sensitivity / identifiability")
    print("    %-56s %9s %9s %11s %9s"
          % ("case", "s(200)mm", "d_s %", "t_arr 90mm", "RMSE"))
    for r in sensitivity_table(data, T_P_C=T_P_C, liquid_model=args.liquid_model,
                               h_bar=h_bar):
        print("    %-56s %9.2f %9.2f %11.2f %9.2f"
              % (str(r["case"])[:56], r["s200_mm"], r["d_s200_pct"], r["t_arr_90mm_s"],
                 r["rmse_global_C"]))

    # -- 7. figure ---------------------------------------------------------------------
    print("\n[7] Figures")
    figure_path = make_figure(prob, hist, data, fd=None, title_extra="")
    print("    -> %s  (analytical vs experiment)" % os.path.basename(figure_path))
    fd_plot = fd_reference(prob, t_end=200.0, out_t=np.linspace(1.0, 200.0, 120), nx=800)
    figure_supp_path = make_figure(prob, hist, data, path=FIG_OUT_SUPP, fd=fd_plot,
                                  residual_panel=True, title_extra="")
    print("    -> %s  (adds the FD reference and a residual panel)"
          % os.path.basename(figure_supp_path))

    # -- 8. interface position vs time -------------------------------------------------
    print("\n[8] Interface position vs time")
    if not os.path.exists(CSV_POSITION):
        t_pos, P_pos = digitise_position_figure()
    else:
        t_pos, P_pos = load_position_data()
        print("    reusing %s" % os.path.basename(CSV_POSITION))
    fit_pos = fit_h_bar_position(t_pos, P_pos, h_i=h_i, T_P_C=T_P_C,
                                 liquid_model=args.liquid_model)
    print("    h_bar from the interface kinetics alone : %.0f W/m2K  (RMSE %.2f mm)"
          % (fit_pos["h_bar"], fit_pos["rmse_mm"]))
    print("    h_bar from the cooling curves           : %.0f W/m2K" % h_bar)
    print("    the two differ by a factor %.2f - the two published experimental figures"
          % (h_bar / fit_pos["h_bar"]))
    print("    are not mutually consistent (see the report):")
    print("      position figure : front at 90 mm at t = %.0f s"
          % ((90.0 / POSITION_FIT_A) ** (1.0 / POSITION_FIT_B)))
    for tc, xc, _, _, lab in interface_from_cooling_curves(data):
        if lab == "90mm":
            print("      cooling curves  : 90 mm probe crosses 660 degC at t = %.0f s" % tc)
    make_position_figure(data, h_bar, t_pos, P_pos, fit_pos, h_i=h_i, T_P_C=T_P_C,
                         liquid_model=args.liquid_model)
    print("    -> %s" % os.path.basename(FIG_POSITION))
    velocity_figure_path = make_velocity_position_figure(hist, data, h_bar)
    print("    -> %s" % os.path.basename(velocity_figure_path))
    gradient_figure_path = make_gradient_position_figure(prob, hist, data, h_bar)
    print("    -> %s" % os.path.basename(gradient_figure_path))

    # -- 8. verification ---------------------------------------------------------------
    chk = run_verification()
    print("\nDone.  Outputs written to %s" % HERE)
    return 1 if chk.n_fail else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generate only the analytical-versus-numerical comparison deliverables.

    The former experimental audit remains available as a library implementation in
    ``_legacy_experimental_main`` but is deliberately not part of the normal run.
    """
    # Import by module name so Windows resolves the Unicode working-directory path
    # itself; passing HERE through a second loader can corrupt accented path names.
    import numerical_stefan_model_pure_al as comparison
    return comparison.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
