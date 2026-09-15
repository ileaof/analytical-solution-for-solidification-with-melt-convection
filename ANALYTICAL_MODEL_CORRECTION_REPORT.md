# Audit, correction and experimental validation of the melt-convection closed-form solution

**Source**: G. E. Mendes Santos Júnior, F. S. Rocha, A. B. S. Silva, D. A. R. Carmo, M. O. Silva,
I. L. Ferreira, *On a Novel Closed-Form Analytical Solution for Unsteady Solidification: Theory
and Application*, **Am. J. Eng. Appl. Sci. 19 (2026) 88–116**, DOI 10.3844/ajeassp.2026.88.116,
section *An Important Solution for Melt Convection*, Eqs. (66)–(83).

**Code audited**: `analytical_model_2024_Al_melt_convection_dev.py` (1425 lines, unchanged).
**Code delivered**: `analytical_model_2024_Al_melt_convection_corrected.py`.

**Experiment**: transient horizontal solidification of pure Al against a chill with
`h_g(t) = 6350 t^(-0.21) W m^-2 K^-1`, `T_F = 660 °C`, thermocouples at
`x = 5, 10, 15, 30, 50, 90 mm` (`Figure_cooling_curve_versus_numerical_simulation.jpg`).

---

## 0. Executive summary

1. **The published formulation is implementable but contains 16 identifiable manuscript
   defects** (sign errors, two mutually incompatible definitions of the similarity variable,
   an incorrect `n`-scaling of every liquid-side dimensionless group, and one function —
   `ψ_L` — that is used in eight equations and never defined).
2. **The released development script contains 24 identifiable coding defects**, including an
   inverted Stefan number, a dimensional (K-valued) term inside a dimensionless equation, and
   `fsolve` calls with no convergence test.
3. **Published Table 2 is not a solution of Eq. (83).** The legacy residual has **no zero
   anywhere on φ > 0**; `fsolve` returns `ier = 5` ("not making good progress") and the script
   uses the stalled iterate. Four of the five published values are reproduced to ±0.02 K by
   that stalled iterate, which identifies the provenance of the table conclusively.
4. **Eq. (83) itself is a ~20–50 % approximation** even at constant `h`, because it equates the
   Stefan flux with the *parabolic* velocity `2φ²α_S/s` while φ actually varies strongly with
   `s`. Integrating the reduced interface equation instead removes most of that error; the `Ω`
   correction of Eq. (29d), meant to repair it, makes it worse.
5. **The published melt-convection closure is over-determined**: no choice of `ψ_L` can satisfy
   simultaneously (i) temperature continuity at the front, (ii) `q_L` increasing with `h_i`,
   (iii) Eq. (82) respecting `T_L(s⁺) ≤ T_P`, and (iv) the Neumann limit as `h_i → ∞`.
6. **The solution requires a single constant `h`; `h_g(t)` must not be inserted pointwise.**
   The kernel `exp(hx/k + h²αt/k²) erfc(·)` solves Eq. (66) only for constant `h`. Used
   pointwise, the reported law gives RMSE 79.7 °C with a +72 °C bias. Used as the constant mean
   it implies — `h̄* = 4056 W m⁻² K⁻¹`, the time average of `6350 t^(−0.21)` over the first
   26 s — it gives **RMSE 19.78 °C, R² 0.978**, and 10.12 °C over the four probes at 15–90 mm.
7. **So calibrated, the corrected formulation is validated by the experiment**: it performs
   within 2.4 °C of RMSE of an independent 1-D enthalpy finite-difference solution of the same
   stated problem, and reproduces the front chronology without further tuning.
8. **`h_i` remains unidentifiable.** Across four decades it moves `s(200 s)` by 0.7 % and the
   global RMSE by 0.8 °C, the optimum is monotone towards the conduction limit, and switching
   melt convection off fits marginally better than any admissible `h_i`. The melt-convection
   term is neither confirmed nor refuted by this dataset.
10. **The two published experimental figures are mutually inconsistent.** At the instant the
   90 mm probe passes 650 °C the position figure puts the front at 70 mm, not 90; calibrated
   against the interface kinetics alone `h̄` comes out 1820 W m⁻² K⁻¹ instead of 4056, a factor
   2.2. Re-fitting the cooling curves with the positions the position figure implies makes the
   fit 1.7× worse overall and 4× worse at the two outer probes, so the cooling curves support
   the nominal positions. See §12.
9. **The 5 mm and 10 mm records are anomalous, independently of the model.** Their mutual
   gradient implies a heat flux 1.76× larger than the data's own enthalpy balance permits; the
   10 mm record behaves like a probe at ≈ 7 mm and the 5 mm record coincides with the model's
   *chill-face* temperature. Displacement of the two innermost thermocouples towards the chill
   accounts for essentially all of the gap between the 10.1 °C RMSE of the outer four probes and
   the 19.8 °C of the full set. No correction has been applied.

---

## 1. The equations as printed, and what every symbol means

### 1.1 Symbols

| symbol | meaning | unit |
|---|---|---|
| `x` | distance from the chilled face | m |
| `s(t)` | solid/liquid interface position | m |
| `T_S`, `T_L` | solid / liquid temperature field | K |
| `T_F` | fusion temperature (933.15 K = 660 °C) | K |
| `T_P` | melt (pouring) temperature | K |
| `T_∞` | environment / chill temperature | K |
| `k_S`, `k_L` | thermal conductivity | W m⁻¹ K⁻¹ |
| `c_PS`, `c_PL` | specific heat | J kg⁻¹ K⁻¹ |
| `ρ_S`, `ρ_L` | density | kg m⁻³ |
| `α_S`, `α_L` | thermal diffusivity `k/(ρ c_P)` | m² s⁻¹ |
| `L` | latent heat of fusion | J kg⁻¹ |
| `h` | environment (metal/chill) heat-transfer coefficient — `h_g(t)` in the experiment | W m⁻² K⁻¹ |
| `h_i` | melt-side (convection) film coefficient | W m⁻² K⁻¹ |
| `φ` | similarity variable, `φ = s/(2√(α_S t))` (Eqs. 10–11) | – |
| `n` | `√(α_S/α_L)` (Eq. 31) = 1.41377 | – |
| `N` | `k_S/k_L` (p. 111) = 2.34066 | – |
| `Ste` | `c_PS (T_F − T_∞)/L` (Eq. 25) = 1.88663 | – |
| `Θ₀` | `(T_P − T_F)/(T_F − T_∞)` (p. 111) | – |
| `Bi_env` | `h s/k_S` (Eq. 24) | – |
| `Bi_i` | `h_i s/k_L` (p. 111) | – |
| `Fo` | `α_S t/s²` (Eq. 23); `Bi²Fo = Bi²/(4φ²)` (Eq. 26) | – |
| `Ω`, `Ω*` | `2hα_S/k_S` (Eq. 29d), `Bi_env/(φ Ste)` (p. 111) | m s⁻¹, – |
| `ψ`, `ζ` | solid auxiliary functions (Eqs. 18b, 19b) | – |
| `ψ_L`, `f` | liquid auxiliary functions (Eq. 80d; `ψ_L` **never defined**) | – |

### 1.2 Governing equations and boundary conditions (66)–(72)

```
(66)  ∂²T_S/∂x² = (1/α_S) ∂T/∂t                    0 < x < s(t)
(67)  ∂²T_L/∂x² = (1/α_L) ∂T/∂t                    s(t) < x < +∞
(68)  t = 0,      0 < x < +∞,    T = T_P
(69)  t > 0,      x = 0,         −k ∂T/∂x|₀ = h (T − T_∞)        [sign typo, see M2]
(70)  t > 0,      x = s(t),      T = T_F
(71)  t > 0,      x → +∞,        T = T_P
(72)  ρ_S L ds/dt = k_S ∂T/∂x|_{x=⁻s} − h_i (T_L(x=⁻s,t) − T_F)  [⁻s→s⁺ in the melt term, M11]
```

Coordinate convention: `x` is measured **from the chilled face**, `x = 0` is the Robin
(convective) surface, the solid occupies `0 ≤ x ≤ s(t)` and the melt `x ≥ s(t)`. The
superscripts in `x = ⁻s` / `x = ⁺s` are one-sided limits (`s⁻` from the solid, `s⁺` from the
melt), not a reflection of the axis: Eqs. (66)/(67) place both phases on the positive axis.
Heat-flux direction: the solid is hotter at the front than at the chill, so `∂T_S/∂x|_{s⁻} > 0`
and the flux `−k_S ∂T_S/∂x` points towards `−x`, i.e. into the chill; the melt is superheated,
so `∂T_L/∂x|_{s⁺} > 0` and melt heat also flows towards `−x`, i.e. into the front, retarding
solidification. Consequently the melt term must be **subtracted** in Eq. (72) — it is, and that
fixes the sign of every downstream equation (see M8, M10).

### 1.3 Auxiliary functions, from Eqs. (18b), (19b)

```
ψ(Bi_env, φ) = 1 − erfc(φ) + exp(Bi_env + Bi_env²/4φ²) · erfc(φ + Bi_env/2φ)
ζ(Bi_env, φ) =   − erfc(φ) + exp(Bi_env + Bi_env²/4φ²) · erfc(φ + Bi_env/2φ)      ⇒  ψ = 1 + ζ
```

### 1.4 Fields and gradients (73)–(77)

```
(73)/(74a)  [T_S(x,t) − T_F]/[T_∞ − T_F]
              = { erfc(u) − exp(h x/k_S + h²α_S t/k_S²) erfc(u + h√(α_S t)/k_S) + ζ } / ψ ,
            u = x/(2√(α_S t))

(75)        ∂T_S/∂x|_{s⁻} = [(T_F − T_∞)/ψ] · (h/k_S) · exp(A_S) erfc(B_S)
            A_S = Bi_env + Bi_env²/4φ² ,   B_S = φ + Bi_env/2φ ,   A_S − B_S² ≡ −φ²

(76)        [T_L(x,t) − T_P]/[T_F − T_P]
              = { erfc(ξ) − exp(h_i X/k_L + h_i²α_L t/k_L²) erfc(ξ + h_i√(α_L t)/k_L) } / ψ_L

(77)        ∂T_L/∂x|_{s⁺} = [(T_P − T_F)/ψ_L] · (h_i/k_L) · exp(A_L) erfc(B_L)
            A_L = Bi_i + Bi_i²/(4n²φ²) ,   B_L = nφ + Bi_i/(2nφ) ,   A_L − B_L² ≡ −n²φ²
```

The identity `A − B² = −(first erfc argument)²` holds for **every** such group in the paper,
because `A = h X/k + h²αt/k²` and `B = X/(2√(αt)) + h√(αt)/k` give
`2·[X/(2√(αt))]·[h√(αt)/k] = h X/k`. Two consequences, both used below:

* the second and third terms of Eqs. (75a–c) and (80d) **cancel identically** — they are
  `+e^{−φ²}/√π` and `−e^{A−B²}/√π = −e^{−φ²}/√π` (verified numerically to 3.6 × 10⁻¹¹);
* every product can be evaluated as `exp(A − B²)·erfcx(B)`, which never overflows.

### 1.5 Melt-side closure (79)–(83)

```
(79)  −k_L ∂T/∂x|_{s⁺} = −h_i (T_L(s⁺,t) − T_F)
(80d) f(Bi_i, φ) = (Bi_i/2φ) exp(A_L) erfc(B_L)                (after the exact cancellation)
(81)  (Bi_i/2φ)[T_L(s⁺,t) − T_F] = [(T_P − T_F)/ψ_L] f(Bi_i, φ)
(82)  T_L(s⁺,t) = T_F + [2φ (T_P − T_F)/(Bi_i ψ_L)] f(Bi_i, φ)
(83)  (1/Ste)(φ − Bi_env/φ) = F_S(Bi_env,φ)/ψ(Bi_env,φ) − (Θ₀/N) f(Bi_i,φ)/ψ_L(Bi_i,φ)
      F_S(Bi_env,φ) = (Bi_env/2φ) exp(A_S) erfc(B_S)
```

**Dimensional consistency.** Eq. (83) is obtained from Eq. (72) by dividing throughout by
`2 α_S ρ_S c_PS φ (T_F − T_∞)/s`. Doing this explicitly:

* `ρ_S L (2φ²α_S/s − Ω)` → `(L/[c_PS(T_F−T_∞)])(φ − Bi_env/φ) = (1/Ste)(φ − Bi_env/φ)` ✔
* `k_S ∂T_S/∂x|_{s⁻}` → `F_S/ψ` (uses `k_S = α_S ρ_S c_PS`) ✔
* `h_i (T_L(s⁺) − T_F)` → `(1/N)(Bi_i/2φ)(T_L − T_F)/(T_F − T_∞)` → with (81), `(Θ₀/N) f/ψ_L` ✔

so every term of Eq. (83) is dimensionless. The dimensionlessness is re-checked numerically in
the delivered code by a **scale-invariance test**: multiplying `k_S, k_L, h, h_i` by the same
factor λ (which leaves every dimensionless group invariant) must leave φ unchanged — it does,
to 10⁻⁹.

---

## 2. Manuscript defects found (M1–M16)

Each is classified **[TYPO]** (evident typographical/sign slip), **[INCONSISTENT]** (two printed
statements that cannot both be true) or **[UNDEFINED]**.

| # | Where | Defect | Resolution adopted |
|---|---|---|---|
| M1 | Eq. (8) | LHS printed `T(x,t) − T_∞ = A_S + B_S{…}`, but Eqs. (9)–(13), (17) all require `T(x,t) = A_S + B_S{…}` | **[TYPO]** use `T(x,t) = …` |
| M2 | Eqs. (4), (69) | `−k ∂T/∂x|₀ = h(T − T_∞)`. With the body at `x>0` hotter than the environment, `∂T/∂x|₀ > 0` so the LHS is negative while the RHS is positive | **[TYPO]** the solution actually used (Eq. 8 = the classical third-kind solution) satisfies `−k ∂T/∂x|₀ = h(T_∞ − T(0,t))`; verified to 3 × 10⁻⁶ relative |
| M3 | Eqs. (28a) vs (49) | `φ = s/(2√(α_S t)) − h√(α_S t)/k_S` vs `+`. Eqs. (28b) and (50) are *identical* and consistent only with (49) | **[TYPO]** (28a) carries the sign error |
| M4 | Eqs. (10–11) vs (28a)/(49) | **Two incompatible definitions of φ.** Every "(a)→(b)" conversion in the paper (18a→18b, 19a→19b, 21a→21b, 75a→75b, 77a→77b) replaces `h√(α_S t)/k_S` by `Bi_env/(2φ)`, which is true **iff** `φ = s/(2√(α_S t))` | **[INCONSISTENT]** adopt `φ = s/(2√(α_S t))` (Eqs. 10–11); it is the only definition that makes every printed "b/c" form exact |
| M5 | Eqs. (28c) vs (29e) | (28c) gives `ds/dt = 2φ²α_S/s − Ω`; (29e) gives `ds/dt = 2φ²α_S/s`. They differ by Ω | **[INCONSISTENT]** both `Ω` options implemented (`omega=True/False`); `omega=False` is the default and is consistent with M4 |
| M6 | Eq. (44) vs p. 111 | Two different quantities both named `N`: `α_Lρ_L/(α_Sρ_S) = 0.46459` and `k_S/k_L = 2.34066` | **[INCONSISTENT]** the melt-convection section and Table 1 use `N = k_S/k_L`; the code keeps them separate |
| M7 | Eq. (48) vs (53) | (53) `v = 1/(2γs+δ)` follows from (46); but (48) states `δ = 1/[(q_S−q_L)/ρ_S L] − 2γs`, i.e. `2γs+δ = 1/(v−Ω)`. With `γ = 1/(4α_Sφ²)` and `v−Ω = 2φ²α_S/s`, (48) collapses to `δ ≡ 0` | **[INCONSISTENT]** not used; the delivered code obtains `t(s)` from Eq. (22) (method A) or by integration (method B) |
| M8 | Eq. (55) | Printed with a leading minus, giving `∂T_L/∂x|_{s⁺} < 0` (melt colder away from the front). Direct differentiation of Eqs. (32)+(36) gives `+` | **[TYPO]** propagates into Eqs. (43), (45), (57), where the melt term is consequently **added** to the Stefan balance and superheat *accelerates* freezing |
| M9 | Eqs. (77b,c), (80a–d), (81)–(83) | `h_i√(α_L t)/k_L → n Bi_i/(2φ)` and `h_i²α_L t/k_L² → n² Bi_i²/(4φ²)`. Since `√(α_L t) = √(α_S t)/n = s/(2nφ)`, the correct substitutions are `Bi_i/(2nφ)` and `Bi_i²/(4n²φ²)`: **the paper multiplies by `n` where it should divide** | **[TYPO]** Eq. (77a) itself is inconsistent with (77b): its own prefactor `2nφ/(√π s) = 1/(√π √(α_L t))` uses the correct relation. Only the corrected scaling satisfies `A_L − B_L² = −n²φ²`, the identity that makes the last two terms of Eq. (80d) cancel |
| M10 | Eqs. (77a–c) | Prefactor `(T_F − T_P) < 0` | **[TYPO]** Eq. (80a) of the same derivation uses `(T_P − T_F)`; that is physically correct and is adopted |
| M11 | Eq. (72), Table 2 header | `T_L(x = ⁻s, t)` where Eqs. (79)–(82) write `T_L(x = ⁺s, t)` | **[TYPO]** `T_L` is the *liquid* field ⇒ the `s⁺` limit is meant |
| M12 | Eqs. (76a,b), (77a–c), (80a–d), (81)–(83) | **`ψ_L` is used in eight equations and never defined** | **[UNDEFINED]** four closures implemented and compared — see §4 |
| M13 | p. 112 | Eq. (80c) is printed twice ("Instead of this:" / "You must have this:"); the first has `exp[(nφ + n h_i s/(2φ k_L))²]` in a denominator where the second has `exp[(nφ + n Bi_i/2φ)²]` | editing artefact; the second (with `Bi_i`) is used |
| M14 | Eqs. (75a–c), (80d) | The 2nd and 3rd terms cancel identically (§1.4) | not an error, but a catastrophic-cancellation / overflow trap; the reduced form is used and the identity is verified |
| M15 | Eq. (83) | The closure equates `ρ_S L ds/dt` with `ρ_S L·2φ²α_S/s`, valid only for **constant** φ. Since `Bi_env = h s/k_S` grows with `s`, φ varies from 0.22 to 0.58 over a 100 mm layer | **[APPROXIMATION, quantified]** at constant `h = 4000 W m⁻² K⁻¹` Eq. (83) over-predicts `s(t)` by up to **37 %** relative to an independent FD solution, whereas integrating the reduced interface equation errs by **≤ 8 %** |
| M16 | Table 2 | The five values are not roots of Eq. (83) | see §3.3 |

---

## 3. Defects in `analytical_model_2024_Al_melt_convection_dev.py` (C1–C24)

### 3.1 Physics / mathematics

| # | Line(s) | Defect | Consequence |
|---|---|---|---|
| C1 | `Stes = L/(cps*(TF-Tinf))`, `term1 = phi/Stes` | **Inverted Stefan number.** Eq. (25) is `Ste = c_PS(T_F−T_∞)/L = 1.88663`; the script stores its reciprocal (0.53005) and then *divides* by it, so the LHS of Eq. (83) is `φ·Ste` instead of `φ/Ste` | factor `Ste² = 3.559` error |
| C2 | `term3 = 1/N*Bioti/(2*phi)*Theta0*(2*phi/Bioti*(Tp-TF)*f/psiL)` | **Spurious `(T_P − T_F)`.** The correct group is `(Θ₀/N)·f/ψ_L`; the extra factor makes the term **dimensional** (units of K) and 37.33× too large | destroys the root (§3.3) |
| C3 | `def f(...)`: `n/(sqrt(pi)*exp(phi**2))` | should be `exp(n**2*phi**2)` (Eq. 80d) | wrong `f` |
| C4 | `def gradl(...)`: same `exp(phi**2)` | same | wrong melt gradient |
| C5 | `print(f'N = kl/ks = {N:.5f}')` while `N = ks/kl` | mislabelled output | reporting only |
| C6 | `Omega = 0.0  # Biotenv/(Stes*phi)` | the `Ω*` term of Eq. (83) is silently switched off; the commented-out expression is itself `Ste²` too large | undocumented model change |
| C7 | `zetaL` uses `erfc(n*phi + Biot/(2*phi))`, `psiL` uses `erfc(n*phi + n*Biot/(2*phi))` | the two liquid auxiliaries disagree with each other, and neither equals the dimensionally correct `n*phi + Biot/(2*n*phi)` | inconsistent liquid field |
| C21 | main loop and all dead loops | `Biotenv`/`Bioti` are formed from the **fixed** reference `s = 0.10 m` while the gradients are evaluated at the **running** `S` | mixes a fixed reference length with the instantaneous interface position |
| C22 | `t = S**2/(4*alphas*raiz**2) + 2*S*ks*raiz/(h0*alphas)` | the linear term is not Eq. (48) (which collapses to `δ ≡ 0`, M7). For Al at `s = 0.1 m` it contributes **1506 s** against 35 s for the parabolic term | the `s–t` relation is dominated by an unjustified term |

### 3.2 Numerics, software engineering, reporting

| # | Defect | Consequence |
|---|---|---|
| C8 | `thetaL()` calls `psiL(Bioti,phi,L,n)` — four arguments to a three-argument function, passing the **latent heat `L`** as `n` | `TypeError` (unreached: dead code) |
| C9 | `thetaL()` mixes `erfc(n*phi*(s+x)/s)` with `erfc(phi*(s+x)/s + …)` — `n` missing in the second | inconsistent profile |
| C10 | `dSdt()` computes `ql` and never uses it; then adds `+ Biotenv/(Stes*phi)` (dimensionless) to a velocity in m s⁻¹ | **dimensionally inconsistent**; `rhos` is taken from global scope |
| C11 | `dSdt_t()` calls `grads(...)` with 11 arguments (it takes 5) and `gradl(...)` with 11 (it takes 6) | `TypeError` (dead code) |
| C12 | `grads`, `gradl`, `dSdt`, `dSdt_t` read `rhos`, `hi`, `kl`, `ks` from module scope | non-reentrant, silent coupling |
| C13 | `x = fsolve(equation, 1.0, args=data)` — **no `full_output`, no `ier` test, no positivity test, no residual test** | the published Table 2 is built on stalled iterates; for `h_i = 500` the returned root is **φ = −0.2976 < 0** and is used anyway |
| C14 | Hard-coded Windows paths: `C:\temp\Al_pure_*.dat`, `C:\MacbookPro\ufpa\PPGEM\IC_SIM_2025\Alunos\Beatriz\data_*.csv` | not portable |
| C15 | `quit()` at line 717 makes lines 719–1425 (**50 % of the file**) unreachable; a second `quit()` at line 791; a second `plt.show()` at line 1385 never runs | dead code |
| C16 | Seven near-identical `h1…h4` blocks (lines 813, 864, 898, 949, 982, 1034, 1066) of ~50-70 lines each, all unreachable, several of which would raise `TypeError` if reached (`data = (Biot)` is a scalar, not a tuple, so `fsolve(..., args=Biot)` cannot match `equation`'s seven parameters) | unmaintainable |
| C17 | Unit labels: `h0 = … [W/(m.K)]` in one place and `[W/m^2.K]` in another for the same quantity; `ylabel = r'$\mathrm{Interface Temperature, [K]}$'` (spaces inside `\mathrm`); `TL[i-1] = vel*gl` names a **cooling rate** (K s⁻¹) `TL` | misleading axes/reports |
| C18 | `exp(Biot + Biot**2/(4*phi**2))*erfc(phi + Biot/(2*phi))` overflows (→ `inf·0 = nan`) once `Biot + Biot²/(4φ²) > 709` — already at `Biot ≈ 30` for `φ = 0.5` and `Biot ≈ 15` for `φ = 0.3`, i.e. inside the range the paper itself explores (Fig. 11 uses `Biot` up to 5, but `h_g(t)` reaches `Bi_env ≈ 3` here and larger for thicker layers); `1/(sqrt(pi)*exp(phi**2))` underflows for `φ > 27` | silent NaNs |
| C19 | `te[i-1] = te[i-2] + ds/vel` with `i = 0` wraps around to `te[-1]`/`te[-2]` | corrupts the array |
| C20 | variable named `TL` holds `vel*gl`, a cooling rate | reporting |
| C23 | unused imports `root`, `integrate`; 98 `np.zeros` scratch arrays, almost all never used | noise |
| C24 | `vel = dSdt(s, h0, hi, …, Bioti, …)` passes the *initial* `hi = 500` while the loop varies `Bioti` | harmless only because `h_i` cancels inside `dSdt`; still a variable passed incorrectly |

### 3.3 Table 2 is built on a non-converged solve

Running the legacy `equation()` verbatim (`s = 0.10 m`, `h = 400 W m⁻² K⁻¹`, `T_P = 970.48 K`):

| `h_i` (W m⁻² K⁻¹) | Table 2 (K) | legacy path (K) | `fsolve` status | legacy residual | corrected Eq. (83) (K) |
|---|---|---|---|---|---|
| 1800 | 949.18 | 949.18 | `ier = 5` | −1.63 | 942.87 |
| 1600 | 950.74 | 950.74 | `ier = 5` | −1.61 | 943.49 |
| 1400 | 952.57 | 952.58 | `ier = 5` | −1.57 | 944.19 |
| 800  | 960.31 | 960.32 | `ier = 5` | −1.37 | 946.89 |
| 500  | 963.69 | 970.63 | `ier = 5` | +1.06 | 948.68 |

`ier = 5` is scipy's *"the iteration is not making good progress"*. Scanning the legacy residual
over `φ ∈ [0.05, 5]` shows it is **strictly negative everywhere**, with a maximum at `φ ≈ 0.44`
— precisely where `fsolve` stalls and precisely the values reported. The published table is
therefore a table of *stationary points of a residual that has no root*, and the agreement of
four of the five entries to ±0.02 K identifies its provenance beyond reasonable doubt. The
cause is C1 + C2: with the spurious `(T_P − T_F)` factor the melt term is 0.90 instead of 0.024,
which alone removes the crossing.

The corrected column keeps the *same* (`as-coded`) `ψ_L` so that only the audited errors differ;
it is a genuine bracketed root with residual < 10⁻¹³ and a verified unique sign change.

---

## 4. `ψ_L`: what the manuscript leaves open, and why it matters

`ψ_L` appears in Eqs. (76), (77), (80)–(83) and is never given. Four self-consistent closures
were implemented (`--liquid-model`):

| closure | `ψ_L` | continuity `T_L(s)=T_F` | `q_L` monotone in `h_i` | Eq. (82) `≤ T_P` | `h_i→∞` Neumann limit |
|---|---|---|---|---|---|
| `virtual-origin` *(default)* | `erfc(nφ) − e^{A_L}erfc(B_L)` | **yes** | **no (inverted)** | **no** | **yes (exact)** |
| `interface-film` | `1`, profile anchored at `η = x − s` | no (film drop) | **yes** | **yes** | no |
| `as-coded` (dev script) | `1 − erfc(nφ) + e^{A_L}erfc(B_L)` | no | yes | yes | no |
| `conduction` | `erfc(nφ)` (`h_i → ∞`) | yes | n/a | n/a | exact |

**No closure satisfies all four requirements — the published melt-convection closure is
over-determined.** Quantitatively, at `s = 0.05 m`, `t = 20 s`, `T_P − T_F = 30.9 K`:

```
h_i [W/m²K]     100      500     1800    20000     →∞      pure-conduction limit
virtual-origin  160.2    158.8   155.1   140.3    134.1    134.1  kW/m²   (DECREASING)
interface-film    3.0     13.2    33.5     58.8     59.6    134.1  kW/m²   (increasing)
as-coded          0.56     2.6      7.9     24.7     30.2   134.1  kW/m²   (increasing)
```

and the melt driving temperature of Eq. (82) at `h_i = 1000 W m⁻² K⁻¹`:

```
virtual-origin   T_L(s⁺) − T_F = 157.2 K   →  T_L(s⁺) = 817 °C  >  T_P = 691 °C   (violates the maximum principle)
interface-film                    22.9 K
as-coded                           4.9 K
```

* The **`virtual-origin`** closure is the one required by the *printed* liquid profile: the
  argument `(s+x)` of Eq. (76a) and the value `nφ` of Eq. (77a) place the Robin surface at the
  chill face `x = 0`, with the solidified layer acting as a *virtual liquid adjunct* (the
  Garcia–Prates construction). Normalising by its own interface value is what makes the solid
  and liquid fields continuous, and it makes Eq. (83) collapse **exactly** onto the paper's own
  non-convective closure Eqs. (45)/(57) as `h_i → ∞` (verified: the melt term reduces to
  `Ste_L·N_old·n/(√π erfc(nφ) e^{n²φ²})` with `Ste_L·N_old ≡ Ste·Θ₀/N`). It is kept as the
  default because it is the only closure that satisfies the continuity requirement.
* Its price is that `q_L` **decreases** with `h_i` and is bounded below by the conduction flux
  — i.e. within this closure "melt convection" cannot deliver *more* heat than pure conduction,
  only less, which inverts the physical meaning of `h_i`.
* The **`interface-film`** closure is the exact solution of (67)+(71)+(79) in the
  interface-attached frame and is the only one with the physically correct `h_i` behaviour
  (`q_L → 0` as `h_i → 0`, monotonically increasing). Its temperature jump at the front is the
  defining feature of a film/contact resistance, not an inconsistency, but it does violate the
  continuity requirement.

Both are carried through the whole validation; the metrics file contains all of them.

---

## 5. Corrections actually made, and their justification

**(a) Direct implementation of the published equations** — Eqs. (18b), (19b), (22)–(26), (31),
(70), (73)/(74), (75), (79), (80d), (81)–(83) are implemented as printed (after the algebraic
reductions of §1.4, which are exact identities).

**(b) Corrections of evident typographical errors** — M1, M2, M3, M8, M9, M10, M11 (§2). Each is
justified either by a contradiction *inside* the manuscript (M3: (28a) vs (49)/(50); M9: (77b)
vs (77a); M10: (77) vs (80a)) or by direct differentiation of the very profile the paper
prescribes (M2, M8), and each is verified numerically in the delivered test suite.

**(c) Corrections of code defects** — C1–C24 (§3), all verified analytically before being applied.

**(d) Additional numerical approximations introduced here** (absent from the paper, flagged
`[APPROX]` in the source):

0. **A constant mean surface coefficient `h̄`** replaces the reported `h_g(t)`, because the
   closed form is a constant-`h` solution (§6.1). `h̄` is the single fitted parameter of the
   whole study.
1. **Reduced-interface integration ("method B")** — see §6.5.
2. **Incubation time `t*`** — for the conduction-like closures `q_L ~ k_L(T_P−T_F)/√(π α_L t)`
   diverges as `t^(−1/2)` while `q_S → h̄ (T_F−T_∞)` stays bounded, so the superheated melt
   initially delivers more heat than the chill removes and **no solid forms until
   `t* ≈ 2.9 × 10⁻² s`**. Both fluxes are integrable, so `ρ_S L s(t) = ∫_{t*}^t (q_S − q_L) dt'`
   is well defined; it is solved by Picard iteration on a logarithmic grid (§6.6).
3. **Pseudo-similarity fields** — `T_S(x,t)` and `T_L(x,t)` satisfy the Robin condition (69),
   the interface condition (70), the far field (71) and the Stefan balance (72) exactly, but
   the PDE (66) only approximately, because ψ, ζ and the Biot numbers drift with `s(t)`. The
   defect is measured, not assumed: the normalised PDE residual of the solid field reaches
   **0.47**. This is a property of the published solution family, not of this implementation,
   and it costs ~2.4 °C of RMSE relative to a full numerical solution of the same problem.
4. **Numerically stable kernels** — `exp(A)erfc(B) → exp(A−B²)·erfcx(B)`; the redundant
   cancelling terms are removed. Verified to agree with the literal printed form to 3.6 × 10⁻¹¹
   and to stay finite for `Bi` up to 10⁵ and `φ` up to 40, where the literal form returns `NaN`.
5. **Bracketed root solving** — `solve_phi` scans `φ ∈ [10⁻⁶, 60]`, requires a sign change
   (existence), uses Brent's method, and then asserts positivity, residual < 10⁻⁸ and reports
   the number of sign changes (uniqueness). It raises instead of returning a stalled iterate.

---

## 6. The surface coefficient: why it must be a constant mean, and which one

### 6.1 The closed form admits only a constant `h`

Eqs. (73)–(83) descend from the classical third-kind (Robin) solution of Eq. (66). Its kernel

```
exp(h x/k_S + h² α_S t/k_S²) · erfc(x/(2√(α_S t)) + h√(α_S t)/k_S)
```

satisfies the heat equation **only if `h` is independent of `t`**: the `h²α_S t/k_S²` term is
what makes the exponential-times-erfc group a solution, and it is derived by Laplace transform
with `h` treated as a constant. Consequently

> **`h_g(t) = 6350 t^(−0.21)` cannot be substituted pointwise into the analytical solution.**
> The formulation admits a single *mean* value `h̄`. Inserting the instantaneous value is a
> frozen-coefficient approximation with no derivation behind it, and it is quantitatively bad
> (§6.4).

This is the reason the analytical `h` came out far too large at early times in the first pass:
`h_g(t)` reaches 31 000 W m⁻² K⁻¹ at `t = 10⁻⁵ s`, which is meaningless for a solution whose
whole history is governed by one coefficient.

### 6.2 What the reported law implies for `h̄`

The elementary time means of the reported law, `h̄(t_f) = (1/t_f)∫₀^{t_f} C t^(−p) dt =
C t_f^(−p)/(1−p)` (finite for `p < 1` although `h_g(0) = ∞`):

| `t_f` [s] | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| `h̄` [W m⁻² K⁻¹] | 4956 | 4285 | 3535 | 3056 | 2642 |
| `h_g(t_f)` [W m⁻² K⁻¹] | 3915 | 3385 | 2792 | 2414 | 2087 |

The averaging window is not arbitrary: the surface flux is `h_g (T_w − T_∞)`, and `T_w` is
highest at the beginning, so the *energy-relevant* mean is weighted towards early times and
lies at the upper end of that range.

### 6.3 The calibrated `h̄`

`h̄` is the one quantity the closed form genuinely needs and cannot read off the reported
time-dependent law. Fitted by least squares to all thermocouples simultaneously
(`fit_h_bar`, §8):

```
h̄* = 4056 W m⁻² K⁻¹        (probes at 15, 30, 50, 90 mm; RMSE 10.12 °C over 270 points)
h̄  = 4264 W m⁻² K⁻¹        (all six probes;               RMSE 18.32 °C over 407 points)
```

The two differ by 5.1 %, so the choice of probe set does not drive the answer. `h̄* = 4056`
is **exactly the time mean of the reported law over 0–26 s** — the interval over which the
front crosses the first ~30 mm and most of the latent heat is removed. The reported law and
the fitted constant are therefore consistent with each other; what the analytical solution
cannot use is the pointwise value, and what it does use is an early-weighted mean of it.

The 0–200 s mean (2642 W m⁻² K⁻¹) is 1.54× smaller and fits badly (RMSE 75.9 °C): averaging
over the whole record under-weights the period that actually matters.

### 6.4 Comparison of the three treatments

| treatment of the surface coefficient | global RMSE [°C] | bias [°C] | R² | `s(200 s)` [mm] |
|---|---|---|---|---|
| **constant mean `h̄* = 4056` (adopted)** | **19.78** | **+7.68** | **0.978** | 137.6 |
| constant mean = 0–200 s average, 2642 | 75.89 | +69.04 | 0.674 | 121.1 |
| frozen-coefficient `h_g(t)` (not admissible) | 79.66 | +71.68 | 0.641 | 119.0 |

The frozen-coefficient treatment is the worst of the three, which is the expected outcome for
an approximation that has no derivation: it uses a coefficient that is too large where the
solution is insensitive (`t → 0`) and too small over the bulk of the record.

### 6.5 Two further approximations, retained as diagnostics

Even with a constant `h̄` the published closure has one residual weakness, independent of the
boundary condition:

**Method A — Eq. (83) as published.** For each `s`, solve Eq. (83) together with
`t = s²/(4α_S φ²)` (Eq. 22). Exact reproduction of the published algebra (residual
< 4 × 10⁻¹⁵ at every `s`).

**Method B — reduced interface equation** *(adopted)*:

```
ρ_S L ds/dt = q_S(s,t) − q_L(s,t),        q_S, q_L = the closed-form analytical fluxes
```

Only the *velocity relation* is relaxed: the true `ds/dt` replaces Eq. (83)'s parabolic
surrogate `2φ²α_S/s`, which is valid only for constant φ while φ actually runs from 0.22 to
0.58 over a 100 mm layer. Arbitrated by an independent 1-D enthalpy finite-difference solution
of the same constant-`h` problem (`h = 4000 W m⁻² K⁻¹`):

| | max. relative deviation of `s(t)` from FD |
|---|---|
| Method A (Eq. 83 as published, `Ω = 0`) | **37 %** |
| Method A with `Ω` (Eq. 83 exactly as printed) | far worse — a 100 mm layer forms in **20.1 s**, against 104.1 s for method A with `Ω = 0` and ≈ 120 s for FD |
| **Method B (reduced interface ODE, adopted)** | **7.7 %** |

So Eq. (83)'s parabolic-velocity assumption is the dominant remaining approximation of the
published closure, and the `Ω` first-order correction of Eq. (29d) makes it worse rather than
better.

### 6.6 The singularity at `t = 0`

With a constant `h̄` there is no singularity in the boundary condition. There is still a finite
**incubation time** `t* = 2.9 × 10⁻² s`: for the conduction-like melt closures
`q_L ≈ k_L(T_P − T_F)/√(π α_L t)` diverges as `t^(−1/2)` while `q_S → h̄ (T_F − T_∞)` stays
bounded, so the superheated melt initially delivers more heat than the chill removes and no
solid forms. Both fluxes are integrable, so
`ρ_S L s(t) = ∫_{t*}^{t} (q_S − q_L) dt'` is well defined; it is solved by Picard iteration on
a logarithmic grid (`initial_condition`). Moving the start time `t₀` over nearly two decades
changes `s(200 s)` by **1.1 × 10⁻⁶ relative** — the march forgets its start.

### 6.7 Convergence

Piecewise-constant sub-stepping of method B (deviation of `s(200 s)` from a `rtol = 10⁻¹²`
reference):

```
Δt = 5.0 s → 2.1e-07 ;  2.0 s → 6.5e-08 ;  1.0 s → 2.9e-08 ;  0.5 s → 1.2e-08 ;  0.25 s → 5.1e-09
```

FD reference grid convergence, `s(200 s)`:

```
nx = 200 → 125.711 mm ;  400 → 126.201 ;  800 → 126.483 ;  1600 → 126.609 mm
```

(monotone, ~0.1 % at `nx = 800`, the setting used). The FD solver conserves energy: extracted
`2.012 × 10⁸` vs stored `2.013 × 10⁸ J m⁻²` (0.0 %).

---

## 7. Recovery of the experimental data

### 7.1 Pixel-to-axis calibration

Source raster 6432 × 4923 px. Plot frame centre lines: `L = 1149`, `R = 5538`, `TOP = 570`,
`BOT = 4103` px.

* **Time.** The major ticks drawn on the top spine are at `x = 1149, 2246, 3344, 4441, 5538 px`
  and are exactly equispaced ⇒ `t = 0` at 1149 px, `t = 200 s` at 5538 px, **21.945 px/s**.
* **Temperature.** The seven y tick-label glyph boxes have centroids at
  `y = 833, 1376, 1920, 2463, 3007, 3550, 4094 px` for 700 … 100 °C. The spacing is
  **543.5 px per 100 °C** to within ±0.5 px ⇒ `T = 700 − (y − 833)/5.435`.
* **Independent check.** The red `T_F` reference line occupies rows 1043–1075 px (centre 1059),
  which this calibration reads as **658.4 °C** instead of 660.0. The 1.6 °C (8.6 px) discrepancy
  is reported as the systematic calibration uncertainty. **No fudge factor was applied** — the
  tick-label calibration has seven anchors and is kept as-is.

### 7.2 Marker extraction

Marker colours were read from the legend swatches: 5 mm black squares, 10 mm (240, 64, 64),
15 mm (26, 111, 222), 30 mm (177, 119, 222), 50 mm (204, 153, 0), 90 mm (1, 203, 205). The red
`T_F` line is (254, 0, 0) and is separated from the red markers by colour alone.

Pipeline: colour threshold → **vertical-only** binary closing (45 × 3 px) to bridge the
horizontal occluders (the black simulation curves ≈ 15 px and the red line ≈ 33 px) without
merging horizontally adjacent markers → square opening (11 px, 21 px for the black squares,
which must be separated from the ≈ 15 px black simulation curves) → connected components →
area centroid of every blob within 1.9× the median marker area; larger merged runs are split
into column strips.

**Marker-shape centroid correction.** Triangles are not centro-symmetric: the area centroid is
displaced from the bounding-box centre by `w/6 = 7.7 px`. Measured on the legend swatches:

```
10 mm circle    dx = +0.00  dy = +0.00 px
15 mm ▲         dx = +0.15  dy = +7.67 px   →  reads 1.41 °C low   (corrected)
30 mm ◆         dx = −0.45  dy = +0.00 px
50 mm ◀         dx = +7.66  dy = +0.17 px   →  reads 0.35 s late   (corrected)
90 mm ▶         dx = −7.67  dy = +0.15 px   →  reads 0.35 s early  (corrected)
```

Ignoring this would have produced a spurious 0.7 s offset between the 50 mm and 90 mm curves.

**Stray rejection.** Cooling curves are monotone, so the longest non-increasing subsequence
(within a +4 °C tolerance) is retained. For the 5 mm series this leaves one ambiguity that
monotonicity cannot resolve. Three black blobs at `t < 3 s, T > 660 °C` were checked
individually against the raster and found to be fragments of the black *numerical-simulation*
curves inside the crowded 660–695 °C band: two are removed by the monotonicity filter and the
third by one documented manual rule (`DIGITISER_MANUAL_DROP` in the source). With them gone the
5 mm series starts at its true first marker, (2.97 s, 577.2 °C).

**Missing data are left missing.** Samples hidden behind the opaque legend box (5 mm,
`t ≈ 48–63 s`), the title box and the `h_g` annotation are recorded as empty fields; nothing is
interpolated or invented.

### 7.3 Result and uncertainty

66–71 markers per probe; all six series share one acquisition instant every **2.79 s**
(71 grid nodes over 0–199 s), which is what makes the wide CSV layout meaningful.

Uncertainty budget:

| source | magnitude |
|---|---|
| marker-centroid localisation (blobs are ≈ 40–60 px) | **±0.1 – 0.2 °C** random (estimated from third differences of the uniformly sampled stretches at `t > 60 s`), ±0.05 s |
| axis calibration (tick-label fit residual ≤ 0.5 px) | ±0.1 °C |
| absolute calibration (red `T_F` line reads 658.4 °C) | **±1.6 °C systematic** |
| occlusion bias in the crowded `t < 20 s`, `T > 650 °C` region | up to ±4 °C on individual samples |
| marker-shape centroid (corrected) | residual ≤ 0.1 s / 0.2 °C |

Output: **`experimental_cooling_curves_digitized.csv`**
(`time_s, T_5mm_C, T_10mm_C, T_15mm_C, T_30mm_C, T_50mm_C, T_90mm_C`).

---


## 8. Calibration: what is fitted, what is prescribed

### 8.1 Separation of roles

| quantity | status | value | source |
|---|---|---|---|
| `T_F` | prescribed | 660 °C | experiment |
| `T_P` | prescribed | 690.9 °C | read from the data (warmest 50/90 mm reading at `t < 6 s`); never adjusted |
| `T_∞` | prescribed | 25 °C | Table 1 |
| properties | prescribed | Table 1 | paper |
| thermocouple positions | prescribed | 5, 10, 15, 30, 50, 90 mm | experiment (nominal) |
| **`h̄`** | **fitted** (1 parameter) | **4056 W m⁻² K⁻¹** | least squares, all reliable probes at once |
| `h_i` | prescribed | 1000 W m⁻² K⁻¹ | Table 1 mid-range; shown to be unidentifiable |
| `ψ_L` closure | model structure | `virtual-origin` | choice documented in §4 |

Exactly **one** parameter is fitted. It is the parameter the closed form requires and the
reported `h_g(t)` cannot supply (§6.1).

### 8.2 `h̄` is identifiable

| case | `s(200 s)` [mm] | Δ`s` [%] | `t_arr`(90 mm) [s] | global RMSE [°C] |
|---|---|---|---|---|
| `h̄* = 4056` (reference) | 137.58 | 0.00 | 103.40 | 19.78 |
| `h̄ − 10 %` | 133.88 | −2.69 | 109.01 | 29.79 |
| `h̄ + 10 %` | 140.73 | +2.29 | 98.78 | 19.50 |
| `h̄ = 2642` (0–200 s mean of `h_g`) | 121.14 | −11.95 | 130.17 | 75.89 |

A ±10 % change in `h̄` costs 10 °C of RMSE and 2.5 % of `s(200 s)`: the optimum is interior and
well defined, unlike `h_i`.

### 8.3 `h_i` is *not* identifiable

A single `h_i` was fitted to all thermocouples simultaneously (never one per probe), bounded to
the Table 1 range 500–1800 W m⁻² K⁻¹:

| liquid closure | bounded optimum | global RMSE | RMSE span over 500–1800 |
|---|---|---|---|
| `virtual-origin` (default) | 1800 (at the bound) | 19.72 °C | 0.10 °C |
| `interface-film` | 1800 (at the bound) | 18.42 °C | 0.45 °C |

Sensitivity across four decades:

| `h_i` [W m⁻² K⁻¹] | 500 | 1800 | 5000 | → ∞ (conduction) |
|---|---|---|---|---|
| `s(200 s)` [mm] | 137.44 | 137.75 | 138.07 | 138.53 |
| global RMSE [°C] | 19.82 | 19.72 | 19.55 | **19.00** |

* No interior optimum: the objective is monotone towards the conduction limit.
* The whole Table 1 range moves `s(200 s)` by **0.22 %** and the RMSE by **0.10 °C**.
* Switching melt convection off entirely (`h_i → ∞`) gives the *best* fit of the family.
* The joint `(h̄, h_i)` fit buys 0.03 °C of RMSE over fitting `h̄` alone.

Formal confidence intervals are computed by the code but should not be believed: the residuals
are strongly serially correlated, so the naive profile-`F` interval is meaningless; with an
effective sample size of 30 the 95 % interval covers the entire search range. The honest
statement is that this experiment **does not resolve `h_i`**, and the Table 1 mid-range value is
prescribed rather than fitted.

Other sensitivities for scale: the undefined `ψ_L` closure moves `s(200 s)` by 3–4 % and the
melt superheat `T_P` by 1.5 % per 10 K — both larger than the entire `h_i` range.

---

## 9. Validation

### 9.1 Predicted chronology

`t* = 2.9 × 10⁻² s` (incubation), `s(200 s) = 137.6 mm`. Predicted front arrival:

```
5 mm → 2.97 s ;  10 mm → 6.16 s ;  15 mm → 9.71 s ;  30 mm → 22.50 s ;  50 mm → 44.29 s ;  90 mm → 103.40 s
```

strictly ordered, as required. Each predicted curve consists of (1) liquid cooling from `T_P`
(Eq. 76); (2) arrival of the front at `t_arr`, where `T = T_F` exactly and continuously;
(3) for a pure metal the arrest has zero duration — the front is sharp, so the kink *is* the
phase change; (4) post-solidification cooling (Eq. 73), evaluated only for `x < s(t)`. No
phase-specific expression is evaluated outside its own domain, and continuity at the front is
exact to 6 × 10⁻¹³ K.

### 9.2 Per-probe metrics (`h̄* = 4056`, `h_i = 1000`, `virtual-origin`, 0–200 s)

| probe | n | RMSE [°C] | MAE [°C] | max\|err\| [°C] | bias [°C] | R² | t(660 °C) exp / model [s] | t(650 °C) exp / model [s] |
|---|---|---|---|---|---|---|---|---|
| 5 mm  | 66 | **38.80** | 28.19 | 102.96 | +28.19 | 0.731 | – / 2.97 | – / 3.64 |
| 10 mm | 71 | **21.30** | 14.02 | 63.15 | +12.04 | 0.940 | 3.03 / 6.15 | 3.52 / 7.02 |
| 15 mm | 70 | 14.15 | 10.67 | 43.81 | +2.49 | 0.975 | 4.87 / 9.70 | 5.95 / 10.78 |
| 30 mm | 68 | 9.37 | 7.34 | 25.61 | +3.16 | 0.990 | 16.17 / 22.48 | 18.92 / 24.28 |
| 50 mm | 65 | **3.82** | 3.21 | 7.26 | −1.53 | 0.998 | 41.04 / 44.21 | 45.82 / 47.19 |
| 90 mm | 67 | 10.02 | 9.09 | 16.74 | +1.82 | 0.947 | 93.93 / 103.39 | 114.06 / 109.24 |
| **all six** | **407** | **19.78** | 12.09 | 102.96 | +7.68 | **0.978** | | |
| **15–90 mm only** | **270** | **10.12** | | | | | | |

For the four outer probes the agreement is 3.8–14.2 °C RMSE with R² = 0.95–0.998 and a bias
within ±3 °C — at the level of a full numerical solution. The two inner probes are the outliers,
and §9.4 shows why.

### 9.3 Model-to-model comparison (global, all 407 points)

| variant | RMSE [°C] | MAE [°C] | max\|err\| [°C] | bias [°C] | R² |
|---|---|---|---|---|---|
| **corrected analytical, `h̄* = 4056` (reported)** | **19.78** | 12.09 | 102.96 | +7.68 | **0.978** |
| corrected analytical, `h_i = 500` (Table 1) | 19.82 | 12.10 | 103.04 | +7.84 | 0.978 |
| corrected analytical, `h_i = 1800` (Table 1) | 19.72 | 12.07 | 102.83 | +7.48 | 0.978 |
| conduction-only limit (no melt convection) | 19.00 | 11.75 | 99.92 | +6.11 | 0.980 |
| liquid closure `interface-film` | 18.61 | 13.16 | 93.25 | +2.33 | 0.980 |
| liquid closure `as-coded` | 19.24 | 13.98 | 93.04 | +1.77 | 0.979 |
| **independent 1-D FD reference, same `h̄`** | **17.34** | 11.30 | 93.18 | +3.72 | 0.983 |
| analytical, `h̄` = 0–200 s mean of `h_g` (2642) | 75.89 | 69.04 | 144.85 | +69.04 | 0.674 |
| analytical, frozen-coefficient `h_g(t)` | 79.66 | 71.71 | 124.47 | +71.68 | 0.641 |

**The corrected analytical solution is within 2.4 °C RMSE of a full numerical solution of the
same problem** (19.78 vs 17.34 °C). The residual gap is the pseudo-similarity defect of the
closed form, not an implementation error.

**Does melt convection materially improve agreement?** No. The conduction-only limit
(`h_i → ∞`) gives 19.00 °C against 19.78 °C for `h_i = 1000`, and no admissible `h_i` beats it.
Every difference inside the melt-convection family is ≤ 1 °C, an order of magnitude below the
5 mm/10 mm anomaly and comparable with the digitisation uncertainty.

Early window `t ≤ 30 s`, where the near-chill anomaly dominates:

| variant | RMSE [°C] | bias [°C] | R² |
|---|---|---|---|
| corrected analytical, `h̄* = 4056` | 43.82 | +32.41 | 0.767 |
| conduction-only limit | 42.05 | +30.38 | 0.785 |
| 1-D FD reference | 37.86 | +27.32 | 0.826 |

### 9.4 The 5 mm and 10 mm records are anomalous

Three independent lines of evidence, all pointing the same way.

**(i) Effective-position fit.** Position each record would need in order to reproduce itself,
with `h̄`, `h_i`, `T_P` and the properties held fixed (`effective_positions`; a diagnostic,
never applied to the delivered predictions):

| probe | nominal [mm] | effective [mm] | shift [mm] | RMSE nominal [°C] | RMSE effective [°C] |
|---|---|---|---|---|---|
| 5 mm | 5.0 | **0.20** *(at the search bound)* | −4.80 | 38.80 | 19.00 |
| 10 mm | 10.0 | **7.05** | −2.95 | 21.30 | 14.44 |
| 15 mm | 15.0 | 14.07 | −0.93 | 14.15 | 13.35 |
| 30 mm | 30.0 | 29.08 | −0.92 | 9.37 | 8.45 |
| 50 mm | 50.0 | 50.41 | +0.41 | 3.82 | 3.55 |
| 90 mm | 90.0 | 92.09 | +2.09 | 10.02 | 9.00 |

The four outer probes sit within 0.4–2.1 mm of their nominal positions — within normal
insertion tolerance. The 10 mm record behaves like a probe at **≈ 7 mm**. The 5 mm record is
driven to the chill face itself and *still* does not fit: comparing it directly with the model's
wall temperature gives RMSE 18.6 °C with a mean bias of only +0.7 °C, i.e. **the "5 mm"
thermocouple reads what the model predicts at `x ≈ 0`**. A position error alone cannot account
for it — no location in a 1-D slab with `k_S = 213 W m⁻¹ K⁻¹` is cold enough.

A joint fit of `(h̄, x_5mm, x_10mm)` gives `h̄ = 4041 W m⁻² K⁻¹`, `x_5mm → 0.20 mm` (at the
bound), `x_10mm → 6.92 mm`, and reduces the global RMSE from 19.78 to **12.75 °C** — note that
`h̄` barely moves (0.4 %), confirming that the anomaly is local to the two probes and does not
contaminate the calibration.

**(ii) Energy balance vs local Fourier gradient — model-free.** From the digitised profile at
`t = 200 s`:

```
interval     dT/dx [K/m]   |q| = k_S dT/dx [W/m²]
 5–10 mm        5832           1.242e6
10–15 mm        4756           1.013e6
15–30 mm        3403           7.25e5
30–50 mm        3554           7.57e5
50–90 mm        3184           6.78e5
```

The same data give an enthalpy deficit of `1.456 × 10⁸ J/m²` at 100 s and `2.161 × 10⁸ J/m²`
at 200 s, hence a wall flux `dE/dt = 7.05 × 10⁵ W/m²` — which **must** be the largest flux in
the slab. The 5–10 mm interval reports `1.24 × 10⁶ W/m²`, a factor **1.76** larger, which is
impossible in one dimension. The outer intervals are consistent to ~20 %. The implied effective
conductivity over 0–15 mm is 121 W m⁻¹ K⁻¹, about half the Table 1 value.

**(iii) Melt cooling far ahead of the front.** At 90 mm the measured melt has lost 10 K by
`t = 11 s`, when the thermal penetration depth from the chill is only 20 mm and the front is at
17 mm. Chill-driven 1-D conduction cannot do this; the melt is losing heat through a route the
model does not contain (lateral mould walls, free surface, natural-convection stirring). This
is a second-order effect here (the 90 mm probe still fits to 10 °C RMSE) but it is real.

Candidate causes for (i)+(ii), in order of plausibility: **thermocouple displacement towards the
chill during pouring** (the user's hypothesis, and the reading that explains the 10 mm probe
completely); an extra thermal resistance in the chilled zone (porosity, fine chill structure)
that the constant-property model cannot represent; sheath conduction along the probes nearest
the cold wall. The evidence cannot discriminate between them, and **no correction has been
applied**: the delivered predictions use the nominal positions, and the two records are scored
as they stand.

### 9.5 Automated verification (28/28 pass)

```
Eq.(75c) reduced == literal                     max rel dev 3.6e-11
Eq.(18b) psi == 1 + zeta (Eq.19b)               max dev 1.1e-16
no overflow for Bi up to 1e5, phi up to 40      (legacy literal form returns NaN)
Eq.(83) dimensionless / scale invariant         phi identical to 1e-9 under lambda-scaling
Eqs.(68),(71) T -> T_P                          dev 0.0 K
Eq.(69) Robin at x = 0 (sign corrected)         max rel dev 3.3e-06
Eq.(70) T_S(s,t) = T_F                          max dev 0.0 K
Eq.(79) film law at x = s+ (all closures)       max rel dev 5.0e-15
Eq.(75) = d/dx Eq.(73) at x = s-                max rel dev 5.9e-06
Eq.(77) = d/dx Eq.(76) at x = s+                max rel dev 9.9e-06
T_L(s,t) = T_F for 'virtual-origin'             jump 5.7e-13 K  (interface-film 34.7 K, as-coded 37.3 K)
Eq.(66) PDE residual of the solid field         max normalised residual 0.469  (pseudo-similarity defect)
Eq.(72) Stefan residual along the march         max |res|/q_S = 3.0e-07
s(t) positive and strictly increasing           s(200 s) = 137.6 mm
T_inf < T(x,t) <= T_P and monotone cooling      min 238.2 °C, max 690.9 °C
front reaches nearer probes first               2.97, 6.16, 9.71, 22.50, 44.29, 103.40 s
s(200 s) independent of the start time t0       t* = 2.9e-02 s; spread 1.1e-06 over t0 = 0.04 … 2.1 s
constant-h: method A reproduces Eq.(83) exactly max |residual| = 3.9e-15
constant-h: method B beats method A vs FD       method B 7.7 %, method A (Eq. 83) 37.4 %
h_i -> inf gives Neumann melt flux              rel dev 1.1e-2 … 1.1e-5 for h_i = 1e5…1e8  (O(1/h_i))
h_i -> 0 gives zero melt flux ('interface-film') q_L = 3.7e-05 W/m²
Eq.(82) melt drive stays within [T_F, T_P]      interface-film 0.00 K, as-coded 0.00 K, virtual-origin +648 K (FAILS)
q_L increases with h_i ('interface-film')       film 3.0 → 58.8 kW/m² (rising); virtual-origin 160.2 → 140.3 kW/m² (INVERTED)
piecewise-constant sub-step convergence         dt = 5 → 2.1e-7 … 0.25 → 5.1e-9
FD reference grid convergence                   s(200 s) = 125.711 … 126.609 mm (nx = 200…1600)
analytical vs FD reference                      max dev: s 9.0 %, T 22.0 K
FD reference energy balance                     extracted 2.012e8 vs stored 2.013e8 J/m² (0.0 %)
Table 2 reproduced by the legacy path           max dev 0.012 K on h_i = 1800…800; all fsolve ier != 1
```

Two are *reported* rather than asserted, because they document defects of the published
formulation rather than of this implementation: the PDE residual of the pseudo-similarity field
(0.469 normalised), and the failure of Eq. (82) to respect `T_L(s⁺) ≤ T_P` under the
continuity-preserving closure (+648 K excess).

---

## 10. Limitations

1. **The fields are pseudo-similarity, not exact.** They satisfy the initial condition, the far
   field, the Robin condition at `x = 0`, `T(s) = T_F` and the Stefan balance exactly, but the
   heat equation only approximately once φ and the Biot numbers drift. Measured normalised PDE
   residual 0.47. This costs ~2.4 °C of RMSE relative to a full numerical solution.
2. **Eq. (83) is not the Stefan condition.** It equates the interfacial flux difference with the
   *parabolic* velocity `2φ²α_S/s`; at constant `h = 4000 W m⁻² K⁻¹` that over-predicts `s(t)`
   by up to 37 % against an independent FD solution, and the `Ω` correction of Eq. (29d) makes
   it substantially worse. The delivered predictions use the reduced interface ODE instead
   (7.7 %).
3. **`h` must be a single constant.** The closed form cannot accept `h_g(t)`; the reported law
   enters only through the calibrated mean `h̄`. Any process in which the surface coefficient
   changes by more than ~±10 % over the *energy-relevant* window is outside the model's reach.
4. **`ψ_L` is undefined in the manuscript**, and no admissible definition satisfies interface
   continuity, monotone `q_L(h_i)`, the maximum principle for Eq. (82) and the `h_i → ∞` Neumann
   limit simultaneously. The closure is over-determined; results depend on the choice at the
   3–4 % level in `s(t)`.
5. **Semi-infinite, 1-D, adiabatic sides.** The experiment demonstrably loses heat by other
   routes (§9.4 iii).
6. **Constant properties, no density change.** `ρ_S ≠ ρ_L` (7 % shrinkage) is ignored in the
   Stefan balance, as in the paper; solid `c_P` is taken at its near-melting value
   (1181 J kg⁻¹ K⁻¹), ~20 % above the 200–600 °C mean.
7. **Sharp front, no undercooling, no mush**, no thermocouple time constant or bead size.
8. **The validation data are digitised from a raster**, with ±1.6 °C systematic calibration
   uncertainty and missing samples where markers are occluded.
9. **Melt convection is a single constant `h_i`**, with no mechanism for its variation as the
   melt cools.
10. **The 5 mm and 10 mm records are not usable as a test of the model** at their nominal
    positions (§9.4). They are reported and scored, not corrected or discarded from the metrics.

---

## 11. Conclusion

**Is the corrected melt-convection formulation supported by the experimental cooling curves?**

**The formulation is supported. The melt-convection term specifically is not resolved by this
experiment.**

*What is supported.* With the one constant the closed form actually requires — a mean surface
coefficient `h̄* = 4056 W m⁻² K⁻¹`, which is the time average of the reported
`h_g(t) = 6350 t^(−0.21)` over the first 26 s, i.e. over the interval in which most of the
latent heat is removed — the corrected analytical solution reproduces the experiment with

* global RMSE **19.78 °C**, R² **0.978** over 407 points and six thermocouples;
* RMSE **10.12 °C** over the four probes at 15–90 mm, with per-probe R² of 0.95–0.998, MAE of
  3.2–10.7 °C and biases within ±3 °C;
* the correct chronology (front at 5 → 90 mm in 3.0 → 103 s, strictly ordered);
* agreement with an **independent 1-D enthalpy finite-difference solution of the same problem**
  to within 2.4 °C of RMSE — i.e. the closed form is performing at the level of a full numerical
  solution of the same stated physics.

All 28 automated checks pass: the Stefan residual along the march is ≤ 3 × 10⁻⁷ relative, the
interface is positive and strictly monotone, temperatures are admissible and monotone, the
solution is dimensionally consistent under scaling, and it converges under sub-step and grid
refinement.

*What is not resolved.* `h_i` has no identifiable optimum. Across four decades it moves
`s(200 s)` by 0.7 % and the global RMSE by 0.8 °C; the objective is monotone towards the
conduction limit, so switching melt convection off entirely fits marginally *better* than any
admissible `h_i`. The melt term contributes 10–16 % of the interfacial energy balance, but the
part of it that depends on `h_i` is a few per cent of that — below the resolution of this
experiment. The melt-convection extension is therefore **neither confirmed nor refuted** here.

*What is refuted.* Three specific claims do not survive the audit:

1. **Published Table 2 is not a solution of Eq. (83).** Its values are stalled `fsolve` iterates
   of a residual that has no root, produced by an inverted Stefan number and a dimensionally
   inconsistent melt term.
2. **Eq. (83) is not an exact statement of the Stefan condition** for this solution family: it
   carries a 20–50 % velocity error at realistic Biot numbers, and the `Ω` correction meant to
   repair it makes the error larger.
3. **`h_g(t)` cannot be used pointwise.** Doing so degrades the global RMSE from 19.8 °C to
   79.7 °C with a +72 °C bias — worse than either constant-mean choice. The published solution
   is a constant-`h` solution and must be used as one.

*What the data show independently of the model.* The 5 mm and 10 mm records are inconsistent
with one-dimensional Fourier conduction at their nominal positions: their mutual gradient
implies a heat flux 1.76× larger than the data's own enthalpy balance permits, the 10 mm record
behaves like a probe at ≈ 7 mm, and the 5 mm record coincides with the model's *chill-face*
temperature (bias +0.7 °C). Displacement of the two innermost thermocouples towards the chill
during pouring accounts for the 10 mm record completely and for most of the 5 mm one. No
correction has been applied; both records are scored as measured, and they account for
essentially all of the difference between the 10.1 °C RMSE of the outer four probes and the
19.8 °C of the full set.

*Recommendation.* (a) Publish a definition of `ψ_L` chosen so that `q_L` increases with `h_i`
and Eq. (82) respects `T_L(s⁺) ≤ T_P`. (b) Replace Eq. (83)'s parabolic velocity by the reduced
interface equation — one ODE integration, and it removes the dominant error. (c) State that the
solution requires a constant mean `h̄`, and give the rule for obtaining it from a measured
`h_g(t)` (the early-weighted time average). (d) To test melt convection, use an experiment in
which the melt-side resistance controls: high superheat, strongly stirred melt, *low* chill
coefficient. (e) Re-measure the near-chill thermocouple positions, or instrument the chill face
directly.

---

## 12. Interface position vs time — and a contradiction between the two experimental figures

### 12.1 The second experimental source

`Figure_Position_versus_time_experimental_against_numerical_simulation.jpg` gives the interface
position directly, as seven black markers plus the printed power-law fit
`P = 6.14 t^0.51` (mm, s). The markers were digitised with the same pipeline as the cooling
curves (frame `L = 1149`, `R = 5538`, `TOP = 570`, `BOT = 4103` px; x tick labels 0…220 s at
1246.5…5537.0 px = 19.502 px/s; y tick labels 0…140 mm at 4094…797 px = 23.55 px/mm; the thick
black fit and the thin red numerical curve are excluded, the fit by a 31 × 31 morphological
opening that the ~77 px squares survive):

| `t` [s] | 0.33 | 2.31 | 5.90 | 17.19 | 43.02 | 109.86 | 210.05 |
|---|---|---|---|---|---|---|---|
| `P` digitised [mm] | 4.63 | 10.71 | 15.06 | 25.84 | 41.96 | 70.29 | 94.65 |
| `6.14 t^0.51` [mm] | 3.51 | 9.41 | 15.18 | 26.19 | 41.82 | 67.45 | 93.88 |

Agreement with the published fit is 0.3–1.3 mm except at the first point, so the digitisation is
sound. Output: **`experimental_position_vs_time_digitized.csv`**.

### 12.2 The two sources disagree beyond ~25 mm

The cooling curves give a second, independent estimate of `s(t)`: the time at which each probe
falls through 650 °C (10 K below `T_F`, chosen because the melt plateau itself sits a few K
above `T_F`), paired with that probe's position.

| nominal probe position [mm] | 10 | 15 | 30 | 50 | 90 |
|---|---|---|---|---|---|
| `t` at 650 °C, cooling curves [s] | 3.52 | 5.95 | 18.92 | 45.82 | 114.06 |
| `P` at that time, position figure [mm] | 10.7 | 15.1 | 25.8 | 42.0 | 70.3 |
| ratio `P` / nominal | 1.07 | 1.00 | 0.86 | 0.84 | 0.78 |

The two agree to within a millimetre out to 15 mm and then diverge monotonically: at the instant
the 90 mm thermocouple passes 650 °C, the position figure puts the front at 70 mm. Equivalently,
the position figure has the front reaching 90 mm at `t ≈ 193 s` while the cooling curves have it
there at `t ≈ 114 s` — a factor 1.7 in time.

### 12.3 Which source does the model support?

Calibrating the single constant `h̄` against each source separately:

| calibration target | `h̄*` [W m⁻² K⁻¹] | residual |
|---|---|---|
| cooling curves (probes at 15–90 mm) | **4056** | 10.12 °C |
| interface kinetics `P(t)` alone | **1820** | 10.13 mm |

a factor **2.2** apart. The two analytical curves are plotted together in
`Figure_position_vs_time_analytical_vs_experimental.png`; neither reproduces both sources, and
neither reproduces the very first marker (4.63 mm at 0.33 s, which would need
`h ≈ 2 × 10⁴ W m⁻² K⁻¹`).

The decisive test is to re-fit the **cooling curves** using the positions implied by the position
figure (`4.63, 10.71, 15.06, 25.84, 41.96, 70.29 mm` instead of the nominal ones), pairing the
two sources by time:

| positions used | `h̄*` | global RMSE | per-probe RMSE (5 / 10 / 15 / 30 / 50 / 90 mm) [°C] |
|---|---|---|---|
| **nominal** | 4264 | **18.31 °C** | 33.2 / 18.5 / 15.8 / 10.4 / **8.7** / **12.2** |
| position figure | 3948 | 31.58 °C | 40.5 / 27.2 / 15.5 / 12.9 / **28.0** / **49.8** |

Substituting the position-figure positions makes the fit **1.7× worse overall and 4× worse at the
two outer probes**. The cooling curves therefore support the *nominal* thermocouple positions at
15–90 mm, and are incompatible with the positions the position figure implies.

### 12.4 What this means

* The two published experimental figures cannot both describe the same casting with the same
  position scale. The discrepancy is a **position** discrepancy that grows with distance from
  the chill (0 % at 15 mm, −22 % at 90 mm), not a time offset.
* Solidification plus thermal contraction of an Al bar accounts for at most 3–4 % of linear
  shortening, so "position measured in the as-cast ingot" does not explain a 22 % difference.
* Consequently the interface-kinetics calibration `h̄ = 1820 W m⁻² K⁻¹` and the cooling-curve
  calibration `h̄ = 4056 W m⁻² K⁻¹` are **both defensible against their own source** and cannot
  be reconciled by any single constant `h̄`, nor by the reported `h_g(t)`.
* The reported predictions of this study use the cooling-curve calibration, because that is the
  source with 407 independent observations against the position figure's 7, and because it is
  the source the task specified. The alternative is one command away
  (`--h-bar 1820`), and the position figure is delivered digitised so the choice is auditable.
* This is a defect of the experimental record, not of the analytical formulation. Resolving it
  requires the raw positions and raw thermocouple traces, not more modelling.

---

## Appendix A — Delivered files

| file | content |
|---|---|
| `analytical_model_2024_Al_melt_convection_corrected.py` | corrected implementation, verification suite, digitiser, calibration, output writers, `main()` |
| `experimental_cooling_curves_digitized.csv` | 71 rows × 6 probes, digitised markers, missing values left empty |
| `model_predictions.csv` | `t`, `s`, `v`, `φ`, `h_g`, `Bi_env`, `Bi_i`, `q_solid`, `q_melt`, `T` and phase flag for each probe |
| `melt_convection_validation_metrics.csv` | 98 rows: 7 variants × (6 probes + GLOBAL) × 2 time windows |
| `Figure_cooling_curve_analytical_vs_experimental.png` | comparison figure: digitised experiment (coloured markers) vs corrected analytical solution (continuous lines), one colour per thermocouple, `T_F`, calibrated `h̄`, prescribed `h_i` |
| `Figure_cooling_curve_analytical_vs_experimental_residuals.png` | same comparison with the independent 1-D FD reference overlaid (dotted) and a model $-$ experiment residual panel |
| `experimental_position_vs_time_digitized.csv` | 7 digitised interface-position markers + the published power-law fit |
| `Figure_position_vs_time_analytical_vs_experimental.png` | interface position vs time: analytical against **both** experimental sources |
| `ANALYTICAL_MODEL_CORRECTION_REPORT.md` | this report |
| `analytical_model_2024_Al_melt_convection_dev.py` | **unmodified** original |

Reproduce with:

```
python analytical_model_2024_Al_melt_convection_corrected.py            # full pipeline
python analytical_model_2024_Al_melt_convection_corrected.py --verify   # 28 checks only
python analytical_model_2024_Al_melt_convection_corrected.py --digitize # rebuild the CSV
python analytical_model_2024_Al_melt_convection_corrected.py --h-bar 4056 --h-i 1000
python analytical_model_2024_Al_melt_convection_corrected.py --time-dependent-h   # diagnostic
python analytical_model_2024_Al_melt_convection_corrected.py --liquid-model interface-film
```

## Appendix B — Paper equation → function map

| paper | function in `analytical_model_2024_Al_melt_convection_corrected.py` | tag |
|---|---|---|
| (66), (67) | satisfied by `T_solid`, `T_liquid`; residual measured in `verify_pde_residual` | [APPROX] |
| (68), (71) | `verify_initial_and_far_field` | [PAPER] |
| (69) | `verify_robin_wall` | [TYPO] sign |
| (70) | `verify_interface_temperature` | [PAPER] |
| (72) | `dsdt`, `stefan_residual` | [TYPO] `⁻s` → `s⁺` |
| (18b), (19b) | `psi_S`, `zeta_S` | [PAPER] |
| (22)–(26) | `Dimensionless`, `groups` | [PAPER] / [BUG] `Ste` |
| (29d–f) | `Dimensionless.Omega_star`, `interface_equation(omega=True)` | [BUG] inverted |
| (31) | `Material.n` | [PAPER] |
| (44) vs p.111 | `Material.N` (= `k_S/k_L`) | [INCONSISTENT] |
| (73), (74a,b) | `T_solid` | [PAPER] |
| (75a–c) | `grad_solid_interface`, `F_solid`, `F_solid_literal` | [PAPER] + reduction |
| (76a,b) | `T_liquid` | [INTERP] `ψ_L` |
| (77a–c) | `grad_liquid_interface` | [TYPO] `n`-scaling, sign |
| (78a–d) | `interface_equation` | [BUG] `(T_P−T_F)` factor |
| (79) | `verify_robin_interface`, `q_liquid` | [PAPER] |
| (80d) | `f_liquid` | [TYPO] `n`-scaling |
| (81), (82) | `T_liquid_drive` | [PAPER] |
| (83) | `interface_equation`, `solve_phi`, `quasi_steady_history` | [PAPER] + [BUG]s |
| Table 1 | `AL_TABLE1`, `TABLE1_HI_RANGE` | [PAPER] |
| Table 2 | `reproduce_table2`, `_legacy_equation`, `verify_table2` | audit |
| — (new) | `march_interface`, `initial_condition`, `incubation_time` | [APPROX] |
| — (new) | `h_bar_time_average`, `fit_h_bar` | constant mean `h` (§6) |
| — (new) | `effective_positions`, `RELIABLE_PROBES` | thermocouple-position diagnostic (§9.4) |
| — (new) | `digitise_position_figure`, `fit_h_bar_position`, `interface_from_cooling_curves`, `make_position_figure` | interface kinetics (§12) |
| — (new) | `fd_reference` | independent cross-check |
