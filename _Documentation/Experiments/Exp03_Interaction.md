# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

# Exp03 — Interaction (Coulomb-like)

## Abstract

Exp03 validates that the emergent potential field in CAELIX carries a measurable **interaction energy**: a separation-dependent energy-like quantity that distinguishes attraction-like and repulsion-like configurations.

By placing two static, localised sources in the lattice and solving for the steady state under diffusive transport, we measure total field energy proxies as a function of separation distance \(d\). The system's interaction term is **negative** for opposite-sign source pairs and **positive** for like-sign pairs, consistent with attraction-like vs repulsion-like behaviour.

This experiment bridges the gap between the static potentials characterised in Exp01 and the dynamical particle experiments planned later in the suite. In standard field theory language, a conservative force is obtained from an interaction potential \(U\) via \(F = -\nabla U\); Exp03 establishes the prerequisite: a well-defined, distance-structured \(U(d)\) arising from local transport.

---

## Theoretical predictions

Exp03 is designed with explicit success criteria.

If the steady scalar field behaves as a Newtonian potential (\(\phi \propto 1/r\)), then the interaction energy associated with overlapping fields should follow the standard 3D scaling with separation distance \(d\).

**Energy scaling (expected):**

A common field-theoretic interaction proxy is gradient energy (field “tension”). For point-like sources in 3D, the overlap integral implies a leading-order scaling:

\[
U(d) \propto \frac{1}{d}
\]

**Force derivation (interpretation):**

For a conservative interaction, the force magnitude follows from the separation derivative of the interaction energy:

\[
F(d) = -\frac{dU}{dd}
\]

So if \(U(d) \propto d^{-1}\), the implied force scaling is:

\[
F(d) \propto \frac{1}{d^2}
\]

**Signatures:**

- Like sources \((+,+)\): \(U(d) > 0\) with \(dU/dd < 0\) (repulsion-like: increasing \(d\) lowers energy).
- Opposite sources \((+,-)\): \(U(d) < 0\) with \(dU/dd > 0\) (attraction-like: decreasing \(d\) lowers energy).

These statements are the interpretive target for Exp03: the experiment measures \(U(d)\) proxies and tests whether the scaling and sign structure follow these expectations.

---

## Methodology and constraints

What the evolution **does**:

- Evolves a real-valued scalar field \(\phi\) on a 3D cubic lattice using a **local neighbour-coupled transport** rule.
- For Exp03, the transport mode is constrained to **diffuse** (steady relaxation).
- Injects two compact sources (charges) at positions \(x_1, x_2\) about the lattice centre.
- Solves three steady states per separation \(d\):
  - **Pair solve:** both sources present.
  - **Self solves:** each source alone (two positions).


- It does **not** compute inter-particle forces directly.
- It does **not** hardcode a Coulomb potential \(\propto 1/r\).
- It does **not** use distance-based kernels, analytic Green's functions, or explicit charge–charge interaction terms.

Diagnostics (analysis) **do** compute interaction proxies:

- Two energy-like metrics are reported:
  - **E_phi:** \(\sum \phi^2\) (scalar intensity; useful but susceptible to bulk offsets).
  - **E_grad:** \(\sum |\nabla\phi|^2\) (field tension; typically the more physically informative channel).
- Interaction terms are formed by subtracting self contributions from the pair solution.
- The gradient channel is emphasised in fits because forces in many field theories arise from the elasticity/tension of field gradients.

---

## Implementation

Exp03 is implemented in `coulomb.py`.

For each separation \(d\) in a configured sweep:

1) Choose positions \(x_1 = c - d/2\), \(x_2 = c + d/2\) (with centre index \(c=n/2\)).
2) Solve the **pair** configuration to convergence (or a bounded iteration budget).
3) Solve the **self** configurations for each source alone.
   - When the geometry is symmetric about the centre, the second self solve may be **reused** by symmetry (reported explicitly in the CSV).
4) Compute energy-like observables for pair and self fields.
5) Form interaction observables by subtraction:

\[
E_{\text{int}} = E_{\text{pair}} - (E_{\text{self1}} + E_{\text{self2}})
\]

Two interaction channels are reported:

- **Phi channel:** a \(\phi\)-derived energy proxy.
- **Gradient channel:** a \(\nabla\phi\)-derived energy proxy.

The experiment then fits simple interaction models as a function of separation.

---

## Boundary conditions and interpretation

Exp03 is a steady-state solve on a finite lattice, so boundary conditions can influence long-range behaviour.

Two boundary regimes are relevant:

- **Hard / default boundary:** closed-domain behaviour; may introduce weak container effects at the largest separations.
- **Sponge layer (absorbing boundary):** reduces reflections by damping \(\phi\) near edges, but is not physical space.

For auditability, Exp03 records boundary mode and sponge parameters in the CSV header.


A robust interaction law should be primarily an **interior transport property**: after renormalisation (see below), the interaction curve should remain stable under reasonable boundary choices.

### Hard vs sponge (empirical equivalence)

Exp03 was rerun with an absorbing sponge boundary (ABC) and compared against the hard-boundary configuration. Within measurement precision, the inferred interaction law is unchanged.

For the inverse (Coulomb-like) fit coefficient \(A\) in \(\hat{E}(d)=A\,(1/d)+B\):

- Like sources \((+,+)\): hard \(A=5.4052\), sponge \(A=5.4049\) (\(<0.01\%\) difference)
- Opposite sources \((+,-)\): hard \(A=-5.7623\), sponge \(A=-5.7623\) (no measurable difference)

This supports the interpretation that the measured long-range interaction is a **bulk transport property** rather than an artefact of reflective box edges.

---

## Renormalisation

To remove additive offsets (often boundary-related), Exp03 also reports renormalised interaction observables:

- `E_int_phi_renorm`
- `E_int_grad_renorm`

Renormalisation is performed by subtracting the far-field value at the maximum sampled separation \(d_{\max}\):

\[
E_{\text{int,renorm}}(d) = E_{\text{int}}(d) - E_{\text{int}}(d_{\max})
\]

This makes the interaction tail explicitly approach zero at \(d_{\max}\) and improves comparability across runs.

---

## Models and fits

Exp03 fits two common forms to the interaction curve (for a chosen fit metric, typically `E_int_grad`):

1) **Inverse form (Coulomb-like):**
\[
\hat{E}(d) = A\,(1/d) + B
\]

2) **Yukawa / screened form:**
\[
\hat{E}(d) = A\,\frac{e^{-k d}}{d} + B
\]


Yukawa fits (\(e^{-k d}/d\)) are included as a diagnostic for **screening**: a non-zero \(k>0\) indicates an effective interaction range, which can arise from boundary screening, finite-volume effects, or lattice discretisation (in field-theory language, it resembles an effective mass term).

In the Exp03 sweeps reported to date, \(k\) remains small (typically \(\sim 0.02\)–\(0.03\) in lattice units), and does not materially change between hard and sponge boundaries. This suggests the sponge does not introduce artificial screening beyond the finite-volume curvature already present in the discrete solve.

Fit parameters and goodness-of-fit (\(R^2\)) are recorded in the CSV.

Additionally, Exp03 writes per-point predictions and residuals (raw and renormalised), so plots and diagnostic checks can be reproduced without re-implementing the models.

---

## Outputs

Exp03 produces a single structured CSV containing:

- A provenance/comment header describing solver, boundary, and fit metadata.
- One row per sampled separation \(d\), with:
  - solver iteration counts and convergence diagnostics
  - pair/self energy observables
  - interaction observables (raw and renormalised)
  - fit parameters (repeated per row)
  - per-point model predictions and residuals

---

## CSV structure

The output CSV is a comment-headed, multi-section file.

### Header fields

The header includes (at minimum):

- lattice: `n`, centre `c`, placement safety `safe`
- sweep: `d_min`, `d_max`, `d_step`, charge `q`, sign
- solver: tolerances and iteration limits (including `traffic_iters` / coulomb budgets)
- boundary: `traffic_boundary`, `traffic_sponge_width`, `traffic_sponge_strength`
- fit: `fit_metric`, `fit_n`, `fit_d_min`, `fit_d_max`, model parameters and \(R^2\)
- renorm anchors:
  - `coulomb_renorm_d_far`
  - `coulomb_renorm_E_int_phi_far`
  - `coulomb_renorm_E_int_grad_far`
  - `coulomb_pred_invr_yhat_far`
  - `coulomb_pred_yuk_yhat_far`

### Data columns

Each data row contains:

- geometry: `d`, `x1`, `x2`, `c`, `safe`
- solver work:
  - `iters_pair`, `iters_self1`, `iters_self2`, `iters_self`
  - `iters_self_logical` (accounts for symmetry reuse)
  - `self2_reused` (1 when self2 is reused by symmetry)
- convergence diagnostics:
  - `pair_converged`, `pair_max_delta`
  - `self1_converged`, `self1_max_delta`
  - `self2_converged`, `self2_max_delta`
- energy observables:
  - pair: `E_pair_phi`, `E_pair_grad`
  - self: `E_self1_phi`, `E_self1_grad`, `E_self2_phi`, `E_self2_grad`
  - totals: `E_self_phi`, `E_self_grad`
- interaction observables:
  - raw: `E_int_phi`, `E_int_grad`
  - renorm: `E_int_phi_renorm`, `E_int_grad_renorm`
- fit parameters (repeated):
  - inverse: `invr_A`, `invr_B`, `invr_r2`
  - yukawa: `yuk_A`, `yuk_B`, `yuk_k`, `yuk_r2`, `yuk_half_range`
- per-point model predictions and residuals:
  - inverse: `invr_yhat`, `invr_resid`, `invr_yhat_renorm`, `invr_resid_renorm`
  - yukawa: `yuk_yhat`, `yuk_resid`, `yuk_yhat_renorm`, `yuk_resid_renorm`

---

## How to read the result

Typical checks:

1) **Convergence:** `*_converged` should be true across the sweep; large `*_max_delta` indicates insufficient iterations or too-loose `check_every`.
2) **Sign structure:** like charges should produce an interaction curve of opposite sign to opposite charges under the same observable definition.
3) **Long-range tail:** after renorm, the interaction should trend smoothly to 0 at \(d_{\max}\).
4) **Model diagnostics:** compare `*_resid` and `*_resid_renorm` to understand whether deviations are boundary offsets, screening-like curvature, or discretisation effects.
5) **Derived force:** a conservative force follows from the separation derivative of the interaction energy. As a practical check, estimate \(F(d) \approx -\Delta U/\Delta d\) from the measured curve and confirm whether the scaling is consistent with an inverse-square trend (\(\propto 1/d^2\)) over the fit window. If the Yukawa parameter \(k\) is measurably non-zero, interpret deviations (often a steeper apparent slope) as screening / finite-range curvature rather than a failure of the interaction sign structure. In the sponge-boundary reruns, the finite-difference force estimate is consistent with an inverse-square trend and yields representative coefficients \(F(d)\approx +3.21/d^2\) for like sources and \(F(d)\approx -3.30/d^2\) for opposite sources (fit window dependent).

---

## Notes

- Exp03 is a **diagnostic interaction proxy**, not a claim that the lattice is electromagnetism. Its role is to determine whether the substrate supports stable, distance-structured interactions under local transport.
- Boundary choice can matter in steady solves; Exp03 therefore records boundary metadata and provides renormalised curves for controlled comparisons.