# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

# CAELIX Glossary

A compact, operational glossary of CAELIX terms.
Definitions describe how terms are used in this codebase and papers.

---

## Boundaries and domains

**Hard boundary (hard clamp)**
A non-periodic boundary condition where updates at or beyond the domain edge are prevented or clamped, producing reflections.

**Sponge**
A damping boundary region that absorbs outgoing energy to reduce reflections from the domain edge.

**Clamp**
A boundary rule that pins a field to a fixed value (often 0) at the boundary or within a region.

**Mask**
A boolean or weighted field used to enable, disable, or modulate updates in selected voxels.

**Wire**
A masked channel region that confines propagation to a guided path through the domain.

**Dump chamber**
A widened region attached to a wire or junction, designed to absorb or disperse a routed signal.

**Junction (T-junction)**
A wire topology where one guided channel splits or merges, used to test routing and fan-out behaviours.

**Obstacle**
A masked region that blocks propagation, used for confinement and routing experiments.

**Periodic boundary**
A boundary condition where opposite faces wrap and neighbour references roll across edges.

**Non-periodic boundary**
A boundary condition where neighbour references do not wrap across edges.

---

## Substrate, microstate, and coupling

**Microstate (s)**
The discrete substrate field, typically taking values in {−1, 0, +1}.

**Balanced ternary**
The signed ternary alphabet {−1, 0, +1} used for the substrate microstate.

**Constructive algorithmics**
The methodological principle behind CAELIX: building continuum-like observables strictly from discrete, local update rules without injecting top-down analytic solutions.

**Load (load functional)**
A deterministic mapping from microstate to a non-negative structural proxy, used to couple discrete structure into carrier dynamics.

**Interface density**
A proxy for local boundary or defect content, typically encoded via the load functional.

**Coupling**
A rule that uses substrate-derived quantities (for example load) to modulate carrier evolution parameters locally.

**k-grid / stiffness grid**
A spatially varying coefficient field that modulates an update rule, often derived from load or masks.

**Source term**
A local injection into a field update, used for driving, seeding, or forcing.

**Sink**
A local removal or damping rule, used for absorption or stabilisation.

---

## Carriers and update rules

**Carrier field (phi, φ)**
The continuous (floating-point) field evolved over the domain by local stencil updates.

**Diffuse mode (diffusion)**
A relaxational update that smooths the carrier field by averaging neighbours; often used to reach an equilibrium profile.

**Telegraph mode**
A second-order-in-time wave-like update (with optional damping/forcing), used for propagating packets and oscillators.

**Stencil**
The set of neighbour offsets used by an update rule, for example a 6-neighbour axis-aligned stencil.

**Discrete Laplacian**
A stencil-based approximation to ∇², typically neighbour-sum minus centre, used in diffusion and wave updates.

**Jacobi iteration**
A relaxation method that updates each site from neighbour values using a double buffer; diffusion mode commonly has this character.

**Double buffer**
Two arrays (old/new) are alternated so updates are synchronous and deterministic.

**dt**
The simulation time step.

**c (wave speed) / c²**
Wave speed parameterisation for telegraph updates; c² often appears directly in the update rule.

**Damping / decay**
A term that reduces amplitude over time, used for stability and to model loss.

**Injection**
A term that adds energy or amplitude, used to drive oscillators or maintain a source.

**Dispersion (numerical)**
A discretisation artefact where phase/group velocity depends on wavelength or direction.

**Back-reaction**
A perturbative feedback mechanism where local field gradients (computed from the continuous carrier) dynamically update the trajectory or momentum of a discrete moving source.

**Dirichlet boundary**
A hard boundary condition where the field value is explicitly clamped to a constant (usually 0), used in confinement and ringdown experiments.

**Neumann boundary**
A boundary condition specifying the derivative, typically forcing the normal gradient to 0 to minimise reflections or allow glide. Used in routing and junction experiments.

---

## Observables and diagnostics

**Vacuum profile**
The equilibrium carrier field profile obtained in the absence of explicit distance kernels, typically measured radially.

**Inverse-radius scaling (≈ 1/r)**
Far-field decay proportional to 1/r, treated as a scaling law rather than an exact identity on the grid.

**Radial fit**
A regression of a measured profile against r, often used to estimate scaling exponents and goodness-of-fit.

**Isotropy**
Directional uniformity of propagation speed/shape; tested by comparing multiple directions and shells.

**Seam metric**
A measure of anisotropy or shell-to-shell variance, used to detect grid imprinting.

**Latency proxy**
A measured delay (tick count, peak arrival time, phase shift) used as a kinematic or refraction-style proxy.

**Lensing proxy**
A measured deflection or delay of rays/packets under a spatially varying index or wave-speed map.

**Kinematic-dilation proxy**
A measured delay in peak arrival times or phase drift between a moving observer and a stationary reference, used to study transport behaviour without asserting strict Lorentz invariance.

**Mode-locking (artefact)**
A numerical artefact in FDTD solvers where moving sources or boundaries synchronise with the discrete grid spacing, producing spurious resonances or trapping. CAELIX uses windowed peak detection to bypass this.

**Falsifier (explicit falsifier)**
A designated control experiment (for example disabling decay or enforcing isotropy) designed to show that an observed continuum-like signature is a consequence of the intended coupling rather than a numerical artefact.

**Provenance header**
A structured metadata block written with outputs, recording parameters, seeds, and run identifiers.

**Run preset (experiment preset)**
A named configuration used by `experiments.py` to run a specific scenario without long CLI invocations.

**Identifier (01A, 07Q, etc.)**
A short code for an experiment preset; accepted as shorthand for the full preset name.

---

## Nonlinear regimes and solitons

**Nonlinear carrier**
A carrier update that includes a nonlinear restoring term such as φ⁴ or sine–Gordon.

**φ⁴ (phi-four)**
A quartic potential term that supports kink-like solutions under appropriate parameters.

**Sine–Gordon (SG)**
A nonlinear restoring term proportional to −k sin(φ) supporting kinks and breathers.

**Kink**
A topological soliton connecting distinct vacua.

**Breather**
A localised oscillatory bound state in sine–Gordon.

**Topological charge**
An integer-like invariant associated with kink configurations.

**Sprite**
A stored patch of a stable soliton (or other structure) saved for reuse in collision/routing experiments.

**Sprite extraction**
The process of detecting, cropping, and saving a stable structure from a run.

**Collider**
A scenario where two or more sprites/packets are launched to study interaction outcomes.

---

## Implementation and performance

**Numba**
A JIT compiler used to accelerate the core update loops.

**Kernel**
A compiled numerical routine for updating the carrier fields over the grid.

**Work units**
A progress accounting scheme used to report meaningful completion for sweeps.

---

## Notes

If a term appears in outputs but is not defined here, add it with a short operational definition and (if relevant) the module or CLI flag where it is controlled.