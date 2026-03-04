# CAELIX
**Constructive Algorithmics for Emergent Lattice Interaction eXperiments**

CAELIX is a falsifiable, discrete-to-continuous simulation laboratory. It is designed to test whether continuum-like field signatures—such as 1/r profiles, kinematic latency, and stable topological solitons—can be generated, routed, and measured using strictly local update rules on a minimal substrate. 

The architecture enforces a strict separation of concerns:
1. **Substrate:** A discrete balanced-ternary lattice (s ∈ {-1, 0, +1}) acts as a topological boundary generator.
2. **Coupling:** A deterministic load functional maps the microstate topology to a non-negative structural proxy.
3. **Carrier:** Continuous field dynamics (φ) are evolved via diffusion and telegraph surrogate equations using Jacobi-style double-buffering. 

The project deliberately avoids metaphysical claims, treating expected numerical behaviours (such as the 3D discrete Laplacian 1/r asymptote) as regression-safe baselines. The focus is on topological routing, spatial stiffness grids, and reproducible non-linear particle extraction (Sine-Gordon / φ⁴).

## Methodological Controls
To ensure computational phenomena are not confused with physical emergence, CAELIX implements strict diagnostic constraints:
* **Isotropy Calibration (`isotropy.py`):** Automated bounding of grid-scale directional anisotropy and numerical dispersion.
* **Separation of Kinematics:** Explicit falsifiers to separate genuine kinematic-dilation proxies from FDTD mode-locking artefacts.
* **Explicit Non-Linearity:** Soliton regimes do not claim to emerge organically from linear rules; standard non-linear field models are explicitly injected to test extraction and routing stability.

## Primary Reference
- **Emergent Field Physics from Balanced-Ternary Microstates** (Zenodo, 2026): [10.5281/zenodo.18823977](https://doi.org/10.5281/zenodo.18823977)
  *Details the constructive substrate approach, experimental controls, and the CAELIX FDTD pipeline.*
- For citation metadata (software), see CITATION.cff.

<details>
<summary><b>Theoretical Background (Foundational Papers)</b></summary>
The underlying logic defining the necessity and forced arithmetic of the balanced-ternary microstates:

- On the Necessity of Existence (Zenodo, 2026): [10.5281/zenodo.18797375](https://doi.org/10.5281/zenodo.18797375)
- Balanced Ternary by Necessity (Zenodo, 2026): [10.5281/zenodo.18806015](https://doi.org/10.5281/zenodo.18806015)
- Constants from Balanced Ternary (Zenodo, 2026): [10.5281/zenodo.18810282](https://doi.org/10.5281/zenodo.18810282)
</details>

## Repository Structure
- `_Documentation/` — Codebase overview, experiment suite analysis, glossary.
- `_Code/` — Core execution spine (`core.py`, `pipeline.py`), solvers (`traffic.py`), and experiment drivers.
- `_Code/_Output/` — Destination for provenance-logged CSVs, pre-extracted HDF5 sprite assets, and diagnostic plots.

The repository includes the full suite of run outputs for the published experiment sets. Experiments 09A, 09B, and 09C rely on pre-extracted Sprite assets being present.

## Quick Start
CAELIX expects a standard Python scientific environment (NumPy, Numba, SciPy).

```bash
# Clone and setup
git clone https://github.com/caelixuk/caelix.git
cd caelix

# Install dependencies
# Create/activate your preferred environment, then install the core deps:
pip install -U pip
pip install numpy numba scipy
```

## Versions used for the Zenodo runs
These are the versions used for the published Zenodo experiment outputs. Newer versions may work, but are not guaranteed.

```
numpy >= 2.3.4
numba >= 0.63.1
scipy >= 1.16.2
```

## Experiment runner
CAELIX provides a single stable entrypoint for running the definitive experiments in a sensible order, without retyping long CLI invocations.

### Usage

```bash
# run from the Codebase main folder
cd _Code

# Discover available experiment presets
python experiments.py --list

# Run a specific preset by name
python experiments.py --run 01A_pipeline_baseline

# Shorthand: the identifier alone is sufficient
python experiments.py --run 01A

# Run all experiments
python experiments.py --run-all
```

You may pass either the full preset name or just its identifier (e.g. 01A).

### Canonical experiment order
This is the intended discovery to complexity progression. Use `--run-all` to execute this order end-to-end.

Note: all runs automatically generate a provenance header in the output CSVs detailing exact kernel parameters and random seeds used.


### Licence
GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See LICENSE.
Commercial licensing is available for organisations that require non-copyleft terms (e.g. closed-source distribution or hosted services). See COMMERCIAL.md.

### Project policies
- Commercial licensing: see COMMERCIAL.md.
- Contributor agreement (for significant contributions): see CLA.md.
- Contributing guidelines: see CONTRIBUTING.md.
- Trademark and attribution policy: see TRADEMARK.md.
- Security reporting: see SECURITY.md.
- Governance: see GOVERNANCE.md.
- Changelog: see CHANGELOG.md.


### Copyright
Copyright (C) 2026 Alan Ball.
