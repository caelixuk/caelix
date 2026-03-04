# Contributing

Thanks for your interest in CAELIX.

The project aims to be a rigorous, reproducible research codebase. Contributions that improve correctness, clarity, reproducibility, performance, and documentation are welcome.

## What to contribute

Useful contributions include:

- bug reports with minimal reproductions (commands, parameters, and outputs)
- documentation fixes (README, experiment notes, glossary)
- new experiments or ablation controls that follow the canonical runner structure
- performance improvements with before/after timings and validation that results are unchanged
- tests and diagnostics that improve regression safety

## Development principles

Please keep changes:

- deterministic where possible (explicit seeds, logged provenance)
- fail-loud (prefer clear errors over silent fallbacks)
- minimal and well-scoped (small PRs are easier to review)

If your change affects outputs, include:

- the exact command used
- the before/after CSV(s) (or a brief diff/summary)
- a short note explaining why the change is correct

## Licensing and contributions

CAELIX is licensed under AGPL-3.0-or-later.

By submitting a contribution (code, documentation, or data) you agree that it may be included in the project and redistributed under the project licence.

To keep the project maintainable and to preserve the ability to offer alternative licensing terms, significant contributions may require signing the Contributor Licence Agreement (CLA). See CLA.md.

## How to contribute

1. Open an issue describing the change (bug report, proposal, or experiment write-up).
2. Keep PRs small and focused.
3. Include reproduction steps and any relevant artefacts.

## Contact

For questions, collaboration proposals, or licensing enquiries:

alan@caelix.co.uk