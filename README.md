[![Rust CI](https://github.com/RPG-Alex/ms2atoms/actions/workflows/rust.yml/badge.svg)](https://github.com/RPG-Alex/ms2atoms/actions/workflows/rust.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/RPG-Alex/ms2atoms/branch/main/graph/badge.svg)](https://codecov.io/gh/RPG-Alex/ms2atoms)

# spectra scribe

ms2atoms is a Rust research prototype for training and evaluating machine-learning models for MS/MS element identification. It uses the [Burn](https://burn.dev/) deep-learning framework to learn multi-label element predictions from binned tandem mass spectra.

The current focus is to make the experimental pipeline discrete, reproducible, and easy enough for collaborators to inspect before discussing broader experimental questions.

## What it does

ms2atoms currently runs a baseline MS/MS element-detection experiment:

1. Load annotated MS2 spectra.
2. Convert spectra into fixed-width binned intensity vectors.
3. Derive element-presence labels from each molecular formula.
4. Generate repeated train/validation holdouts.
5. Train a multilayer perceptron with Burn.
6. Evaluate predictions across multiple decision thresholds.
7. Write class-distribution and metric reports as CSV files.

## Getting started

```bash
git clone https://github.com/RPG-Alex/ms2atoms.git
cd ms2atoms
cargo fmt --all -- --check
cargo clippy --all-targets --all-features
cargo test --all-targets --all-features
cargo run --release
```

## Research status

ms2atoms is an exploratory research tool. Model architecture, input representation, thresholds, class weighting, and evaluation protocol are expected to change as the experimental task becomes clearer.

## License

MIT
