//! Command-line entry point for running `SpectraScribe` experiments.

use spectra_scribe::{error::SpectraError, experiments::experiment1};
use tracing_subscriber::{EnvFilter, fmt};

fn main() -> Result<(), SpectraError> {
    fmt().with_env_filter(EnvFilter::from_default_env()).init();

    experiment1::run()
}
