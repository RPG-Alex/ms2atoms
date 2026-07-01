//! Command-line entry point for running `ms2atoms` experiments.

use ms2atoms::{error::Ms2AtomsError, experiments::experiment1};
use tracing_subscriber::{EnvFilter, fmt};

fn main() -> Result<(), Ms2AtomsError> {
    fmt().with_env_filter(EnvFilter::from_default_env()).init();

    experiment1::run()
}
