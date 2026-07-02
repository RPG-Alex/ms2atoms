//! Command-line entry point for running `ms2atoms` experiments.

use std::env;

use ms2atoms::error::Ms2AtomsError;
use ms2atoms::experiments::run_by_number;
use ms2atoms::experiments::selection::{
    Command, init_tracing, parse_command, write_experiment_list, write_usage,
};

fn main() -> Result<(), Ms2AtomsError> {
    init_tracing();

    match parse_command(env::args())? {
        Command::Run { experiment_number } => run_by_number(experiment_number),
        Command::List => write_experiment_list(),
        Command::Help { program_name } => write_usage(&program_name),
    }
}
