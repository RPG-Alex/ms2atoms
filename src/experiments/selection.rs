use crate::{error::Ms2AtomsError, experiments};
use std::io::Write;
use tracing_subscriber::{EnvFilter, fmt};

/// Represents CLI runtime options
pub enum Command {
    /// Takes the experiment number to be run
    Run {
        /// the number of the experiment to run (obtainable from `List` option)
        experiment_number: usize,
    },
    /// Lists all experiments available
    List,
    /// Help options
    Help {
        /// Takes the program name as a parameter
        program_name: String,
    },
}

/// takes and filters the input
pub fn init_tracing() {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
}

/// Parses the command and routes based on the `Command` variant
///
/// # Errors
/// - Returns a commandline error if unable to parse input
pub fn parse_command(mut args: impl Iterator<Item = String>) -> Result<Command, Ms2AtomsError> {
    let program_name = args.next().unwrap_or_else(|| "ms2atoms".to_owned());

    let Some(first_argument) = args.next() else {
        return Ok(Command::Help { program_name });
    };

    if args.next().is_some() {
        return Err(Ms2AtomsError::command_line(
            "expected exactly one argument: an experiment number, --list, or --help",
        ));
    }

    match first_argument.as_str() {
        "-h" | "--help" => Ok(Command::Help { program_name }),
        "--list" => Ok(Command::List),
        value => parse_experiment_number(value),
    }
}

/// Takes the experiment number a `str` and attempts to parse it as a `usize`
///
/// # Errors
/// - Returns a command line error if unable to parse a valid experiment number
pub fn parse_experiment_number(value: &str) -> Result<Command, Ms2AtomsError> {
    let experiment_number = value.parse::<usize>().map_err(|error| {
        Ms2AtomsError::command_line(format!("invalid experiment number '{value}': {error}"))
    })?;

    Ok(Command::Run { experiment_number })
}

/// Outputs the intended usage for running the repo
///
/// # Errors
/// - Returns command line error if unable to write to the standard output
pub fn write_usage(program_name: &str) -> Result<(), Ms2AtomsError> {
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();

    writeln!(stdout, "Usage:")?;
    writeln!(stdout, "  {program_name} <EXPERIMENT_NUMBER>")?;
    writeln!(stdout, "  {program_name} --list")?;
    writeln!(stdout, "  {program_name} --help")?;
    writeln!(stdout)?;
    writeln!(stdout, "Examples:")?;
    writeln!(stdout, "  cargo run -- 1")?;
    writeln!(stdout, "  RUST_LOG=debug cargo run -- 1")?;
    writeln!(stdout)?;
    write_experiments(&mut stdout)
}

/// Outputs the experiment list to the std out
///
/// # Errors
/// - Returns command line error if unable to write to the standard output
pub fn write_experiment_list() -> Result<(), Ms2AtomsError> {
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();

    write_experiments(&mut stdout)
}

/// Writes the list of experiments from the `experiments` module
///
/// # Errors
/// - Returns
pub fn write_experiments(writer: &mut impl std::io::Write) -> Result<(), Ms2AtomsError> {
    writeln!(writer, "Available experiments:")?;

    for experiment in experiments::available_experiments() {
        let number = experiment.number;
        let name = experiment.name;
        writeln!(writer, "  {number} - {name}")?;
    }

    Ok(())
}
