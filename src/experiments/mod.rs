//! Concrete experiment entry points.

use crate::error::Ms2AtomsError;

/// Baseline Burn MLP experiment.
pub mod experiment1;
/// Linfa logistic-regression baseline experiment.
pub mod experiment2;
/// Linfa log regression experiment with max to min reduced to 5
pub mod experiment3;

/// Selection logic for selecting experiments to run
pub mod selection;

/// Callable experiment entry-point type.
pub type ExperimentRun = fn() -> Result<(), Ms2AtomsError>;

/// Metadata and entry point for one registered experiment.
#[derive(Clone, Copy)]
pub struct ExperimentDefinition {
    /// Numeric experiment identifier used by the command-line interface.
    pub number: usize,
    /// Human-readable experiment name.
    pub name: &'static str,
    /// Function that runs the experiment.
    pub run: ExperimentRun,
}

const EXPERIMENTS: &[ExperimentDefinition] = &[
    ExperimentDefinition {
        number: 1,
        name: "single_holdout-burn-mlp-baseline",
        run: experiment1::run,
    },
    ExperimentDefinition {
        number: 2,
        name: "single_holdout-linfa-logistic-baseline",
        run: experiment2::run,
    },
    ExperimentDefinition {
        number: 3,
        name: "single_holdout-linfa-logistic-max-to-min-five",
        run: experiment3::run,
    },
];

/// Returns all experiments registered for command-line execution.
#[must_use]
pub const fn available_experiments() -> &'static [ExperimentDefinition] {
    EXPERIMENTS
}

/// Runs the experiment matching the provided numeric identifier.
///
/// # Errors
/// Returns [`Ms2AtomsError`] if no experiment is registered with the provided number,
/// or if the selected experiment fails.
pub fn run_by_number(number: usize) -> Result<(), Ms2AtomsError> {
    let Some(experiment) = EXPERIMENTS
        .iter()
        .find(|experiment| experiment.number == number)
    else {
        return Err(Ms2AtomsError::UnknownExperimentNumber { number });
    };

    (experiment.run)()
}
