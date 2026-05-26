use crate::{
    error::SpectraError,
    experiment_config::{run_experiment, standard::StandardConfig},
};

fn experiment_3_config() -> StandardConfig {
    StandardConfig {
        // Experiment variable changed. No weights added
        weight_range: None,
        experiment_num: 3,
        ..StandardConfig::default()
    }
}

pub fn run() -> Result<(), SpectraError> {
    run_experiment(experiment_3_config())
}
