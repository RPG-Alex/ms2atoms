use crate::{
    error::SpectraError,
    experiment_config::{run_experiment, standard::StandardConfig},
};

fn experiment_5_config() -> StandardConfig {
    StandardConfig {
        // Experiment variable changed. No weights added
        batch_size: 32,
        experiment_num: 5,
        ..StandardConfig::default()
    }
}

pub fn run() -> Result<(), SpectraError> {
    run_experiment(experiment_5_config())
}
