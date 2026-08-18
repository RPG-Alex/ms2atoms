use crate::{
    error::Ms2AtomsError,
    experiment::{
        EvaluationConfig, ExperimentConfig, FeatureConfig, RunConfig, StratifiedRetryProtocol,
        run_experiment,
    },
    models::{linfa::config::LinfaLogisticConfig, spec::ModelSpec},
};

/// Runs the Linfa logistic-regression baseline experiment.
///
/// This experiment uses the same holdout settings as experiment 1, but swaps the
/// Burn MLP for independent binary logistic-regression classifiers.
///
/// # Errors
/// Returns [`Ms2AtomsError`] if experiment execution fails.
pub fn run() -> Result<(), Ms2AtomsError> {
    let config = ExperimentConfig {
        run: RunConfig {
            experiment_num: 2,
            name: "single_holdout-linfa-logistic-baseline".to_owned(),
        },
        features: FeatureConfig { bin_size: 1000 },
        protocol: StratifiedRetryProtocol {
            number_of_holdouts: 1,
            random_seed: 42,
            training_size: 0.8,
            retries_per_holdout: 100,
        },
        model: ModelSpec::LinfaLogistic(LinfaLogisticConfig {
            max_iterations: 100,
            gradient_tolerance: 1e-4,
            alpha: 1.0,
            random_seed: 42,
            max_majority_to_minority_ratio: Some(50),
            max_samples_per_class: Some(50_000),
        }),
        evaluation: EvaluationConfig {
            thresholds: vec![0.5],
        },
    };

    run_experiment(&config)
}
