use crate::{
    error::Ms2AtomsError,
    experiment::{
        EvaluationConfig, ExperimentConfig, FeatureConfig, RunConfig, StratifiedRetryProtocol,
        run_experiment,
    },
    models::{
        burn::config::{
            BurnLossConfig, BurnMlpExperimentConfig, BurnTrainingConfig, ClassWeighting,
            MlpArchitectureConfig,
        },
        spec::ModelSpec,
    },
};

/// Runs the first baseline `ms2atoms` experiment.
///
/// This experiment uses stratified retry holdouts, a Burn MLP model,
/// inverse-frequency class weighting, and fixed evaluation thresholds.
///
/// # Errors
/// Returns [`Ms2AtomsError`] if experiment execution fails.
pub fn run() -> Result<(), Ms2AtomsError> {
    let config = ExperimentConfig {
        run: RunConfig {
            experiment_num: 1,
            name: "single_holdout-burn-mlp-baseline".to_owned(),
        },
        features: FeatureConfig { bin_size: 1000 },
        protocol: StratifiedRetryProtocol {
            number_of_holdouts: 1,
            random_seed: 42,
            training_size: 0.8,
            retries_per_holdout: 100,
        },
        model: ModelSpec::BurnMlp(BurnMlpExperimentConfig {
            architecture: MlpArchitectureConfig {
                hidden_size: 100,
                dropout: 0.5,
            },
            training: BurnTrainingConfig {
                epochs: 10,
                batch_size: 64,
                workers: 4,
                learning_rate: 1.0e-4,
            },
            loss: BurnLossConfig {
                class_weighting: ClassWeighting::InverseFrequency { clamp: (0.1, 10.0) },
            },
        }),
        evaluation: EvaluationConfig {
            thresholds: vec![0.5],
        },
    };

    run_experiment(&config)
}
