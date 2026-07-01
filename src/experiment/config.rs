use crate::models::spec::ModelSpec;

#[derive(Clone, Debug)]
/// Complete configuration for one experiment run.
pub struct ExperimentConfig<P> {
    /// Metadata identifying this experiment run.
    pub run: RunConfig,
    /// Feature extraction and input-shape settings.
    pub features: FeatureConfig,
    /// Protocol used to generate train/validation holdout splits.
    pub protocol: P,
    /// Model-family configuration and hyperparameters.
    pub model: ModelSpec,
    /// Evaluation settings.
    pub evaluation: EvaluationConfig,
}

#[derive(Clone, Debug)]
/// Metadata identifying an experiment run.
pub struct RunConfig {
    /// Numeric experiment identifier.
    pub experiment_num: usize,
    /// Human-readable experiment name.
    pub name: String,
}

/// Feature extraction and model input settings.
#[derive(Clone, Copy, Debug)]
pub struct FeatureConfig {
    /// Number of fixed-width bins used to represent each spectrum.
    pub bin_size: usize,
}

/// Evaluation configuration.
#[derive(Clone, Debug)]
pub struct EvaluationConfig {
    /// Decision thresholds used to convert prediction scores into labels.
    pub thresholds: Vec<f64>,
}
