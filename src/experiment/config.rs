#[derive(Clone, Debug)]
/// Complete configuration for one experiment run.
pub struct ExperimentConfig<P> {
    /// Metadata identifying this experiment run.
    pub run: RunConfig,
    /// Feature extraction and input-shape settings.
    pub features: FeatureConfig,
    /// Protocol used to generate train/validation holdout splits.
    pub protocol: P,
    /// MLP model architecture settings.
    pub model: MlpModelConfig,
    /// Training hyperparameters.
    pub training: TrainingParams,
    /// Loss-function settings.
    pub loss: LossConfig,
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

/// Multilayer perceptron architecture settings.
#[derive(Clone, Copy, Debug)]
pub struct MlpModelConfig {
    /// Number of neurons in the first hidden layer.
    pub hidden_size: usize,
    /// Dropout probability applied during training.
    pub dropout: f64,
}

/// Training hyperparameters.
#[derive(Clone, Copy, Debug)]
pub struct TrainingParams {
    /// Number of training epochs.
    pub epochs: usize,
    /// Number of samples per training batch.
    pub batch_size: usize,
    /// Number of data-loader workers.
    pub workers: usize,
    /// Optimizer learning rate.
    pub learning_rate: f64,
}

/// Loss-function configuration.
#[derive(Clone, Copy, Debug)]
pub struct LossConfig {
    /// Class-weighting strategy used by the loss function.
    pub class_weighting: ClassWeighting,
}

/// Class-weighting strategy for multi-label element classification.
#[derive(Clone, Copy, Debug)]
pub enum ClassWeighting {
    /// Do not apply class weights.
    None,
    /// Weight classes by inverse positive frequency, clamped to a fixed range.
    InverseFrequency {
        /// Minimum and maximum allowed class weights.
        clamp: (f32, f32),
    },
}

/// Evaluation configuration.
#[derive(Clone, Debug)]
pub struct EvaluationConfig {
    /// Decision thresholds used to convert prediction scores into labels.
    pub thresholds: Vec<f64>,
}
