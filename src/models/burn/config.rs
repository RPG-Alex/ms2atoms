/// Full Burn MLP configuration for one experiment.
#[derive(Clone, Debug)]
pub struct BurnMlpExperimentConfig {
    /// MLP architecture settings.
    pub architecture: MlpArchitectureConfig,
    /// Burn training hyperparameters.
    pub training: BurnTrainingConfig,
    /// Burn loss-function settings.
    pub loss: BurnLossConfig,
}

/// Multilayer perceptron architecture settings.
#[derive(Clone, Copy, Debug)]
pub struct MlpArchitectureConfig {
    /// Number of neurons in the first hidden layer.
    pub hidden_size: usize,
    /// Dropout probability applied during training.
    pub dropout: f64,
}

/// Burn training hyperparameters.
#[derive(Clone, Copy, Debug)]
pub struct BurnTrainingConfig {
    /// Number of training epochs.
    pub epochs: usize,
    /// Number of samples per training batch.
    pub batch_size: usize,
    /// Number of data-loader workers.
    pub workers: usize,
    /// Optimizer learning rate.
    pub learning_rate: f64,
}

/// Burn loss-function configuration.
#[derive(Clone, Copy, Debug)]
pub struct BurnLossConfig {
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
