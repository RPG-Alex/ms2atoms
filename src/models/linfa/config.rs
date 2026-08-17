//! Module for Linfa model configs

/// Configuration for the Linfa logistic-regression.
#[derive(Clone, Copy, Debug)]
pub struct LinfaLogisticConfig {
    /// Maximum optimizer iterations for each binary classifier.
    pub max_iterations: u64,
    /// L2 regularization weight used by Linfa logistic regression.
    pub alpha: f64,
    /// Seed used when deterministic class-balancing samples are drawn.
    pub random_seed: u64,
    /// Maximum ratio between the larger and smaller binary classes.
    pub max_majority_to_minority_ratio: Option<usize>,
    /// Maximum number of samples retained from either binary class.
    pub max_samples_per_class: Option<usize>,
}
