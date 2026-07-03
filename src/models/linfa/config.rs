/// Configuration for the Linfa one-vs-rest logistic-regression baseline.
#[derive(Clone, Copy, Debug)]
pub struct LinfaLogisticConfig {
    /// Maximum optimizer iterations for each binary classifier.
    pub max_iterations: u64,
    /// L2 regularization weight used by Linfa logistic regression.
    pub alpha: f64,
    /// Seed used when deterministic class-balancing samples are drawn.
    pub random_seed: u64,
    /// Maximum negative-to-positive ratio used while fitting each binary classifier.
    pub max_negative_to_positive_ratio: Option<usize>,
    /// Absolute cap on the number of negative examples used while fitting each binary classifier.
    pub max_negative_samples: Option<usize>,
}
