/// Configuration for the Linfa one-vs-rest logistic-regression baseline.
#[derive(Clone, Copy, Debug)]
pub struct LinfaLogisticConfig {
    /// Maximum optimizer iterations for each binary classifier.
    pub max_iterations: u64,
    /// L2 regularization weight used by Linfa logistic regression.
    pub alpha: f64,
}
