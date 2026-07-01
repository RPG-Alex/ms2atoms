use crate::models::{burn::config::BurnMlpExperimentConfig, linfa::config::LinfaLogisticConfig};

/// Model-family choice and model-specific configuration for one experiment.
#[derive(Clone, Debug)]
pub enum ModelSpec {
    /// Burn-backed MLP multi-label classifier.
    BurnMlp(BurnMlpExperimentConfig),
    /// Linfa one-vs-rest logistic-regression baseline.
    LinfaLogistic(LinfaLogisticConfig),
}
