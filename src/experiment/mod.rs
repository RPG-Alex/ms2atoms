//! Experiment configuration, split protocols, and model-agnostic runner.

/// Experiment configuration types.
pub mod config;
/// Experiment protocols for generating train/validation holdout splits.
pub mod protocol;
/// Experiment runner for training, inference, and report generation.
pub mod runner;

pub use config::{EvaluationConfig, ExperimentConfig, FeatureConfig, RunConfig};
pub use protocol::{ExperimentProtocol, RandomSplitProtocol, StratifiedRetryProtocol};
pub use runner::run_experiment;
