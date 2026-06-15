/// Experiment configuration types.
pub mod config;
/// Experiment protocols for generating train/validation holdout splits.
pub mod protocol;
/// Experiment runner for training, inference, and report generation.
pub mod runner;

pub use config::*;
pub use protocol::*;
pub use runner::run_experiment;
