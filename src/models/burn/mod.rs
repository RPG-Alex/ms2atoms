//! Burn-backed model implementations.

/// Converts domain samples into Burn batches.
pub mod batcher;
/// Burn MLP experiment configuration.
pub mod config;
/// Burn dataset trait implementation for shared dataset types.
pub mod dataset;
/// Burn model inference helpers.
pub mod inference;
/// Burn training-time metrics.
pub mod metrics;
/// Burn MLP model implementation.
pub mod mlp;
/// Burn training orchestration.
pub mod training;
