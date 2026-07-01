//! Model-agnostic evaluation utilities.

/// Confusion-matrix construction utilities.
pub mod confusion;
/// Evaluation metric computation utilities.
pub mod metrics;
/// Model-agnostic prediction score matrix.
pub mod prediction;
/// Report-building helpers.
pub mod reports;

pub use confusion::create_confusion_matrices;
pub use metrics::{aggregate_metrics, element_metrics_from_matrices};
pub use prediction::PredictionMatrix;
