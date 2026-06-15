/// Confusion-matrix construction utilities.
pub mod confusion;
/// Evaluation metric computation utilities.
pub mod metrics;

pub use confusion::create_confusion_matrices;
pub use metrics::{aggregate_metrics, element_metrics_from_matrices};
