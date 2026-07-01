//! Dataset loading, feature extraction, and class-statistics utilities.

/// Class-frequency and target-discovery helpers.
pub mod class_stats;
/// Feature extraction helpers.
pub mod features;
/// Annotated spectrum loading.
pub mod load;
/// In-memory spectrum dataset.
pub mod spectra_data;

pub use class_stats::observed_class_indices;
pub use spectra_data::SpectraData;
