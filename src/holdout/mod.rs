//! Holdout split types and holdout reports.

/// Basic holdout implementation.
pub mod basic;
/// Holdout reporting utilities.
pub mod report;

pub use basic::BasicHoldout;
pub use report::class_distribution_report;

use crate::{dataset::SpectraData, domain::sample::SpectrumSample};

/// Defines the methods for a single holdout split.
pub trait Holdout {
    /// Number of output classes this holdout trains/evaluates.
    fn num_classes(&self) -> usize {
        self.class_indices().len()
    }

    /// Returns the indices of the classes from the crate element list.
    fn class_indices(&self) -> &[usize];

    /// Returns which split this is, e.g. 0, 1, 2, ...
    fn holdout_number(&self) -> usize;

    /// Returns the random seed that produced this holdout.
    fn random_seed(&self) -> u64;

    /// Returns the training [`SpectraData`] set.
    fn train_dataset(&self) -> &SpectraData;

    /// Returns the validation [`SpectraData`] set.
    fn validation_dataset(&self) -> &SpectraData;

    /// Returns a tuple of training and validation [`SpectrumSample`] slices.
    fn split(&self) -> (&[SpectrumSample], &[SpectrumSample]) {
        (
            self.train_dataset().samples(),
            self.validation_dataset().samples(),
        )
    }

    /// Returns the total spectra in the holdout's training set.
    fn training_len(&self) -> usize {
        self.split().0.len()
    }

    /// Returns the total spectra in the holdout's validation set.
    fn validation_len(&self) -> usize {
        self.split().1.len()
    }
}
