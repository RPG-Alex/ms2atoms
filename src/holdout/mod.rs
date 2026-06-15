/// Basic holdout implementation
pub mod basic_holdout;
/// Holdout reporting utilities
pub mod report;

pub use basic_holdout::BasicHoldout;
pub use report::class_distribution_report;

use crate::{data::SpectrumSample, dataset::SpectraData};

/// Defines the methods for a single holdout
pub trait Holdout {
    /// Number of output classes this holdout trains/evaluates.
    fn num_classes(&self) -> usize {
        self.class_indices().len()
    }
    /// Returns the indices of the classes from the `ELEMENTS` constant
    fn class_indices(&self) -> &[usize];
    /// Which split this is on, e.g. 0,1,2, ...
    fn holdout_number(&self) -> usize;
    /// the value of the random seed that produced this holdout
    fn random_seed(&self) -> u64;
    /// Returns the training [`SpectraData`] set
    fn train_dataset(&self) -> &SpectraData;
    /// Returns the validation [`SpectraData`] set
    fn validation_dataset(&self) -> &SpectraData;
    /// Returns a tuple of slices of the training and validation [`SpectrumSample`]
    fn split(&self) -> (&[SpectrumSample], &[SpectrumSample]) {
        (
            self.train_dataset().samples(),
            self.validation_dataset().samples(),
        )
    }
    /// the total spectra that are in the holdout's training set
    fn training_len(&self) -> usize {
        self.split().0.len()
    }
    /// the total spectra in the holdout's validation set
    fn validation_len(&self) -> usize {
        self.split().1.len()
    }
}
