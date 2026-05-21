pub mod basic_holdout;

pub use basic_holdout::BasicHoldout;

use elements_rs::Element;

use crate::{data::SpectrumSample, dataset::SpectraData};

/// Defines the methods for a single holdout
pub trait Holdout {
    /// Returns a slice of [`Element`] that constitute the class
    fn classes(&self) -> &[Element];
    /// Returns the indices of the classes from the `ELEMENTS` constant
    fn class_indices(&self) -> &[usize];
    /// The iteration of the holdout
    fn holdout_number(&self) -> usize;
    /// the value of the random seed for the holdout
    fn random_seed(&self) -> usize;
    /// Returns the training [`SpectraData`] set
    fn train_dataset(&self) -> &SpectraData;
    /// Returns the validation [`SpectraData`] set
    fn validation_dataset(&self) -> &SpectraData;
    /// Returns a tuple of slices of the training and validation [`SpectrumSample`]
    fn split(&self) -> (&[SpectrumSample], &[SpectrumSample]) {
        (
            &self.train_dataset().dataset,
            &self.validation_dataset().dataset,
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
