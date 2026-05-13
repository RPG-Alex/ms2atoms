use elements_rs::Element;

use crate::data::Spectra;

/// Defines the methods for a single holdout
pub trait Holdout {
    /// Returns a slice of [`Element`] that constitute the class
    fn classes(&self) -> &[Element]; 
    /// The iteration of the holdout
    fn holdout_number(&self) -> usize;
    /// the value of the random seed for the holdout
    fn random_seed(&self) -> usize;
    /// Returns a tuple of slices of the training and validation [`Spectra`]
    fn split(&self) -> (&[Spectra], &[Spectra]);
}