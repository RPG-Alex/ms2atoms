use crate::{
    dataset::{class_stats::class_weights_for_samples, load::load_spectra},
    domain::sample::SpectrumSample,
    error::Ms2AtomsError,
};

#[derive(Clone, Debug)]
/// Dataset of binned MS/MS spectra and element-presence labels.
pub struct SpectraData {
    dataset: Vec<SpectrumSample>,
    bin_size: usize,
}

impl SpectraData {
    /// Loads annotated MS/MS spectra and converts them into [`SpectrumSample`] values.
    ///
    /// # Parameters
    /// - `bin_size` - Number of fixed-width bins used to represent each spectrum.
    ///
    /// # Errors
    /// Returns [`Ms2AtomsError`] if annotated spectra loading or spectrum binning fails.
    pub fn new(bin_size: usize) -> Result<Self, Ms2AtomsError> {
        let data = load_spectra(bin_size)?;
        Ok(Self {
            dataset: data,
            bin_size,
        })
    }

    /// Creates a dataset from precomputed spectrum samples.
    ///
    /// # Parameters
    /// - `dataset` - Precomputed spectrum samples.
    /// - `bin_size` - Number of bins used to represent each sample spectrum.
    #[must_use]
    pub const fn from_samples(dataset: Vec<SpectrumSample>, bin_size: usize) -> Self {
        Self { dataset, bin_size }
    }

    /// Returns all spectrum samples in this dataset.
    #[must_use]
    pub fn samples(&self) -> &[SpectrumSample] {
        &self.dataset
    }

    /// Returns the number of samples in this dataset.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Returns whether this dataset contains no samples.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    /// Computes inverse-frequency class weights for the provided class indices.
    ///
    /// # Errors
    /// Returns [`Ms2AtomsError`] if a class index is invalid.
    pub fn class_weights_for(
        &self,
        class_indices: &[usize],
        weight_range: (f32, f32),
    ) -> Result<Vec<f32>, Ms2AtomsError> {
        class_weights_for_samples(&self.dataset, class_indices, weight_range)
    }

    /// Returns the number of bins used by each spectrum sample.
    #[must_use]
    pub const fn bin_size(&self) -> usize {
        self.bin_size
    }
}
