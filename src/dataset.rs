use mascot_rs::mascot_generic_format::MGFVec;
use mass_spectrometry::traits::Spectrum;
use molecular_formulas::{ChemicalFormula, MolecularFormula};

use burn::data::dataset::Dataset;

use crate::{
    data::{ELEMENT_COUNT, ELEMENTS, SpectrumSample},
    error::SpectraError,
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
    pub fn new(bin_size: usize) -> Result<Self, SpectraError> {
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
    pub const fn from_samples(dataset: Vec<SpectrumSample>, bin_size: usize) -> Self {
        Self { dataset, bin_size }
    }

    /// Returns all spectrum samples in this dataset.
    pub fn samples(&self) -> &[SpectrumSample] {
        &self.dataset
    }

    /// Returns the number of samples in this dataset.
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Returns whether this dataset contains no samples.
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    /// Computes inverse-frequency class weights for the provided class indices.
    ///
    /// # Parameters
    /// - `class_indices` - Element class indices to compute weights for.
    /// - `weight_range` - Minimum and maximum class-weight values.
    pub fn class_weights_for(
        &self,
        class_indices: &[usize],
        weight_range: (f32, f32),
    ) -> Result<Vec<f32>, SpectraError> {
        let (min_weight, max_weight) = weight_range;
        let n_samples = self.dataset.len() as f32;
        let n_classes = class_indices.len() as f32;
        let mut weights = Vec::with_capacity(class_indices.len());

        for &class_index in class_indices {
            if class_index >= ELEMENT_COUNT {
                return Err(SpectraError::InvalidArray);
            }

            let positive_count = self
                .dataset
                .iter()
                .filter(|sample| {
                    sample
                        .element_present()
                        .get(class_index)
                        .copied()
                        .unwrap_or(false)
                })
                .count() as f32;

            let positive_count = positive_count.max(1.0);
            let weight = n_samples / (positive_count * n_classes);

            weights.push(weight.clamp(min_weight, max_weight));
        }
        Ok(weights)
    }

    /// Returns the number of bins used by each spectrum sample.
    pub const fn bin_size(&self) -> usize {
        self.bin_size
    }
}

/// Returns the element class indices observed in the provided samples.
///
/// # Parameters
/// - `samples` - Spectrum samples to scan for observed element classes.
pub fn observed_class_indices(samples: &[SpectrumSample]) -> Vec<usize> {
    let mut observed = vec![false; ELEMENT_COUNT];
    for sample in samples {
        for (index, present) in sample.element_present().iter().enumerate() {
            if *present {
                if let Some(observed_class) = observed.get_mut(index) {
                    *observed_class = true;
                }
            }
        }
    }
    observed
        .into_iter()
        .enumerate()
        .filter_map(|(index, present)| present.then_some(index))
        .collect()
}

/// Loads annotated MS/MS spectra from the local data directory.
fn load_spectra(bin_size: usize) -> Result<Vec<SpectrumSample>, SpectraError> {
    let load = pollster::block_on(
        MGFVec::<f64>::annotated_ms2()
            .target_directory("data")
            .load(),
    )?;
    let mut output: Vec<SpectrumSample> = Vec::with_capacity(load.spectra().len());
    for spec in load.spectra() {
        let Some(formula) = spec.metadata().formula() else {
            continue;
        };
        let spectra = spec
            .linear_binned_intensities(0.0, 1000.0, bin_size)?
            .clone();
        output.push(SpectrumSample::new(
            spectra,
            formula_element_occurrence(formula),
        ));
    }
    Ok(output)
}

/// Returns a fixed-width element-presence array for a molecular formula.
fn formula_element_occurrence(formula: &ChemicalFormula<u32, i32>) -> [bool; ELEMENT_COUNT] {
    let mut element_present = [false; ELEMENT_COUNT];
    for (present, &element) in element_present.iter_mut().zip(ELEMENTS.iter()) {
        *present = formula.contains_element(element);
    }

    element_present
}

impl Dataset<SpectrumSample> for SpectraData {
    fn get(&self, index: usize) -> Option<SpectrumSample> {
        self.dataset.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}
