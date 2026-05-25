use mascot_rs::mascot_generic_format::MGFVec;
use mass_spectrometry::traits::Spectrum;
use molecular_formulas::{ChemicalFormula, MolecularFormula};

use burn::data::dataset::Dataset;

use crate::{
    data::{ELEMENT_COUNT, ELEMENTS, SpectrumSample},
    error::SpectraError,
};

#[derive(Clone, Debug)]
pub struct SpectraData {
    pub(crate) dataset: Vec<SpectrumSample>,
    pub(crate) bin_size: usize,
}

impl SpectraData {
    pub fn new(bin_size: usize) -> Result<Self, SpectraError> {
        let data = load_spectra(bin_size)?;
        Ok(Self {
            dataset: data,
            bin_size,
        })
    }

    pub fn class_weights_for(&self, class_indices: &[usize], weight_range: (f32, f32)) -> Vec<f32> {
        let (min_weight, max_weight) = weight_range;
        let n_samples = self.dataset.len() as f32;
        let n_classes = class_indices.len() as f32;
        class_indices
            .iter()
            .map(|&class_index| {
                let positive_count = self
                    .dataset
                    .iter()
                    .filter(|sample| sample.element_present[class_index])
                    .count() as f32;
                let positive_count = positive_count.max(1.0);
                let weight = n_samples / (positive_count * n_classes);
                weight.clamp(min_weight, max_weight)
            })
            .collect()
    }
    pub fn bin_size(&self) -> usize {
        self.bin_size
    }
}

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
            .to_vec();
        output.push(SpectrumSample {
            spectra,
            element_present: *spec_occurrence(formula)
                .as_array::<ELEMENT_COUNT>()
                .ok_or(SpectraError::InvalidArray)?,
        });
    }
    Ok(output)
}

fn spec_occurrence(formula: &ChemicalFormula<u32, i32>) -> [bool; ELEMENT_COUNT] {
    let mut elements_occurrence = [false; ELEMENT_COUNT];
    for (i, &e) in ELEMENTS.iter().enumerate() {
        if formula.contains_element(e) {
            elements_occurrence[i] = true;
        }
    }
    elements_occurrence
}

impl Dataset<SpectrumSample> for SpectraData {
    fn get(&self, index: usize) -> Option<SpectrumSample> {
        self.dataset.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}
