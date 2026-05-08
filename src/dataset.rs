use mascot_rs::mascot_generic_format::MGFVec;
use mass_spectrometry::traits::Spectrum;
use molecular_formulas::{ChemicalFormula, MolecularFormula};
use rand::{prelude::*, rngs::ChaCha8Rng, seq::SliceRandom};

use burn::data::dataset::Dataset;

use crate::{
    data::{BIN_SIZE, ELEMENT_COUNT, ELEMENTS, Spectra},
    error::SpectraError,
};

pub struct SpectraData {
    pub(crate) dataset: Vec<Spectra>,
    pub(crate) class_weights: Vec<f32>,
}

impl SpectraData {
    pub fn train(&self, seed: u64) -> Self {
        let mut data = self.dataset.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        data.shuffle(&mut rng);
        let len = data.len();
        let Some(subset) = data.get(0..(len * 8 / 10)) else {
            unreachable!("Problem loading vec")
        };
        Self {
            dataset: subset.to_vec(),
            class_weights: self.class_weights.clone(),
        }
    }

    pub fn new() -> Result<Self, SpectraError> {
        let data = load_spectra()?;
        let weights = get_class_weights(&data);
        Ok(Self {
            dataset: data,
            class_weights: weights,
        })
    }
    pub fn test(&self, seed: u64) -> Self {
        let mut data = self.dataset.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        data.shuffle(&mut rng);
        let len = data.len();
        let Some(subset) = self.dataset.get((len * 8 / 10)..len) else {
            unreachable!("Vec subsetting failed")
        };

        Self { dataset: subset.to_vec(), class_weights: self.class_weights.clone() }
    }
}
fn load_spectra() -> Result<Vec<Spectra>, SpectraError> {
    let load = pollster::block_on(
        MGFVec::<f64>::annotated_ms2()
            .target_directory("data")
            .load(),
    )?;
    let mut output: Vec<Spectra> = Vec::with_capacity(load.spectra().len());
    for spec in load.spectra() {
        let Some(formula) = spec.metadata().formula() else {
            continue;
        };
        output.push(Spectra {
            spectra: *spec
                .linear_binned_intensities(0.0, 1000.0, BIN_SIZE)?
                .as_array::<BIN_SIZE>()
                .ok_or(SpectraError::InvalidArray)?,
            element_present: *spec_occurrence(formula)?
                .as_array::<ELEMENT_COUNT>()
                .ok_or(SpectraError::InvalidArray)?,
        });
    }
    Ok(output)
}

fn spec_occurrence(
    formula: &ChemicalFormula<u32, i32>,
) -> Result<[bool; ELEMENT_COUNT], SpectraError> {
    let mut elements_occurrence = [false; ELEMENT_COUNT];
    for (i, &e) in ELEMENTS.iter().enumerate() {
        if formula.contains_element(e) {
            elements_occurrence[i] = true;
        }
    }
    Ok(elements_occurrence)
}

fn get_class_weights(data: &[Spectra]) -> Vec<f32> {
    let mut output: Vec<f32> = vec![0.0; ELEMENT_COUNT];
    let n_samples = data.len() as f32;
    let n_classes = ELEMENT_COUNT as f32;
    for d in data {
        for (i, &element_present) in d.element_present.iter().enumerate() {
            if element_present {
                output[i] += 1.0;
            }
        }
    }
    for weight in output.iter_mut() {
        *weight = n_samples / (*weight * n_classes);
    }
    output
}

impl Dataset<Spectra> for SpectraData {
    fn get(&self, index: usize) -> Option<Spectra> {
        self.dataset.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}