use crate::data::{ELEMENT_COUNT, SpectrumSample};

pub mod experiment1;
pub mod experiment2;

/// Returns the indices for elements that occur in the spectra data. Useful for experiments wanting to control for only elements present in data.
pub(crate) fn observed_class_indices(samples: &[SpectrumSample]) -> Vec<usize> {
    let mut observed = vec![false; ELEMENT_COUNT];
    for sample in samples {
        for (index, present) in sample.element_present.iter().enumerate() {
            if *present {
                observed[index] = true;
            }
        }
    }
    observed
        .into_iter()
        .enumerate()
        .filter_map(|(index, present)| present.then_some(index))
        .collect()
}
