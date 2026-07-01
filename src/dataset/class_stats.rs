use crate::{
    domain::{elements::ELEMENT_COUNT, sample::SpectrumSample},
    error::Ms2AtomsError,
};

/// Returns the element class indices observed in the provided samples.
///
/// # Parameters
/// - `samples` - Spectrum samples to scan for observed element classes.
#[must_use]
pub fn observed_class_indices(samples: &[SpectrumSample]) -> Vec<usize> {
    let mut observed = vec![false; ELEMENT_COUNT];

    for sample in samples {
        for (index, present) in sample.element_present().iter().enumerate() {
            if *present && let Some(observed_class) = observed.get_mut(index) {
                *observed_class = true;
            }
        }
    }

    observed
        .into_iter()
        .enumerate()
        .filter_map(|(index, present)| present.then_some(index))
        .collect()
}

/// Computes inverse-frequency class weights for the provided class indices.
///
/// # Errors
/// Returns [`Ms2AtomsError`] if a class index is invalid.
pub fn class_weights_for_samples(
    samples: &[SpectrumSample],
    class_indices: &[usize],
    weight_range: (f32, f32),
) -> Result<Vec<f32>, Ms2AtomsError> {
    let (min_weight, max_weight) = weight_range;
    let n_samples = count_as_f32(samples.len());
    let n_classes = count_as_f32(class_indices.len());
    let mut weights = Vec::with_capacity(class_indices.len());

    for &class_index in class_indices {
        if class_index >= ELEMENT_COUNT {
            return Err(Ms2AtomsError::InvalidClassIndex { class_index });
        }

        let positive_count = samples
            .iter()
            .filter(|sample| sample.is_element_present(class_index).unwrap_or(false))
            .count();

        let weight = if positive_count == 0 || class_indices.is_empty() {
            max_weight
        } else {
            n_samples / (count_as_f32(positive_count) * n_classes)
        };

        weights.push(weight.clamp(min_weight, max_weight));
    }

    Ok(weights)
}

/// Converts a count into `f32` for class-weight computation.
#[allow(
    clippy::cast_precision_loss,
    reason = "Class-weight computation uses dataset counts far below the precision limit of f32."
)]
const fn count_as_f32(count: usize) -> f32 {
    count as f32
}
