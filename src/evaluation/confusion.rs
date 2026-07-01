use serde::{Deserialize, Serialize};

use crate::{
    domain::{elements::ELEMENTS, sample::SpectrumSample},
    error::Ms2AtomsError,
    evaluation::prediction::PredictionMatrix,
};

/// Per-element confusion-matrix counts for one evaluation threshold.
#[derive(Serialize, Deserialize, Debug)]
pub struct ConfusionMatrix {
    /// Chemical symbol for the evaluated element.
    pub element: String,
    /// Number of true positive predictions.
    pub true_positive: u32,
    /// Number of true negative predictions.
    pub true_negative: u32,
    /// Number of false positive predictions.
    pub false_positive: u32,
    /// Number of false negative predictions.
    pub false_negative: u32,
}

impl ConfusionMatrix {
    /// Creates an empty confusion matrix for one element.
    ///
    /// # Parameters
    /// - `element` - Chemical symbol for the evaluated element.
    #[must_use]
    pub const fn new(element: String) -> Self {
        Self {
            element,
            true_positive: 0,
            true_negative: 0,
            false_positive: 0,
            false_negative: 0,
        }
    }
}

/// Creates per-element confusion matrices from model-agnostic prediction scores.
///
/// # Parameters
/// - `predictions` - Prediction scores as `[n_samples, n_classes]`.
/// - `items` - Spectrum samples corresponding to the prediction rows.
/// - `threshold` - Decision threshold used to convert prediction scores into labels.
///
/// # Errors
/// Returns [`Ms2AtomsError`] if class indices, prediction rows, or sample labels are invalid.
pub fn create_confusion_matrices(
    predictions: &PredictionMatrix,
    items: &[SpectrumSample],
    threshold: f64,
) -> Result<Vec<ConfusionMatrix>, Ms2AtomsError> {
    if predictions.len() != items.len() {
        return Err(Ms2AtomsError::InconsistentFeatureLength {
            expected: items.len(),
            actual: predictions.len(),
        });
    }

    let class_indices = predictions.class_indices();
    let mut confusion_matrices = Vec::with_capacity(class_indices.len());

    for &class_index in class_indices {
        let element = ELEMENTS
            .get(class_index)
            .ok_or(Ms2AtomsError::InvalidClassIndex { class_index })?
            .symbol()
            .to_owned();

        confusion_matrices.push(ConfusionMatrix::new(element));
    }

    for (prediction_row, sample) in predictions.scores().iter().zip(items.iter()) {
        if prediction_row.len() != class_indices.len() {
            return Err(Ms2AtomsError::InconsistentFeatureLength {
                expected: class_indices.len(),
                actual: prediction_row.len(),
            });
        }

        for (prediction_column, (&class_index, &score)) in
            class_indices.iter().zip(prediction_row.iter()).enumerate()
        {
            let predicted_atom = score > threshold;
            let true_atom = sample
                .is_element_present(class_index)
                .ok_or(Ms2AtomsError::InvalidClassIndex { class_index })?;

            let matrix = confusion_matrices
                .get_mut(prediction_column)
                .ok_or(Ms2AtomsError::InvalidArray)?;

            match (predicted_atom, true_atom) {
                (true, true) => matrix.true_positive += 1,
                (false, false) => matrix.true_negative += 1,
                (true, false) => matrix.false_positive += 1,
                (false, true) => matrix.false_negative += 1,
            }
        }
    }

    Ok(confusion_matrices)
}
