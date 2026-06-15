use serde::{Deserialize, Serialize};

use crate::{
    data::{ELEMENTS, SpectrumSample},
    error::SpectraError,
};

use burn::{prelude::*, tensor::Transaction};

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

/// Creates per-element confusion matrices from model predictions.
///
/// # Parameters
/// - `predictions` - Model prediction tensor as `[n_samples, n_classes]`.
/// - `items` - Spectrum samples corresponding to the prediction rows.
/// - `class_indices` - Element class indices corresponding to prediction columns.
/// - `threshold` - Decision threshold used to convert prediction scores into labels.
///
/// # Errors
/// - Returns [`SpectraError`] if an element class index is invalid.
/// - Returns [`SpectraError`] if prediction tensor extraction fails.
/// - Returns [`SpectraError`] if a prediction row is missing an expected class value.
/// - Returns [`SpectraError`] if a sample is missing an expected element label.
/// - Returns [`SpectraError`] if a thresholded prediction is not binary.
pub fn create_confusion_matrices<B: Backend>(
    predictions: Tensor<B, 2>,
    items: &[SpectrumSample],
    class_indices: &[usize],
    threshold: f64,
) -> Result<Vec<ConfusionMatrix>, SpectraError> {
    let mut confusion_matrices = Vec::with_capacity(class_indices.len());

    for &class_index in class_indices {
        let element = ELEMENTS
            .get(class_index)
            .ok_or(SpectraError::InvalidArray)?
            .symbol()
            .to_owned();

        confusion_matrices.push(ConfusionMatrix::new(element));
    }

    let predicted_tensor = predictions.greater_elem(threshold).int();
    let prediction_rows = predicted_tensor.iter_dim(0);

    for (prediction_row, sample) in prediction_rows.zip(items.iter()) {
        let [output_data] = Transaction::default()
            .register(prediction_row)
            .execute()
            .try_into()
            .map_err(|_| SpectraError::InvalidArray)?;

        let predicted_values = output_data.as_slice::<i32>()?;

        for (prediction_column, &class_index) in class_indices.iter().enumerate() {
            let predicted_atom = predicted_values
                .get(prediction_column)
                .copied()
                .ok_or(SpectraError::InvalidArray)?;

            let true_atom = sample
                .is_element_present(class_index)
                .ok_or(SpectraError::InvalidArray)?;

            let matrix = confusion_matrices
                .get_mut(prediction_column)
                .ok_or(SpectraError::InvalidArray)?;

            match (predicted_atom, true_atom) {
                (1, true) => matrix.true_positive += 1,
                (0, false) => matrix.true_negative += 1,
                (1, false) => matrix.false_positive += 1,
                (0, true) => matrix.false_negative += 1,
                _ => return Err(SpectraError::InvalidArray),
            }
        }
    }

    Ok(confusion_matrices)
}
