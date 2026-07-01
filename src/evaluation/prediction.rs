use crate::error::Ms2AtomsError;

/// Model-agnostic prediction scores for a validation/test split.
#[derive(Clone, Debug)]
pub struct PredictionMatrix {
    class_indices: Vec<usize>,
    scores: Vec<Vec<f64>>,
}

impl PredictionMatrix {
    /// Creates a prediction matrix after validating row widths.
    ///
    /// # Parameters
    /// - `class_indices` - Element class indices corresponding to prediction columns.
    /// - `scores` - Prediction scores with shape `[n_samples, n_classes]`.
    ///
    /// # Errors
    /// Returns [`Ms2AtomsError`] when any score row has the wrong width.
    pub fn new(class_indices: Vec<usize>, scores: Vec<Vec<f64>>) -> Result<Self, Ms2AtomsError> {
        let expected = class_indices.len();

        for row in &scores {
            let actual = row.len();
            if actual != expected {
                return Err(Ms2AtomsError::InconsistentFeatureLength { expected, actual });
            }
        }

        Ok(Self {
            class_indices,
            scores,
        })
    }

    /// Returns the element class indices corresponding to prediction columns.
    #[must_use]
    pub fn class_indices(&self) -> &[usize] {
        &self.class_indices
    }

    /// Returns prediction score rows.
    #[must_use]
    pub fn scores(&self) -> &[Vec<f64>] {
        &self.scores
    }

    /// Returns the number of prediction rows.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.scores.len()
    }

    /// Returns whether the prediction matrix has no rows.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }
}
