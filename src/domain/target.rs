use crate::{domain::elements::ELEMENT_COUNT, error::Ms2AtomsError};

/// Element-class targets selected for one experiment or holdout.
#[derive(Clone, Debug)]
pub struct TargetSpec {
    class_indices: Vec<usize>,
}

impl TargetSpec {
    /// Creates a target specification after validating class indices.
    ///
    /// # Errors
    /// Returns [`Ms2AtomsError`] if any class index is outside the supported element list.
    pub fn new(class_indices: Vec<usize>) -> Result<Self, Ms2AtomsError> {
        for &class_index in &class_indices {
            if class_index >= ELEMENT_COUNT {
                return Err(Ms2AtomsError::InvalidClassIndex { class_index });
            }
        }

        Ok(Self { class_indices })
    }

    /// Returns the selected element-class indices.
    #[must_use]
    pub fn class_indices(&self) -> &[usize] {
        &self.class_indices
    }
}
