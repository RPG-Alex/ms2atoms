use crate::domain::elements::ELEMENT_COUNT;

#[derive(Clone, Debug)]
/// One preprocessed MS/MS spectrum sample as a vector of `f64` and its element-presence in reference to the [`ELEMENT_COUNT`].
pub struct SpectrumSample {
    /// molecular group identifier
    group_id: String,
    spectra: Vec<f64>,
    element_present: [bool; ELEMENT_COUNT],
}

impl SpectrumSample {
    /// Creates a new spectrum sample from binned intensities and element labels.
    ///
    /// # Parameters
    /// - `spectra` - Binned spectrum intensity values.
    /// - `element_present` - Fixed-width element-presence labels aligned with the element list.
    #[must_use]
    pub const fn new(
        group_id: String,
        spectra: Vec<f64>,
        element_present: [bool; ELEMENT_COUNT],
    ) -> Self {
        Self {
            group_id,
            spectra,
            element_present,
        }
    }

    /// Returns the Sample's `group_id`
    #[must_use]
    pub fn group_id(&self) -> &str {
        &self.group_id
    }

    /// Returns the binned spectrum intensity values.
    #[must_use]
    pub fn spectra(&self) -> &[f64] {
        &self.spectra
    }

    /// Returns all element-presence labels for this sample.
    #[must_use]
    pub const fn element_present(&self) -> &[bool; ELEMENT_COUNT] {
        &self.element_present
    }

    /// Returns whether the element at `class_index` is present in this sample.
    ///
    /// # Parameters
    /// - `class_index` - Index into the crate element list.
    #[must_use]
    pub fn is_element_present(&self, class_index: usize) -> Option<bool> {
        self.element_present.get(class_index).copied()
    }
}
