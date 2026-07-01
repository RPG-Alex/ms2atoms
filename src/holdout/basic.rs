use crate::dataset::SpectraData;

use super::Holdout;

#[derive(Clone, Debug)]
/// Basic concrete holdout split used by experiment protocols.
pub struct BasicHoldout {
    train: SpectraData,
    validation: SpectraData,
    class_indices: Vec<usize>,
    holdout_number: usize,
    random_seed: u64,
}

impl BasicHoldout {
    /// Creates a new basic holdout split.
    ///
    /// # Parameters
    /// - `train` - Training dataset for this holdout.
    /// - `validation` - Validation dataset for this holdout.
    /// - `class_indices` - Element class indices included in this holdout.
    /// - `holdout_number` - Sequential identifier for this holdout.
    /// - `random_seed` - Random seed used to generate this holdout.
    #[must_use]
    pub const fn new(
        train: SpectraData,
        validation: SpectraData,
        class_indices: Vec<usize>,
        holdout_number: usize,
        random_seed: u64,
    ) -> Self {
        Self {
            train,
            validation,
            class_indices,
            holdout_number,
            random_seed,
        }
    }
}

impl Holdout for BasicHoldout {
    fn class_indices(&self) -> &[usize] {
        &self.class_indices
    }

    fn holdout_number(&self) -> usize {
        self.holdout_number
    }

    fn random_seed(&self) -> u64 {
        self.random_seed
    }

    fn train_dataset(&self) -> &SpectraData {
        &self.train
    }

    fn validation_dataset(&self) -> &SpectraData {
        &self.validation
    }
}
