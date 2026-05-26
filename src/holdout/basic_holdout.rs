use super::Holdout;

use crate::dataset::SpectraData;

#[derive(Clone, Debug)]
pub struct BasicHoldout {
    train: SpectraData,
    validation: SpectraData,
    class_indices: Vec<usize>,
    holdout_number: usize,
    random_seed: usize,
}

impl BasicHoldout {
    pub fn new(
        train: SpectraData,
        validation: SpectraData,
        class_indices: Vec<usize>,
        holdout_number: usize,
        random_seed: usize,
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

    fn random_seed(&self) -> usize {
        self.random_seed
    }

    fn train_dataset(&self) -> &SpectraData {
        &self.train
    }

    fn validation_dataset(&self) -> &SpectraData {
        &self.validation
    }
}
