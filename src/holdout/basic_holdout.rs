use elements_rs::Element;

use super::Holdout;

use crate::{data::SpectrumSample, dataset::SpectraData};

#[derive(Clone, Debug)]
pub struct BasicHoldout {
    train: SpectraData,
    validation: SpectraData,
    classes: Vec<Element>,
    class_indices: Vec<usize>,
    holdout_number: usize,
    random_seed: usize,
}

impl BasicHoldout {
    pub fn new(
        train: SpectraData,
        validation: SpectraData,
        classes: Vec<Element>,
        class_indices: Vec<usize>,
        holdout_number: usize,
        random_seed: usize,
    ) -> Self {
        Self {
            train,
            validation,
            classes,
            class_indices,
            holdout_number,
            random_seed,
        }
    }

    pub fn train_dataset(&self) -> &SpectraData {
        &self.train
    }

    pub fn validation_dataset(&self) -> &SpectraData {
        &self.validation
    }
}

impl Holdout for BasicHoldout {
    fn classes(&self) -> &[Element] {
        &self.classes
    }

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
