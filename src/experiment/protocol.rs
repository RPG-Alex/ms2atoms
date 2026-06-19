use rand::{SeedableRng, rngs::ChaCha8Rng, seq::SliceRandom};

use crate::{
    data::SpectrumSample,
    dataset::{SpectraData, observed_class_indices},
    holdout::{BasicHoldout, Holdout},
};

/// Protocol for generating train/validation holdout splits.
///
/// Experiment protocols define how many holdouts to generate, which random seed
/// to use, how large the training split should be, and how holdout datasets are
/// produced from the full dataset.
pub trait ExperimentProtocol {
    /// Concrete holdout type produced by this protocol.
    type HoldoutType: Holdout;

    /// Returns the number of holdout splits to generate.
    fn number_of_holdouts(&self) -> usize;

    /// Returns the base random seed used by this protocol.
    fn random_seed(&self) -> u64;

    /// Returns the fraction of samples assigned to training.
    fn training_size(&self) -> f32;

    /// Returns the fraction of samples assigned to validation.
    fn validation_size(&self) -> f32 {
        1.0 - self.training_size()
    }

    /// Generates holdout splits from the provided dataset.
    ///
    /// # Parameters
    ///
    /// - `dataset` - Dataset to split into train/validation holdouts.
    fn generate_holdouts(&self, dataset: &SpectraData) -> Vec<Self::HoldoutType>;
}

/// Protocol that generates random train/validation splits.
pub struct RandomSplitProtocol {
    /// Number of holdout splits to generate.
    pub number_of_holdouts: usize,

    /// Base random seed used to generate holdout splits.
    pub random_seed: u64,

    /// Fraction of samples assigned to training.
    pub training_size: f32,
}

impl ExperimentProtocol for RandomSplitProtocol {
    type HoldoutType = BasicHoldout;

    fn number_of_holdouts(&self) -> usize {
        self.number_of_holdouts
    }

    fn random_seed(&self) -> u64 {
        self.random_seed
    }

    fn training_size(&self) -> f32 {
        self.training_size
    }

    fn generate_holdouts(&self, dataset: &SpectraData) -> Vec<Self::HoldoutType> {
        let class_indices = observed_class_indices(dataset.samples());

        (0..self.number_of_holdouts)
            .map(|holdout_number| {
                let seed = holdout_seed(self.random_seed, holdout_number, 0);

                make_random_holdout(
                    dataset,
                    &class_indices,
                    holdout_number,
                    seed,
                    self.training_size,
                )
            })
            .collect()
    }
}

/// Protocol that retries random splits and keeps the most class-balanced split.
///
/// This protocol is useful for sparse multi-label data where a purely random
/// split can easily place rare classes only in training or only in validation.
pub struct StratifiedRetryProtocol {
    /// Number of holdout splits to generate.
    pub number_of_holdouts: usize,

    /// Base random seed used to generate holdout splits.
    pub random_seed: u64,

    /// Fraction of samples assigned to training.
    pub training_size: f32,

    /// Number of random split attempts evaluated for each holdout.
    pub retries_per_holdout: usize,
}

impl ExperimentProtocol for StratifiedRetryProtocol {
    type HoldoutType = BasicHoldout;

    fn number_of_holdouts(&self) -> usize {
        self.number_of_holdouts
    }

    fn random_seed(&self) -> u64 {
        self.random_seed
    }

    fn training_size(&self) -> f32 {
        self.training_size
    }

    fn generate_holdouts(&self, dataset: &SpectraData) -> Vec<Self::HoldoutType> {
        let class_indices = observed_class_indices(dataset.samples());
        let attempts = self.retries_per_holdout.max(1);

        (0..self.number_of_holdouts)
            .map(|holdout_number| {
                let initial_seed = holdout_seed(self.random_seed, holdout_number, 0);
                let (initial_train, initial_validation) =
                    random_split(dataset, initial_seed, self.training_size);

                let mut best_score = split_score(
                    &initial_train,
                    &initial_validation,
                    &class_indices,
                    self.training_size,
                );
                let mut best_train = initial_train;
                let mut best_validation = initial_validation;
                let mut best_seed = initial_seed;

                for attempt in 1..attempts {
                    let seed = holdout_seed(self.random_seed, holdout_number, attempt);
                    let (train, validation) = random_split(dataset, seed, self.training_size);
                    let score =
                        split_score(&train, &validation, &class_indices, self.training_size);

                    if score < best_score {
                        best_score = score;
                        best_train = train;
                        best_validation = validation;
                        best_seed = seed;
                    }
                }

                BasicHoldout::new(
                    SpectraData::from_samples(best_train, dataset.bin_size()),
                    SpectraData::from_samples(best_validation, dataset.bin_size()),
                    class_indices.clone(),
                    holdout_number,
                    best_seed,
                )
            })
            .collect()
    }
}

/// Creates one random holdout from a dataset.
fn make_random_holdout(
    dataset: &SpectraData,
    class_indices: &[usize],
    holdout_number: usize,
    holdout_seed: u64,
    training_size: f32,
) -> BasicHoldout {
    let (train, validation) = random_split(dataset, holdout_seed, training_size);

    BasicHoldout::new(
        SpectraData::from_samples(train, dataset.bin_size()),
        SpectraData::from_samples(validation, dataset.bin_size()),
        class_indices.to_vec(),
        holdout_number,
        holdout_seed,
    )
}

/// Splits a dataset into shuffled training and validation samples.
fn random_split(
    dataset: &SpectraData,
    seed: u64,
    training_size: f32,
) -> (Vec<SpectrumSample>, Vec<SpectrumSample>) {
    let mut samples = dataset.samples().to_vec();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    samples.shuffle(&mut rng);

    let split_index = split_index(samples.len(), training_size);
    let validation = samples.split_off(split_index);
    let train = samples;

    (train, validation)
}

/// Computes the number of samples assigned to the training split.
///
/// # Parameters
/// - `sample_count` - Total number of samples available for splitting.
/// - `training_size` - Fraction of samples to assign to the training split.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Holdout splitting uses small dataset counts and a validated training fraction between 0.0 and 1.0."
)]
fn split_index(sample_count: usize, training_size: f32) -> usize {
    debug_assert!((0.0..=1.0).contains(&training_size));

    (sample_count as f32 * training_size).floor() as usize
}

/// Scores a split by how well it preserves positive examples in validation.
fn split_score(
    train: &[SpectrumSample],
    validation: &[SpectrumSample],
    class_indices: &[usize],
    training_size: f32,
) -> f32 {
    let expected_validation_fraction = 1.0 - training_size;
    let mut score = 0.0;

    for &class_index in class_indices {
        let train_positive = count_positive_samples(train, class_index);
        let validation_positive = count_positive_samples(validation, class_index);
        let total_positive = train_positive + validation_positive;

        if total_positive == 0 {
            continue;
        }

        if train_positive == 0 {
            score += 1_000.0;
        }

        if validation_positive == 0 && total_positive > 1 {
            score += 1_000.0;
        }

        let actual_validation_fraction = fraction_f32(validation_positive, total_positive);
        score += (actual_validation_fraction - expected_validation_fraction).abs();
    }

    score
}

/// Returns `part / total` as an `f32`, or `0.0` when `total` is zero.
#[allow(
    clippy::cast_precision_loss,
    reason = "Holdout split scoring uses dataset counts far below the precision limit of f32."
)]
fn fraction_f32(part: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        part as f32 / total as f32
    }
}

/// Counts samples where the selected element class is present.
fn count_positive_samples(samples: &[SpectrumSample], class_index: usize) -> usize {
    samples
        .iter()
        .filter(|sample| sample.is_element_present(class_index).unwrap_or(false))
        .count()
}

/// Computes the deterministic seed for one holdout attempt.
const fn holdout_seed(base_seed: u64, holdout_number: usize, attempt: usize) -> u64 {
    base_seed + holdout_number as u64 * 10_000 + attempt as u64
}


#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use proptest::prelude::*;
    use proptest::test_runner::TestCaseError;

    use super::*;
    use crate::data::ELEMENT_COUNT;

    const BIN_SIZE: usize = 1;
    const TEST_CLASS_INDEX: usize = 0;

    fn sample(id: usize, class_present: bool) -> SpectrumSample {
        let mut element_present = [false; ELEMENT_COUNT];
        if let Some(present) = element_present.get_mut(TEST_CLASS_INDEX) {
            *present = class_present;
        }
        SpectrumSample::new(vec![id_as_f64(id)], element_present)
    }

    #[allow(clippy::cast_precision_loss, reason = "Test samples")]
    fn id_as_f64(id: usize) -> f64 {
        id as f64
    }
    fn sample_id(sample: &SpectrumSample) -> usize {
        sample.spectra().first().copied().unwrap_or_default() as usize
    }
}