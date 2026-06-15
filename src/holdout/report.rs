use serde::Serialize;

use crate::{
    data::{ELEMENTS, SpectrumSample},
    error::SpectraError,
    holdout::Holdout,
};

#[derive(Debug, Serialize)]
/// Per-element class-balance report for one holdout split.
pub struct ClassDistribution {
    /// Index of the element class in `crate::data::ELEMENTS`.
    pub class_index: usize,
    /// Chemical symbol for the element class.
    pub element: String,
    /// Number of positive samples in the training partition.
    pub train_positive: usize,
    /// Number of positive samples in the validation partition.
    pub validation_positive: usize,
    /// Total number of positive samples across training and validation.
    pub total_positive: usize,
    /// Fraction of positive samples assigned to the training partition.
    pub train_fraction_of_positives: f64,
    /// Fraction of positive samples assigned to the validation partition.
    pub validation_fraction_of_positives: f64,
    /// Warning string describing potentially problematic class balance.
    pub warning: String,
}

/// Builds a per-class distribution report for a holdout split.
///
/// # Parameters
/// - `holdout` - Holdout split to inspect for class-balance reporting.
///
/// # Errors
/// - Returns [`SpectraError`] if a holdout class index is invalid.
pub fn class_distribution_report<H: Holdout>(
    holdout: &H,
) -> Result<Vec<ClassDistribution>, SpectraError> {
    holdout
        .class_indices()
        .iter()
        .map(|&class_index| {
            let element = element_symbol(class_index)?;

            let train_positive =
                count_positive_samples(holdout.train_dataset().samples(), class_index)?;

            let validation_positive =
                count_positive_samples(holdout.validation_dataset().samples(), class_index)?;

            let total_positive = train_positive + validation_positive;
            let train_fraction_of_positives = fraction(train_positive, total_positive);
            let validation_fraction_of_positives = fraction(validation_positive, total_positive);

            let warning = match (train_positive, validation_positive, total_positive) {
                (0, _, _) => "NO_TRAIN_POSITIVES".to_owned(),
                (_, 0, total) if total > 1 => "NO_VALIDATION_POSITIVES".to_owned(),
                _ => String::new(),
            };

            Ok(ClassDistribution {
                class_index,
                element,
                train_positive,
                validation_positive,
                total_positive,
                train_fraction_of_positives,
                validation_fraction_of_positives,
                warning,
            })
        })
        .collect()
}

/// Returns the chemical symbol for one element class index.
fn element_symbol(class_index: usize) -> Result<String, SpectraError> {
    Ok(ELEMENTS
        .get(class_index)
        .ok_or(SpectraError::InvalidArray)?
        .symbol()
        .to_owned())
}

/// Counts samples where the selected element class is present.
fn count_positive_samples(
    samples: &[SpectrumSample],
    class_index: usize,
) -> Result<usize, SpectraError> {
    samples.iter().try_fold(0, |count, sample| {
        let is_present = sample
            .is_element_present(class_index)
            .ok_or(SpectraError::InvalidArray)?;

        Ok(if is_present { count + 1 } else { count })
    })
}

/// Returns `part / total`, or `0.0` when `total` is zero.
#[allow(
    clippy::cast_precision_loss,
    reason = "f64 casting loss is not a concern here"
)]
fn fraction(part: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 / total as f64
    }
}
