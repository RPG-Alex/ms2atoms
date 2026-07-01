use linfa::DatasetBase;
use ndarray::{Array1, Array2};

use crate::{
    domain::{elements::ELEMENT_COUNT, sample::SpectrumSample},
    error::Ms2AtomsError,
};

/// Linfa dataset type used by the binary logistic-regression baseline.
pub type BinaryDataset = DatasetBase<Array2<f64>, Array1<usize>>;

/// Builds a feature matrix with shape `[n_samples, n_features]`.
///
/// # Errors
/// Returns [`Ms2AtomsError`] if the samples are empty or have inconsistent feature widths.
pub fn feature_matrix(samples: &[SpectrumSample]) -> Result<Array2<f64>, Ms2AtomsError> {
    let Some(first_sample) = samples.first() else {
        return Err(Ms2AtomsError::EmptyDataset {
            context: "Linfa feature matrix".to_owned(),
        });
    };

    let n_samples = samples.len();
    let n_features = first_sample.spectra().len();

    let capacity = n_samples
        .checked_mul(n_features)
        .ok_or(Ms2AtomsError::InvalidArray)?;

    let mut values = Vec::with_capacity(capacity);

    for sample in samples {
        if sample.spectra().len() != n_features {
            return Err(Ms2AtomsError::InconsistentFeatureLength {
                expected: n_features,
                actual: sample.spectra().len(),
            });
        }

        values.extend(sample.spectra().iter().copied());
    }

    Array2::from_shape_vec((n_samples, n_features), values).map_err(|_| Ms2AtomsError::InvalidArray)
}

/// Builds binary targets for one element class.
///
/// # Errors
/// Returns [`Ms2AtomsError`] if the class index is invalid.
pub fn binary_targets(
    samples: &[SpectrumSample],
    class_index: usize,
) -> Result<Array1<usize>, Ms2AtomsError> {
    if class_index >= ELEMENT_COUNT {
        return Err(Ms2AtomsError::InvalidClassIndex { class_index });
    }

    let targets = samples
        .iter()
        .map(|sample| {
            sample
                .is_element_present(class_index)
                .map(usize::from)
                .ok_or(Ms2AtomsError::InvalidClassIndex { class_index })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Array1::from(targets))
}

/// Builds a Linfa binary dataset for one element class.
///
/// # Errors
/// Returns [`Ms2AtomsError`] if feature extraction or target extraction fails.
pub fn binary_dataset(
    samples: &[SpectrumSample],
    class_index: usize,
) -> Result<BinaryDataset, Ms2AtomsError> {
    Ok(DatasetBase::new(
        feature_matrix(samples)?,
        binary_targets(samples, class_index)?,
    ))
}
