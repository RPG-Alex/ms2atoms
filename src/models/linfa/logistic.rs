use linfa::traits::Fit;
use linfa_logistic::{FittedLogisticRegression, LogisticRegression};
use rand::{SeedableRng, rngs::ChaCha8Rng, seq::IndexedRandom};
use std::{fmt::Write, fs, path::Path};

use crate::{
    domain::{elements::ELEMENTS, sample::SpectrumSample},
    error::Ms2AtomsError,
    evaluation::prediction::PredictionMatrix,
    experiment::progress::Reporter,
    holdout::Holdout,
    models::linfa::{
        config::LinfaLogisticConfig,
        dataset::{binary_dataset_from_indices, feature_matrix},
    },
};

/// Trained one-vs-rest Linfa logistic-regression model.
pub(crate) struct TrainedLinfaLogisticModel {
    classifiers: Vec<ElementLogisticClassifier>,
}

impl ElementLogisticClassifier {
    const fn class_index(&self) -> usize {
        match self {
            Self::Fitted { class_index, .. } | Self::Constant { class_index, .. } => *class_index,
        }
    }
}

/// One binary classifier for one target element.
pub(crate) enum ElementLogisticClassifier {
    /// A fitted Linfa binary logistic-regression model.
    Fitted {
        /// Element class index handled by this classifier.
        class_index: usize,
        /// Number of positive training samples used for fitting.
        used_positives: usize,
        /// Number of negative training samples used for fitting.
        used_negatives: usize,
        /// Original number of positive training samples before any balancing.
        original_positives: usize,
        /// Original number of negative training samples before any balancing.
        original_negatives: usize,
        /// Fitted Linfa model.
        model: FittedLogisticRegression<f64, usize>,
    },
    /// Fallback classifier used when a split contains only positives or only negatives.
    Constant {
        /// Element class index handled by this classifier.
        class_index: usize,
        /// Constant probability returned for every validation item.
        probability: f64,
        /// Original number of positive training samples.
        original_positives: usize,
        /// Original number of negative training samples.
        original_negatives: usize,
    },
}

struct SelectedTrainingIndices {
    indices: Vec<usize>,
    original_positives: usize,
    original_negatives: usize,
    used_positives: usize,
    used_negatives: usize,
}

/// Trains one binary logistic classifier per holdout class and predicts validation scores.
///
/// # Parameters
/// - `config` - Linfa logistic-regression configuration.
/// - `holdout` - Holdout split used for training and validation.
/// - `artifact_dir` - Directory where a small training summary will be written.
/// - `progress` - Progress reporter for non-Burn model status updates.
///
/// # Errors
/// Returns [`Ms2AtomsError`] if feature construction, model fitting, or artifact writing fails.
pub(crate) fn train_and_predict<H>(
    config: &LinfaLogisticConfig,
    holdout: &H,
    artifact_dir: &Path,
    progress: &mut dyn Reporter,
) -> Result<PredictionMatrix, Ms2AtomsError>
where
    H: Holdout,
{
    fs::create_dir_all(artifact_dir)?;

    progress.report("preparing Linfa one-vs-rest logistic regression")?;

    let model = train(
        config,
        holdout.train_dataset().samples(),
        holdout.class_indices(),
        progress,
    )?;
    write_training_summary(&model, artifact_dir)?;

    let predictions = predict(&model, holdout.validation_dataset().samples(), progress)?;
    progress.finish("Linfa logistic training and inference complete")?;

    Ok(predictions)
}

/// Trains one binary classifier for each selected element class.
fn train(
    config: &LinfaLogisticConfig,
    samples: &[SpectrumSample],
    class_indices: &[usize],
    progress: &mut dyn Reporter,
) -> Result<TrainedLinfaLogisticModel, Ms2AtomsError> {
    let mut classifiers = Vec::with_capacity(class_indices.len());

    progress.report_step(0, class_indices.len(), "starting classifier training")?;

    for (position, &class_index) in class_indices.iter().enumerate() {
        let step_number = position + 1;
        let class_label = class_label(class_index);
        let selection = selected_training_indices(samples, class_index, config)?;

        let class_summary = format!(
            "class_index={class_index} ({class_label}), positives={}, negatives={}",
            selection.original_positives, selection.original_negatives
        );
        progress.report_step(step_number, class_indices.len(), &class_summary)?;

        let classifier = match (selection.original_positives, selection.original_negatives) {
            (0, _) => ElementLogisticClassifier::Constant {
                class_index,
                probability: 0.0,
                original_positives: selection.original_positives,
                original_negatives: selection.original_negatives,
            },
            (_, 0) => ElementLogisticClassifier::Constant {
                class_index,
                probability: 1.0,
                original_positives: selection.original_positives,
                original_negatives: selection.original_negatives,
            },
            _ => {
                let fit_summary = format!(
                    "fitting class_index={class_index} ({class_label}) with {} positives \
                     and {} negatives; original negatives={}",
                    selection.used_positives,
                    selection.used_negatives,
                    selection.original_negatives
                );
                progress.report_step(step_number, class_indices.len(), &fit_summary)?;

                let dataset =
                    binary_dataset_from_indices(samples, class_index, &selection.indices)?;
                let model = LogisticRegression::default()
                    .max_iterations(config.max_iterations)
                    .gradient_tolerance(config.gradient_tolerance)
                    .alpha(config.alpha)
                    .fit(&dataset)
                    .map_err(Ms2AtomsError::model_training)?;

                ElementLogisticClassifier::Fitted {
                    class_index,
                    used_positives: selection.used_positives,
                    used_negatives: selection.used_negatives,
                    original_positives: selection.original_positives,
                    original_negatives: selection.original_negatives,
                    model,
                }
            }
        };

        let done_summary = format!("finished {class_summary}");
        progress.report_step(step_number, class_indices.len(), &done_summary)?;
        classifiers.push(classifier);
    }

    Ok(TrainedLinfaLogisticModel { classifiers })
}

/// Selects all positives and a deterministic capped sample of negatives for one binary fit.
fn selected_training_indices(
    samples: &[SpectrumSample],
    class_index: usize,
    config: &LinfaLogisticConfig,
) -> Result<SelectedTrainingIndices, Ms2AtomsError> {
    let mut positive_indices = Vec::new();
    let mut negative_indices = Vec::new();

    for (sample_index, sample) in samples.iter().enumerate() {
        match sample.is_element_present(class_index) {
            Some(true) => positive_indices.push(sample_index),
            Some(false) => negative_indices.push(sample_index),
            None => return Err(Ms2AtomsError::InvalidClassIndex { class_index }),
        }
    }

    let original_positives = positive_indices.len();
    let original_negatives = negative_indices.len();

    if original_positives == 0 || original_negatives == 0 {
        return Ok(SelectedTrainingIndices {
            indices: Vec::new(),
            original_positives,
            original_negatives,
            used_positives: original_positives,
            used_negatives: original_negatives,
        });
    }

    let (used_positives, used_negatives) =
        selected_class_counts(original_positives, original_negatives, config);

    let mut rng = ChaCha8Rng::seed_from_u64(class_seed(config.random_seed, class_index)?);

    let mut indices = Vec::with_capacity(used_positives.saturating_add(used_negatives));

    indices.extend(select_indices(&positive_indices, used_positives, &mut rng));

    indices.extend(select_indices(&negative_indices, used_negatives, &mut rng));

    Ok(SelectedTrainingIndices {
        indices,
        original_positives,
        original_negatives,
        used_positives,
        used_negatives,
    })
}

/// Returns a tuple of positive and negative samples (p,n)
fn selected_class_counts(
    positives: usize,
    negatives: usize,
    config: &LinfaLogisticConfig,
) -> (usize, usize) {
    if positives == 0 || negatives == 0 {
        return (positives, negatives);
    }
    let sample_cap = config.max_samples_per_class.unwrap_or(usize::MAX);

    let mut used_positives = positives.min(sample_cap);
    let mut used_negatives = negatives.min(sample_cap);

    if let Some(ratio) = config.max_majority_to_minority_ratio {
        if used_positives > used_negatives {
            used_positives = used_positives.min(used_negatives.saturating_mul(ratio));
        } else {
            used_negatives = used_negatives.min(used_positives.saturating_mul(ratio));
        }
    }
    (used_positives.max(1), used_negatives.max(1))
}

fn class_seed(base_seed: u64, class_index: usize) -> Result<u64, Ms2AtomsError> {
    let class_index =
        u64::try_from(class_index).map_err(|_| Ms2AtomsError::InvalidClassIndex { class_index })?;

    Ok(base_seed ^ class_index.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn class_label(class_index: usize) -> String {
    ELEMENTS
        .get(class_index)
        .map_or_else(|| "unknown".to_owned(), |element| format!("{element:?}"))
}

/// Predicts per-class probabilities for validation samples.
fn predict(
    trained_model: &TrainedLinfaLogisticModel,
    samples: &[SpectrumSample],
    progress: &mut dyn Reporter,
) -> Result<PredictionMatrix, Ms2AtomsError> {
    progress.report("building validation feature matrix")?;
    let features = feature_matrix(samples)?;
    let total_classifiers = trained_model.classifiers.len();
    let mut scores = vec![Vec::with_capacity(total_classifiers); samples.len()];

    progress.report_step(0, total_classifiers, "starting prediction")?;

    for (position, classifier) in trained_model.classifiers.iter().enumerate() {
        let step_number = position + 1;

        match classifier {
            ElementLogisticClassifier::Fitted {
                class_index,
                model: fitted_model,
                ..
            } => {
                let summary = format!(
                    "predicting class_index={class_index} ({})",
                    class_label(*class_index)
                );
                progress.report_step(step_number, total_classifiers, &summary)?;

                let probabilities = fitted_model.predict_probabilities(&features);
                let invert_probability = should_invert_probability(fitted_model)?;

                for (row, probability) in scores.iter_mut().zip(probabilities.iter()) {
                    let presence_probability = if invert_probability {
                        1.0 - *probability
                    } else {
                        *probability
                    };

                    row.push(presence_probability);
                }
            }
            ElementLogisticClassifier::Constant {
                class_index,
                probability,
                ..
            } => {
                let summary = format!(
                    "using constant prediction for class_index={class_index} ({})",
                    class_label(*class_index)
                );
                progress.report_step(step_number, total_classifiers, &summary)?;

                for row in &mut scores {
                    row.push(*probability);
                }
            }
        }
    }
    let class_indices = trained_model
        .classifiers
        .iter()
        .map(ElementLogisticClassifier::class_index)
        .collect();
    PredictionMatrix::new(class_indices, scores)
}

/// Writes a lightweight human-readable summary of the logistic baseline artifacts.
fn write_training_summary(
    model: &TrainedLinfaLogisticModel,
    artifact_dir: &Path,
) -> Result<(), Ms2AtomsError> {
    let mut summary = String::from("Linfa one-vs-rest logistic regression\n");
    for classifier in &model.classifiers {
        match classifier {
            ElementLogisticClassifier::Fitted {
                class_index,
                used_positives,
                used_negatives,
                original_positives,
                original_negatives,
                ..
            } => {
                writeln!(
                    summary,
                    "class_index={class_index} ({}): fitted, \
                     used_positives={used_positives}, used_negatives={used_negatives}, \
                     original_positives={original_positives}, \
                     original_negatives={original_negatives}",
                    class_label(*class_index)
                )
                .map_err(Ms2AtomsError::model_artifact)?;
            }
            ElementLogisticClassifier::Constant {
                class_index,
                probability,
                original_positives,
                original_negatives,
            } => {
                writeln!(
                    summary,
                    "class_index={class_index} ({}): constant_probability={probability}, \
                     original_positives={original_positives}, \
                     original_negatives={original_negatives}",
                    class_label(*class_index)
                )
                .map_err(Ms2AtomsError::model_artifact)?;
            }
        }
    }

    fs::write(artifact_dir.join("model_summary.txt"), summary)?;
    Ok(())
}

/// Returns whether Linfa's probability must be inverted to represent
fn should_invert_probability(
    model: &FittedLogisticRegression<f64, usize>,
) -> Result<bool, Ms2AtomsError> {
    let labels = model.labels();

    match (labels.pos.class, labels.neg.class) {
        (1, 0) => Ok(false),
        (0, 1) => Ok(true),
        (positive_label, negative_label) => Err(Ms2AtomsError::ModelInference(format!(
            "expected binary labels `0` and `1`, found positive \
                 label {positive_label} and negative label {negative_label}"
        ))),
    }
}

fn select_indices(indices: &[usize], amount: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
    if amount >= indices.len() {
        indices.to_vec()
    } else {
        indices.sample(rng, amount).copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use linfa::{Dataset, traits::Fit};
    use ndarray::array;

    use super::*;

    fn fitted_test_model(
        targets: [usize; 4],
    ) -> Result<FittedLogisticRegression<f64, usize>, Ms2AtomsError> {
        let features = array![[0.0], [0.5], [1.0], [1.5],];

        let dataset = Dataset::new(features, ndarray::Array::from_iter(targets));

        LogisticRegression::default()
            .max_iterations(100)
            .alpha(1.0)
            .fit(&dataset)
            .map_err(Ms2AtomsError::model_training)
    }
    #[test]
    fn does_not_invert_when_present_is_positive() -> Result<(), Ms2AtomsError> {
        let model = fitted_test_model([0, 1, 1, 1])?;

        assert!(!should_invert_probability(&model)?);

        Ok(())
    }

    #[test]
    fn inverts_when_absent_is_positive() -> Result<(), Ms2AtomsError> {
        let model = fitted_test_model([0, 0, 0, 1])?;

        assert!(should_invert_probability(&model)?);

        Ok(())
    }

    #[test]
    fn caps_positive_majority_for_common_element() {
        let config = LinfaLogisticConfig {
            max_iterations: 100,
            gradient_tolerance: 1e-4,
            alpha: 1.0,
            random_seed: 42,
            max_majority_to_minority_ratio: Some(50),
            max_samples_per_class: Some(50_000),
        };

        let counts = selected_class_counts(355_195, 5, &config);

        assert_eq!(counts, (250, 5));
    }

    #[test]
    fn caps_negative_majority_for_rare_element() {
        let config = LinfaLogisticConfig {
            max_iterations: 100,
            gradient_tolerance: 1e-4,
            alpha: 1.0,
            random_seed: 42,
            max_majority_to_minority_ratio: Some(50),
            max_samples_per_class: Some(50_000),
        };

        let counts = selected_class_counts(100, 300_000, &config);

        assert_eq!(counts, (100, 5_000));
    }

    use crate::domain::elements::ELEMENT_COUNT;

    fn test_config() -> LinfaLogisticConfig {
        LinfaLogisticConfig {
            max_iterations: 100,
            gradient_tolerance: 1e-4,
            alpha: 1.0,
            random_seed: 42,
            max_majority_to_minority_ratio: Some(5),
            max_samples_per_class: Some(50_000),
        }
    }

    fn samples_for_class(
        class_index: usize,
        positives: usize,
        negatives: usize,
    ) -> Result<Vec<SpectrumSample>, Ms2AtomsError> {
        let mut samples = Vec::with_capacity(positives.saturating_add(negatives));

        for _ in 0..positives {
            let mut elements = [false; ELEMENT_COUNT];

            let Some(element) = elements.get_mut(class_index) else {
                return Err(Ms2AtomsError::InvalidClassIndex { class_index });
            };

            *element = true;

            samples.push(SpectrumSample::new(
                "positive".to_owned(),
                vec![0.0],
                elements,
            ));
        }

        for _ in 0..negatives {
            samples.push(SpectrumSample::new(
                "negative".to_owned(),
                vec![0.0],
                [false; ELEMENT_COUNT],
            ));
        }

        Ok(samples)
    }

    #[test]
    fn constant_positive_class_selects_zero_fitting_indices() -> Result<(), Ms2AtomsError> {
        let class_index = 0;
        let samples = samples_for_class(class_index, 10, 0)?;

        let selection = selected_training_indices(&samples, class_index, &test_config())?;

        assert_eq!(selection.indices, [] as [usize; 0]);
        assert_eq!(selection.original_positives, 10);
        assert_eq!(selection.original_negatives, 0);
        assert_eq!(selection.used_positives, 10);
        assert_eq!(selection.used_negatives, 0);

        Ok(())
    }

    #[test]
    fn constant_negative_class_selects_zero_fitting_indices() -> Result<(), Ms2AtomsError> {
        let class_index = 0;
        let samples = samples_for_class(class_index, 0, 10)?;

        let selection = selected_training_indices(&samples, class_index, &test_config())?;

        assert_eq!(selection.indices, [] as [usize; 0]);
        assert_eq!(selection.original_positives, 0);
        assert_eq!(selection.original_negatives, 10);
        assert_eq!(selection.used_positives, 0);
        assert_eq!(selection.used_negatives, 10);

        Ok(())
    }

    #[test]
    fn selection_is_deterministic_for_same_seed() -> Result<(), Ms2AtomsError> {
        let class_index = 0;
        let samples = samples_for_class(class_index, 20, 100)?;

        let config = LinfaLogisticConfig {
            max_majority_to_minority_ratio: Some(2),
            max_samples_per_class: Some(10),
            ..test_config()
        };

        let first = selected_training_indices(&samples, class_index, &config)?;

        let second = selected_training_indices(&samples, class_index, &config)?;

        assert_eq!(first.indices, second.indices);

        Ok(())
    }

    #[test]
    fn selected_indices_contain_exact_pos_neg_counts() -> Result<(), Ms2AtomsError> {
        let class_index = 0;
        let samples = samples_for_class(class_index, 4, 20)?;

        let config = LinfaLogisticConfig {
            max_majority_to_minority_ratio: Some(2),
            max_samples_per_class: None,
            ..test_config()
        };

        let selection = selected_training_indices(&samples, class_index, &config)?;

        let mut positives = 0;
        let mut negatives = 0;

        for sample_index in &selection.indices {
            let sample = samples
                .get(*sample_index)
                .ok_or(Ms2AtomsError::InvalidArray)?;

            match sample.is_element_present(class_index) {
                Some(true) => positives += 1,
                Some(false) => negatives += 1,
                None => return Err(Ms2AtomsError::InvalidClassIndex { class_index }),
            }
        }

        assert_eq!(positives, 4);
        assert_eq!(negatives, 8);
        assert_eq!(selection.indices.len(), 12);

        Ok(())
    }
}
