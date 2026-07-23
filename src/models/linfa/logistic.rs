use linfa::traits::Fit;
use linfa_logistic::{FittedLogisticRegression, LogisticRegression};
use rand::{SeedableRng, rngs::ChaCha8Rng, seq::SliceRandom};
use std::{fmt::Write, fs, path::Path};

use crate::{
    domain::{elements::ELEMENTS, sample::SpectrumSample},
    error::Ms2AtomsError,
    evaluation::prediction::PredictionMatrix,
    experiment::progress::Reporter,
    holdout::Holdout,
    models::linfa::{
        config::LinfaLogisticConfig,
        dataset::{binary_dataset_from_indices, binary_targets, feature_matrix},
    },
};

/// Trained one-vs-rest Linfa logistic-regression model.
pub struct TrainedLinfaLogisticModel {
    classifiers: Vec<AtomLogisticClassifier>,
    class_indices: Vec<usize>,
}

/// One binary classifier for one target atom.
pub enum AtomLogisticClassifier {
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
            (0, _) => AtomLogisticClassifier::Constant {
                class_index,
                probability: 0.0,
                original_positives: selection.original_positives,
                original_negatives: selection.original_negatives,
            },
            (_, 0) => AtomLogisticClassifier::Constant {
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
                    .alpha(config.alpha)
                    .fit(&dataset)
                    .map_err(Ms2AtomsError::model_training)?;

                AtomLogisticClassifier::Fitted {
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

    Ok(TrainedLinfaLogisticModel {
        classifiers,
        class_indices: class_indices.to_vec(),
    })
}

/// Selects all positives and a deterministic capped sample of negatives for one binary fit.
fn selected_training_indices(
    samples: &[SpectrumSample],
    class_index: usize,
    config: &LinfaLogisticConfig,
) -> Result<SelectedTrainingIndices, Ms2AtomsError> {
    let targets = binary_targets(samples, class_index)?;
    let mut positive_indices = Vec::new();
    let mut negative_indices = Vec::new();

    for (sample_index, target) in targets.iter().enumerate() {
        if *target == 1 {
            positive_indices.push(sample_index);
        } else {
            negative_indices.push(sample_index);
        }
    }

    let original_positives = positive_indices.len();
    let original_negatives = negative_indices.len();
    let max_negatives = max_negative_count(original_positives, original_negatives, config);
    let used_negatives = max_negatives.min(original_negatives);

    let mut rng = ChaCha8Rng::seed_from_u64(class_seed(config.random_seed, class_index)?);
    negative_indices.shuffle(&mut rng);

    let mut indices = Vec::with_capacity(original_positives.saturating_add(used_negatives));
    indices.extend(positive_indices);
    indices.extend(negative_indices.into_iter().take(used_negatives));
    indices.shuffle(&mut rng);

    Ok(SelectedTrainingIndices {
        indices,
        original_positives,
        original_negatives,
        used_positives: original_positives,
        used_negatives,
    })
}

fn max_negative_count(positives: usize, negatives: usize, config: &LinfaLogisticConfig) -> usize {
    if positives == 0 || negatives == 0 {
        return negatives;
    }

    let ratio_limit = config
        .max_negative_to_positive_ratio
        .map_or(negatives, |ratio| positives.saturating_mul(ratio));
    let cap_limit = config.max_negative_samples.unwrap_or(negatives);

    ratio_limit.min(cap_limit).max(1).min(negatives)
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
    let mut scores = vec![Vec::with_capacity(trained_model.class_indices.len()); samples.len()];
    let total_classifiers = trained_model.classifiers.len();

    progress.report_step(0, total_classifiers, "starting prediction")?;

    for (position, classifier) in trained_model.classifiers.iter().enumerate() {
        let step_number = position + 1;

        match classifier {
            AtomLogisticClassifier::Fitted {
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

                for (row, probability) in scores.iter_mut().zip(probabilities.iter()) {
                    row.push(element_presence_probability(fitted_model, *probability)?);
                }
            }
            AtomLogisticClassifier::Constant {
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

    PredictionMatrix::new(trained_model.class_indices.clone(), scores)
}

/// Writes a lightweight human-readable summary of the logistic baseline artifacts.
fn write_training_summary(
    model: &TrainedLinfaLogisticModel,
    artifact_dir: &Path,
) -> Result<(), Ms2AtomsError> {
    let mut summary = String::from("Linfa one-vs-rest logistic regression\n");
    for classifier in &model.classifiers {
        match classifier {
            AtomLogisticClassifier::Fitted {
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
            AtomLogisticClassifier::Constant {
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

/// Helper to refine probability of element present
fn element_presence_probability(
    model: &FittedLogisticRegression<f64, usize>,
    probability: f64,
) -> Result<f64, Ms2AtomsError> {
    let labels = model.labels();

    match (labels.pos.class, labels.neg.class) {
        // Label is already correct, the element is present
        (1,0) => Ok(probability),
        // Label shows the element is absent
        (0,1) => Ok(1.0 - probability),
        // anything else isn't binary and should be an error
        (positive_label, negative_label) => Err(Ms2AtomsError::ModelInference(
            format!("expected binary labels of `0` and `1`, found labels: positive {positive_label} and negative: {negative_label}")
        ))

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
        let features = array![
            [0.0],
            [0.5],
            [1.0],
            [1.5],
        ];

        let dataset = Dataset::new(features, ndarray::Array::from_iter(targets));

        LogisticRegression::default()
            .max_iterations(100)
            .alpha(1.0)
            .fit(&dataset)
            .map_err(Ms2AtomsError::model_training)
    }

    #[test]
    fn preserves_probability_when_present_is_positive(
    ) -> Result<(), Ms2AtomsError> {
        let model = fitted_test_model([0,1,1,1])?;

        assert_eq!(model.labels().pos.class, 1);
        assert_eq!(model.labels().neg.class, 0);

        let input_probability = 0.8;
        let probability =
            element_presence_probability(&model, input_probability)?;

        assert_eq!(probability, input_probability);

        Ok(())
    }

    #[test]
fn inverts_probability_when_absent_is_positive(
) -> Result<(), Ms2AtomsError> {
    let model = fitted_test_model([0, 0, 0, 1])?;

    assert_eq!(model.labels().pos.class, 0);
    assert_eq!(model.labels().neg.class, 1);

    let probability = element_presence_probability(&model, 0.8)?;

    assert!((probability - 0.2).abs() < f64::EPSILON);

    Ok(())
}
}