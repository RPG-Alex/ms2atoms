use linfa::traits::Fit;
use linfa_logistic::{FittedLogisticRegression, LogisticRegression};
use std::fmt::Write;
use std::{fs, path::Path};

use crate::{
    error::Ms2AtomsError,
    evaluation::prediction::PredictionMatrix,
    holdout::Holdout,
    models::linfa::{
        config::LinfaLogisticConfig,
        dataset::{binary_dataset, binary_targets, feature_matrix},
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
        /// Fitted Linfa model.
        model: FittedLogisticRegression<f64, usize>,
    },
    /// Fallback classifier used when a split contains only positives or only negatives.
    Constant {
        /// Element class index handled by this classifier.
        class_index: usize,
        /// Constant probability returned for every validation item.
        probability: f64,
    },
}

/// Trains one binary logistic classifier per holdout class and predicts validation scores.
///
/// # Parameters
/// - `config` - Linfa logistic-regression configuration.
/// - `holdout` - Holdout split used for training and validation.
/// - `artifact_dir` - Directory where a small training summary will be written.
///
/// # Errors
/// Returns [`Ms2AtomsError`] if feature construction, model fitting, or artifact writing fails.
pub fn train_and_predict<H>(
    config: &LinfaLogisticConfig,
    holdout: &H,
    artifact_dir: &Path,
) -> Result<PredictionMatrix, Ms2AtomsError>
where
    H: Holdout,
{
    fs::create_dir_all(artifact_dir)?;

    let model = train(
        config,
        holdout.train_dataset().samples(),
        holdout.class_indices(),
    )?;
    write_training_summary(&model, artifact_dir)?;

    predict(&model, holdout.validation_dataset().samples())
}

/// Trains one binary classifier for each selected element class.
fn train(
    config: &LinfaLogisticConfig,
    samples: &[crate::domain::sample::SpectrumSample],
    class_indices: &[usize],
) -> Result<TrainedLinfaLogisticModel, Ms2AtomsError> {
    let mut classifiers = Vec::with_capacity(class_indices.len());

    for &class_index in class_indices {
        let targets = binary_targets(samples, class_index)?;
        let positives = targets.iter().filter(|&&target| target == 1).count();
        let negatives = targets.len().saturating_sub(positives);

        let classifier = match (positives, negatives) {
            (0, _) => AtomLogisticClassifier::Constant {
                class_index,
                probability: 0.0,
            },
            (_, 0) => AtomLogisticClassifier::Constant {
                class_index,
                probability: 1.0,
            },
            _ => {
                let dataset = binary_dataset(samples, class_index)?;
                let model = LogisticRegression::default()
                    .max_iterations(config.max_iterations)
                    .alpha(config.alpha)
                    .fit(&dataset)
                    .map_err(Ms2AtomsError::model_training)?;

                AtomLogisticClassifier::Fitted { class_index, model }
            }
        };

        classifiers.push(classifier);
    }

    Ok(TrainedLinfaLogisticModel {
        classifiers,
        class_indices: class_indices.to_vec(),
    })
}

/// Predicts per-class probabilities for validation samples.
fn predict(
    model: &TrainedLinfaLogisticModel,
    samples: &[crate::domain::sample::SpectrumSample],
) -> Result<PredictionMatrix, Ms2AtomsError> {
    let features = feature_matrix(samples)?;
    let mut scores = vec![Vec::with_capacity(model.class_indices.len()); samples.len()];

    for classifier in &model.classifiers {
        match classifier {
            AtomLogisticClassifier::Fitted { model, .. } => {
                let probabilities = model.predict_probabilities(&features);

                for (row, probability) in scores.iter_mut().zip(probabilities.iter()) {
                    row.push(*probability);
                }
            }
            AtomLogisticClassifier::Constant { probability, .. } => {
                for row in &mut scores {
                    row.push(*probability);
                }
            }
        }
    }

    PredictionMatrix::new(model.class_indices.clone(), scores)
}

/// Writes a lightweight human-readable summary of the logistic baseline artifacts.
fn write_training_summary(
    model: &TrainedLinfaLogisticModel,
    artifact_dir: &Path,
) -> Result<(), Ms2AtomsError> {
    let mut summary = String::from("Linfa one-vs-rest logistic regression\n");

    for classifier in &model.classifiers {
        match classifier {
            AtomLogisticClassifier::Fitted { class_index, .. } => {
                let _ = writeln!(summary, "class_index={class_index}: fitted\n");
            }
            AtomLogisticClassifier::Constant {
                class_index,
                probability,
            } => {
                let _ = writeln!(
                    summary,
                    "class_index={class_index}: constant_probability={probability}\n"
                );
            }
        }
    }

    fs::write(artifact_dir.join("model_summary.txt"), summary)?;
    Ok(())
}
