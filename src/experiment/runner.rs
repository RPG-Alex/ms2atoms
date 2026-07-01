use std::{
    collections::BTreeMap,
    fs,
    fs::File,
    path::{Path, PathBuf},
};

use csv::Writer;
use serde::Serialize;
use tracing::info;

use crate::{
    dataset::SpectraData,
    domain::sample::SpectrumSample,
    error::Ms2AtomsError,
    evaluation::{
        PredictionMatrix, aggregate_metrics, confusion::ConfusionMatrix, create_confusion_matrices,
        element_metrics_from_matrices, metrics::AggregateMetrics,
    },
    experiment::{ExperimentConfig, ExperimentProtocol},
    holdout::{Holdout, class_distribution_report},
    models::{self, spec::ModelSpec},
};

type ExperimentConfusions = BTreeMap<String, Vec<ConfusionMatrix>>;

/// Runs one configured experiment, trains every holdout, and writes artifact and CSV reports.
///
/// # Parameters
/// - `config` - Experiment configuration to execute.
///
/// # Errors
/// Returns [`Ms2AtomsError`] if loading, training, inference, or report generation fails.
pub fn run_experiment<P>(config: &ExperimentConfig<P>) -> Result<(), Ms2AtomsError>
where
    P: ExperimentProtocol,
{
    let experiment_name = experiment_slug(config.run.experiment_num, &config.run.name);
    let paths = prepare_experiment_dirs(&experiment_name)?;

    info!("Loading spectra for experiment: {experiment_name}");
    let dataset = SpectraData::new(config.features.bin_size)?;
    info!("Spectra loaded.");

    let holdouts = config.protocol.generate_holdouts(&dataset);
    let mut holdout_aggregate_rows = Vec::new();
    let mut experiment_confusions_by_threshold = ExperimentConfusions::new();

    for holdout in holdouts {
        run_holdout(
            config,
            &experiment_name,
            &paths,
            &holdout,
            &mut holdout_aggregate_rows,
            &mut experiment_confusions_by_threshold,
        )?;
    }

    write_csv(
        paths
            .summary_results
            .join("aggregate_metrics_by_holdout.csv"),
        &holdout_aggregate_rows,
    )?;

    write_experiment_summary(
        &experiment_name,
        &paths.summary_results,
        experiment_confusions_by_threshold,
    )?;

    Ok(())
}

struct ExperimentPaths {
    artifact: PathBuf,
    holdout_results: PathBuf,
    summary_results: PathBuf,
}

fn prepare_experiment_dirs(experiment_name: &str) -> Result<ExperimentPaths, Ms2AtomsError> {
    let artifact_root = PathBuf::from("./experiments").join(experiment_name);
    let results_root = PathBuf::from("./results").join(experiment_name);
    let holdout_results_root = results_root.join("holdouts");
    let summary_results_root = results_root.join("summary");

    fs::create_dir_all(&artifact_root)?;
    fs::create_dir_all(&holdout_results_root)?;
    fs::create_dir_all(&summary_results_root)?;

    Ok(ExperimentPaths {
        artifact: artifact_root,
        holdout_results: holdout_results_root,
        summary_results: summary_results_root,
    })
}

fn run_holdout<P>(
    config: &ExperimentConfig<P>,
    experiment_name: &str,
    paths: &ExperimentPaths,
    holdout: &P::HoldoutType,
    holdout_aggregate_rows: &mut Vec<HoldoutAggregateMetricsRow>,
    experiment_confusions_by_threshold: &mut ExperimentConfusions,
) -> Result<(), Ms2AtomsError>
where
    P: ExperimentProtocol,
{
    debug_assert_eq!(holdout.num_classes(), holdout.class_indices().len());

    let holdout_number = holdout.holdout_number();
    let holdout_label = format_holdout(holdout_number);
    let holdout_results_dir = paths.holdout_results.join(&holdout_label);
    let artifact_dir = paths.artifact.join(&holdout_label);

    fs::create_dir_all(&holdout_results_dir)?;

    info!(
        "Running {holdout_label} with seed {}: {} training samples, {} validation samples.",
        holdout.random_seed(),
        holdout.training_len(),
        holdout.validation_len(),
    );

    let distribution = class_distribution_report(holdout)?;
    write_csv(
        holdout_results_dir.join("class_distribution.csv"),
        &distribution,
    )?;

    let validation_items = holdout.validation_dataset().samples().to_vec();
    let predictions = train_and_predict_holdout(&config.model, holdout, &artifact_dir)?;

    for threshold in &config.evaluation.thresholds {
        evaluate_threshold(
            &ThresholdEvaluation {
                experiment_name,
                holdout_results_dir: &holdout_results_dir,
                holdout,
                validation_items: &validation_items,
                threshold: *threshold,
            },
            &predictions,
            holdout_aggregate_rows,
            experiment_confusions_by_threshold,
        )?;
    }

    Ok(())
}

struct ThresholdEvaluation<'a, H: Holdout> {
    experiment_name: &'a str,
    holdout_results_dir: &'a Path,
    holdout: &'a H,
    validation_items: &'a [SpectrumSample],
    threshold: f64,
}

fn evaluate_threshold<H>(
    evaluation: &ThresholdEvaluation<'_, H>,
    predictions: &PredictionMatrix,
    holdout_aggregate_rows: &mut Vec<HoldoutAggregateMetricsRow>,
    experiment_confusions_by_threshold: &mut ExperimentConfusions,
) -> Result<(), Ms2AtomsError>
where
    H: Holdout,
{
    let threshold_label = format_threshold(evaluation.threshold);
    let threshold_results_dir = evaluation
        .holdout_results_dir
        .join(format!("threshold_{threshold_label}"));

    fs::create_dir_all(&threshold_results_dir)?;

    let confusion_matrices = create_confusion_matrices(
        predictions,
        evaluation.validation_items,
        evaluation.threshold,
    )?;

    write_csv(
        threshold_results_dir.join("confusion_matrix.csv"),
        &confusion_matrices,
    )?;

    merge_confusion_matrices(
        experiment_confusions_by_threshold
            .entry(threshold_label.clone())
            .or_default(),
        &confusion_matrices,
    );

    let element_metrics = element_metrics_from_matrices(&confusion_matrices);
    write_csv(
        threshold_results_dir.join("element_metrics.csv"),
        &element_metrics,
    )?;

    let aggregate = aggregate_metrics(&element_metrics);
    let row = aggregate_metrics_row(
        evaluation.experiment_name,
        evaluation.holdout.holdout_number(),
        evaluation.holdout.random_seed(),
        evaluation.threshold,
        &threshold_label,
        &aggregate,
    );

    write_csv(
        threshold_results_dir.join("aggregate_metrics.csv"),
        std::slice::from_ref(&row),
    )?;

    holdout_aggregate_rows.push(row);

    Ok(())
}

fn train_and_predict_holdout<H>(
    model: &ModelSpec,
    holdout: &H,
    artifact_dir: &Path,
) -> Result<PredictionMatrix, Ms2AtomsError>
where
    H: Holdout,
{
    match model {
        ModelSpec::BurnMlp(config) => {
            models::burn::training::train_and_predict(config, holdout, artifact_dir)
        }
        ModelSpec::LinfaLogistic(config) => {
            models::linfa::logistic::train_and_predict(config, holdout, artifact_dir)
        }
    }
}

fn write_experiment_summary(
    experiment_name: &str,
    summary_results_root: &Path,
    experiment_confusions_by_threshold: ExperimentConfusions,
) -> Result<(), Ms2AtomsError> {
    let mut experiment_aggregate_rows = Vec::new();

    for (threshold_label, confusion_matrices) in experiment_confusions_by_threshold {
        let threshold_summary_dir =
            summary_results_root.join(format!("threshold_{threshold_label}"));

        fs::create_dir_all(&threshold_summary_dir)?;

        write_csv(
            threshold_summary_dir.join("confusion_matrix.csv"),
            &confusion_matrices,
        )?;

        let element_metrics = element_metrics_from_matrices(&confusion_matrices);
        write_csv(
            threshold_summary_dir.join("element_metrics.csv"),
            &element_metrics,
        )?;

        let aggregate = aggregate_metrics(&element_metrics);
        write_csv(
            threshold_summary_dir.join("aggregate_metrics.csv"),
            &[experiment_aggregate_metrics_row(
                experiment_name,
                &threshold_label,
                &aggregate,
            )],
        )?;

        experiment_aggregate_rows.push(experiment_aggregate_metrics_row(
            experiment_name,
            &threshold_label,
            &aggregate,
        ));
    }

    write_csv(
        summary_results_root.join("aggregate_metrics_by_threshold.csv"),
        &experiment_aggregate_rows,
    )?;

    Ok(())
}

/// One aggregate-metrics row for a specific holdout and threshold.
#[derive(Clone, Debug, Serialize)]
struct HoldoutAggregateMetricsRow {
    experiment: String,
    holdout: usize,
    seed: u64,
    threshold: f64,
    threshold_label: String,
    macro_precision: f64,
    macro_recall: f64,
    macro_f1: f64,
    macro_mcc: f64,
    micro_precision: f64,
    micro_recall: f64,
    micro_f1: f64,
    micro_mcc: f64,
}

/// One aggregate-metrics row for all holdouts at a specific threshold.
#[derive(Debug, Serialize)]
struct ExperimentAggregateMetricsRow {
    experiment: String,
    threshold_label: String,
    macro_precision: f64,
    macro_recall: f64,
    macro_f1: f64,
    macro_mcc: f64,
    micro_precision: f64,
    micro_recall: f64,
    micro_f1: f64,
    micro_mcc: f64,
}

/// Converts one aggregate metric into a CSV row that includes holdout metadata.
fn aggregate_metrics_row(
    experiment: &str,
    holdout: usize,
    seed: u64,
    threshold: f64,
    threshold_label: &str,
    metrics: &AggregateMetrics,
) -> HoldoutAggregateMetricsRow {
    HoldoutAggregateMetricsRow {
        experiment: experiment.to_owned(),
        holdout,
        seed,
        threshold,
        threshold_label: threshold_label.to_owned(),
        macro_precision: metrics.macro_precision,
        macro_recall: metrics.macro_recall,
        macro_f1: metrics.macro_f1,
        macro_mcc: metrics.macro_mcc,
        micro_precision: metrics.micro_precision,
        micro_recall: metrics.micro_recall,
        micro_f1: metrics.micro_f1,
        micro_mcc: metrics.micro_mcc,
    }
}

/// Converts whole-experiment aggregate metrics into a threshold-level CSV row.
fn experiment_aggregate_metrics_row(
    experiment: &str,
    threshold_label: &str,
    metrics: &AggregateMetrics,
) -> ExperimentAggregateMetricsRow {
    ExperimentAggregateMetricsRow {
        experiment: experiment.to_owned(),
        threshold_label: threshold_label.to_owned(),
        macro_precision: metrics.macro_precision,
        macro_recall: metrics.macro_recall,
        macro_f1: metrics.macro_f1,
        macro_mcc: metrics.macro_mcc,
        micro_precision: metrics.micro_precision,
        micro_recall: metrics.micro_recall,
        micro_f1: metrics.micro_f1,
        micro_mcc: metrics.micro_mcc,
    }
}

/// Adds per-holdout confusion matrices into the experiment-level confusion matrix accumulator.
fn merge_confusion_matrices(destination: &mut Vec<ConfusionMatrix>, source: &[ConfusionMatrix]) {
    for source_matrix in source {
        if let Some(destination_matrix) = destination
            .iter_mut()
            .find(|matrix| matrix.element == source_matrix.element)
        {
            destination_matrix.true_positive += source_matrix.true_positive;
            destination_matrix.true_negative += source_matrix.true_negative;
            destination_matrix.false_positive += source_matrix.false_positive;
            destination_matrix.false_negative += source_matrix.false_negative;
        } else {
            destination.push(ConfusionMatrix {
                element: source_matrix.element.clone(),
                true_positive: source_matrix.true_positive,
                true_negative: source_matrix.true_negative,
                false_positive: source_matrix.false_positive,
                false_negative: source_matrix.false_negative,
            });
        }
    }
}

/// Writes a slice of serializable rows to a CSV file.
fn write_csv<T: Serialize>(path: PathBuf, rows: &[T]) -> Result<(), Ms2AtomsError> {
    let file = File::create(path)?;
    let mut writer = Writer::from_writer(file);

    for row in rows {
        writer.serialize(row)?;
    }

    writer.flush()?;
    Ok(())
}

/// Builds a stable directory name from the experiment number and human-readable experiment name.
fn experiment_slug(experiment_num: usize, experiment_name: &str) -> String {
    let slug = sanitize_for_path(experiment_name);
    format!("experiment{experiment_num:02}_{slug}")
}

/// Formats holdout numbers for ease of sorting.
fn format_holdout(holdout_number: usize) -> String {
    format!("holdout_{holdout_number:03}")
}

/// Formats floating-point thresholds into filesystem-safe names.
fn format_threshold(threshold: f64) -> String {
    format!("{threshold:.2}").replace('.', "_")
}

/// Converts arbitrary experiment names into lowercase filesystem-safe slugs.
fn sanitize_for_path(value: &str) -> String {
    let mut sanitized = String::new();
    let mut last_was_separator = false;

    for character in value.chars() {
        if character.is_ascii_alphanumeric() {
            sanitized.push(character.to_ascii_lowercase());
            last_was_separator = false;
        } else if !last_was_separator {
            sanitized.push('-');
            last_was_separator = true;
        }
    }

    let trimmed = sanitized.trim_matches('-');

    if trimmed.is_empty() {
        "unnamed-experiment".to_owned()
    } else {
        trimmed.to_owned()
    }
}
