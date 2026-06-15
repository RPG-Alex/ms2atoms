use crate::evaluation::confusion::ConfusionMatrix;
use serde::Serialize;

/// Per-element evaluation metrics derived from a confusion matrix.
#[derive(Debug, Serialize)]
pub struct ElementMetrics {
    /// Chemical symbol for the evaluated element.
    pub element: String,
    /// Number of true positive predictions.
    pub true_positive: u32,
    /// Number of true negative predictions.
    pub true_negative: u32,
    /// Number of false positive predictions.
    pub false_positive: u32,
    /// Number of false negative predictions.
    pub false_negative: u32,
    /// Precision score for this element.
    pub precision: f64,
    /// Recall score for this element.
    pub recall: f64,
    /// F1 score for this element.
    pub f1: f64,
    /// Matthews correlation coefficient for this element.
    pub mcc: f64,
}

/// Aggregate evaluation metrics across all evaluated element classes.
#[derive(Debug, Serialize)]
pub struct AggregateMetrics {
    /// Macro-averaged precision.
    pub macro_precision: f64,
    /// Macro-averaged recall.
    pub macro_recall: f64,
    /// Macro-averaged F1 score.
    pub macro_f1: f64,
    /// Macro-averaged Matthews correlation coefficient.
    pub macro_mcc: f64,
    /// Micro-averaged precision.
    pub micro_precision: f64,
    /// Micro-averaged recall.
    pub micro_recall: f64,
    /// Micro-averaged F1 score.
    pub micro_f1: f64,
    /// Micro-averaged Matthews correlation coefficient.
    pub micro_mcc: f64,
}

/// Computes per-element metrics from confusion matrices.
///
/// # Parameters
/// - `matrices` - Confusion matrices to convert into per-element metrics.
#[must_use]
pub fn element_metrics_from_matrices(matrices: &[ConfusionMatrix]) -> Vec<ElementMetrics> {
    matrices
        .iter()
        .map(|matrix| {
            let true_positive = f64::from(matrix.true_positive);
            let true_negative = f64::from(matrix.true_negative);
            let false_positive = f64::from(matrix.false_positive);
            let false_negative = f64::from(matrix.false_negative);

            let precision = safe_div(true_positive, true_positive + false_positive);
            let recall = safe_div(true_positive, true_positive + false_negative);
            let f1 = safe_div(2.0 * precision * recall, precision + recall);
            let mcc = mcc(true_positive, true_negative, false_positive, false_negative);

            ElementMetrics {
                element: matrix.element.clone(),
                true_positive: matrix.true_positive,
                true_negative: matrix.true_negative,
                false_positive: matrix.false_positive,
                false_negative: matrix.false_negative,
                precision,
                recall,
                f1,
                mcc,
            }
        })
        .collect()
}

/// Computes macro and micro aggregate metrics from per-element metrics.
///
/// # Parameters
/// - `metrics` - Per-element metrics to aggregate.
#[must_use]
#[allow(
    clippy::similar_names,
    reason = "macro and micro both have f1 as standard names. No need to change"
)]
pub fn aggregate_metrics(metrics: &[ElementMetrics]) -> AggregateMetrics {
    let macro_precision = mean_metric(metrics, |metric| metric.precision);
    let macro_recall = mean_metric(metrics, |metric| metric.recall);
    let macro_f1 = mean_metric(metrics, |metric| metric.f1);

    let true_positive = metrics
        .iter()
        .map(|metric| f64::from(metric.true_positive))
        .sum::<f64>();
    let true_negative = metrics
        .iter()
        .map(|metric| f64::from(metric.true_negative))
        .sum::<f64>();
    let false_positive = metrics
        .iter()
        .map(|metric| f64::from(metric.false_positive))
        .sum::<f64>();
    let false_negative = metrics
        .iter()
        .map(|metric| f64::from(metric.false_negative))
        .sum::<f64>();

    let micro_precision = safe_div(true_positive, true_positive + false_positive);
    let micro_recall = safe_div(true_positive, true_positive + false_negative);
    let micro_f1 = safe_div(
        2.0 * micro_precision * micro_recall,
        micro_precision + micro_recall,
    );

    AggregateMetrics {
        macro_precision,
        macro_recall,
        macro_f1,
        macro_mcc: mean_metric(metrics, |metric| metric.mcc),
        micro_precision,
        micro_recall,
        micro_f1,
        micro_mcc: mcc(true_positive, true_negative, false_positive, false_negative),
    }
}

/// Computes the arithmetic mean of a selected metric value.
fn mean_metric<F>(metrics: &[ElementMetrics], select: F) -> f64
where
    F: Fn(&ElementMetrics) -> f64,
{
    let mut total = 0.0;
    let mut count = 0.0;

    for metric in metrics {
        total += select(metric);
        count += 1.0;
    }

    safe_div(total, count)
}

/// Divides two values and returns `0.0` when the denominator is effectively zero.
fn safe_div(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() < f64::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

/// Computes Matthews correlation coefficient from confusion-matrix counts.
fn mcc(true_positive: f64, true_negative: f64, false_positive: f64, false_negative: f64) -> f64 {
    let numerator = false_positive.mul_add(-false_negative, true_positive * true_negative);
    let denominator = ((true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative))
        .sqrt();

    safe_div(numerator, denominator)
}
