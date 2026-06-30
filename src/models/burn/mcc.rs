use burn::{
    Tensor,
    prelude::*,
    tensor::{activation::sigmoid, backend::Backend},
    train::{
        MultiLabelClassificationOutput,
        metric::{
            Adaptor, Metric, Numeric, NumericAttributes, NumericEntry,
            state::{FormatOptions, NumericMetricState},
        },
    },
};

use core::marker::PhantomData;
use std::sync::Arc;

#[derive(Clone)]

/// Burn training metric for multi-label Matthews correlation coefficient.
pub struct MatthewsCorrelationMetric<B: Backend> {
    name: Arc<String>,
    state: NumericMetricState,
    threshold: f32,
    sigmoid: bool,
    _b: PhantomData<B>,
}

/// Input batch used by [`MatthewsCorrelationMetric`].
pub struct MCCInput<B: Backend> {
    outputs: Tensor<B, 2>,
    targets: Tensor<B, 2, Int>,
}

impl<B: Backend> MatthewsCorrelationMetric<B> {
    /// Creates a new MCC metric with the default threshold.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the decision threshold used to convert outputs into labels.
    ///
    /// # Parameters
    /// - `threshold` - The threshold above which an output is treated as a positive prediction.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self.name = Arc::new(format!("MCC Score @ Threshold({threshold})"));
        self
    }

    /// Enables or disables sigmoid activation before thresholding.
    ///
    /// # Parameters
    /// - `sigmoid` - Whether to apply sigmoid before computing binary predictions.
    #[must_use]
    pub const fn with_sigmoid(mut self, sigmoid: bool) -> Self {
        self.sigmoid = sigmoid;
        self
    }
}

impl<B: Backend> Default for MatthewsCorrelationMetric<B> {
    fn default() -> Self {
        let threshold = 0.5;
        let name = Arc::new(format!("MCC Score @ Threshold({threshold})"));

        Self {
            name,
            state: NumericMetricState::default(),
            threshold,
            sigmoid: false,
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Metric for MatthewsCorrelationMetric<B>
where
    B::FloatElem: Into<f64>,
{
    type Input = MCCInput<B>;
    fn update(
        &mut self,
        input: &Self::Input,
        _metadata: &burn::train::metric::MetricMetadata,
    ) -> burn::train::metric::SerializedEntry {
        let [batch_size, _n_classes] = input.outputs.dims();
        let targets = input.targets.clone();
        let mut outputs = input.outputs.clone();

        if self.sigmoid {
            outputs = sigmoid(outputs);
        }

        let predictions = outputs.greater_elem(self.threshold).float();
        let targets = targets.float();

        let ones = Tensor::<B, 2>::ones_like(&predictions);

        let true_positives = (predictions.clone() * targets.clone()).sum_dim(1);

        let true_negatives =
            ((ones.clone() - predictions.clone()) * (ones.clone() - targets.clone())).sum_dim(1);

        let false_positives = (predictions.clone() * (ones.clone() - targets.clone())).sum_dim(1);

        let false_negatives = ((ones - predictions) * targets).sum_dim(1);

        let numerator = true_positives.clone() * true_negatives.clone()
            - false_positives.clone() * false_negatives.clone();

        let denominator = ((true_positives.clone() + false_positives.clone())
            * (true_positives + false_negatives.clone())
            * (true_negatives.clone() + false_positives)
            * (true_negatives + false_negatives))
            .sqrt();

        let mcc = numerator / denominator.clamp_min(1e-12);
        let mcc_value = mcc.mean().into_scalar();

        self.state.update(
            mcc_value.into(),
            batch_size,
            FormatOptions::new(self.name()).precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }

    fn name(&self) -> burn::train::metric::MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> burn::train::metric::MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: true,
        }
        .into()
    }
}

impl<B: Backend> Numeric for MatthewsCorrelationMetric<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

impl<B: Backend> Adaptor<MCCInput<B>> for MultiLabelClassificationOutput<B> {
    fn adapt(&self) -> MCCInput<B> {
        MCCInput {
            outputs: self.output.clone(),
            targets: self.targets.clone(),
        }
    }
}
