use linfa::prelude::*;
use linfa_logistic::{FittedLogisticRegression, LogisticRegression};
use ndarray::{Array1, Array2};

use crate::data::SpectrumSample;
use crate::models::Model;

/// The settings for the Logistic Model for training
pub struct LinfaLogisticModel {
    /// Maximum number of iterations
    pub max_iterations: u64,
    /// The alpha value
    pub alpha: f64,
}

/// Structure for the trained Logistic Model
pub struct TrainedLogisticModel {
    model: FittedLogisticRegression<f64, usize>,
}

impl LinfaLogisticModel {
    fn to_dataset(data: &[SpectrumSample]) -> (Array2<f64>, Array1<usize>) {
        let n = data.len();
        let f = data[0].spectra().len();

        let mut x = Array2::<f64>::zeros((n, f));
        let mut y = Array1::<usize>::zeros(n);

        for (i, sample) in data.iter().enumerate() {
            for (j, v) in sample.spectra().iter().enumerate() {
                x[[i, j]] = *v;
            }

            y[i] = sample.element_present()[0] as usize; // placeholder (binary test)
        }

        (x, y)
    }
}

impl Model for LinfaLogisticModel {
    type Input = Vec<SpectrumSample>;
    type Output = Vec<f64>;
    type Trained = TrainedLogisticModel;

    fn train(&self, data: Self::Input) -> Self::Trained {
        let (x, y) = Self::to_dataset(&data);

        let dataset = Dataset::new(x, y);

        let model = LogisticRegression::default()
            .max_iterations(self.max_iterations)
            .alpha(self.alpha)
            .fit(&dataset)
            .expect("training failed");

        TrainedLogisticModel { model }
    }

    fn predict(&self, trained: &Self::Trained, input: Self::Input) -> Self::Output {
        let (x, _) = Self::to_dataset(&input);

        trained
            .model
            .predict(&x)
            .iter()
            .map(|v| *v as f64)
            .collect()
    }
}
