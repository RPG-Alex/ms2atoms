use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::{TensorData, Transaction},
};

use crate::{
    domain::sample::SpectrumSample,
    error::Ms2AtomsError,
    evaluation::PredictionMatrix,
    models::burn::{batcher::ElementBatcher, training::TrainingConfig},
};

/// Runs inference using a trained Burn model artifact directory.
///
/// # Parameters
/// - `artifact_dir` - Directory containing `config.json` and the saved model record.
/// - `device` - Backend device used to load the model and run inference.
/// - `items` - Spectrum samples to evaluate.
///
/// # Errors
/// Returns [`Ms2AtomsError`] if configuration, record loading, batching, or tensor extraction fails.
pub fn infer<B: Backend>(
    artifact_dir: &str,
    device: &B::Device,
    items: Vec<SpectrumSample>,
) -> Result<PredictionMatrix, Ms2AtomsError>
where
    B::FloatElem: Into<f64>,
{
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .map_err(Ms2AtomsError::model_artifact)?;

    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .map_err(Ms2AtomsError::model_artifact)?;

    let model = config.model().init::<B>(device).load_record(record);
    let batcher = ElementBatcher::new(config.class_indices().to_vec(), config.model().bin_size());
    let batch = batcher.batch(items, device);
    let predictions = model.forward(batch.spectra);

    tensor_to_prediction_matrix(predictions, config.class_indices())
}

/// Converts a Burn prediction tensor into the model-agnostic prediction matrix.
fn tensor_to_prediction_matrix<B: Backend>(
    predictions: Tensor<B, 2>,
    class_indices: &[usize],
) -> Result<PredictionMatrix, Ms2AtomsError>
where
    B::FloatElem: Into<f64>,
{
    let [n_rows, n_classes] = predictions.dims();

    if n_classes != class_indices.len() {
        return Err(Ms2AtomsError::InconsistentFeatureLength {
            expected: class_indices.len(),
            actual: n_classes,
        });
    }

    let [data] = Transaction::default()
        .register(predictions)
        .execute()
        .try_into()
        .map_err(|_| Ms2AtomsError::InvalidArray)?;

    tensor_data_to_prediction_matrix::<B::FloatElem>(&data, n_rows, n_classes, class_indices)
}

/// Converts Burn tensor data into the model-agnostic prediction matrix.
fn tensor_data_to_prediction_matrix<E>(
    data: &TensorData,
    n_rows: usize,
    n_classes: usize,
    class_indices: &[usize],
) -> Result<PredictionMatrix, Ms2AtomsError>
where
    E: burn::tensor::Element + Into<f64>,
{
    if n_classes == 0 {
        return PredictionMatrix::new(class_indices.to_vec(), vec![Vec::new(); n_rows]);
    }

    let values = data
        .as_slice::<E>()
        .map_err(Ms2AtomsError::model_inference)?;

    let mut scores = Vec::with_capacity(n_rows);

    for row in values.chunks(n_classes) {
        scores.push(row.iter().copied().map(Into::into).collect());
    }

    PredictionMatrix::new(class_indices.to_vec(), scores)
}
