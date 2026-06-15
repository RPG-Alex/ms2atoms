use crate::{
    data::{SpectraScribeBatcher, SpectrumSample},
    error::SpectraError,
    training::TrainingConfig,
};

use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
    record::{CompactRecorder, Recorder},
};

/// Runs inference using a trained model artifact directory.
///
/// # Parameters
/// - `artifact_dir` - Directory containing `config.json` and the saved model record.
/// - `device` - Backend device used to load the model and run inference.
/// - `items` - Spectrum samples to evaluate.
///
/// # Errors
/// - Returns [`SpectraError`] if training configuration loading fails.
/// - Returns [`SpectraError`] if model record loading fails.
pub fn infer<B: Backend>(
    artifact_dir: &str,
    device: &B::Device,
    items: Vec<SpectrumSample>,
) -> Result<Tensor<B, 2>, SpectraError> {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))?;
    let record = CompactRecorder::new().load(format!("{artifact_dir}/model").into(), device)?;

    let model = config.model().init::<B>(device).load_record(record);

    let batcher =
        SpectraScribeBatcher::new(config.class_indices().to_vec(), config.model().bin_size());
    let batch = batcher.batch(items, device);
    Ok(model.forward(batch.spectra))
}
