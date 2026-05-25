use crate::{
    data::{ELEMENTS, SpectraScribeBatcher, SpectrumSample},
    error::SpectraError,
    output::ConfusionMatrix,
    training::TrainingConfig,
};

use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::Transaction,
};

pub fn infer<B: Backend>(
    artifact_dir: &str,
    device: &B::Device,
    items: Vec<SpectrumSample>,
) -> Tensor<B, 2> {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("Trained model should exist; run train first");

    let model = config.model.init::<B>(device).load_record(record);

    let batcher = SpectraScribeBatcher::new(config.class_indices.clone(), config.model.bin_size());
    let batch = batcher.batch(items, device);
    model.forward(batch.spectra)
}

pub fn create_confusion_matrices<B: Backend>(
    predictions: Tensor<B, 2>,
    items: &[SpectrumSample],
    class_indices: &[usize],
    threshold: f64,
) -> Result<Vec<ConfusionMatrix>, SpectraError> {
    let mut confusion_matrices: Vec<ConfusionMatrix> = class_indices
        .iter()
        .map(|&class_index| ConfusionMatrix::new(&ELEMENTS[class_index]))
        .collect::<Vec<_>>();

    let predicted_tensor = predictions.greater_elem(threshold).int();
    let pred_iter = predicted_tensor.iter_dim(0);

    for (prediction_row, sample) in pred_iter.into_iter().zip(items.iter()) {
        let [output_data] = Transaction::default()
            .register(prediction_row)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let predicted_values = output_data.as_slice::<i32>()?;

        for (prediction_column, &class_index) in class_indices.iter().enumerate() {
            let predicted_atom = predicted_values[prediction_column];
            let true_atom = sample.element_present[class_index];

            match (predicted_atom, true_atom) {
                (1, true) => confusion_matrices[prediction_column].true_positive += 1,
                (0, false) => confusion_matrices[prediction_column].true_negative += 1,
                (1, false) => confusion_matrices[prediction_column].false_positive += 1,
                (0, true) => confusion_matrices[prediction_column].false_negative += 1,
                (i32::MIN..=-1_i32 | 2_i32..=i32::MAX, _) => {
                    unreachable!("Values outside of 0 and 1 should be unreachable")
                }
            }
        }
    }

    Ok(confusion_matrices)
}
