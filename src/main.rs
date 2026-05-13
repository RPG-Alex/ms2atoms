#![recursion_limit = "256"]

mod data;
mod dataset;
mod error;
mod inference;
mod mcc;
mod model;
mod output;
mod training;
mod experimental_setup;
mod holdout;

use crate::{
    data::ELEMENT_COUNT, dataset::SpectraData, error::SpectraError,
    inference::create_confusion_matrices, model::ModelConfig, training::TrainingConfig,
};
use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};
use csv::Writer;
use std::fs::File;

fn main() -> Result<(), SpectraError> {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "./first_attempt";

    println!("Loading spectra.");
    let dataset = SpectraData::new()?;
    println!("Finished loading spectra");
    let model_config = ModelConfig::new(ELEMENT_COUNT, 100)
        .with_class_weights(Some(dataset.class_weights.clone()));

    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        &dataset,
        TrainingConfig::new(model_config, AdamConfig::new()),
        device.clone(),
    );

    let results =
        crate::inference::infer::<MyBackend>(artifact_dir, device, dataset.test(42).dataset);

    let confusion_matrices = create_confusion_matrices(results, dataset.test(42).dataset, 0.5);

    let file = File::create("results.csv")?;
    let mut wtr = Writer::from_writer(file);
    
    for matrix in confusion_matrices? {
        wtr.serialize(matrix)?;
    }

    wtr.flush()?;
    Ok(())
}
