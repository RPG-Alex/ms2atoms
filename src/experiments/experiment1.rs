use std::{
    fs::{self, File},
    path::PathBuf,
};

use burn::backend::{Autodiff, Wgpu};
use csv::Writer;
use rand::{SeedableRng, rngs::ChaCha8Rng, seq::SliceRandom};

use crate::{
    data::{ELEMENT_COUNT, ELEMENTS},
    dataset::SpectraData,
    error::SpectraError,
    experiment_config::ExperimentConfig,
    experiments::observed_class_indices,
    holdout::{BasicHoldout, Holdout},
    inference::create_confusion_matrices,
    model::ModelConfig,
    training::TrainingConfig,
};

pub struct Experiment1Config {
    pub number_of_holdouts: usize,
    pub random_seed: u64,
    pub training_size: f32,
}

impl Default for Experiment1Config {
    fn default() -> Self {
        Self {
            number_of_holdouts: 1,
            random_seed: 42,
            training_size: 0.8,
        }
    }
}

impl ExperimentConfig for Experiment1Config {
    type HoldoutType = BasicHoldout;

    fn number_of_holdouts(&self) -> usize {
        self.number_of_holdouts
    }

    fn random_seed(&self) -> u64 {
        self.random_seed
    }

    fn training_size(&self) -> f32 {
        self.training_size
    }

    fn generate_holdouts(&self, dataset: &SpectraData) -> Vec<Self::HoldoutType> {
        let class_indices = observed_class_indices(&dataset.dataset);

        let classes = class_indices
            .iter()
            .map(|&index| ELEMENTS[index])
            .collect::<Vec<_>>();

        let mut holdouts = Vec::with_capacity(self.number_of_holdouts());

        for holdout_number in 0..self.number_of_holdouts() {
            let holdout_seed = self.random_seed() + holdout_number as u64;

            let mut samples = dataset.dataset.clone();
            let mut rng = ChaCha8Rng::seed_from_u64(holdout_seed);
            samples.shuffle(&mut rng);

            let split_index = (samples.len() as f32 * self.training_size()) as usize;

            let train = SpectraData {
                dataset: samples[..split_index].to_vec(),
                class_weights: dataset.class_weights.clone(),
            };

            let validation = SpectraData {
                dataset: samples[split_index..].to_vec(),
                class_weights: dataset.class_weights.clone(),
            };

            let holdout = BasicHoldout::new(
                train,
                validation,
                classes.clone(),
                class_indices.clone(),
                holdout_number,
                holdout_seed as usize,
            );

            holdouts.push(holdout);
        }

        holdouts
    }
}

pub fn run() -> Result<(), SpectraError> {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    // Training values
    let epochs: usize = 10;
    let batch_size: usize = 64;
    let workers: usize = 4;
    let learning_rate: f64 = 1.0e-4;

    let experiment_config = Experiment1Config::default();

    let device = burn::backend::wgpu::WgpuDevice::default();

    let experiment_dir = "./experiments/experiment1";
    let results_dir = "./results";

    fs::create_dir_all(experiment_dir)?;
    fs::create_dir_all(results_dir)?;

    println!("Loading spectra.");
    let dataset = SpectraData::new()?;
    println!("Finished loading spectra.");

    let holdouts = experiment_config.generate_holdouts(&dataset);

    println!("Generated {} holdout(s).", holdouts.len());

    for holdout in holdouts {
        println!(
            "Running holdout {} with seed {}.",
            holdout.holdout_number(),
            holdout.random_seed(),
        );

        println!(
            "Training samples: {}, validation samples: {}.",
            holdout.training_len(),
            holdout.validation_len(),
        );

        let artifact_dir = format!("{experiment_dir}/holdout_{}", holdout.holdout_number(),);

        let model_config = ModelConfig::new(ELEMENT_COUNT, 100)
            .with_class_weights(Some(holdout.train_dataset().class_weights.clone()));

        let training_config = TrainingConfig::new_with_values(
            model_config,
            epochs,
            batch_size,
            workers,
            holdout.random_seed() as u64,
            learning_rate,
        );

        crate::training::train_holdout::<MyAutodiffBackend, _>(
            &artifact_dir,
            &holdout,
            training_config,
            device.clone(),
        );

        let validation_items = holdout.validation_dataset().dataset.clone();

        let predictions =
            crate::inference::infer::<MyBackend>(&artifact_dir, &device, validation_items.clone());

        let confusion_matrices = create_confusion_matrices(predictions, &validation_items, 0.5)?;

        let results_path = PathBuf::from(results_dir).join(format!(
            "experiment1_holdout_{}_results.csv",
            holdout.holdout_number(),
        ));

        let file = File::create(results_path)?;
        let mut writer = Writer::from_writer(file);

        for matrix in confusion_matrices {
            writer.serialize(matrix)?;
        }

        writer.flush()?;
    }

    Ok(())
}
