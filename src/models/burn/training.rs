use std::path::Path;

use burn::{
    backend::{Autodiff, Wgpu},
    data::dataloader::DataLoaderBuilder,
    nn::loss::BinaryCrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        InferenceStep, Learner, MultiLabelClassificationOutput, SupervisedTraining, TrainOutput,
        TrainStep,
        metric::{HammingScore, LossMetric},
    },
};

use crate::{
    error::Ms2AtomsError,
    evaluation::PredictionMatrix,
    holdout::Holdout,
    models::burn::{
        batcher::{ElementBatch, ElementBatcher},
        config::{BurnMlpExperimentConfig, ClassWeighting},
        inference,
        metrics::MatthewsCorrelationMetric,
        mlp::{MLPConfig, MLPModel},
    },
};

/// Concrete Burn backend used by the current experiment runner.
type BurnBackend = Wgpu<f32, i32>;
/// Autodiff wrapper for the concrete Burn backend.
type BurnAutodiffBackend = Autodiff<BurnBackend>;

/// Computes logits, weighted binary cross-entropy loss, and activated predictions.
///
/// # Parameters
/// - `model` - Model used to compute logits and predictions.
/// - `spectra` - Binned spectra features with shape `[batch_size, bin_size]`.
/// - `targets` - Multi-label element targets with shape `[batch_size, num_classes]`.
fn forward_classification<B: Backend>(
    model: &MLPModel<B>,
    spectra: Tensor<B, 2>,
    targets: Tensor<B, 2, Int>,
) -> MultiLabelClassificationOutput<B> {
    let logits = model.forward_logits(spectra);

    let loss_bce = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .with_weights(model.class_weights())
        .init(&logits.device())
        .forward(logits.clone(), targets.clone());

    let lambda = 1e-4;
    let logit_regularization = logits.clone().powf_scalar(2.0).mean();
    let loss = loss_bce + logit_regularization * lambda;

    MultiLabelClassificationOutput::new(loss, model.activate_logits(logits), targets)
}

impl<B: AutodiffBackend> TrainStep for MLPModel<B> {
    type Input = ElementBatch<B>;
    type Output = MultiLabelClassificationOutput<B>;

    fn step(&self, batch: Self::Input) -> burn::train::TrainOutput<Self::Output> {
        let item = forward_classification(self, batch.spectra, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for MLPModel<B> {
    type Input = ElementBatch<B>;
    type Output = MultiLabelClassificationOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        forward_classification(self, batch.spectra, batch.targets)
    }
}

#[derive(Config, Debug)]
/// Configuration used to train one Burn MLP model on one holdout split.
pub struct TrainingConfig {
    /// Model architecture and initialization settings.
    model: MLPConfig,
    /// Optimizer configuration used during training.
    optimizer: AdamConfig,
    /// Number of training epochs.
    num_epochs: usize,
    /// Number of samples per batch.
    batch_size: usize,
    /// Number of workers used by the data loaders.
    num_workers: usize,
    /// Random seed used for reproducible training.
    seed: u64,
    /// Optimizer learning rate.
    learning_rate: f64,
    /// Element class indices included in this training run.
    class_indices: Vec<usize>,
}

impl TrainingConfig {
    /// Creates a new [`TrainingConfig`] from explicit training values.
    ///
    /// # Parameters
    /// - `model` - The [`MLPConfig`] used to initialize the model.
    /// - `num_epochs` - The number of training epochs.
    /// - `batch_size` - The number of samples per training batch.
    /// - `num_workers` - The number of data-loader workers.
    /// - `seed` - The random seed used for reproducible training.
    /// - `learning_rate` - The optimizer learning rate.
    /// - `class_indices` - The class indices included in this training run.
    #[must_use]
    pub fn new_with_values(
        model: MLPConfig,
        num_epochs: usize,
        batch_size: usize,
        num_workers: usize,
        seed: u64,
        learning_rate: f64,
        class_indices: Vec<usize>,
    ) -> Self {
        Self {
            model,
            optimizer: AdamConfig::new(),
            num_epochs,
            batch_size,
            num_workers,
            seed,
            learning_rate,
            class_indices,
        }
    }

    /// Returns the model configuration used for training.
    pub(crate) const fn model(&self) -> &MLPConfig {
        &self.model
    }

    /// Returns the class indices included in this training run.
    pub(crate) fn class_indices(&self) -> &[usize] {
        &self.class_indices
    }
}

/// Trains a Burn MLP on one holdout split and returns validation predictions.
///
/// # Parameters
/// - `config` - Burn MLP experiment configuration.
/// - `holdout` - Holdout split used for training and validation.
/// - `artifact_dir` - Directory where Burn artifacts will be written.
///
/// # Errors
/// Returns [`Ms2AtomsError`] if training, artifact writing, or inference fails.
pub fn train_and_predict<H>(
    config: &BurnMlpExperimentConfig,
    holdout: &H,
    artifact_dir: &Path,
) -> Result<PredictionMatrix, Ms2AtomsError>
where
    H: Holdout,
{
    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir_string = artifact_dir.to_string_lossy().into_owned();
    let training_config = training_config_for_holdout(config, holdout)?;

    train_holdout::<BurnAutodiffBackend, _>(
        &artifact_dir_string,
        holdout,
        &training_config,
        &device,
    )?;

    inference::infer::<BurnBackend>(
        &artifact_dir_string,
        &device,
        holdout.validation_dataset().samples().to_vec(),
    )
}

/// Builds the low-level Burn training configuration for one holdout.
fn training_config_for_holdout<H>(
    config: &BurnMlpExperimentConfig,
    holdout: &H,
) -> Result<TrainingConfig, Ms2AtomsError>
where
    H: Holdout,
{
    let class_weights = match config.loss.class_weighting {
        ClassWeighting::None => None,
        ClassWeighting::InverseFrequency { clamp } => Some(
            holdout
                .train_dataset()
                .class_weights_for(holdout.class_indices(), clamp)?,
        ),
    };

    let model_config = MLPConfig::new(
        holdout.num_classes(),
        config.architecture.hidden_size,
        holdout.train_dataset().bin_size(),
        config.architecture.dropout,
    )
    .with_class_weights(class_weights);

    Ok(TrainingConfig::new_with_values(
        model_config,
        config.training.epochs,
        config.training.batch_size,
        config.training.workers,
        holdout.random_seed(),
        config.training.learning_rate,
        holdout.class_indices().to_vec(),
    ))
}

/// Recreates the artifact directory for a fresh training run.
fn create_artifact_dir(artifact_dir: &str) -> Result<(), Ms2AtomsError> {
    match std::fs::remove_dir_all(artifact_dir) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }

    std::fs::create_dir_all(artifact_dir)?;

    Ok(())
}

/// Trains a model on one holdout split and saves the resulting artifacts.
///
/// # Parameters
/// - `artifact_dir` - Directory where training artifacts will be written.
/// - `holdout` - Holdout split used for training and validation.
/// - `config` - Training configuration for this holdout.
/// - `device` - Backend device used for training.
///
/// # Errors
/// Returns [`Ms2AtomsError`] if artifact setup, training config saving, or model record saving fails.
pub fn train_holdout<B, H>(
    artifact_dir: &str,
    holdout: &H,
    config: &TrainingConfig,
    device: &B::Device,
) -> Result<(), Ms2AtomsError>
where
    B: AutodiffBackend,
    H: Holdout,
{
    create_artifact_dir(artifact_dir)?;
    config
        .save(format!("{artifact_dir}/config.json"))
        .map_err(Ms2AtomsError::model_artifact)?;
    B::seed(device, config.seed);

    let batcher = ElementBatcher::new(
        holdout.class_indices().to_vec(),
        holdout.train_dataset().bin_size(),
    );

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(holdout.train_dataset().clone());

    let dataloader_validation = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(holdout.validation_dataset().clone());

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_validation)
        .metrics((
            MatthewsCorrelationMetric::new(),
            LossMetric::new(),
            HammingScore::new(),
        ))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.model.init::<B>(device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .map_err(Ms2AtomsError::model_artifact)?;

    Ok(())
}
