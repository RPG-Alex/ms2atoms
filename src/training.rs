use crate::{
    data::{SpectraScribeBatch, SpectraScribeBatcher},
    holdout::Holdout,
    mcc::MatthewsCorrelationMetric,
    model::{Model, ModelConfig},
};

use burn::{
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

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        spectra: Tensor<B, 2>,
        targets: Tensor<B, 2, Int>,
    ) -> MultiLabelClassificationOutput<B> {
        let logits = self.forward_logit(spectra);
        let loss_bce = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .with_weights(self.class_weights())
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());

        let lambda = 1e-4;
        let logit_reg = logits.clone().powf_scalar(2.0).mean();
        let loss = loss_bce + logit_reg * lambda;
        MultiLabelClassificationOutput::new(loss, self.activation.forward(logits), targets)
    }
}

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = SpectraScribeBatch<B>;
    type Output = MultiLabelClassificationOutput<B>;
    fn step(&self, batch: Self::Input) -> burn::train::TrainOutput<Self::Output> {
        let item = self.forward_classification(batch.spectra, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = SpectraScribeBatch<B>;
    type Output = MultiLabelClassificationOutput<B>;
    fn step(&self, batch: Self::Input) -> Self::Output {
        self.forward_classification(batch.spectra, batch.targets)
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub num_workers: usize,
    pub seed: u64,
    pub learning_rate: f64,
    pub class_indices: Vec<usize>,
}

impl TrainingConfig {
    pub fn new_with_values(
        model: ModelConfig,
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
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train_holdout<B, H>(
    artifact_dir: &str,
    holdout: &H,
    config: TrainingConfig,
    device: B::Device,
) where
    B: AutodiffBackend,
    H: Holdout,
{
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");
    B::seed(&device, config.seed);

    let batcher = SpectraScribeBatcher::new(
        holdout.class_indices().to_vec(),
        holdout.train_dataset().bin_size(),
    );

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(holdout.train_dataset().clone());

    let dataloader_validation = DataLoaderBuilder::new(batcher.clone())
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

    let model = config.model.init::<B>(&device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
