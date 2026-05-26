use crate::{dataset::SpectraData, holdout::Holdout};
pub mod runner;
pub mod standard;

pub use runner::run_experiment;

/// Defines the methods for setting up an experiment
pub trait ExperimentConfig {
    type HoldoutType: Holdout;
    /// How many independent train/validation splits to run for this experiment.
    fn number_of_holdouts(&self) -> usize;

    /// Base seed used to make dataset shuffling and holdout generation reproducible.
    fn random_seed(&self) -> u64;

    /// Fraction of the dataset used for training in each holdout.
    fn training_size(&self) -> f32;

    /// Fraction of the dataset held back for validation in each holdout.
    fn validation_size(&self) -> f32 {
        1.0 - self.training_size()
    }

    /// Minimum and maximum bounds applied to class weights after frequency weighting.
    fn weight_range(&self) -> Option<(f32, f32)>;

    /// Number of full passes the model makes over the training data.
    fn epochs(&self) -> usize;

    /// Number of samples processed together in one training step.
    fn batch_size(&self) -> usize;

    /// Number of background workers used to load batches during training.
    fn num_workers(&self) -> usize;

    /// Step size used by the optimizer when updating model weights.
    fn learning_rate(&self) -> f64;

    /// Width of the model’s hidden layer; bigger means more capacity but more training cost.
    fn hidden_size(&self) -> usize;

    /// Probability cutoff for turning model scores into yes/no element predictions.
    fn threshold(&self) -> f64 {
        0.5
    }

    /// Number of intensity bins used to convert each raw spectrum into a fixed-length model input.
    fn bin_size(&self) -> usize;

    /// Builds the concrete train/validation holdouts this experiment will run.
    fn generate_holdouts(&self, dataset: &SpectraData) -> Vec<Self::HoldoutType>;

    /// Returns the number of the current experiment
    fn experiment_num(&self) -> usize;

    /// The current experiment's dropout number
    fn dropout(&self) -> f64;
}
