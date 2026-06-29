//! Module for individual module implementations, such MLP or Logistic Regression.

/// The Linfa Logistic Regression model implementation
pub mod logistic_regression;
/// The MLP Burn model implementation
pub mod mlp;

/// Trait defining the shared interface for all models.
///
/// Each model defines:
/// - an input type
/// - an output type
/// - a trained model representation used for inference
pub trait Model {
    /// The input data for the model
    type Input;
    /// The Output the trained model should give with the input data
    type Output;
    /// The trained model for predicting outputs
    type Trained;

    /// Trains the model using the data
    fn train(&self, data: Self::Input) -> Self::Trained;

    /// Applies the trained model to an input dataset and returns the predicted output
    fn predict(&self, model: &Self::Trained, input: Self::Input) -> Self::Output;
}
