use burn::{
    nn::{BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu, Sigmoid},
    prelude::*,
};

#[derive(Module, Debug)]
/// Definition for the model
pub struct MLPModel<B: Backend> {
    linear1: Linear<B>,
    batch_norm1: BatchNorm<B>,
    linear2: Linear<B>,
    batch_norm2: BatchNorm<B>,
    linear3: Linear<B>,
    dropout: Dropout,
    inner_activation: Relu,
    pub(crate) activation: Sigmoid,
    class_weights: Option<Vec<f32>>,
}

#[derive(Config, Debug)]
/// Configuration for constructing a `SpectraScribe` multi-label classification model.
pub struct MLPConfig {
    /// Number of element classes predicted by the model.
    num_classes: usize,
    /// Number of neurons in the first hidden layer.
    hidden_size: usize,
    /// Number of binned intensity features in each input spectrum.
    bin_size: usize,
    /// Dropout probability applied during training.
    dropout: f64,
    /// Optional per-class weights used by the binary cross-entropy loss.
    class_weights: Option<Vec<f32>>,
}

impl MLPConfig {
    /// Initializes a [`Model`] from this configuration on the provided backend device.
    ///
    /// # Parameters
    ///
    /// - `device` - The backend device used to initialize model parameters.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLPModel<B> {
        MLPModel {
            linear1: LinearConfig::new(self.bin_size, self.hidden_size).init(device),
            batch_norm1: BatchNormConfig::new(self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size / 2).init(device),
            batch_norm2: BatchNormConfig::new(self.hidden_size / 2).init(device),
            linear3: LinearConfig::new(self.hidden_size / 2, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: Sigmoid::new(),
            inner_activation: Relu::new(),
            class_weights: self.class_weights.clone(),
        }
    }

    /// Returns the number of binned intensity features expected by the model.
    #[must_use]
    pub const fn bin_size(&self) -> usize {
        self.bin_size
    }
}

impl<B: Backend> MLPModel<B> {
    /// Computes raw element-class logits for a batch of binned spectra.
    ///
    /// # Parameters
    /// - `spectra` - Binned spectra features as `[batch_size, bin_size]`.
    pub fn forward_logits(&self, spectra: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, binned_spectrum_size] = spectra.dims();

        let x = spectra.reshape([batch_size, binned_spectrum_size]);

        let x = self.linear1.forward(x);
        let x = self.batch_norm1.forward(x);
        let x = self.inner_activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.linear2.forward(x);
        let x = self.batch_norm2.forward(x);
        let x = self.inner_activation.forward(x);
        let x = self.dropout.forward(x);

        self.linear3.forward(x)
    }

    /// Runs inference on spectra and returns activated multi-label predictions.
    ///
    /// # Parameters
    /// - `spectra` - Binned spectra features with shape `[batch_size, bin_size]`.
    pub fn forward(&self, spectra: Tensor<B, 2>) -> Tensor<B, 2> {
        let logits = self.forward_logits(spectra);
        self.activation.forward(logits)
    }

    /// Returns the optional per-class loss weights.
    pub fn class_weights(&self) -> Option<Vec<f32>> {
        self.class_weights.clone()
    }

    /// Applies the output activation to raw logits.
    pub fn activate_logits(&self, logits: Tensor<B, 2>) -> Tensor<B, 2> {
        self.activation.forward(logits)
    }
}
