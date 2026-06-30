use burn::{Tensor, data::dataloader::batcher::Batcher, tensor::{Bool, Int, TensorData, backend::{Backend, BackendTypes}}};

use crate::data::SpectrumSample;

#[derive(Clone)]
/// Converts [`SpectrumSample`] values into Burn training batches.
pub struct ElementBatcher {
    class_indices: Vec<usize>,
    bin_size: usize,
}

impl ElementBatcher {
    /// Creates a new batcher for selected element classes and spectrum bin size.
    ///
    /// # Parameters
    /// - `class_indices` - Element class indices to include as prediction targets.
    /// - `bin_size` - Number of binned intensity features in each spectrum.
    #[must_use]
    pub const fn new(class_indices: Vec<usize>, bin_size: usize) -> Self {
        Self {
            class_indices,
            bin_size,
        }
    }

    /// Returns the number of element classes produced by this batcher.
    #[must_use]
    pub const fn num_classes(&self) -> usize {
        self.class_indices.len()
    }

    /// Returns the number of binned intensity features expected per spectrum.
    #[must_use]
    pub const fn bin_size(&self) -> usize {
        self.bin_size
    }

    /// Returns the element class indices included by this batcher.
    #[must_use]
    pub fn class_indices(&self) -> &[usize] {
        &self.class_indices
    }
}

#[derive(Clone, Debug)]
/// Burn tensor batch containing spectra and multi-label element targets.
pub struct ElementBatch<B: Backend> {
    /// Binned spectra tensor with shape `[batch_size, bin_size]`.
    pub spectra: Tensor<B, 2>,
    /// Multi-label target tensor with shape `[batch_size, num_classes]`.
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<B, SpectrumSample, ElementBatch<B>> for ElementBatcher {
    fn batch(
        &self,
        items: Vec<SpectrumSample>,
        device: &<B as BackendTypes>::Device,
    ) -> ElementBatch<B> {
        let spectra = items
            .iter()
            .map(|item| TensorData::from(item.spectra()).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 1>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, self.bin_size()]))
            .collect();
        let targets = items
            .iter()
            .map(|item| {
                let selected_targets = self
                    .class_indices()
                    .iter()
                    .map(|&class_index| item.is_element_present(class_index).unwrap_or(false))
                    .collect::<Vec<_>>();
                Tensor::<B, 1, Bool>::from_data(selected_targets.as_slice(), device)
                    .reshape([1, self.num_classes()])
                    .int()
            })
            .collect();
        let spectra = Tensor::cat(spectra, 0);
        let targets = Tensor::cat(targets, 0);
        ElementBatch { spectra, targets }
    }
}
