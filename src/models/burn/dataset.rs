use burn::data::dataset::Dataset;

use crate::{dataset::SpectraData, domain::sample::SpectrumSample};

impl Dataset<SpectrumSample> for SpectraData {
    fn get(&self, index: usize) -> Option<SpectrumSample> {
        self.samples().get(index).cloned()
    }

    fn len(&self) -> usize {
        self.samples().len()
    }
}
