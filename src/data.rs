use burn::{data::dataloader::batcher::Batcher, prelude::*, tensor::backend::BackendTypes};
use molecular_formulas::prelude::*;

#[derive(Clone)]
/// Converts [`SpectrumSample`] values into Burn training batches.
pub struct SpectraScribeBatcher {
    class_indices: Vec<usize>,
    bin_size: usize,
}

impl SpectraScribeBatcher {
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
pub struct SpectraScribeBatch<B: Backend> {
    /// Binned spectra tensor with shape `[batch_size, bin_size]`.
    pub spectra: Tensor<B, 2>,
    /// Multi-label target tensor with shape `[batch_size, num_classes]`.
    pub targets: Tensor<B, 2, Int>,
}

/// Ordered list of elements used as model output classes.
pub const ELEMENTS: &[Element; 118] = &[
    Element::Ac,
    Element::Ag,
    Element::Al,
    Element::Am,
    Element::Ar,
    Element::As,
    Element::At,
    Element::Au,
    Element::B,
    Element::Ba,
    Element::Be,
    Element::Bh,
    Element::Bi,
    Element::Bk,
    Element::Br,
    Element::C,
    Element::Ca,
    Element::Cd,
    Element::Ce,
    Element::Cf,
    Element::Cl,
    Element::Cm,
    Element::Cn,
    Element::Co,
    Element::Cr,
    Element::Cs,
    Element::Cu,
    Element::Db,
    Element::Ds,
    Element::Dy,
    Element::Er,
    Element::Es,
    Element::Eu,
    Element::F,
    Element::Fe,
    Element::Fl,
    Element::Fm,
    Element::Fr,
    Element::Ga,
    Element::Gd,
    Element::Ge,
    Element::H,
    Element::He,
    Element::Hf,
    Element::Hg,
    Element::Ho,
    Element::Hs,
    Element::I,
    Element::In,
    Element::Ir,
    Element::K,
    Element::Kr,
    Element::La,
    Element::Li,
    Element::Lr,
    Element::Lu,
    Element::Lv,
    Element::Mc,
    Element::Md,
    Element::Mg,
    Element::Mn,
    Element::Mo,
    Element::Mt,
    Element::N,
    Element::Na,
    Element::Nb,
    Element::Nd,
    Element::Ne,
    Element::Nh,
    Element::Ni,
    Element::No,
    Element::Np,
    Element::O,
    Element::Og,
    Element::Os,
    Element::P,
    Element::Pa,
    Element::Pb,
    Element::Pd,
    Element::Pm,
    Element::Po,
    Element::Pr,
    Element::Pt,
    Element::Pu,
    Element::Ra,
    Element::Rb,
    Element::Re,
    Element::Rf,
    Element::Rg,
    Element::Rh,
    Element::Rn,
    Element::Ru,
    Element::S,
    Element::Sb,
    Element::Sc,
    Element::Se,
    Element::Sg,
    Element::Si,
    Element::Sm,
    Element::Sn,
    Element::Sr,
    Element::Ta,
    Element::Tb,
    Element::Tc,
    Element::Te,
    Element::Th,
    Element::Ti,
    Element::Tl,
    Element::Tm,
    Element::Ts,
    Element::U,
    Element::V,
    Element::W,
    Element::Xe,
    Element::Y,
    Element::Yb,
    Element::Zn,
    Element::Zr,
];

/// Number of supported element classes.
pub const ELEMENT_COUNT: usize = ELEMENTS.len();

#[derive(Clone, Debug)]
/// One preprocessed MS/MS spectrum sample and its element-presence labels.
pub struct SpectrumSample {
    spectra: Vec<f64>,
    element_present: [bool; ELEMENT_COUNT],
}

impl SpectrumSample {
    /// Creates a new spectrum sample from binned intensities and element labels.
    ///
    /// # Parameters
    /// - `spectra` - Binned spectrum intensity values.
    /// - `element_present` - Fixed-width element-presence labels aligned with [`ELEMENTS`].
    #[must_use]
    pub const fn new(spectra: Vec<f64>, element_present: [bool; ELEMENT_COUNT]) -> Self {
        Self {
            spectra,
            element_present,
        }
    }
    /// Returns the binned spectrum intensity values.
    #[must_use]
    pub fn spectra(&self) -> &[f64] {
        &self.spectra
    }

    /// Returns all element-presence labels for this sample.
    #[must_use]
    pub const fn element_present(&self) -> &[bool; ELEMENT_COUNT] {
        &self.element_present
    }

    /// Returns whether the element at `class_index` is present in this sample.
    ///
    /// # Parameters
    ///
    /// - `class_index` - Index into [`ELEMENTS`].
    #[must_use]
    pub fn is_element_present(&self, class_index: usize) -> Option<bool> {
        self.element_present.get(class_index).copied()
    }
}

impl<B: Backend> Batcher<B, SpectrumSample, SpectraScribeBatch<B>> for SpectraScribeBatcher {
    fn batch(
        &self,
        items: Vec<SpectrumSample>,
        device: &<B as BackendTypes>::Device,
    ) -> SpectraScribeBatch<B> {
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
        SpectraScribeBatch { spectra, targets }
    }
}
