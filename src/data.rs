use burn::{data::dataloader::batcher::Batcher, prelude::*};
use molecular_formulas::prelude::*;
#[derive(Clone, Default)]
pub struct SpectraScribeBatcher {
    class_indices: Vec<usize>,
}

impl SpectraScribeBatcher {
    pub fn new(class_indices: Vec<usize>) -> Self {
        Self { class_indices }
    }
    pub fn num_classes(&self) -> usize {
        self.class_indices.len()
    }
}

#[derive(Clone, Debug)]
pub struct SpectraScribeBatch<B: Backend> {
    pub spectra: Tensor<B, 2>,
    pub targets: Tensor<B, 2, Int>,
}

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

pub const ELEMENT_COUNT: usize = ELEMENTS.len();
pub const BIN_SIZE: usize = 1000; // need to figure out what is optimal value

#[derive(Clone, Debug)]
pub struct SpectrumSample {
    pub(crate) spectra: [f64; BIN_SIZE],
    pub(crate) element_present: [bool; ELEMENT_COUNT],
}

impl<B: Backend> Batcher<B, SpectrumSample, SpectraScribeBatch<B>> for SpectraScribeBatcher {
    fn batch(
        &self,
        items: Vec<SpectrumSample>,
        device: &<B as Backend>::Device,
    ) -> SpectraScribeBatch<B> {
        let spectra = items
            .iter()
            .map(|item| TensorData::from(item.spectra).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 1>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, BIN_SIZE]))
            .collect();
        let targets = items.iter().map(|item| {
            let selected_targets = self.class_indices.iter().map(|&class_index| item.element_present[class_index]).collect::<Vec<_>>();
            Tensor::<B,1, Bool>::from_data(selected_targets.as_slice(), device)
                .reshape([1, self.num_classes()])
                .int()
        }).collect();
        let spectra = Tensor::cat(spectra, 0);
        let targets = Tensor::cat(targets, 0);
        SpectraScribeBatch { spectra, targets }
    }
}
