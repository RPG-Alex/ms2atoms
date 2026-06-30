use molecular_formulas::prelude::*;

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

