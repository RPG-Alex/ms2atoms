use molecular_formulas::{ChemicalFormula, MolecularFormula};

use crate::domain::elements::{ELEMENT_COUNT, ELEMENTS};

/// Returns a fixed-width element-presence array for a molecular formula.
#[must_use]
pub fn formula_element_occurrence(formula: &ChemicalFormula<u32, i32>) -> [bool; ELEMENT_COUNT] {
    let mut element_present = [false; ELEMENT_COUNT];

    for (present, &element) in element_present.iter_mut().zip(ELEMENTS.iter()) {
        *present = formula.contains_element(element);
    }

    element_present
}
