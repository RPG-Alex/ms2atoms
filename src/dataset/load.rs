use mascot_rs::mascot_generic_format::MGFVec;
use mass_spectrometry::traits::Spectrum;

use crate::{
    dataset::features::formula_element_occurrence, domain::sample::SpectrumSample,
    error::Ms2AtomsError,
};

/// Loads annotated MS/MS spectra from the local data directory.
///
/// # Errors
/// Returns [`Ms2AtomsError`] if annotated spectra loading or spectrum binning fails.
pub fn load_spectra(bin_size: usize) -> Result<Vec<SpectrumSample>, Ms2AtomsError> {
    let load = pollster::block_on(
        MGFVec::<f64>::annotated_ms2()
            .target_directory("data")
            .load(),
    )?;

    let mut output = Vec::with_capacity(load.spectra().len());
    // keep track of any spectra with missing smiles
    let mut missing_smiles = 0usize;

    for spec in load.spectra() {
        let Some(formula) = spec.metadata().formula() else {
            continue;
        };
        let Some(smiles) = spec.metadata().smiles() else {
            missing_smiles += 1;
            continue;
        };
        let group_id = smiles.to_string();
        let spectra = spec
            .linear_binned_intensities(0.0, 1000.0, bin_size)?
            .clone();

        output.push(SpectrumSample::new(
            group_id,
            spectra,
            formula_element_occurrence(formula),
        ));
    }

    tracing::info!(
        loaded_samples = output.len(),
        missing_smiles,
        "Loaded annotated spectra"
    );

    Ok(output)
}
