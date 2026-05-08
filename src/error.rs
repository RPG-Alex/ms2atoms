use mass_spectrometry::structs::similarity_errors::SimilarityComputationError;

use mascot_rs::error::MascotError;

#[derive(Debug)]
pub enum SpectraError {
    InvalidArray,
    Mascot(MascotError),
    SimilarityComputation(SimilarityComputationError),
    Io(std::io::Error)
}

impl From<MascotError> for SpectraError {
    fn from(error: MascotError) -> Self {
        Self::Mascot(error)
    }
}

impl From<SimilarityComputationError> for SpectraError {
    fn from(error: SimilarityComputationError) -> Self {
        Self::SimilarityComputation(error)
    }
}

impl From<std::io::Error> for SpectraError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}