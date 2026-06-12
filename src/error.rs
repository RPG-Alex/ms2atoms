use burn::{config::ConfigError, record::RecorderError, tensor::DataError};
use mass_spectrometry::structs::similarity_errors::SimilarityComputationError;

use mascot_rs::error::MascotError;

#[derive(Debug)]
/// Error type used throughout the SpectraScribe pipeline.
pub enum SpectraError {
    /// Returned when an array or tensor-compatible input has an invalid shape or value.
    InvalidArray,
    /// Error returned by MASCOT spectrum parsing or loading.
    Mascot(MascotError),
    /// Error returned during mass-spectrometry similarity computation.
    SimilarityComputation(SimilarityComputationError),
    /// Error returned by filesystem operations.
    Io(std::io::Error),
    /// Error returned by Burn tensor data conversion.
    BurnData(DataError),
    /// Error returned while reading or writing CSV files.
    Csv(csv::Error),
    /// Error returned while saving or loading Burn model records.
    BurnRecord(RecorderError),
    /// Error returned while saving or loading Burn configuration files.
    BurnConfig(ConfigError),
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

impl From<DataError> for SpectraError {
    fn from(value: DataError) -> Self {
        Self::BurnData(value)
    }
}

impl From<RecorderError> for SpectraError {
    fn from(value: RecorderError) -> Self {
        Self::BurnRecord(value)
    }
}

impl From<ConfigError> for SpectraError {
    fn from(value: ConfigError) -> Self {
        Self::BurnConfig(value)
    }
}

impl From<csv::Error> for SpectraError {
    fn from(value: csv::Error) -> Self {
        Self::Csv(value)
    }
}

impl core::fmt::Display for SpectraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArray => write!(f, "The array is invalid"),
            Self::Mascot(mascot_error) => write!(f, "MASCOT ERROR: {mascot_error}"),
            Self::SimilarityComputation(similarity_computation_error) => write!(
                f,
                "Similarity Computation Error: {similarity_computation_error}"
            ),
            Self::Io(error) => write!(f, "IO ERROR: {error}"),
            Self::BurnData(data_error) => write!(f, "BURN DATA ERROR: {data_error}"),
            Self::Csv(error) => write!(f, "CSV ERROR: {error}"),
            Self::BurnRecord(record_error) => write!(f, "BURN RECORD ERROR: {record_error}"),
            Self::BurnConfig(config_error) => write!(f, " BURN CONFIG ERROR: {config_error}"),
        }
    }
}
