use mascot_rs::error::MascotError;
use mass_spectrometry::structs::similarity_errors::SimilarityComputationError;

/// Error type used throughout the `ms2atoms` pipeline.
#[derive(Debug)]
pub enum Ms2AtomsError {
    /// Returned when an array, matrix, or tensor-compatible input has an invalid shape or value.
    InvalidArray,
    /// Returned when an element class index is outside the supported element list.
    InvalidClassIndex {
        /// Invalid class index that was requested.
        class_index: usize,
    },
    /// Returned when a dataset is unexpectedly empty.
    EmptyDataset {
        /// Context describing where the empty dataset was encountered.
        context: String,
    },
    /// Returned when samples in one dataset do not have the same feature width.
    InconsistentFeatureLength {
        /// Expected feature length.
        expected: usize,
        /// Actual feature length found on a later sample.
        actual: usize,
    },
    /// Error returned by MASCOT spectrum parsing or loading.
    Mascot(MascotError),
    /// Error returned during mass-spectrometry similarity computation.
    SimilarityComputation(SimilarityComputationError),
    /// Error returned by filesystem operations.
    Io(std::io::Error),
    /// Error returned while reading or writing CSV files.
    Csv(csv::Error),
    /// Error returned while training a model.
    ModelTraining(String),
    /// Error returned while running model inference.
    ModelInference(String),
    /// Error returned while saving or loading model artifacts.
    ModelArtifact(String),
}

impl Ms2AtomsError {
    /// Creates a model-training error from a displayable message.
    #[allow(clippy::needless_pass_by_value)]
    pub fn model_training(error: impl ToString) -> Self {
        Self::ModelTraining(error.to_string())
    }
    /// Creates a model-inference error from a displayable message.
    #[allow(clippy::needless_pass_by_value)]
    pub fn model_inference(error: impl ToString) -> Self {
        Self::ModelInference(error.to_string())
    }

    /// Creates a model-artifact error from a displayable message.
    #[allow(clippy::needless_pass_by_value)]
    pub fn model_artifact(error: impl ToString) -> Self {
        Self::ModelArtifact(error.to_string())
    }
}

impl From<MascotError> for Ms2AtomsError {
    fn from(error: MascotError) -> Self {
        Self::Mascot(error)
    }
}

impl From<SimilarityComputationError> for Ms2AtomsError {
    fn from(error: SimilarityComputationError) -> Self {
        Self::SimilarityComputation(error)
    }
}

impl From<std::io::Error> for Ms2AtomsError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<csv::Error> for Ms2AtomsError {
    fn from(value: csv::Error) -> Self {
        Self::Csv(value)
    }
}

impl core::fmt::Display for Ms2AtomsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArray => write!(f, "array or matrix shape/value is invalid"),
            Self::InvalidClassIndex { class_index } => {
                write!(f, "invalid element class index: {class_index}")
            }
            Self::EmptyDataset { context } => write!(f, "empty dataset: {context}"),
            Self::InconsistentFeatureLength { expected, actual } => write!(
                f,
                "inconsistent feature length: expected {expected}, found {actual}"
            ),
            Self::Mascot(error) => write!(f, "MASCOT error: {error}"),
            Self::SimilarityComputation(error) => {
                write!(f, "similarity computation error: {error}")
            }
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Csv(error) => write!(f, "CSV error: {error}"),
            Self::ModelTraining(error) => write!(f, "model training error: {error}"),
            Self::ModelInference(error) => write!(f, "model inference error: {error}"),
            Self::ModelArtifact(error) => write!(f, "model artifact error: {error}"),
        }
    }
}

impl std::error::Error for Ms2AtomsError {}
