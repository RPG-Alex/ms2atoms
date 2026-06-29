//! # `SpectraScribe`
//!
//! `spectra_scribe` trains and evaluates machine-learning models for MS/MS
//! element identification.
//!
//! The crate currently provides an experimental pipeline for:
//!
//! - loading annotated MS/MS spectra,
//! - converting spectra into binned intensity vectors,
//! - training multi-label element classifiers,
//! - evaluating predictions across multiple thresholds,
//! - writing per-holdout and whole-experiment CSV reports.
//!
//! This is research software. APIs, model architecture, output formats, and
//! experiment protocols are expected to change while the task definition is
//! being refined.

#![recursion_limit = "256"]

/// Embedded or static data used by `spectra_scribe` experiments.
pub mod data;
/// Dataset loading, preprocessing, and sample representation.
pub mod dataset;
/// Error types used throughout the `spectra_scribe` pipeline.
pub mod error;
/// Evaluation utilities for confusion matrices, metrics, and reports.
pub mod evaluation;
/// Core experiment configuration, protocols, and runner logic.
pub mod experiment;
/// Concrete experiment definitions.
pub mod experiments;
/// Holdout generation and class-distribution reporting.
pub mod holdout;
/// Inference utilities for loading trained models and generating predictions.
pub mod inference;
/// [Matthews correlation coefficient](https://en.wikipedia.org/wiki/Phi_coefficient) utilities.
pub mod mcc;
/// Model architecture and model configuration.
pub mod models;
/// Training configuration and holdout training routines.
pub mod training;
