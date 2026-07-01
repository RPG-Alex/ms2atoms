//! # `ms2atoms`
//!
//! `ms2atoms` trains and evaluates machine-learning models for MS/MS
//! element-identification experiments.
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

/// Dataset loading, preprocessing, and sample representation.
pub mod dataset;
/// Domain-specific types for spectra and element targets.
pub mod domain;
/// Error types used throughout the `ms2atoms` pipeline.
pub mod error;
/// Evaluation utilities for confusion matrices, metrics, and reports.
pub mod evaluation;
/// Core experiment configuration, protocols, and runner logic.
pub mod experiment;
/// Concrete experiment definitions.
pub mod experiments;
/// Holdout generation and class-distribution reporting.
pub mod holdout;
/// Model architecture and model configuration.
pub mod models;
