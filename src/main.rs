#![recursion_limit = "256"]

use crate::{
    error::SpectraError,
    experiments::{experiment1, experiment2, experiment3},
};

mod data;
mod dataset;
mod error;
mod experiment_config;
mod experiments;
mod holdout;
mod inference;
mod mcc;
mod model;
mod output;
mod training;

fn main() -> Result<(), SpectraError> {
    experiment1::run()?;
    experiment2::run()?;
    experiment3::run()
}
