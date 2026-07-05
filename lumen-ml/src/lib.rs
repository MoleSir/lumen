pub mod linear;
pub mod cluster;
pub mod neighbor;
pub mod tree;
pub mod datasets;
pub mod metrics;
pub mod ensemble;
pub mod naive_bayes;
pub mod preprocessing;
pub mod decomposition;
pub mod utils;
pub mod pipeline;

mod core;
pub use core::*;
mod error;
pub use error::*;