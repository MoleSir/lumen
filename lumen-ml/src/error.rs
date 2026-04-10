use crate::datasets::RegressionOptionBuilderError;


#[thiserrorctx::context_error]
pub enum MlError {
    #[error(transparent)]
    Core(#[from] lumen_core::Error),

    #[error(transparent)]
    Linalg(#[from] lumen_linalg::LinalgError),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    RegressionOption(#[from] RegressionOptionBuilderError),

    #[error("knn error: {0}")]
    Knn(String),

    #[error("Mismatched number of samples: x has {x_samples}, y has {y_samples}")]
    SampleSizeMismatch {
        x_samples: usize,
        y_samples: usize,
    },

}