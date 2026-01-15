
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("Index out of bounds for wrapped dataset size: {0} >= {1} when {2}")]
    IndexOutOfRangeWhenSelectDataset(usize, usize, &'static str),

    #[error("Use 0 to split dataset")]
    NumSplitZeroWhenSelectDataset,
}

pub type DatasetResult<T> = std::result::Result<T, DatasetError>;