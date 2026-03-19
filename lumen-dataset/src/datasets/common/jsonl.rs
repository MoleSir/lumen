use serde::de::DeserializeOwned;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    marker::PhantomData,
    path::Path,
};

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

use crate::Dataset;

pub struct JsonlDataset<T: DeserializeOwned> {
    file: File,
    line_spans: Vec<(u64, usize)>, 
    _marker: PhantomData<T>,
}

impl<T: DeserializeOwned> JsonlDataset<T> {
    pub fn new<P: AsRef<Path>>(path: P) -> JsonlDatasetResult<Self> {
        let file = File::open(&path)?;
        let mut reader = BufReader::new(File::open(&path)?); // 用另一个文件句柄来做初始化

        let mut line_spans = vec![];
        let mut current_offset = 0;
        let mut buffer = Vec::new();

        loop {
            buffer.clear();
            let bytes_read = reader.read_until(b'\n', &mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            
            line_spans.push((current_offset, bytes_read));
            current_offset += bytes_read as u64;
        }

        Ok(Self {
            file,
            line_spans,
            _marker: PhantomData,
        })
    }
}

impl<T: DeserializeOwned> Dataset for JsonlDataset<T> {
    type Error = JsonlDatasetError;
    type Item = T;

    fn len(&self) -> usize {
        self.line_spans.len()
    }

    fn get(&self, index: usize) -> Result<Option<Self::Item>, Self::Error> {
        if index >= self.line_spans.len() {
            return Ok(None);
        }

        let (offset, length) = self.line_spans[index];
        let mut buffer = vec![0u8; length];

        #[cfg(unix)]
        self.file.read_exact_at(&mut buffer, offset)?;
        #[cfg(windows)]
        self.file.seek_read(&mut buffer, offset)?;

        let value: T = serde_json::from_slice(&buffer)?;
        
        Ok(Some(value))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum JsonlDatasetError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
}

pub type JsonlDatasetResult<T> = Result<T, JsonlDatasetError>;