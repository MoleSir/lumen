use std::io::BufWriter;
use std::{fs::File, io::Write};
use std::path::Path;
use lumen_core::{DType, DynTensor};
use memmap2::MmapOptions;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

use crate::utils;

///
/// +-----------------------+-------------------------------+-------------------------------------+
/// |      N (8 bytes)      |         Header (N bytes)      |          Data (Rest of file)        |
/// +-----------------------+-------------------------------+-------------------------------------+
/// |     len of header     |             json              |              Tensor data            |
/// +-----------------------+-------------------------------+-------------------------------------+
///

#[derive(Debug, Deserialize, Serialize)]
struct SafeTensorsInfo {
    dtype: SafeTensorsDType,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SafeTensorsDType {
    #[serde(rename = "BOOL")]
    Bool,
    #[serde(rename = "U8")]
    U8,
    #[serde(rename = "I8")]
    I8,

    #[serde(rename = "F8_E4M3")]
    F8E4M3,
    #[serde(rename = "F8_E5M2")]
    F8E5M2,

    #[serde(rename = "U16")]
    U16,
    #[serde(rename = "I16")]
    I16,
    #[serde(rename = "F16")]
    F16, 
    #[serde(rename = "BF16")]
    Bf16,

    #[serde(rename = "U32")]
    U32,
    #[serde(rename = "I32")]
    I32,
    #[serde(rename = "F32")]
    F32,

    #[serde(rename = "U64")]
    U64,
    #[serde(rename = "I64")]
    I64,
    #[serde(rename = "F64")]
    F64,
}

impl TryInto<DType> for SafeTensorsDType {
    type Error = SafeTensorsError;
    fn try_into(self) -> Result<DType, Self::Error> {
        match self {
            Self::Bool => Ok(DType::Bool),
            Self::U8 => Ok(DType::U8),
            Self::I32 => Ok(DType::I32),
            Self::U32 => Ok(DType::U32),
            Self::F32 => Ok(DType::F32),
            Self::F64 => Ok(DType::F64),
            _ => Err(SafeTensorsError::UnsupportDType(self))
        }
    }
}

impl From<DType> for SafeTensorsDType {
    fn from(value: DType) -> Self {
        match value {
            DType::Bool => Self::Bool,
            DType::F32 => Self::F32,
            DType::F64 => Self::F64,
            DType::U8 => Self::U8,
            DType::I32 => Self::I32,
            DType::U32 => Self::U32,
        }
    }
}

pub struct SafeTensorsContent {
    pub metadata: Option<HashMap<String, String>>,
    pub tensors: HashMap<String, DynTensor>,
}

pub fn load_file<P: AsRef<Path>>(path: P) -> SafeTensorsResult<SafeTensorsContent> {
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    // len of header
    let header_size_bytes = &mmap[0..8];
    let header_size = u64::from_le_bytes(header_size_bytes.try_into()?) as usize;

    // header
    let header_slice = &mmap[8..8 + header_size];
    let header: HashMap<String, serde_json::Value> = serde_json::from_slice(header_slice)?;

    // data
    let data_start_position = 8 + header_size;
    let mut metadata = None;
    let mut tensors = HashMap::new();
    for (name, value) in header {
        if name == "__metadata__" {
            metadata = Some( serde_json::from_value(value)? );
            continue;
        }

        let info: SafeTensorsInfo = serde_json::from_value(value)?;
        let (start_offset, end_offset) = info.data_offsets;
        let absolute_start = data_start_position + start_offset;
        let absolute_end = data_start_position + end_offset;

        let raw_bytes = &mmap[absolute_start..absolute_end];
        let dtype = info.dtype.try_into()?;
    
        let tensor = utils::load_tensor(dtype, info.shape, raw_bytes)?;
        tensors.insert(name, tensor);
    }

    Ok(SafeTensorsContent { metadata, tensors } )
}

pub fn load<R: std::io::Read>(reader: &mut R) -> SafeTensorsResult<SafeTensorsContent> {
    // Header Size
    let mut header_size_bytes = [0u8; 8];
    reader.read_exact(&mut header_size_bytes)?;
    let header_size = usize::from_le_bytes(header_size_bytes);

    // Header
    let mut json_bytes = vec![0u8; header_size];
    reader.read_exact(&mut json_bytes)?;
    let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&json_bytes)?;

    // data
    let mut data_buffer = Vec::new();
    reader.read_to_end(&mut data_buffer)?;

    let mut metadata = None;
    let mut tensors = HashMap::new();
    for (name, value) in header {
        if name == "__metadata__" {
            metadata = Some( serde_json::from_value(value)? );
            continue;
        }

        let info: SafeTensorsInfo = serde_json::from_value(value)?;
        let (start, end) = info.data_offsets;
        if end > data_buffer.len() {
            return Err(SafeTensorsError::DataOffsetOutOfRange(data_buffer.len(), end))?;
        }

        let raw_bytes = &data_buffer[start..end];
        let dtype = info.dtype.try_into()?;
    
        let tensor = utils::load_tensor(dtype, info.shape, raw_bytes)?;
        tensors.insert(name, tensor);
    }

    Ok(SafeTensorsContent { metadata, tensors } )
}

pub fn save_file<P: AsRef<Path>>(tensors: &HashMap<String, DynTensor>, metadata: Option<&HashMap<String, String>>, path: P) -> SafeTensorsResult<()> {    
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    save(tensors, metadata, &mut writer)
}

pub fn save<W: Write>(tensors: &HashMap<String, DynTensor>, metadata: Option<&HashMap<String, String>>, writer: &mut W) -> SafeTensorsResult<()> {
    let mut header_map = BTreeMap::new();
    let mut current_offset = 0;
    let tensors_ordered: BTreeMap<&String, &DynTensor> = tensors.iter().collect();

    // get tensor info
    for (name, tensor) in tensors_ordered.iter() {
        let n_elements = tensor.shape().element_count();
        let dtype_size = tensor.dtype().size_of();
        let n_bytes = n_elements * dtype_size;
        
        let info = SafeTensorsInfo {
            dtype: tensor.dtype().into(),
            shape: tensor.shape().dims().to_vec(),
            data_offsets: (current_offset, current_offset + n_bytes),
        };

        current_offset += n_bytes;
        header_map.insert(name.as_str(), info);
    }

    // insert metadata
    let mut header_value = serde_json::to_value(&header_map)?;    
    if let Some(metadata) = metadata {
        let meta_value = serde_json::to_value(metadata)?;
        if let Some(obj) = header_value.as_object_mut() {
            obj.insert("__metadata__".to_string(), meta_value);
        }
    }

    // header len
    let header_bytes = serde_json::to_vec(&header_value)?;
    let header_size_u64 = header_bytes.len() as u64; 
    let header_size_bytes = header_size_u64.to_le_bytes();

    // write header
    writer.write_all(&header_size_bytes)?;
    writer.write_all(&header_bytes)?;

    // write data iter
    for (_, tensor) in tensors_ordered {
        utils::write_tensor(writer, tensor)?;
    }
    writer.flush()?;

    Ok(())
}

impl SafeTensorsContent {
    pub fn save_file<P: AsRef<Path>>(&self, path: P) -> SafeTensorsResult<()> {
        save_file(&self.tensors, self.metadata.as_ref(), path)
    }
}

#[thiserrorctx::Error]
pub enum SafeTensorsError {
    #[error(transparent)]
    Lumen(#[from] lumen_core::ErrorCtx),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Parse(#[from] std::array::TryFromSliceError),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error("invalid format {0}")]
    InvalidFormat(String),

    #[error("unsupport dtype {0:?}")]
    UnsupportDType(SafeTensorsDType),

    #[error("Data offset out of range, total {0}, but try get {1}")]
    DataOffsetOutOfRange(usize, usize),
}

#[cfg(test)]
mod test {
    use super::load_file;

    #[test]
    fn test_load() {
        println!("{:?}", std::env::current_dir().unwrap());
        let content = load_file("./bench/test1.safetensors").unwrap();
        for (name, tensor) in content.tensors {
            println!("{}", name);
            println!("{:?}", tensor.dtype());
        }
    }
}